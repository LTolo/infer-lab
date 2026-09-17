"""FastAPI serving layer.

Endpoints are the ones an SRE expects to find, not just the fun one:

    POST /generate          run a prompt through the engine
    POST /tokenize          byte-tokenizer round-trip helper
    GET  /health            liveness  -- is the process up?
    GET  /ready             readiness -- is the engine actually able to serve?
    GET  /metrics           Prometheus text exposition
    GET  /stats             deep engine/scheduler/KV introspection (debugging)
    GET  /info              static build + config metadata
    GET  /kernels           which kernel backends this machine actually has
    POST /admin/reset_prefix_cache
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from infer_lab import __version__
from infer_lab.config import EngineConfig, ModelConfig
from infer_lab.engine.llm_engine import LLMEngine
from infer_lab.engine.request import SamplingParams
from infer_lab.kernels.registry import get_registry
from infer_lab.model.tokenizer import ByteTokenizer
from infer_lab.server.engine_runner import EngineRunner
from infer_lab.server.metrics import get_metrics
from infer_lab.utils.logging_conf import get_logger, setup_logging

logger = get_logger(__name__)
REQUEST_TIMEOUT_S = float(os.environ.get("INFER_LAB_REQUEST_TIMEOUT", "60"))


# --------------------------------------------------------------------------- schemas
class GenerateRequest(BaseModel):
    prompt: str | None = Field(default=None, description="UTF-8 text prompt")
    prompt_token_ids: list[int] | None = Field(default=None, description="Pre-tokenized ids")
    max_tokens: int = Field(default=32, ge=1, le=2048)
    temperature: float = Field(default=0.0, ge=0.0, le=5.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    seed: int | None = None
    ignore_eos: bool = True


class GenerateResponse(BaseModel):
    request_id: str
    text: str
    output_token_ids: list[int]
    prompt_tokens: int
    output_tokens: int
    finish_reason: str | None
    metrics: dict[str, float]


class TokenizeRequest(BaseModel):
    text: str
    add_bos: bool = True


# --------------------------------------------------------------------------- app
def create_app(model_config: ModelConfig | None = None,
               engine_config: EngineConfig | None = None) -> FastAPI:
    setup_logging()
    metrics = get_metrics()
    tokenizer = ByteTokenizer()

    model_config = model_config or ModelConfig()
    engine_config = engine_config or EngineConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = LLMEngine(model_config, engine_config)
        runner = EngineRunner(engine)
        runner.start()
        app.state.engine = engine
        app.state.runner = runner
        app.state.started_at = time.time()
        metrics.build_info.set(1.0, version=__version__, backend="numpy")
        logger.info("infer-lab server ready",
                    extra={"version": __version__,
                           "params": model_config.param_count(),
                           "kv_pool_mib": engine.kv_cache.snapshot()["pool_mib"]})
        try:
            yield
        finally:
            # Graceful shutdown: in-flight requests are drained, not dropped.
            runner.stop()

    app = FastAPI(
        title="infer-lab",
        version=__version__,
        description="LLM inference engine laboratory: paged KV, continuous batching, "
                    "prefix caching, speculative decoding, kernels and observability.",
        lifespan=lifespan,
    )

    # ----------------------------------------------------------------- middleware
    @app.middleware("http")
    async def access_log(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - started) * 1000.0
        response.headers["X-Response-Time-ms"] = f"{duration_ms:.2f}"
        if request.url.path not in ("/metrics", "/health", "/ready"):
            logger.info("http request", extra={
                "method": request.method, "path": request.url.path,
                "status": response.status_code, "duration_ms": round(duration_ms, 3),
            })
        return response

    # ------------------------------------------------------------------- generate
    @app.post("/generate", response_model=GenerateResponse)
    async def generate(body: GenerateRequest) -> GenerateResponse:
        if body.prompt is None and not body.prompt_token_ids:
            raise HTTPException(400, "provide either 'prompt' or 'prompt_token_ids'")
        # Check the raw field before tokenization. Afterwards it is too late: the
        # tokenizer prepends BOS, so an empty string becomes a valid one-token
        # prompt and the emptiness check below can never fire.
        if body.prompt_token_ids is None and not (body.prompt or "").strip():
            raise HTTPException(400, "prompt must not be empty or whitespace only")
        token_ids = body.prompt_token_ids or tokenizer.encode(body.prompt or "")
        if not token_ids:
            raise HTTPException(400, "prompt is empty after tokenization")
        # Bound the request before it reaches the scheduler. Pydantic validates
        # the shape of each field; these are the cross-field and domain rules it
        # cannot express.
        if len(token_ids) > engine_config.max_model_len:
            raise HTTPException(
                400,
                f"prompt has {len(token_ids)} tokens, exceeding max_model_len "
                f"{engine_config.max_model_len}",
            )
        if body.top_k and body.top_k > model_config.vocab_size:
            raise HTTPException(
                400,
                f"top_k {body.top_k} exceeds vocab_size {model_config.vocab_size}",
            )

        sampling = SamplingParams(
            max_tokens=body.max_tokens, temperature=body.temperature,
            top_p=body.top_p, top_k=body.top_k, seed=body.seed,
            ignore_eos=body.ignore_eos,
        )
        runner: EngineRunner = app.state.runner
        try:
            pending = runner.submit(token_ids, sampling)
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc

        loop = asyncio.get_running_loop()
        try:
            output = await loop.run_in_executor(
                None, runner.wait, pending, REQUEST_TIMEOUT_S
            )
        except TimeoutError as exc:
            metrics.requests_total.inc(outcome="timeout")
            raise HTTPException(504, str(exc)) from exc
        except RuntimeError as exc:
            metrics.requests_total.inc(outcome="error")
            raise HTTPException(400, str(exc)) from exc

        metrics.observe_request(
            outcome="success",
            ttft_ms=output.metrics.get("ttft_ms", float("nan")),
            e2e_ms=output.metrics.get("e2e_ms", float("nan")),
            tpot_ms=output.metrics.get("tpot_ms", float("nan")),
            prompt_tokens=len(output.prompt_token_ids),
            output_tokens=output.num_generated,
        )
        metrics.observe_engine(app.state.engine)

        return GenerateResponse(
            request_id=output.request_id,
            text=tokenizer.decode(output.output_token_ids),
            output_token_ids=output.output_token_ids,
            prompt_tokens=len(output.prompt_token_ids),
            output_tokens=output.num_generated,
            finish_reason=output.finish_reason,
            metrics={k: (v if v == v else -1.0) for k, v in output.metrics.items()},
        )

    @app.post("/tokenize")
    async def tokenize(body: TokenizeRequest) -> dict[str, Any]:
        ids = tokenizer.encode(body.text, add_bos=body.add_bos)
        return {"token_ids": ids, "num_tokens": len(ids),
                "roundtrip": tokenizer.decode(ids)}

    # --------------------------------------------------------------------- probes
    @app.get("/health", response_class=JSONResponse)
    async def health() -> dict[str, Any]:
        """Liveness: cheap, never touches the engine."""
        return {"status": "ok", "version": __version__,
                "uptime_s": round(time.time() - app.state.started_at, 2)}

    @app.get("/ready", response_class=JSONResponse)
    async def ready() -> JSONResponse:
        """Readiness: fails while draining so load balancers stop sending traffic."""
        runner: EngineRunner = app.state.runner
        payload = runner.snapshot()
        code = 200 if runner.is_ready and not payload["draining"] else 503
        return JSONResponse(payload, status_code=code)

    # -------------------------------------------------------------- observability
    @app.get("/metrics", response_class=PlainTextResponse)
    async def prometheus_metrics() -> PlainTextResponse:
        engine: LLMEngine = app.state.engine
        metrics.observe_engine(engine)
        sched = engine.scheduler.stats
        metrics.preemptions_total.values[()] = float(sched.preemptions)
        metrics.prefix_cache_hits.values[()] = float(sched.prefix_cache_hit_tokens)
        metrics.engine_steps_total.values[()] = float(sched.steps)
        return PlainTextResponse(metrics.render(),
                                 media_type="text/plain; version=0.0.4; charset=utf-8")

    @app.get("/stats")
    async def stats() -> dict[str, Any]:
        engine: LLMEngine = app.state.engine
        runner: EngineRunner = app.state.runner
        return {"engine": engine.stats(), "runner": runner.snapshot()}

    @app.get("/info")
    async def info() -> dict[str, Any]:
        engine: LLMEngine = app.state.engine
        return {
            "version": __version__,
            "model": model_config.to_dict(),
            "param_count": model_config.param_count(),
            "engine_config": engine_config.to_dict(),
            "kv_cache": engine.kv_cache.snapshot(),
            "request_timeout_s": REQUEST_TIMEOUT_S,
        }

    @app.get("/kernels")
    async def kernels() -> dict[str, Any]:
        return get_registry().report()

    @app.post("/admin/reset_prefix_cache")
    async def reset_prefix_cache() -> dict[str, Any]:
        engine: LLMEngine = app.state.engine
        if engine.radix_cache is None:
            raise HTTPException(400, "prefix caching is disabled")
        before = engine.radix_cache.num_cached_blocks
        engine.radix_cache.reset()
        return {"released_blocks": before,
                "kv_cache": engine.kv_cache.snapshot()}

    return app


app = create_app()
