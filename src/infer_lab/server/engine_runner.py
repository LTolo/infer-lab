"""Background engine driver.

The engine loop is synchronous and CPU-bound; the HTTP layer is async.  Bridging
them with a dedicated engine thread is the same architecture vLLM uses
(``AsyncLLMEngine``): the event loop never blocks on a forward pass, and the
engine never has to know what asyncio is.

Only ONE thread ever touches the engine, so no locking is needed around the
model or the KV cache -- submissions go through a queue instead.  That is a
deliberate design choice: KV-cache mutation from two threads would be the kind
of bug that shows up once a week in production and never in a test.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field

from infer_lab.engine.llm_engine import EngineOutput, LLMEngine
from infer_lab.engine.request import SamplingParams
from infer_lab.utils.logging_conf import get_logger

logger = get_logger(__name__)


@dataclass
class PendingRequest:
    request_id: str
    done: threading.Event = field(default_factory=threading.Event)
    output: EngineOutput | None = None
    error: str | None = None


class EngineRunner:
    """Owns the engine thread and the request/response plumbing."""

    def __init__(self, engine: LLMEngine, *, idle_sleep_s: float = 0.001) -> None:
        self.engine = engine
        self.idle_sleep_s = idle_sleep_s
        self._submissions: queue.Queue[tuple[list[int], SamplingParams, PendingRequest]] = \
            queue.Queue()
        self._pending: dict[str, PendingRequest] = {}
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()
        self._ready = threading.Event()
        self.started_at: float | None = None
        self.steps = 0

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stopping.clear()
        self._thread = threading.Thread(target=self._loop, name="infer-lab-engine",
                                        daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)
        self.started_at = time.time()
        logger.info("engine runner started")

    def stop(self, timeout: float = 10.0) -> None:
        """Graceful shutdown: stop admitting, drain in-flight work, then exit."""
        if self._thread is None:
            return
        logger.info("engine runner draining", extra={"pending": len(self._pending)})
        self._stopping.set()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logger.error("engine thread did not stop within timeout")
        else:
            logger.info("engine runner stopped", extra={"steps": self.steps})
        self._thread = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set() and self.is_running

    # ---------------------------------------------------------------- submission
    def submit(self, prompt_token_ids: list[int],
               sampling: SamplingParams) -> PendingRequest:
        if self._stopping.is_set():
            raise RuntimeError("server is shutting down and is not accepting new requests")
        pending = PendingRequest(request_id="")
        self._submissions.put((list(prompt_token_ids), sampling, pending))
        return pending

    def wait(self, pending: PendingRequest, timeout: float) -> EngineOutput:
        if not pending.done.wait(timeout=timeout):
            if pending.request_id:
                self.engine.abort_request(pending.request_id)
            raise TimeoutError(f"request timed out after {timeout}s")
        if pending.error:
            raise RuntimeError(pending.error)
        assert pending.output is not None
        return pending.output

    # ---------------------------------------------------------------------- loop
    def _loop(self) -> None:
        self._ready.set()
        while True:
            self._drain_submissions()

            if not self.engine.has_unfinished_requests:
                if self._stopping.is_set() and self._submissions.empty():
                    break
                time.sleep(self.idle_sleep_s)
                continue

            try:
                outputs = self.engine.step()
                self.steps += 1
                if not outputs:
                    # A step that emitted nothing means the engine is blocked on
                    # memory rather than doing work.  Looping at full speed would
                    # pin a core and drive the step counter into the millions
                    # without producing a single token.
                    time.sleep(self.idle_sleep_s)
            except Exception as exc:  # noqa: BLE001 - one bad request must not kill the loop
                logger.exception("engine step failed")
                self._fail_all(f"{type(exc).__name__}: {exc}")
                continue

            for output in outputs:
                if not output.finished:
                    continue
                pending = self._pending.pop(output.request_id, None)
                if pending is not None:
                    pending.output = output
                    pending.done.set()

    def _drain_submissions(self) -> None:
        # While stopping we stop admitting new work but still finish what is in flight.
        while not self._submissions.empty():
            prompt, sampling, pending = self._submissions.get()
            if self._stopping.is_set():
                pending.error = "server is shutting down"
                pending.done.set()
                continue
            try:
                request_id = self.engine.add_request(prompt, sampling)
            except Exception as exc:  # noqa: BLE001 - e.g. prompt too long
                pending.error = f"{type(exc).__name__}: {exc}"
                pending.done.set()
                continue
            pending.request_id = request_id
            self._pending[request_id] = pending

    def _fail_all(self, message: str) -> None:
        for pending in list(self._pending.values()):
            pending.error = message
            pending.done.set()
        self._pending.clear()

    def snapshot(self) -> dict[str, object]:
        return {
            "running": self.is_running,
            "ready": self.is_ready,
            "draining": self._stopping.is_set(),
            "steps": self.steps,
            "pending": len(self._pending),
            "uptime_s": round(time.time() - self.started_at, 2) if self.started_at else 0.0,
        }
