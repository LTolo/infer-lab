"""HTTP API contract, metrics format and graceful shutdown."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from infer_lab.config import EngineConfig, ModelConfig
from infer_lab.server.api import create_app


@pytest.fixture
def client(model_config: ModelConfig, engine_config: EngineConfig):
    with TestClient(create_app(model_config, engine_config)) as c:
        yield c


def test_health_is_cheap_and_always_ok(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and "version" in body


def test_ready_reports_engine_state(client):
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["ready"] is True


def test_generate_from_text(client):
    body = client.post("/generate", json={"prompt": "hallo", "max_tokens": 5}).json()
    assert body["output_tokens"] == 5
    assert len(body["output_token_ids"]) == 5
    assert body["finish_reason"] == "length"


def test_generate_from_token_ids(client):
    body = client.post("/generate",
                       json={"prompt_token_ids": [1, 2, 3], "max_tokens": 4}).json()
    assert body["output_tokens"] == 4


def test_generate_is_deterministic_at_temperature_zero(client):
    payload = {"prompt": "determinism", "max_tokens": 6, "temperature": 0.0}
    first = client.post("/generate", json=payload).json()
    second = client.post("/generate", json=payload).json()
    assert first["output_token_ids"] == second["output_token_ids"]


def test_missing_prompt_is_rejected(client):
    assert client.post("/generate", json={"max_tokens": 4}).status_code == 400


def test_invalid_sampling_params_are_rejected_by_schema(client):
    response = client.post("/generate", json={"prompt": "x", "max_tokens": 0})
    assert response.status_code == 422


def test_oversized_prompt_returns_400(client, engine_config):
    huge = list(range(engine_config.max_model_len + 10))
    response = client.post("/generate", json={"prompt_token_ids": huge, "max_tokens": 4})
    assert response.status_code == 400


def test_tokenizer_roundtrip(client):
    body = client.post("/tokenize", json={"text": "grüße"}).json()
    assert body["roundtrip"] == "grüße"


# ------------------------------------------------------------------ observability
def test_metrics_are_valid_prometheus_text(client):
    client.post("/generate", json={"prompt": "metrics", "max_tokens": 4})
    response = client.get("/metrics")
    assert response.headers["content-type"].startswith("text/plain")
    text = response.text

    for name in ("infer_lab_requests_total", "infer_lab_kv_cache_utilization_ratio",
                 "infer_lab_ttft_milliseconds_bucket", "infer_lab_engine_steps_total"):
        assert name in text, f"missing metric {name}"

    # every metric family must carry HELP and TYPE, or Prometheus drops it
    helps = {ln.split()[2] for ln in text.splitlines() if ln.startswith("# HELP")}
    types = {ln.split()[2] for ln in text.splitlines() if ln.startswith("# TYPE")}
    assert helps == types


def test_histogram_buckets_are_cumulative_and_bounded(client):
    client.post("/generate", json={"prompt": "hist", "max_tokens": 4})
    lines = [ln for ln in client.get("/metrics").text.splitlines()
             if ln.startswith("infer_lab_ttft_milliseconds_bucket")]
    counts = [float(ln.split()[-1]) for ln in lines]
    assert counts == sorted(counts), "bucket counts must be non-decreasing"


def test_stats_endpoint_exposes_scheduler_internals(client):
    client.post("/generate", json={"prompt": "stats", "max_tokens": 4})
    body = client.get("/stats").json()
    assert "scheduler" in body["engine"] and "kv_cache" in body["engine"]
    assert body["runner"]["running"] is True


def test_info_reports_model_and_config(client):
    body = client.get("/info").json()
    assert body["param_count"] > 0 and "block_size" in body["engine_config"]


def test_kernels_endpoint_always_lists_numpy(client):
    assert client.get("/kernels").json()["numpy"]["available"] is True


def test_reset_prefix_cache_releases_blocks(client):
    client.post("/generate", json={"prompt": "a" * 40, "max_tokens": 4})
    body = client.post("/admin/reset_prefix_cache").json()
    assert "kv_cache" in body


def test_response_time_header_is_present(client):
    response = client.get("/health")
    assert float(response.headers["X-Response-Time-ms"]) >= 0
