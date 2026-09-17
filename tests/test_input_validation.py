"""Input validation at the API and engine boundary.

These exist because a single out-of-range token id used to reach the embedding
lookup and raise ``IndexError`` from inside the engine thread -- reported to the
caller as a 500, and only *after* the request had been admitted and KV
allocated.

The rule this file pins down: a malformed request is rejected at the boundary,
cheaply, with a 4xx, and never allowed to affect a request that is already in
flight.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from infer_lab.config import EngineConfig, ModelConfig
from infer_lab.engine.llm_engine import LLMEngine
from infer_lab.engine.request import SamplingParams
from infer_lab.server.api import create_app


@pytest.fixture
def client(model_config: ModelConfig, engine_config: EngineConfig):
    with TestClient(create_app(model_config, engine_config)) as c:
        yield c


def greedy(max_tokens: int = 4) -> SamplingParams:
    return SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)


# ------------------------------------------------------------- engine level
def test_token_id_above_vocab_is_rejected(engine, model_config):
    with pytest.raises(ValueError, match="out of range"):
        engine.add_request([1, 2, model_config.vocab_size], greedy())


def test_negative_token_id_is_rejected(engine):
    with pytest.raises(ValueError, match="negative"):
        engine.add_request([1, -5, 3], greedy())


def test_non_integer_token_id_is_rejected(engine):
    with pytest.raises(ValueError, match="integers"):
        engine.add_request([1, 2.5, 3], greedy())  # type: ignore[list-item]


def test_empty_prompt_is_rejected(engine):
    with pytest.raises(ValueError, match="at least one token"):
        engine.add_request([], greedy())


def test_highest_valid_token_id_is_accepted(engine, model_config):
    """Off-by-one guard: vocab_size - 1 must be valid."""
    engine.add_request([model_config.vocab_size - 1], greedy())


def test_rejection_allocates_no_kv(engine, model_config):
    """A rejected request must leave the KV pool untouched.

    This is the actual point of validating early: failing after admission would
    strand blocks and, in the original bug, poison the whole engine step.
    """
    free_before = engine.kv_cache.allocator.num_free
    with pytest.raises(ValueError):
        engine.add_request([model_config.vocab_size + 10], greedy())
    assert engine.kv_cache.allocator.num_free == free_before


def test_rejection_does_not_disturb_other_requests(model_config, engine_config):
    """A malformed request must not affect one that is already in flight."""
    engine = LLMEngine(model_config, engine_config)
    good = engine.add_request([1, 2, 3, 4], greedy(6))

    with pytest.raises(ValueError):
        engine.add_request([model_config.vocab_size + 1], greedy())

    final = None
    while engine.has_unfinished_requests:
        for out in engine.step():
            if out.finished and out.request_id == good:
                final = out
    assert final is not None and final.num_generated == 6


# ---------------------------------------------------------------- API level
def test_api_rejects_out_of_range_token_id(client, model_config):
    response = client.post("/generate", json={
        "prompt_token_ids": [1, 2, model_config.vocab_size + 100], "max_tokens": 4})
    assert response.status_code == 400
    assert "out of range" in response.text


def test_api_rejects_negative_token_id(client):
    response = client.post("/generate", json={
        "prompt_token_ids": [1, -1], "max_tokens": 4})
    assert response.status_code == 400


def test_api_rejects_oversized_prompt(client, engine_config):
    response = client.post("/generate", json={
        "prompt_token_ids": list(range(engine_config.max_model_len + 5)),
        "max_tokens": 4})
    assert response.status_code == 400


def test_api_rejects_top_k_above_vocab(client, model_config):
    response = client.post("/generate", json={
        "prompt": "hi", "max_tokens": 4, "temperature": 1.0,
        "top_k": model_config.vocab_size + 1})
    assert response.status_code == 400


@pytest.mark.parametrize("payload", [
    {"prompt": "hi", "max_tokens": 0},        # below the schema minimum
    {"prompt": "hi", "max_tokens": -1},
    {"prompt": "hi", "max_tokens": 4, "temperature": -1.0},
    {"prompt": "hi", "max_tokens": 4, "top_p": 1.5},
])
def test_api_rejects_invalid_sampling_params(client, payload):
    """Pydantic handles per-field bounds; assert the contract, not the mechanism."""
    assert client.post("/generate", json=payload).status_code == 422


def test_api_never_returns_500_for_bad_input(client, model_config):
    """The core guarantee: client mistakes are 4xx, never 5xx."""
    bad_payloads = [
        {"prompt_token_ids": [model_config.vocab_size], "max_tokens": 4},
        {"prompt_token_ids": [-1], "max_tokens": 4},
        {"prompt_token_ids": [], "max_tokens": 4},
        {"max_tokens": 4},
        {"prompt": "", "max_tokens": 4},
    ]
    for payload in bad_payloads:
        status = client.post("/generate", json=payload).status_code
        assert 400 <= status < 500, f"{payload} returned {status}"


def test_server_still_serves_after_rejections(client):
    """Rejections must not leave the engine in a degraded state."""
    for _ in range(5):
        client.post("/generate", json={"prompt_token_ids": [99999], "max_tokens": 4})
    body = client.post("/generate", json={"prompt": "ok", "max_tokens": 4}).json()
    assert body["output_tokens"] == 4
