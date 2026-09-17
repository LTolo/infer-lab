"""Shared fixtures.

Configs here are deliberately tiny: the whole suite must run in seconds on a
CPU-only laptop, otherwise nobody runs it before pushing.
"""

from __future__ import annotations

import numpy as np
import pytest

from infer_lab.config import EngineConfig, ModelConfig
from infer_lab.engine.llm_engine import LLMEngine
from infer_lab.kv.paged_cache import PagedKVCache
from infer_lab.model.numpy_model import NumpyTransformer
from infer_lab.model.weights import ModelWeights


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(vocab_size=300, hidden_size=64, intermediate_size=128,
                       num_layers=2, num_heads=4, num_kv_heads=2,
                       max_position_embeddings=512)


@pytest.fixture
def moe_config() -> ModelConfig:
    return ModelConfig(vocab_size=300, hidden_size=64, intermediate_size=128,
                       num_layers=2, num_heads=4, num_kv_heads=2,
                       max_position_embeddings=512,
                       num_experts=4, num_experts_per_tok=2)


@pytest.fixture
def engine_config() -> EngineConfig:
    return EngineConfig(block_size=8, num_gpu_blocks=128, max_num_seqs=8,
                        max_num_batched_tokens=64, max_model_len=256)


@pytest.fixture
def weights(model_config: ModelConfig) -> ModelWeights:
    return ModelWeights.random(model_config, seed=0)


@pytest.fixture
def model(weights: ModelWeights) -> NumpyTransformer:
    return NumpyTransformer(weights)


@pytest.fixture
def kv_cache(model_config: ModelConfig, engine_config: EngineConfig) -> PagedKVCache:
    return PagedKVCache(model_config, engine_config)


@pytest.fixture
def engine(model_config: ModelConfig, engine_config: EngineConfig) -> LLMEngine:
    return LLMEngine(model_config, engine_config)
