"""Numeric instability detection."""

from __future__ import annotations

import numpy as np

from infer_lab.debug.instability import InstabilityDetector, NumericIssue, inject_fault
from infer_lab.model.weights import ModelWeights


def test_healthy_activations_raise_nothing(rng):
    detector = InstabilityDetector()
    for i in range(4):
        detector.observe(f"layer{i}", rng.normal(size=(8, 32)).astype(np.float32), i)
    assert not detector.has_issues
    assert detector.summary()["healthy"] is True


def test_nan_is_detected_and_localised(rng):
    detector = InstabilityDetector()
    detector.observe("layer0", rng.normal(size=(4, 8)).astype(np.float32), 0)
    bad = rng.normal(size=(4, 8)).astype(np.float32)
    bad[1, 2] = np.nan
    detector.observe("layer1", bad, 1)
    detector.observe("layer2", rng.normal(size=(4, 8)).astype(np.float32), 2)

    first = detector.first_bad_layer()
    assert first is not None and first.name == "layer1"
    assert NumericIssue.NAN in first.issues


def test_inf_is_detected():
    detector = InstabilityDetector()
    x = np.ones((2, 4), dtype=np.float32)
    x[0, 0] = np.inf
    assert NumericIssue.INF in detector.observe("l", x).issues


def test_fp16_overflow_is_flagged_even_though_fp32_is_fine():
    """The bug that only appears when you switch the model to half precision."""
    detector = InstabilityDetector()
    x = np.full((2, 4), 1e5, dtype=np.float32)
    report = detector.observe("l", x)
    assert np.isfinite(report.abs_max)
    assert NumericIssue.FP16_OVERFLOW in report.issues


def test_dead_activation_is_flagged():
    detector = InstabilityDetector()
    assert NumericIssue.DEAD_ACTIVATION in \
        detector.observe("l", np.zeros((4, 8), dtype=np.float32)).issues


def test_magnitude_spike_is_flagged(rng):
    detector = InstabilityDetector(spike_factor=10.0, fp16_check=False)
    detector.observe("l0", rng.normal(size=(8, 8)).astype(np.float32) * 0.1)
    report = detector.observe("l1", np.full((8, 8), 50.0, dtype=np.float32))
    assert NumericIssue.MAGNITUDE_SPIKE in report.issues


def test_hook_is_transparent(rng):
    detector = InstabilityDetector()
    x = rng.normal(size=(4, 8)).astype(np.float32)
    np.testing.assert_array_equal(detector.hook("l0", 0)(x), x)
    assert len(detector.reports) == 1


def test_injected_fault_is_caught_on_a_real_model(model_config):
    """End-to-end: corrupt a weight matrix, confirm the detector names the layer."""
    weights = ModelWeights.random(model_config, seed=0)
    inject_fault(weights, layer_index=1, kind="nan")

    detector = InstabilityDetector()
    for i, layer in enumerate(weights.layers):
        detector.observe(f"layer{i}.wq", layer.wq, i)

    first = detector.first_bad_layer()
    assert first is not None and first.layer_index == 1


def test_overflow_fault_is_caught(model_config):
    weights = ModelWeights.random(model_config, seed=0)
    inject_fault(weights, layer_index=0, kind="overflow", scale=1e6)
    detector = InstabilityDetector()
    report = detector.observe("layer0.wq", weights.layers[0].wq, 0)
    assert NumericIssue.FP16_OVERFLOW in report.issues


def test_detector_can_be_disabled(rng):
    detector = InstabilityDetector()
    detector.enabled = False
    detector.hook("l0")(np.full((2, 2), np.nan, dtype=np.float32))
    assert not detector.reports
