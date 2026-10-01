from nolane_personal.native_evaluation import (
    NativeQualityEvidence,
    NativeResourceEvidence,
    decide_native_quality,
    decide_native_resources,
)
from nolane_personal.native_promotion import decide_native_promotion


def _quality(**overrides):
    values = dict(
        test_examples=20,
        anchor_examples=8,
        qwen_decoder_layers_total=28,
        qwen_decoder_calls_forward=0,
        qwen_decoder_calls_generation=0,
        baseline_nll=2.0,
        l14_nll=1.66,
        native_nll=1.67,
        improvement_vs_base=0.33,
        degradation_vs_l14=0.01,
        anchor_baseline_nll=1.5,
        anchor_native_nll=1.54,
        anchor_nll_regression=0.04,
        prompt_scan_equivalence_passed=True,
        native_generation_passed=True,
        cortex_parameters=89_805,
        qwen_model_unchanged=True,
        qwen_gradients_seen=0,
    )
    values.update(overrides)
    return NativeQualityEvidence(**values)


def _resources(**overrides):
    values = dict(
        prompts=8,
        baseline_median_ms=100.0,
        l14_median_ms=55.0,
        native_median_ms=35.0,
        latency_ratio_vs_base=0.35,
        latency_ratio_vs_l14=0.636,
        native_tokens_per_second=50.0,
        artifact_bytes=500_000,
        cortex_parameters=89_805,
        qwen_decoder_calls=0,
    )
    values.update(overrides)
    return NativeResourceEvidence(**values)


def test_native_quality_requires_zero_decoder_execution():
    assert decide_native_quality(_quality()).status == "NATIVE_BOUNDARY_QUALITY_PASS"
    assert "qwen_decoder_forward_executed" in decide_native_quality(
        _quality(qwen_decoder_calls_forward=1)
    ).reasons
    assert "qwen_decoder_generation_executed" in decide_native_quality(
        _quality(qwen_decoder_calls_generation=1)
    ).reasons
    assert "l14_noninferiority_failed" in decide_native_quality(
        _quality(degradation_vs_l14=0.05)
    ).reasons


def test_native_resource_and_promotion_bind_checkpoint_and_spec():
    assert decide_native_resources(_resources())["status"] == "NATIVE_BOUNDARY_RESOURCE_PASS"
    assert "qwen_decoder_execution_detected" in decide_native_resources(
        _resources(qwen_decoder_calls=1)
    )["reasons"]

    promoted = decide_native_promotion(
        quality_status="NATIVE_BOUNDARY_QUALITY_PASS",
        quality_checkpoint_sha256="checkpoint",
        quality_spec_sha256="spec",
        resource_status="NATIVE_BOUNDARY_RESOURCE_PASS",
        resource_checkpoint_sha256="checkpoint",
        resource_spec_sha256="spec",
    )
    assert promoted.status == "NATIVE_BOUNDARY_PROMOTION_PASS"

    blocked = decide_native_promotion(
        quality_status="NATIVE_BOUNDARY_QUALITY_PASS",
        quality_checkpoint_sha256="a",
        quality_spec_sha256="x",
        resource_status="NATIVE_BOUNDARY_RESOURCE_PASS",
        resource_checkpoint_sha256="b",
        resource_spec_sha256="y",
    )
    assert "court_checkpoint_mismatch" in blocked.reasons
    assert "court_spec_mismatch" in blocked.reasons
