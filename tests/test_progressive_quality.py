from nolane_personal.progressive_evaluation import (
    ProgressiveQualityEvidence,
    ProgressiveResourceEvidence,
    decide_progressive_quality,
    decide_progressive_resources,
)
from nolane_personal.progressive_promotion import decide_progressive_promotion


def _quality(**overrides):
    values = dict(
        test_examples=20,
        anchor_examples=8,
        total_layers=28,
        replaced_layers=8,
        stages_completed=4,
        baseline_nll=2.0,
        l9_nll=1.65,
        progressive_nll=1.64,
        improvement_vs_base=0.36,
        degradation_vs_l9=-0.01,
        anchor_baseline_nll=1.5,
        anchor_progressive_nll=1.53,
        anchor_nll_regression=0.03,
        replacement_parameters=84_289,
        cached_generation_passed=True,
        base_model_unchanged=True,
        base_gradients_seen=0,
    )
    values.update(overrides)
    return ProgressiveQualityEvidence(**values)


def _resources(**overrides):
    values = dict(
        prompts=8,
        total_layers=28,
        replaced_layers=8,
        baseline_median_ms=100.0,
        progressive_median_ms=80.0,
        latency_ratio_vs_base=0.80,
        cached_tokens_per_second=35.0,
        replay_safe_tokens_per_second=20.0,
        cached_speedup_vs_replay=1.75,
        artifact_bytes=500_000,
        replacement_parameters=84_289,
    )
    values.update(overrides)
    return ProgressiveResourceEvidence(**values)


def test_progressive_quality_requires_meaningful_depth_and_cache_path():
    assert decide_progressive_quality(_quality()).status == "PROGRESSIVE_REPLACEMENT_QUALITY_PASS"

    bad = decide_progressive_quality(_quality(replaced_layers=3))
    assert "insufficient_transformer_depth_replaced" in bad.reasons

    bad = decide_progressive_quality(_quality(cached_generation_passed=False))
    assert "cached_generation_gate_failed" in bad.reasons

    bad = decide_progressive_quality(_quality(degradation_vs_l9=0.05))
    assert "l9_noninferiority_failed" in bad.reasons


def test_progressive_resource_and_promotion_require_same_checkpoint_and_plan():
    resource = decide_progressive_resources(_resources())
    assert resource["status"] == "PROGRESSIVE_REPLACEMENT_RESOURCE_PASS"

    bad = decide_progressive_resources(_resources(cached_speedup_vs_replay=1.0))
    assert "cached_generation_speed_gate_failed" in bad["reasons"]

    promoted = decide_progressive_promotion(
        quality_status="PROGRESSIVE_REPLACEMENT_QUALITY_PASS",
        quality_checkpoint_sha256="checkpoint",
        quality_plan_sha256="plan",
        resource_status="PROGRESSIVE_REPLACEMENT_RESOURCE_PASS",
        resource_checkpoint_sha256="checkpoint",
        resource_plan_sha256="plan",
    )
    assert promoted.status == "PROGRESSIVE_REPLACEMENT_PROMOTION_PASS"

    blocked = decide_progressive_promotion(
        quality_status="PROGRESSIVE_REPLACEMENT_QUALITY_PASS",
        quality_checkpoint_sha256="checkpoint-a",
        quality_plan_sha256="plan-a",
        resource_status="PROGRESSIVE_REPLACEMENT_RESOURCE_PASS",
        resource_checkpoint_sha256="checkpoint-b",
        resource_plan_sha256="plan-b",
    )
    assert "court_checkpoint_mismatch" in blocked.reasons
    assert "court_plan_mismatch" in blocked.reasons
