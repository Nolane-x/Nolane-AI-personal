from nolane_personal.island_evaluation import (
    IslandQualityEvidence,
    IslandResourceEvidence,
    decide_island_quality,
    decide_island_resources,
)
from nolane_personal.island_promotion import decide_island_promotion


def _quality(**overrides):
    values = dict(
        test_examples=20,
        anchor_examples=8,
        total_layers=28,
        replaced_layers=8,
        island_count=3,
        stages_completed=3,
        baseline_nll=2.0,
        l10_nll=1.64,
        island_nll=1.65,
        improvement_vs_base=0.35,
        degradation_vs_l10=0.01,
        anchor_baseline_nll=1.5,
        anchor_island_nll=1.53,
        anchor_nll_regression=0.03,
        cached_generation_passed=True,
        replacement_parameters=84_289,
        base_model_unchanged=True,
        base_gradients_seen=0,
    )
    values.update(overrides)
    return IslandQualityEvidence(**values)


def _resources(**overrides):
    values = dict(
        prompts=8,
        total_layers=28,
        replaced_layers=8,
        island_count=3,
        baseline_median_ms=100.0,
        l10_median_ms=86.0,
        island_median_ms=82.0,
        latency_ratio_vs_base=0.82,
        latency_ratio_vs_l10=0.953,
        cached_tokens_per_second=38.0,
        replay_safe_tokens_per_second=24.0,
        cached_speedup_vs_replay=1.58,
        artifact_bytes=500_000,
        replacement_parameters=84_289,
    )
    values.update(overrides)
    return IslandResourceEvidence(**values)


def test_island_quality_requires_real_region_compression_and_l10_noninferiority():
    assert decide_island_quality(_quality()).status == "RECURRENT_ISLAND_QUALITY_PASS"

    bad = decide_island_quality(_quality(replaced_layers=4, island_count=3))
    assert "insufficient_transformer_depth_replaced" in bad.reasons or "insufficient_region_compression" in bad.reasons

    bad = decide_island_quality(_quality(degradation_vs_l10=0.05))
    assert "l10_noninferiority_failed" in bad.reasons

    bad = decide_island_quality(_quality(cached_generation_passed=False))
    assert "cached_generation_gate_failed" in bad.reasons


def test_island_resource_and_promotion_bind_checkpoint_and_plan():
    resource = decide_island_resources(_resources())
    assert resource["status"] == "RECURRENT_ISLAND_RESOURCE_PASS"

    bad = decide_island_resources(_resources(latency_ratio_vs_l10=1.01))
    assert "latency_gain_vs_l10_failed" in bad["reasons"]

    promoted = decide_island_promotion(
        quality_status="RECURRENT_ISLAND_QUALITY_PASS",
        quality_checkpoint_sha256="checkpoint",
        quality_plan_sha256="plan",
        resource_status="RECURRENT_ISLAND_RESOURCE_PASS",
        resource_checkpoint_sha256="checkpoint",
        resource_plan_sha256="plan",
    )
    assert promoted.status == "RECURRENT_ISLAND_PROMOTION_PASS"

    blocked = decide_island_promotion(
        quality_status="RECURRENT_ISLAND_QUALITY_PASS",
        quality_checkpoint_sha256="a",
        quality_plan_sha256="plan-a",
        resource_status="RECURRENT_ISLAND_RESOURCE_PASS",
        resource_checkpoint_sha256="b",
        resource_plan_sha256="plan-b",
    )
    assert "court_checkpoint_mismatch" in blocked.reasons
    assert "court_plan_mismatch" in blocked.reasons
