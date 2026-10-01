from nolane_personal.state_space_evaluation import (
    StateSpaceQualityEvidence,
    StateSpaceResourceEvidence,
    decide_state_space_quality,
    decide_state_space_resources,
)
from nolane_personal.state_space_promotion import decide_state_space_promotion


def _quality(**overrides):
    values = dict(
        test_examples=20,
        anchor_examples=8,
        total_layers=28,
        replaced_layers=17,
        stages_completed=4,
        baseline_nll=2.0,
        l11_nll=1.65,
        state_space_nll=1.66,
        improvement_vs_base=0.34,
        degradation_vs_l11=0.01,
        anchor_baseline_nll=1.5,
        anchor_state_space_nll=1.53,
        anchor_nll_regression=0.03,
        cached_generation_passed=True,
        scan_equivalence_passed=True,
        cortex_parameters=72_897,
        base_model_unchanged=True,
        base_gradients_seen=0,
    )
    values.update(overrides)
    return StateSpaceQualityEvidence(**values)


def _resources(**overrides):
    values = dict(
        prompts=8,
        total_layers=28,
        replaced_layers=17,
        baseline_median_ms=100.0,
        l11_median_ms=82.0,
        state_space_median_ms=75.0,
        latency_ratio_vs_base=0.75,
        latency_ratio_vs_l11=0.915,
        cached_tokens_per_second=42.0,
        replay_safe_tokens_per_second=28.0,
        cached_speedup_vs_replay=1.50,
        artifact_bytes=400_000,
        cortex_parameters=72_897,
    )
    values.update(overrides)
    return StateSpaceResourceEvidence(**values)


def test_state_space_quality_requires_major_depth_scan_and_l11_noninferiority():
    assert decide_state_space_quality(_quality()).status == "STATE_SPACE_CORTEX_QUALITY_PASS"

    bad = decide_state_space_quality(_quality(replaced_layers=8))
    assert "insufficient_transformer_depth_replaced" in bad.reasons

    bad = decide_state_space_quality(_quality(degradation_vs_l11=0.05))
    assert "l11_noninferiority_failed" in bad.reasons

    bad = decide_state_space_quality(_quality(scan_equivalence_passed=False))
    assert "state_space_scan_equivalence_failed" in bad.reasons


def test_state_space_resource_and_promotion_bind_checkpoint_and_plan():
    resource = decide_state_space_resources(_resources())
    assert resource["status"] == "STATE_SPACE_CORTEX_RESOURCE_PASS"

    bad = decide_state_space_resources(_resources(latency_ratio_vs_l11=1.0))
    assert "latency_gain_vs_l11_failed" in bad["reasons"]

    promoted = decide_state_space_promotion(
        quality_status="STATE_SPACE_CORTEX_QUALITY_PASS",
        quality_checkpoint_sha256="checkpoint",
        quality_plan_sha256="plan",
        resource_status="STATE_SPACE_CORTEX_RESOURCE_PASS",
        resource_checkpoint_sha256="checkpoint",
        resource_plan_sha256="plan",
    )
    assert promoted.status == "STATE_SPACE_CORTEX_PROMOTION_PASS"

    blocked = decide_state_space_promotion(
        quality_status="STATE_SPACE_CORTEX_QUALITY_PASS",
        quality_checkpoint_sha256="a",
        quality_plan_sha256="plan-a",
        resource_status="STATE_SPACE_CORTEX_RESOURCE_PASS",
        resource_checkpoint_sha256="b",
        resource_plan_sha256="plan-b",
    )
    assert "court_checkpoint_mismatch" in blocked.reasons
    assert "court_plan_mismatch" in blocked.reasons
