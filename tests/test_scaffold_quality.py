from nolane_personal.scaffold_evaluation import (
    ScaffoldQualityEvidence,
    ScaffoldResourceEvidence,
    decide_scaffold_quality,
    decide_scaffold_resources,
)
from nolane_personal.scaffold_promotion import decide_scaffold_promotion


def _quality(**overrides):
    values = dict(
        test_examples=20,
        anchor_examples=8,
        total_layers=28,
        head_layers=4,
        tail_layers=3,
        remaining_qwen_layers=7,
        stages_completed=4,
        baseline_nll=2.0,
        l12_nll=1.66,
        scaffold_nll=1.67,
        improvement_vs_base=0.33,
        degradation_vs_l12=0.01,
        anchor_baseline_nll=1.5,
        anchor_scaffold_nll=1.53,
        anchor_nll_regression=0.03,
        cached_generation_passed=True,
        scan_equivalence_passed=True,
        multiscale_separation_passed=True,
        cortex_parameters=81_249,
        base_model_unchanged=True,
        base_gradients_seen=0,
    )
    values.update(overrides)
    return ScaffoldQualityEvidence(**values)


def _resources(**overrides):
    values = dict(
        prompts=8,
        total_layers=28,
        remaining_qwen_layers=7,
        baseline_median_ms=100.0,
        l12_median_ms=75.0,
        scaffold_median_ms=68.0,
        latency_ratio_vs_base=0.68,
        latency_ratio_vs_l12=0.907,
        cached_tokens_per_second=46.0,
        replay_safe_tokens_per_second=30.0,
        cached_speedup_vs_replay=1.53,
        artifact_bytes=500_000,
        cortex_parameters=81_249,
    )
    values.update(overrides)
    return ScaffoldResourceEvidence(**values)


def test_scaffold_quality_requires_thin_qwen_shell_and_multiscale_contract():
    assert decide_scaffold_quality(_quality()).status == "SHRINKING_SCAFFOLD_QUALITY_PASS"

    bad = decide_scaffold_quality(_quality(remaining_qwen_layers=12))
    assert "qwen_scaffold_too_large" in bad.reasons

    bad = decide_scaffold_quality(_quality(degradation_vs_l12=0.05))
    assert "l12_noninferiority_failed" in bad.reasons

    bad = decide_scaffold_quality(_quality(multiscale_separation_passed=False))
    assert "multiscale_separation_gate_failed" in bad.reasons


def test_scaffold_resource_and_promotion_bind_checkpoint_and_plan():
    resource = decide_scaffold_resources(_resources())
    assert resource["status"] == "SHRINKING_SCAFFOLD_RESOURCE_PASS"

    bad = decide_scaffold_resources(_resources(latency_ratio_vs_l12=0.99))
    assert "latency_gain_vs_l12_failed" in bad["reasons"]

    promoted = decide_scaffold_promotion(
        quality_status="SHRINKING_SCAFFOLD_QUALITY_PASS",
        quality_checkpoint_sha256="checkpoint",
        quality_plan_sha256="plan",
        resource_status="SHRINKING_SCAFFOLD_RESOURCE_PASS",
        resource_checkpoint_sha256="checkpoint",
        resource_plan_sha256="plan",
    )
    assert promoted.status == "SHRINKING_SCAFFOLD_PROMOTION_PASS"

    blocked = decide_scaffold_promotion(
        quality_status="SHRINKING_SCAFFOLD_QUALITY_PASS",
        quality_checkpoint_sha256="a",
        quality_plan_sha256="plan-a",
        resource_status="SHRINKING_SCAFFOLD_RESOURCE_PASS",
        resource_checkpoint_sha256="b",
        resource_plan_sha256="plan-b",
    )
    assert "court_checkpoint_mismatch" in blocked.reasons
    assert "court_plan_mismatch" in blocked.reasons
