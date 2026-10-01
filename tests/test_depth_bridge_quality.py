from nolane_personal.depth_bridge_evaluation import (
    DepthBridgeQualityEvidence,
    decide_depth_bridge_quality,
)


def _evidence(**overrides):
    values = dict(
        test_examples=20,
        anchor_examples=8,
        baseline_nll=2.0,
        l6_nll=1.75,
        l7_hybrid_nll=1.68,
        depth_bridge_nll=1.60,
        improvement_vs_base=0.40,
        improvement_vs_l6=0.15,
        improvement_vs_l7_hybrid=0.08,
        anchor_baseline_nll=1.50,
        anchor_depth_bridge_nll=1.53,
        anchor_nll_regression=0.03,
        bridge_parameters=84_289,
        base_model_unchanged=True,
        base_gradients_seen=0,
    )
    values.update(overrides)
    return DepthBridgeQualityEvidence(**values)


def test_depth_bridge_must_beat_base_l6_and_l7_hybrid():
    assert decide_depth_bridge_quality(_evidence()).status == "DEPTH_BRIDGE_QUALITY_PASS"

    decision = decide_depth_bridge_quality(_evidence(improvement_vs_l6=0.0))
    assert decision.status == "DEPTH_BRIDGE_QUALITY_BLOCKED"
    assert "l6_comparison_gate_failed" in decision.reasons

    decision = decide_depth_bridge_quality(_evidence(improvement_vs_l7_hybrid=-0.01))
    assert "l7_hybrid_comparison_gate_failed" in decision.reasons

    decision = decide_depth_bridge_quality(_evidence(anchor_nll_regression=0.2))
    assert "general_regression_gate_failed" in decision.reasons

    decision = decide_depth_bridge_quality(_evidence(bridge_parameters=100_001))
    assert "parameter_cap_failed" in decision.reasons
