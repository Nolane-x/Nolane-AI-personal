from nolane_personal.bridge_evaluation import (
    BridgeQualityEvidence,
    BridgeQualityThresholds,
    decide_bridge_quality,
)


def _evidence(**overrides):
    values = dict(
        test_examples=20,
        anchor_examples=8,
        baseline_nll=2.0,
        bridge_nll=1.60,
        nll_improvement=0.40,
        l6_nll=1.70,
        bridge_vs_l6_improvement=0.10,
        anchor_baseline_nll=1.5,
        anchor_bridge_nll=1.53,
        anchor_nll_regression=0.03,
        bridge_parameters=84_289,
        base_model_unchanged=True,
        base_gradients_seen=0,
    )
    values.update(overrides)
    return BridgeQualityEvidence(**values)


def test_bridge_quality_requires_gain_over_qwen_and_l6_without_regression():
    decision = decide_bridge_quality(_evidence())
    assert decision["status"] == "BRIDGE_QUALITY_PASS"

    bad = decide_bridge_quality(_evidence(anchor_nll_regression=0.2))
    assert bad["status"] == "BRIDGE_QUALITY_BLOCKED"
    assert "general_regression_gate_failed" in bad["reasons"]

    bad = decide_bridge_quality(_evidence(bridge_parameters=150_000))
    assert "parameter_cap_failed" in bad["reasons"]

    bad = decide_bridge_quality(_evidence(l6_nll=None, bridge_vs_l6_improvement=None))
    assert "missing_l6_comparison" in bad["reasons"]

    bad = decide_bridge_quality(_evidence(l6_nll=1.60, bridge_vs_l6_improvement=0.0))
    assert "l6_comparison_gate_failed" in bad["reasons"]
