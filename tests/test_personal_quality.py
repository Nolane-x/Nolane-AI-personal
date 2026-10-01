from nolane_personal.personal_evaluation import (
    PersonalQualityEvidence,
    PersonalQualityThresholds,
    decide_personal_quality,
)


def _evidence(**overrides):
    values = dict(
        test_examples=20,
        anchor_examples=8,
        baseline_nll=2.0,
        personal_nll=1.7,
        nll_improvement=0.3,
        anchor_baseline_nll=1.5,
        anchor_personal_nll=1.52,
        anchor_nll_regression=0.02,
        adapter_parameters=18_001,
        base_model_unchanged=True,
        base_gradients_seen=0,
    )
    values.update(overrides)
    return PersonalQualityEvidence(**values)


def test_quality_gate_requires_personal_gain_without_general_regression():
    decision = decide_personal_quality(_evidence())
    assert decision.status == "PERSONAL_QUALITY_PASS"

    blocked = decide_personal_quality(_evidence(anchor_nll_regression=0.2))
    assert blocked.status == "PERSONAL_QUALITY_BLOCKED"
    assert "general_regression_gate_failed" in blocked.reasons

    blocked = decide_personal_quality(_evidence(base_gradients_seen=1))
    assert "base_gradient_boundary_failed" in blocked.reasons
