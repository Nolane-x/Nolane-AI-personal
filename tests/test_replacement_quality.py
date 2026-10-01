from nolane_personal.replacement_evaluation import (
    ReplacementQualityEvidence,
    decide_replacement_quality,
)


def _evidence(**overrides):
    values = dict(
        test_examples=20,
        anchor_examples=8,
        skipped_layers=2,
        baseline_nll=2.0,
        l6_nll=1.72,
        l7_nll=1.67,
        l8_nll=1.65,
        replacement_nll=1.655,
        improvement_vs_base=0.345,
        best_prior_nll=1.65,
        degradation_vs_best_prior=0.005,
        anchor_baseline_nll=1.5,
        anchor_replacement_nll=1.53,
        anchor_nll_regression=0.03,
        replacement_parameters=84_289,
        base_model_unchanged=True,
        base_gradients_seen=0,
    )
    values.update(overrides)
    return ReplacementQualityEvidence(**values)


def test_replacement_quality_requires_personal_gain_and_best_prior_noninferiority():
    assert decide_replacement_quality(_evidence()).status == "BLOCK_REPLACEMENT_QUALITY_PASS"

    bad = decide_replacement_quality(_evidence(degradation_vs_best_prior=0.03))
    assert "best_prior_noninferiority_failed" in bad.reasons

    bad = decide_replacement_quality(_evidence(skipped_layers=0))
    assert "no_real_block_replacement" in bad.reasons

    bad = decide_replacement_quality(_evidence(anchor_nll_regression=0.2))
    assert "general_regression_gate_failed" in bad.reasons
