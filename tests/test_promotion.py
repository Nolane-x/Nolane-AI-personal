from nolane_personal.promotion import (
    PromotionEvidence,
    PromotionThresholds,
    decide_promotion,
)


def test_promotion_requires_fidelity_forecast_gain_and_sample_size():
    thresholds = PromotionThresholds(
        min_state_test_cases=10,
        min_return_test_cases=5,
        state_mae_max=0.03,
        state_max_error_max=0.20,
        brier_improvement_min=0.02,
        parameter_cap=100_000,
    )
    evidence = PromotionEvidence(
        protocol_verified=True,
        checkpoint_protocol_match=True,
        state_test_cases=50,
        return_test_cases=20,
        state_mae=0.02,
        state_max_error=0.10,
        baseline_brier=0.25,
        candidate_brier=0.20,
        parameter_count=14_515,
    )
    assert decide_promotion(evidence, thresholds).status == "PROMOTION_PASS"

    weak = PromotionEvidence(
        protocol_verified=True,
        checkpoint_protocol_match=True,
        state_test_cases=50,
        return_test_cases=20,
        state_mae=0.02,
        state_max_error=0.10,
        baseline_brier=0.25,
        candidate_brier=0.245,
        parameter_count=14_515,
    )
    decision = decide_promotion(weak, thresholds)
    assert decision.status == "PROMOTION_BLOCKED"
    assert "return_forecast_gate_failed" in decision.reasons
