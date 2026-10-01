from nolane_personal.surgery import CounterfactualReceipt
from nolane_personal.surgery_court import ShadowAdmissionThresholds, evaluate_shadow_admission


def _receipt(**overrides):
    values = dict(
        schema="NOLANE-L5-COUNTERFACTUAL-SURGERY-V1",
        authority="COUNTERFACTUAL_ONLY_BASELINE_OUTPUT",
        candidate_id="candidate",
        base_model_fingerprint="base",
        latent_digest="latent",
        adapter_digest="adapter",
        adapter_parameters=18_001,
        layer_indices=[2, 4, 6],
        token_scope="last",
        gate=0.02,
        baseline_latency_ms=10.0,
        counterfactual_latency_ms=11.0,
        overhead_ratio=1.1,
        kl_baseline_to_counterfactual=0.02,
        mean_abs_logit_shift=0.01,
        max_abs_logit_shift=0.08,
        cosine_similarity=0.99,
        baseline_top1=10,
        counterfactual_top1=10,
        top1_changed=False,
        base_model_unchanged=True,
    )
    values.update(overrides)
    return CounterfactualReceipt(**values)


def test_shadow_admission_is_not_promotion_and_blocks_model_mutation():
    decision = evaluate_shadow_admission([_receipt()])
    assert decision.status == "SHADOW_ADMISSION_PASS"
    assert decision.summary["authority"] == "SHADOW_ONLY_NO_PROMOTION"

    blocked = evaluate_shadow_admission([_receipt(base_model_unchanged=False)])
    assert blocked.status == "SHADOW_ADMISSION_BLOCKED"
    assert "base_model_mutated" in blocked.reasons


def test_shadow_admission_blocks_mixed_lineage_and_excessive_adapter():
    thresholds = ShadowAdmissionThresholds(parameter_cap=20_000)
    rows = [
        _receipt(),
        _receipt(candidate_id="other", adapter_digest="other-adapter", adapter_parameters=25_000),
    ]
    decision = evaluate_shadow_admission(rows, thresholds)
    assert decision.status == "SHADOW_ADMISSION_BLOCKED"
    assert "mixed_candidate_ids" in decision.reasons
    assert "mixed_adapter_states" in decision.reasons
    assert "parameter_cap_failed" in decision.reasons
