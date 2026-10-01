from nolane_personal.standalone_evaluation import (
    StandaloneParityEvidence,
    StandaloneResourceEvidence,
    decide_standalone_parity,
    decide_standalone_resources,
)
from nolane_personal.standalone_promotion import decide_standalone_promotion


def _parity(**overrides):
    values = dict(
        examples=8,
        max_logit_abs_error=1e-6,
        max_state_abs_error=1e-6,
        greedy_generation_equal=True,
        prompt_scan_equivalence_passed=True,
        cortex_digest_equal_to_l15=True,
        source_mutation_isolated=True,
        runtime_has_qwen_model_reference=False,
        runtime_requires_transformers=False,
        checkpoint_contains_decoder_keys=False,
        cortex_parameters=89_805,
    )
    values.update(overrides)
    return StandaloneParityEvidence(**values)


def _resources(**overrides):
    values = dict(
        prompts=8,
        l15_median_ms=40.0,
        standalone_median_ms=39.0,
        latency_ratio_vs_l15=0.975,
        standalone_tokens_per_second=50.0,
        checkpoint_bytes=400_000_000,
        source_boundary_bytes=390_000_000,
        checkpoint_to_boundary_ratio=1.026,
        runtime_qwen_model_objects=0,
    )
    values.update(overrides)
    return StandaloneResourceEvidence(**values)


def test_parity_requires_owned_qwen_free_runtime_and_exact_behavior():
    assert decide_standalone_parity(_parity()).status == "STANDALONE_PARITY_PASS"
    assert "logit_parity_failed" in decide_standalone_parity(
        _parity(max_logit_abs_error=1e-3)
    ).reasons
    assert "source_tensor_alias_detected" in decide_standalone_parity(
        _parity(source_mutation_isolated=False)
    ).reasons
    assert "runtime_qwen_model_dependency_detected" in decide_standalone_parity(
        _parity(runtime_has_qwen_model_reference=True)
    ).reasons
    assert "decoder_tensor_leak_detected" in decide_standalone_parity(
        _parity(checkpoint_contains_decoder_keys=True)
    ).reasons


def test_resource_and_promotion_bind_standalone_and_source_l15():
    assert decide_standalone_resources(_resources())["status"] == "STANDALONE_RESOURCE_PASS"
    assert "standalone_latency_regression" in decide_standalone_resources(
        _resources(latency_ratio_vs_l15=1.2)
    )["reasons"]

    promoted = decide_standalone_promotion(
        parity_status="STANDALONE_PARITY_PASS",
        parity_checkpoint_sha256="standalone",
        parity_source_l15_checkpoint_sha256="l15",
        resource_status="STANDALONE_RESOURCE_PASS",
        resource_checkpoint_sha256="standalone",
        resource_source_l15_checkpoint_sha256="l15",
    )
    assert promoted.status == "STANDALONE_PROMOTION_PASS"

    blocked = decide_standalone_promotion(
        parity_status="STANDALONE_PARITY_PASS",
        parity_checkpoint_sha256="a",
        parity_source_l15_checkpoint_sha256="l15-a",
        resource_status="STANDALONE_RESOURCE_PASS",
        resource_checkpoint_sha256="b",
        resource_source_l15_checkpoint_sha256="l15-b",
    )
    assert "court_checkpoint_mismatch" in blocked.reasons
    assert "source_l15_checkpoint_mismatch" in blocked.reasons
