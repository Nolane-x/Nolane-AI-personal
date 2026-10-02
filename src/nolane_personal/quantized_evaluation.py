from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class QuantizedQualityEvidence:
    test_examples: int
    anchor_examples: int
    rank: int
    source_factorized_nll: float
    quantized_nll: float
    nll_regression_vs_factorized: float
    anchor_source_nll: float
    anchor_quantized_nll: float
    anchor_nll_regression: float
    greedy_token_agreement: float
    prompt_scan_equivalence_passed: bool
    cortex_digest_equal_to_source: bool
    source_rank_equal: bool
    runtime_requires_qwen_model: bool
    runtime_requires_transformers: bool


@dataclass(slots=True)
class QuantizedQualityThresholds:
    min_test_examples: int = 2
    min_anchor_examples: int = 4
    max_nll_regression_vs_factorized: float = 0.02
    max_anchor_nll_regression: float = 0.03
    min_greedy_token_agreement: float = 0.98


def decide_quantized_quality(evidence, thresholds=None):
    t=thresholds or QuantizedQualityThresholds()
    reasons=[]
    if evidence.test_examples<t.min_test_examples: reasons.append("insufficient_test_examples")
    if evidence.anchor_examples<t.min_anchor_examples: reasons.append("insufficient_anchor_examples")
    if evidence.nll_regression_vs_factorized>t.max_nll_regression_vs_factorized: reasons.append("factorized_noninferiority_failed")
    if evidence.anchor_nll_regression>t.max_anchor_nll_regression: reasons.append("general_regression_gate_failed")
    if evidence.greedy_token_agreement<t.min_greedy_token_agreement: reasons.append("greedy_token_agreement_failed")
    if not evidence.prompt_scan_equivalence_passed: reasons.append("prompt_scan_equivalence_failed")
    if not evidence.cortex_digest_equal_to_source: reasons.append("cortex_digest_drift")
    if not evidence.source_rank_equal: reasons.append("rank_mismatch")
    if evidence.runtime_requires_qwen_model: reasons.append("runtime_qwen_dependency_detected")
    if evidence.runtime_requires_transformers: reasons.append("runtime_transformers_dependency_detected")
    return {
        "status":"QUANTIZED_FACTOR_QUALITY_PASS" if not reasons else "QUANTIZED_FACTOR_QUALITY_BLOCKED",
        "reasons":reasons,
        "evidence":asdict(evidence),
        "thresholds":asdict(t),
    }


@dataclass(slots=True)
class QuantizedResourceEvidence:
    prompts: int
    source_median_ms: float
    quantized_median_ms: float
    latency_ratio_vs_source: float
    source_checkpoint_bytes: int
    quantized_checkpoint_bytes: int
    checkpoint_ratio: float
    source_boundary_storage_bytes: int
    quantized_boundary_storage_bytes: int
    boundary_storage_ratio: float
    tokens_per_second: float


@dataclass(slots=True)
class QuantizedResourceThresholds:
    min_prompts: int = 4
    max_latency_ratio_vs_source: float = 1.75
    max_checkpoint_ratio: float = 0.45
    max_boundary_storage_ratio: float = 0.40
    min_tokens_per_second: float = 1.0


def decide_quantized_resources(evidence, thresholds=None):
    t=thresholds or QuantizedResourceThresholds()
    reasons=[]
    if evidence.prompts<t.min_prompts: reasons.append("insufficient_resource_prompts")
    if evidence.latency_ratio_vs_source>t.max_latency_ratio_vs_source: reasons.append("latency_regression_failed")
    if evidence.checkpoint_ratio>t.max_checkpoint_ratio: reasons.append("checkpoint_compression_failed")
    if evidence.boundary_storage_ratio>t.max_boundary_storage_ratio: reasons.append("boundary_storage_compression_failed")
    if evidence.tokens_per_second<t.min_tokens_per_second: reasons.append("generation_speed_failed")
    return {
        "status":"QUANTIZED_FACTOR_RESOURCE_PASS" if not reasons else "QUANTIZED_FACTOR_RESOURCE_BLOCKED",
        "reasons":reasons,
        "evidence":asdict(evidence),
        "thresholds":asdict(t),
    }
