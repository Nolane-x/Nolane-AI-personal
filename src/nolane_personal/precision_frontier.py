from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class PackedInt4QualityEvidence:
    test_examples: int
    anchor_examples: int
    rank: int
    factorized_nll: float
    int8_nll: float
    int4_nll: float
    int4_regression_vs_factorized: float
    int4_regression_vs_int8: float
    anchor_factorized_nll: float
    anchor_int8_nll: float
    anchor_int4_nll: float
    anchor_int4_regression_vs_factorized: float
    anchor_int4_regression_vs_int8: float
    greedy_agreement_int4_vs_factorized: float
    greedy_agreement_int4_vs_int8: float
    prompt_scan_equivalence_passed: bool
    cortex_digest_equal: bool
    rank_equal: bool
    runtime_requires_qwen_model: bool
    runtime_requires_transformers: bool


@dataclass(slots=True)
class PackedInt4QualityThresholds:
    min_test_examples: int = 2
    min_anchor_examples: int = 4
    max_regression_vs_factorized: float = 0.04
    max_regression_vs_int8: float = 0.025
    max_anchor_regression_vs_factorized: float = 0.05
    max_anchor_regression_vs_int8: float = 0.03
    min_agreement_vs_factorized: float = 0.95
    min_agreement_vs_int8: float = 0.97


def decide_packed_int4_quality(evidence, thresholds=None):
    t = thresholds or PackedInt4QualityThresholds()
    reasons = []
    if evidence.test_examples < t.min_test_examples:
        reasons.append("insufficient_test_examples")
    if evidence.anchor_examples < t.min_anchor_examples:
        reasons.append("insufficient_anchor_examples")
    if evidence.int4_regression_vs_factorized > t.max_regression_vs_factorized:
        reasons.append("factorized_noninferiority_failed")
    if evidence.int4_regression_vs_int8 > t.max_regression_vs_int8:
        reasons.append("int8_noninferiority_failed")
    if evidence.anchor_int4_regression_vs_factorized > t.max_anchor_regression_vs_factorized:
        reasons.append("factorized_anchor_regression_failed")
    if evidence.anchor_int4_regression_vs_int8 > t.max_anchor_regression_vs_int8:
        reasons.append("int8_anchor_regression_failed")
    if evidence.greedy_agreement_int4_vs_factorized < t.min_agreement_vs_factorized:
        reasons.append("factorized_token_agreement_failed")
    if evidence.greedy_agreement_int4_vs_int8 < t.min_agreement_vs_int8:
        reasons.append("int8_token_agreement_failed")
    if not evidence.prompt_scan_equivalence_passed:
        reasons.append("prompt_scan_equivalence_failed")
    if not evidence.cortex_digest_equal:
        reasons.append("cortex_digest_drift")
    if not evidence.rank_equal:
        reasons.append("rank_mismatch")
    if evidence.runtime_requires_qwen_model:
        reasons.append("runtime_qwen_dependency_detected")
    if evidence.runtime_requires_transformers:
        reasons.append("runtime_transformers_dependency_detected")
    return {
        "status": "PACKED_INT4_QUALITY_PASS" if not reasons else "PACKED_INT4_QUALITY_BLOCKED",
        "reasons": reasons,
        "evidence": asdict(evidence),
        "thresholds": asdict(t),
    }


@dataclass(slots=True)
class PackedInt4ResourceEvidence:
    prompts: int
    int8_median_ms: float
    int4_median_ms: float
    latency_ratio_vs_int8: float
    int8_checkpoint_bytes: int
    int4_checkpoint_bytes: int
    checkpoint_ratio_vs_int8: float
    int8_boundary_storage_bytes: int
    int4_boundary_storage_bytes: int
    boundary_storage_ratio_vs_int8: float
    int4_tokens_per_second: float


@dataclass(slots=True)
class PackedInt4ResourceThresholds:
    min_prompts: int = 4
    max_latency_ratio_vs_int8: float = 1.50
    max_checkpoint_ratio_vs_int8: float = 0.70
    max_boundary_storage_ratio_vs_int8: float = 0.60
    min_tokens_per_second: float = 1.0


def decide_packed_int4_resources(evidence, thresholds=None):
    t = thresholds or PackedInt4ResourceThresholds()
    reasons = []
    if evidence.prompts < t.min_prompts:
        reasons.append("insufficient_resource_prompts")
    if evidence.latency_ratio_vs_int8 > t.max_latency_ratio_vs_int8:
        reasons.append("latency_regression_failed")
    if evidence.checkpoint_ratio_vs_int8 > t.max_checkpoint_ratio_vs_int8:
        reasons.append("checkpoint_compression_failed")
    if evidence.boundary_storage_ratio_vs_int8 > t.max_boundary_storage_ratio_vs_int8:
        reasons.append("boundary_storage_compression_failed")
    if evidence.int4_tokens_per_second < t.min_tokens_per_second:
        reasons.append("generation_speed_failed")
    return {
        "status": "PACKED_INT4_RESOURCE_PASS" if not reasons else "PACKED_INT4_RESOURCE_BLOCKED",
        "reasons": reasons,
        "evidence": asdict(evidence),
        "thresholds": asdict(t),
    }


@dataclass(slots=True)
class PrecisionFrontierDecision:
    status: str
    selected_precision: str | None
    reasons: list[str]
    source_factorized_checkpoint_sha256: str | None
    selected_checkpoint_sha256: str | None


def decide_precision_frontier(
    *,
    int8_quality_status: str,
    int8_resource_status: str,
    int8_checkpoint_sha256: str | None,
    int8_source_factorized_checkpoint_sha256: str | None,
    int4_quality_status: str,
    int4_resource_status: str,
    int4_checkpoint_sha256: str | None,
    int4_source_factorized_checkpoint_sha256: str | None,
    int4_quality_reference_int8_checkpoint_sha256: str | None = None,
    int4_resource_reference_int8_checkpoint_sha256: str | None = None,
    int4_resource_checkpoint_sha256: str | None = None,
):
    reasons = []
    if int8_quality_status != "QUANTIZED_FACTOR_QUALITY_PASS":
        reasons.append("int8_fallback_quality_not_passed")
    if int8_resource_status != "QUANTIZED_FACTOR_RESOURCE_PASS":
        reasons.append("int8_fallback_resource_not_passed")
    if not int8_checkpoint_sha256:
        reasons.append("int8_checkpoint_missing")
    if not int8_source_factorized_checkpoint_sha256:
        reasons.append("int8_source_factorized_missing")

    if reasons:
        return PrecisionFrontierDecision(
            status="PRECISION_FRONTIER_BLOCKED",
            selected_precision=None,
            reasons=reasons,
            source_factorized_checkpoint_sha256=None,
            selected_checkpoint_sha256=None,
        )

    int4_lineage_ok = bool(
        int4_checkpoint_sha256
        and int4_source_factorized_checkpoint_sha256
        and int4_source_factorized_checkpoint_sha256
        == int8_source_factorized_checkpoint_sha256
        and (
            int4_quality_reference_int8_checkpoint_sha256 is None
            or int4_quality_reference_int8_checkpoint_sha256 == int8_checkpoint_sha256
        )
        and (
            int4_resource_reference_int8_checkpoint_sha256 is None
            or int4_resource_reference_int8_checkpoint_sha256 == int8_checkpoint_sha256
        )
        and (
            int4_resource_checkpoint_sha256 is None
            or int4_resource_checkpoint_sha256 == int4_checkpoint_sha256
        )
    )
    if (
        int4_quality_status == "PACKED_INT4_QUALITY_PASS"
        and int4_resource_status == "PACKED_INT4_RESOURCE_PASS"
        and int4_lineage_ok
    ):
        return PrecisionFrontierDecision(
            status="PRECISION_FRONTIER_SELECT_INT4",
            selected_precision="int4",
            reasons=[],
            source_factorized_checkpoint_sha256=int8_source_factorized_checkpoint_sha256,
            selected_checkpoint_sha256=int4_checkpoint_sha256,
        )

    fallback_reasons = []
    if int4_quality_status != "PACKED_INT4_QUALITY_PASS":
        fallback_reasons.append("int4_quality_not_passed")
    if int4_resource_status != "PACKED_INT4_RESOURCE_PASS":
        fallback_reasons.append("int4_resource_not_passed")
    if not int4_lineage_ok:
        fallback_reasons.append("int4_source_lineage_mismatch")
    return PrecisionFrontierDecision(
        status="PRECISION_FRONTIER_KEEP_INT8",
        selected_precision="int8",
        reasons=fallback_reasons,
        source_factorized_checkpoint_sha256=int8_source_factorized_checkpoint_sha256,
        selected_checkpoint_sha256=int8_checkpoint_sha256,
    )
