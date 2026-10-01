from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class NativeQualityEvidence:
    test_examples: int
    anchor_examples: int
    qwen_decoder_layers_total: int
    qwen_decoder_calls_forward: int
    qwen_decoder_calls_generation: int
    baseline_nll: float
    l14_nll: float
    native_nll: float
    improvement_vs_base: float
    degradation_vs_l14: float
    anchor_baseline_nll: float
    anchor_native_nll: float
    anchor_nll_regression: float
    prompt_scan_equivalence_passed: bool
    native_generation_passed: bool
    cortex_parameters: int
    qwen_model_unchanged: bool
    qwen_gradients_seen: int


@dataclass(slots=True)
class NativeQualityThresholds:
    min_test_examples: int = 2
    min_anchor_examples: int = 4
    max_qwen_decoder_calls: int = 0
    min_vs_base_improvement: float = 0.005
    max_degradation_vs_l14: float = 0.025
    max_anchor_nll_regression: float = 0.06
    parameter_cap: int = 100_000
    require_prompt_scan_equivalence: bool = True
    require_native_generation: bool = True


@dataclass(slots=True)
class NativeQualityDecision:
    status: str
    reasons: list[str]
    evidence: dict[str, Any]
    thresholds: dict[str, Any]


def decide_native_quality(evidence: NativeQualityEvidence, thresholds=None):
    thresholds = thresholds or NativeQualityThresholds()
    reasons = []
    if evidence.test_examples < thresholds.min_test_examples:
        reasons.append("insufficient_test_examples")
    if evidence.anchor_examples < thresholds.min_anchor_examples:
        reasons.append("insufficient_anchor_examples")
    if evidence.qwen_decoder_calls_forward > thresholds.max_qwen_decoder_calls:
        reasons.append("qwen_decoder_forward_executed")
    if evidence.qwen_decoder_calls_generation > thresholds.max_qwen_decoder_calls:
        reasons.append("qwen_decoder_generation_executed")
    if evidence.improvement_vs_base < thresholds.min_vs_base_improvement:
        reasons.append("base_quality_gate_failed")
    if evidence.degradation_vs_l14 > thresholds.max_degradation_vs_l14:
        reasons.append("l14_noninferiority_failed")
    if evidence.anchor_nll_regression > thresholds.max_anchor_nll_regression:
        reasons.append("general_regression_gate_failed")
    if thresholds.require_prompt_scan_equivalence and not evidence.prompt_scan_equivalence_passed:
        reasons.append("prompt_scan_equivalence_failed")
    if thresholds.require_native_generation and not evidence.native_generation_passed:
        reasons.append("native_generation_gate_failed")
    if evidence.cortex_parameters > thresholds.parameter_cap:
        reasons.append("parameter_cap_failed")
    if not evidence.qwen_model_unchanged:
        reasons.append("qwen_model_mutated")
    if evidence.qwen_gradients_seen:
        reasons.append("qwen_gradient_boundary_failed")
    return NativeQualityDecision(
        status="NATIVE_BOUNDARY_QUALITY_PASS" if not reasons else "NATIVE_BOUNDARY_QUALITY_BLOCKED",
        reasons=reasons,
        evidence=asdict(evidence),
        thresholds=asdict(thresholds),
    )


@dataclass(slots=True)
class NativeResourceEvidence:
    prompts: int
    baseline_median_ms: float
    l14_median_ms: float
    native_median_ms: float
    latency_ratio_vs_base: float
    latency_ratio_vs_l14: float
    native_tokens_per_second: float
    artifact_bytes: int
    cortex_parameters: int
    qwen_decoder_calls: int


@dataclass(slots=True)
class NativeResourceThresholds:
    min_prompts: int = 4
    max_latency_ratio_vs_base: float = 0.50
    max_latency_ratio_vs_l14: float = 0.75
    min_native_tokens_per_second: float = 1.0
    max_artifact_bytes: int = 2_000_000
    parameter_cap: int = 100_000
    max_qwen_decoder_calls: int = 0


def decide_native_resources(evidence: NativeResourceEvidence, thresholds=None):
    thresholds = thresholds or NativeResourceThresholds()
    reasons = []
    if evidence.prompts < thresholds.min_prompts:
        reasons.append("insufficient_resource_prompts")
    if evidence.latency_ratio_vs_base > thresholds.max_latency_ratio_vs_base:
        reasons.append("latency_gain_vs_base_failed")
    if evidence.latency_ratio_vs_l14 > thresholds.max_latency_ratio_vs_l14:
        reasons.append("latency_gain_vs_l14_failed")
    if evidence.native_tokens_per_second < thresholds.min_native_tokens_per_second:
        reasons.append("native_generation_speed_failed")
    if evidence.artifact_bytes > thresholds.max_artifact_bytes:
        reasons.append("artifact_size_gate_failed")
    if evidence.cortex_parameters > thresholds.parameter_cap:
        reasons.append("parameter_cap_failed")
    if evidence.qwen_decoder_calls > thresholds.max_qwen_decoder_calls:
        reasons.append("qwen_decoder_execution_detected")
    return {
        "status": "NATIVE_BOUNDARY_RESOURCE_PASS" if not reasons else "NATIVE_BOUNDARY_RESOURCE_BLOCKED",
        "reasons": reasons,
        "evidence": asdict(evidence),
        "thresholds": asdict(thresholds),
    }
