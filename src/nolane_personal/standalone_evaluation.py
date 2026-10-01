from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class StandaloneParityEvidence:
    examples: int
    max_logit_abs_error: float
    max_state_abs_error: float
    greedy_generation_equal: bool
    prompt_scan_equivalence_passed: bool
    cortex_digest_equal_to_l15: bool
    source_mutation_isolated: bool
    runtime_has_qwen_model_reference: bool
    runtime_requires_transformers: bool
    checkpoint_contains_decoder_keys: bool
    cortex_parameters: int


@dataclass(slots=True)
class StandaloneParityThresholds:
    min_examples: int = 2
    max_logit_abs_error: float = 2e-5
    max_state_abs_error: float = 2e-5
    cortex_parameter_cap: int = 100_000


@dataclass(slots=True)
class StandaloneParityDecision:
    status: str
    reasons: list[str]
    evidence: dict[str, Any]
    thresholds: dict[str, Any]


def decide_standalone_parity(evidence: StandaloneParityEvidence, thresholds=None):
    thresholds = thresholds or StandaloneParityThresholds()
    reasons = []
    if evidence.examples < thresholds.min_examples:
        reasons.append("insufficient_parity_examples")
    if evidence.max_logit_abs_error > thresholds.max_logit_abs_error:
        reasons.append("logit_parity_failed")
    if evidence.max_state_abs_error > thresholds.max_state_abs_error:
        reasons.append("state_parity_failed")
    if not evidence.greedy_generation_equal:
        reasons.append("generation_parity_failed")
    if not evidence.prompt_scan_equivalence_passed:
        reasons.append("prompt_scan_equivalence_failed")
    if not evidence.cortex_digest_equal_to_l15:
        reasons.append("cortex_digest_drift")
    if not evidence.source_mutation_isolated:
        reasons.append("source_tensor_alias_detected")
    if evidence.runtime_has_qwen_model_reference:
        reasons.append("runtime_qwen_model_dependency_detected")
    if evidence.runtime_requires_transformers:
        reasons.append("runtime_transformers_dependency_detected")
    if evidence.checkpoint_contains_decoder_keys:
        reasons.append("decoder_tensor_leak_detected")
    if evidence.cortex_parameters > thresholds.cortex_parameter_cap:
        reasons.append("cortex_parameter_cap_failed")
    return StandaloneParityDecision(
        status="STANDALONE_PARITY_PASS" if not reasons else "STANDALONE_PARITY_BLOCKED",
        reasons=reasons,
        evidence=asdict(evidence),
        thresholds=asdict(thresholds),
    )


@dataclass(slots=True)
class StandaloneResourceEvidence:
    prompts: int
    l15_median_ms: float
    standalone_median_ms: float
    latency_ratio_vs_l15: float
    standalone_tokens_per_second: float
    checkpoint_bytes: int
    source_boundary_bytes: int
    checkpoint_to_boundary_ratio: float
    runtime_qwen_model_objects: int


@dataclass(slots=True)
class StandaloneResourceThresholds:
    min_prompts: int = 4
    max_latency_ratio_vs_l15: float = 1.05
    min_tokens_per_second: float = 1.0
    max_checkpoint_to_boundary_ratio: float = 1.20
    max_runtime_qwen_model_objects: int = 0


def decide_standalone_resources(evidence: StandaloneResourceEvidence, thresholds=None):
    thresholds = thresholds or StandaloneResourceThresholds()
    reasons = []
    if evidence.prompts < thresholds.min_prompts:
        reasons.append("insufficient_resource_prompts")
    if evidence.latency_ratio_vs_l15 > thresholds.max_latency_ratio_vs_l15:
        reasons.append("standalone_latency_regression")
    if evidence.standalone_tokens_per_second < thresholds.min_tokens_per_second:
        reasons.append("standalone_generation_speed_failed")
    if evidence.checkpoint_to_boundary_ratio > thresholds.max_checkpoint_to_boundary_ratio:
        reasons.append("standalone_checkpoint_overhead_failed")
    if evidence.runtime_qwen_model_objects > thresholds.max_runtime_qwen_model_objects:
        reasons.append("runtime_qwen_model_object_detected")
    return {
        "status": "STANDALONE_RESOURCE_PASS" if not reasons else "STANDALONE_RESOURCE_BLOCKED",
        "reasons": reasons,
        "evidence": asdict(evidence),
        "thresholds": asdict(thresholds),
    }
