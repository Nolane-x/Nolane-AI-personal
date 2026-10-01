from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class AnchorQualityEvidence:
    test_examples: int
    anchor_examples: int
    total_layers: int
    head_layers: int
    tail_layers: int
    remaining_qwen_layers: int
    stages_completed: int
    virtual_steps: int
    baseline_nll: float
    l13_nll: float
    anchor_cortex_nll: float
    improvement_vs_base: float
    degradation_vs_l13: float
    anchor_baseline_nll: float
    anchor_cortex_anchor_nll: float
    anchor_nll_regression: float
    cached_generation_passed: bool
    scan_equivalence_passed: bool
    virtual_depth_effect_passed: bool
    cortex_parameters: int
    base_model_unchanged: bool
    base_gradients_seen: int


@dataclass(slots=True)
class AnchorQualityThresholds:
    min_test_examples: int = 2
    min_anchor_examples: int = 4
    max_remaining_qwen_fraction: float = 0.15
    min_head_layers: int = 1
    min_tail_layers: int = 1
    min_stages_completed: int = 2
    min_virtual_steps: int = 4
    min_vs_base_improvement: float = 0.005
    max_degradation_vs_l13: float = 0.020
    max_anchor_nll_regression: float = 0.05
    parameter_cap: int = 100_000
    require_cached_generation: bool = True
    require_scan_equivalence: bool = True
    require_virtual_depth_effect: bool = True


@dataclass(slots=True)
class AnchorQualityDecision:
    status: str
    reasons: list[str]
    evidence: dict[str, Any]
    thresholds: dict[str, Any]


def decide_anchor_quality(evidence: AnchorQualityEvidence, thresholds=None):
    thresholds = thresholds or AnchorQualityThresholds()
    reasons = []
    remaining_fraction = evidence.remaining_qwen_layers / max(1, evidence.total_layers)
    if evidence.test_examples < thresholds.min_test_examples:
        reasons.append("insufficient_test_examples")
    if evidence.anchor_examples < thresholds.min_anchor_examples:
        reasons.append("insufficient_anchor_examples")
    if remaining_fraction > thresholds.max_remaining_qwen_fraction:
        reasons.append("qwen_anchor_shell_too_large")
    if evidence.head_layers < thresholds.min_head_layers:
        reasons.append("missing_head_anchor")
    if evidence.tail_layers < thresholds.min_tail_layers:
        reasons.append("missing_tail_anchor")
    if evidence.stages_completed < thresholds.min_stages_completed:
        reasons.append("insufficient_anchor_shrink_stages")
    if evidence.virtual_steps < thresholds.min_virtual_steps:
        reasons.append("insufficient_virtual_depth")
    if evidence.improvement_vs_base < thresholds.min_vs_base_improvement:
        reasons.append("base_quality_gate_failed")
    if evidence.degradation_vs_l13 > thresholds.max_degradation_vs_l13:
        reasons.append("l13_noninferiority_failed")
    if evidence.anchor_nll_regression > thresholds.max_anchor_nll_regression:
        reasons.append("general_regression_gate_failed")
    if thresholds.require_cached_generation and not evidence.cached_generation_passed:
        reasons.append("cached_generation_gate_failed")
    if thresholds.require_scan_equivalence and not evidence.scan_equivalence_passed:
        reasons.append("scan_equivalence_gate_failed")
    if thresholds.require_virtual_depth_effect and not evidence.virtual_depth_effect_passed:
        reasons.append("virtual_depth_effect_gate_failed")
    if evidence.cortex_parameters > thresholds.parameter_cap:
        reasons.append("parameter_cap_failed")
    if not evidence.base_model_unchanged:
        reasons.append("base_model_mutated")
    if evidence.base_gradients_seen:
        reasons.append("base_gradient_boundary_failed")
    return AnchorQualityDecision(
        status="MINIMAL_ANCHOR_QUALITY_PASS" if not reasons else "MINIMAL_ANCHOR_QUALITY_BLOCKED",
        reasons=reasons,
        evidence=asdict(evidence),
        thresholds=asdict(thresholds),
    )


@dataclass(slots=True)
class AnchorResourceEvidence:
    prompts: int
    total_layers: int
    remaining_qwen_layers: int
    baseline_median_ms: float
    l13_median_ms: float
    anchor_cortex_median_ms: float
    latency_ratio_vs_base: float
    latency_ratio_vs_l13: float
    cached_tokens_per_second: float
    replay_safe_tokens_per_second: float
    cached_speedup_vs_replay: float
    artifact_bytes: int
    cortex_parameters: int


@dataclass(slots=True)
class AnchorResourceThresholds:
    min_prompts: int = 4
    max_remaining_qwen_fraction: float = 0.15
    max_latency_ratio_vs_base: float = 0.65
    max_latency_ratio_vs_l13: float = 0.85
    min_cached_speedup_vs_replay: float = 1.05
    max_artifact_bytes: int = 2_000_000
    parameter_cap: int = 100_000


def decide_anchor_resources(evidence: AnchorResourceEvidence, thresholds=None):
    thresholds = thresholds or AnchorResourceThresholds()
    reasons = []
    remaining_fraction = evidence.remaining_qwen_layers / max(1, evidence.total_layers)
    if evidence.prompts < thresholds.min_prompts:
        reasons.append("insufficient_resource_prompts")
    if remaining_fraction > thresholds.max_remaining_qwen_fraction:
        reasons.append("qwen_anchor_shell_too_large")
    if evidence.latency_ratio_vs_base > thresholds.max_latency_ratio_vs_base:
        reasons.append("latency_gain_vs_base_failed")
    if evidence.latency_ratio_vs_l13 > thresholds.max_latency_ratio_vs_l13:
        reasons.append("latency_gain_vs_l13_failed")
    if evidence.cached_speedup_vs_replay < thresholds.min_cached_speedup_vs_replay:
        reasons.append("cached_generation_speed_gate_failed")
    if evidence.artifact_bytes > thresholds.max_artifact_bytes:
        reasons.append("artifact_size_gate_failed")
    if evidence.cortex_parameters > thresholds.parameter_cap:
        reasons.append("parameter_cap_failed")
    return {
        "status": "MINIMAL_ANCHOR_RESOURCE_PASS" if not reasons else "MINIMAL_ANCHOR_RESOURCE_BLOCKED",
        "reasons": reasons,
        "evidence": asdict(evidence),
        "thresholds": asdict(thresholds),
    }
