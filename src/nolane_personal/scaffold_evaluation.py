from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class ScaffoldQualityEvidence:
    test_examples: int
    anchor_examples: int
    total_layers: int
    head_layers: int
    tail_layers: int
    remaining_qwen_layers: int
    stages_completed: int
    baseline_nll: float
    l12_nll: float
    scaffold_nll: float
    improvement_vs_base: float
    degradation_vs_l12: float
    anchor_baseline_nll: float
    anchor_scaffold_nll: float
    anchor_nll_regression: float
    cached_generation_passed: bool
    scan_equivalence_passed: bool
    multiscale_separation_passed: bool
    cortex_parameters: int
    base_model_unchanged: bool
    base_gradients_seen: int


@dataclass(slots=True)
class ScaffoldQualityThresholds:
    min_test_examples: int = 2
    min_anchor_examples: int = 4
    max_remaining_qwen_fraction: float = 0.35
    min_head_layers: int = 2
    min_tail_layers: int = 2
    min_stages_completed: int = 2
    min_vs_base_improvement: float = 0.005
    max_degradation_vs_l12: float = 0.020
    max_anchor_nll_regression: float = 0.05
    parameter_cap: int = 100_000
    require_cached_generation: bool = True
    require_scan_equivalence: bool = True
    require_multiscale_separation: bool = True


@dataclass(slots=True)
class ScaffoldQualityDecision:
    status: str
    reasons: list[str]
    evidence: dict[str, Any]
    thresholds: dict[str, Any]


def decide_scaffold_quality(evidence: ScaffoldQualityEvidence, thresholds=None):
    thresholds = thresholds or ScaffoldQualityThresholds()
    reasons = []
    remaining_fraction = evidence.remaining_qwen_layers / max(1, evidence.total_layers)
    if evidence.test_examples < thresholds.min_test_examples:
        reasons.append("insufficient_test_examples")
    if evidence.anchor_examples < thresholds.min_anchor_examples:
        reasons.append("insufficient_anchor_examples")
    if remaining_fraction > thresholds.max_remaining_qwen_fraction:
        reasons.append("qwen_scaffold_too_large")
    if evidence.head_layers < thresholds.min_head_layers:
        reasons.append("insufficient_head_scaffold")
    if evidence.tail_layers < thresholds.min_tail_layers:
        reasons.append("insufficient_tail_scaffold")
    if evidence.stages_completed < thresholds.min_stages_completed:
        reasons.append("insufficient_scaffold_stages")
    if evidence.improvement_vs_base < thresholds.min_vs_base_improvement:
        reasons.append("base_quality_gate_failed")
    if evidence.degradation_vs_l12 > thresholds.max_degradation_vs_l12:
        reasons.append("l12_noninferiority_failed")
    if evidence.anchor_nll_regression > thresholds.max_anchor_nll_regression:
        reasons.append("general_regression_gate_failed")
    if thresholds.require_cached_generation and not evidence.cached_generation_passed:
        reasons.append("cached_generation_gate_failed")
    if thresholds.require_scan_equivalence and not evidence.scan_equivalence_passed:
        reasons.append("scan_equivalence_gate_failed")
    if (
        thresholds.require_multiscale_separation
        and not evidence.multiscale_separation_passed
    ):
        reasons.append("multiscale_separation_gate_failed")
    if evidence.cortex_parameters > thresholds.parameter_cap:
        reasons.append("parameter_cap_failed")
    if not evidence.base_model_unchanged:
        reasons.append("base_model_mutated")
    if evidence.base_gradients_seen:
        reasons.append("base_gradient_boundary_failed")
    return ScaffoldQualityDecision(
        status="SHRINKING_SCAFFOLD_QUALITY_PASS" if not reasons else "SHRINKING_SCAFFOLD_QUALITY_BLOCKED",
        reasons=reasons,
        evidence=asdict(evidence),
        thresholds=asdict(thresholds),
    )


@dataclass(slots=True)
class ScaffoldResourceEvidence:
    prompts: int
    total_layers: int
    remaining_qwen_layers: int
    baseline_median_ms: float
    l12_median_ms: float
    scaffold_median_ms: float
    latency_ratio_vs_base: float
    latency_ratio_vs_l12: float
    cached_tokens_per_second: float
    replay_safe_tokens_per_second: float
    cached_speedup_vs_replay: float
    artifact_bytes: int
    cortex_parameters: int


@dataclass(slots=True)
class ScaffoldResourceThresholds:
    min_prompts: int = 4
    max_remaining_qwen_fraction: float = 0.35
    max_latency_ratio_vs_base: float = 0.75
    max_latency_ratio_vs_l12: float = 0.93
    min_cached_speedup_vs_replay: float = 1.05
    max_artifact_bytes: int = 2_000_000
    parameter_cap: int = 100_000


def decide_scaffold_resources(evidence: ScaffoldResourceEvidence, thresholds=None):
    thresholds = thresholds or ScaffoldResourceThresholds()
    reasons = []
    remaining_fraction = evidence.remaining_qwen_layers / max(1, evidence.total_layers)
    if evidence.prompts < thresholds.min_prompts:
        reasons.append("insufficient_resource_prompts")
    if remaining_fraction > thresholds.max_remaining_qwen_fraction:
        reasons.append("qwen_scaffold_too_large")
    if evidence.latency_ratio_vs_base > thresholds.max_latency_ratio_vs_base:
        reasons.append("latency_gain_vs_base_failed")
    if evidence.latency_ratio_vs_l12 > thresholds.max_latency_ratio_vs_l12:
        reasons.append("latency_gain_vs_l12_failed")
    if evidence.cached_speedup_vs_replay < thresholds.min_cached_speedup_vs_replay:
        reasons.append("cached_generation_speed_gate_failed")
    if evidence.artifact_bytes > thresholds.max_artifact_bytes:
        reasons.append("artifact_size_gate_failed")
    if evidence.cortex_parameters > thresholds.parameter_cap:
        reasons.append("parameter_cap_failed")
    return {
        "status": "SHRINKING_SCAFFOLD_RESOURCE_PASS" if not reasons else "SHRINKING_SCAFFOLD_RESOURCE_BLOCKED",
        "reasons": reasons,
        "evidence": asdict(evidence),
        "thresholds": asdict(thresholds),
    }
