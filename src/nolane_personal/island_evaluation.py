from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class IslandQualityEvidence:
    test_examples: int
    anchor_examples: int
    total_layers: int
    replaced_layers: int
    island_count: int
    stages_completed: int
    baseline_nll: float
    l10_nll: float
    island_nll: float
    improvement_vs_base: float
    degradation_vs_l10: float
    anchor_baseline_nll: float
    anchor_island_nll: float
    anchor_nll_regression: float
    cached_generation_passed: bool
    replacement_parameters: int
    base_model_unchanged: bool
    base_gradients_seen: int


@dataclass(slots=True)
class IslandQualityThresholds:
    min_test_examples: int = 2
    min_anchor_examples: int = 4
    min_replaced_fraction: float = 0.25
    min_compression_ratio: float = 2.0
    min_stages_completed: int = 1
    min_vs_base_improvement: float = 0.005
    max_degradation_vs_l10: float = 0.02
    max_anchor_nll_regression: float = 0.05
    parameter_cap: int = 100_000
    require_cached_generation: bool = True


@dataclass(slots=True)
class IslandQualityDecision:
    status: str
    reasons: list[str]
    evidence: dict[str, Any]
    thresholds: dict[str, Any]


def decide_island_quality(evidence: IslandQualityEvidence, thresholds=None):
    thresholds = thresholds or IslandQualityThresholds()
    reasons = []
    replaced_fraction = evidence.replaced_layers / max(1, evidence.total_layers)
    compression_ratio = evidence.replaced_layers / max(1, evidence.island_count)
    if evidence.test_examples < thresholds.min_test_examples:
        reasons.append("insufficient_test_examples")
    if evidence.anchor_examples < thresholds.min_anchor_examples:
        reasons.append("insufficient_anchor_examples")
    if replaced_fraction < thresholds.min_replaced_fraction:
        reasons.append("insufficient_transformer_depth_replaced")
    if compression_ratio < thresholds.min_compression_ratio:
        reasons.append("insufficient_region_compression")
    if evidence.stages_completed < thresholds.min_stages_completed:
        reasons.append("insufficient_island_stages")
    if evidence.improvement_vs_base < thresholds.min_vs_base_improvement:
        reasons.append("base_quality_gate_failed")
    if evidence.degradation_vs_l10 > thresholds.max_degradation_vs_l10:
        reasons.append("l10_noninferiority_failed")
    if evidence.anchor_nll_regression > thresholds.max_anchor_nll_regression:
        reasons.append("general_regression_gate_failed")
    if thresholds.require_cached_generation and not evidence.cached_generation_passed:
        reasons.append("cached_generation_gate_failed")
    if evidence.replacement_parameters > thresholds.parameter_cap:
        reasons.append("parameter_cap_failed")
    if not evidence.base_model_unchanged:
        reasons.append("base_model_mutated")
    if evidence.base_gradients_seen:
        reasons.append("base_gradient_boundary_failed")
    return IslandQualityDecision(
        status="RECURRENT_ISLAND_QUALITY_PASS" if not reasons else "RECURRENT_ISLAND_QUALITY_BLOCKED",
        reasons=reasons,
        evidence=asdict(evidence),
        thresholds=asdict(thresholds),
    )


@dataclass(slots=True)
class IslandResourceEvidence:
    prompts: int
    total_layers: int
    replaced_layers: int
    island_count: int
    baseline_median_ms: float
    l10_median_ms: float
    island_median_ms: float
    latency_ratio_vs_base: float
    latency_ratio_vs_l10: float
    cached_tokens_per_second: float
    replay_safe_tokens_per_second: float
    cached_speedup_vs_replay: float
    artifact_bytes: int
    replacement_parameters: int


@dataclass(slots=True)
class IslandResourceThresholds:
    min_prompts: int = 4
    min_replaced_fraction: float = 0.25
    min_compression_ratio: float = 2.0
    max_latency_ratio_vs_base: float = 0.90
    max_latency_ratio_vs_l10: float = 0.98
    min_cached_speedup_vs_replay: float = 1.05
    max_artifact_bytes: int = 2_000_000
    parameter_cap: int = 100_000


def decide_island_resources(evidence: IslandResourceEvidence, thresholds=None):
    thresholds = thresholds or IslandResourceThresholds()
    reasons = []
    replaced_fraction = evidence.replaced_layers / max(1, evidence.total_layers)
    compression_ratio = evidence.replaced_layers / max(1, evidence.island_count)
    if evidence.prompts < thresholds.min_prompts:
        reasons.append("insufficient_resource_prompts")
    if replaced_fraction < thresholds.min_replaced_fraction:
        reasons.append("insufficient_transformer_depth_replaced")
    if compression_ratio < thresholds.min_compression_ratio:
        reasons.append("insufficient_region_compression")
    if evidence.latency_ratio_vs_base > thresholds.max_latency_ratio_vs_base:
        reasons.append("latency_gain_vs_base_failed")
    if evidence.latency_ratio_vs_l10 > thresholds.max_latency_ratio_vs_l10:
        reasons.append("latency_gain_vs_l10_failed")
    if evidence.cached_speedup_vs_replay < thresholds.min_cached_speedup_vs_replay:
        reasons.append("cached_generation_speed_gate_failed")
    if evidence.artifact_bytes > thresholds.max_artifact_bytes:
        reasons.append("artifact_size_gate_failed")
    if evidence.replacement_parameters > thresholds.parameter_cap:
        reasons.append("parameter_cap_failed")
    return {
        "status": "RECURRENT_ISLAND_RESOURCE_PASS" if not reasons else "RECURRENT_ISLAND_RESOURCE_BLOCKED",
        "reasons": reasons,
        "evidence": asdict(evidence),
        "thresholds": asdict(thresholds),
    }
