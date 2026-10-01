from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class StateSpaceQualityEvidence:
    test_examples: int
    anchor_examples: int
    total_layers: int
    replaced_layers: int
    stages_completed: int
    baseline_nll: float
    l11_nll: float
    state_space_nll: float
    improvement_vs_base: float
    degradation_vs_l11: float
    anchor_baseline_nll: float
    anchor_state_space_nll: float
    anchor_nll_regression: float
    cached_generation_passed: bool
    scan_equivalence_passed: bool
    cortex_parameters: int
    base_model_unchanged: bool
    base_gradients_seen: int


@dataclass(slots=True)
class StateSpaceQualityThresholds:
    min_test_examples: int = 2
    min_anchor_examples: int = 4
    min_replaced_fraction: float = 0.40
    min_stages_completed: int = 2
    min_vs_base_improvement: float = 0.005
    max_degradation_vs_l11: float = 0.020
    max_anchor_nll_regression: float = 0.05
    parameter_cap: int = 100_000
    require_cached_generation: bool = True
    require_scan_equivalence: bool = True


@dataclass(slots=True)
class StateSpaceQualityDecision:
    status: str
    reasons: list[str]
    evidence: dict[str, Any]
    thresholds: dict[str, Any]


def decide_state_space_quality(evidence: StateSpaceQualityEvidence, thresholds=None):
    thresholds = thresholds or StateSpaceQualityThresholds()
    reasons = []
    fraction = evidence.replaced_layers / max(1, evidence.total_layers)
    if evidence.test_examples < thresholds.min_test_examples:
        reasons.append("insufficient_test_examples")
    if evidence.anchor_examples < thresholds.min_anchor_examples:
        reasons.append("insufficient_anchor_examples")
    if fraction < thresholds.min_replaced_fraction:
        reasons.append("insufficient_transformer_depth_replaced")
    if evidence.stages_completed < thresholds.min_stages_completed:
        reasons.append("insufficient_state_space_stages")
    if evidence.improvement_vs_base < thresholds.min_vs_base_improvement:
        reasons.append("base_quality_gate_failed")
    if evidence.degradation_vs_l11 > thresholds.max_degradation_vs_l11:
        reasons.append("l11_noninferiority_failed")
    if evidence.anchor_nll_regression > thresholds.max_anchor_nll_regression:
        reasons.append("general_regression_gate_failed")
    if thresholds.require_cached_generation and not evidence.cached_generation_passed:
        reasons.append("cached_generation_gate_failed")
    if thresholds.require_scan_equivalence and not evidence.scan_equivalence_passed:
        reasons.append("state_space_scan_equivalence_failed")
    if evidence.cortex_parameters > thresholds.parameter_cap:
        reasons.append("parameter_cap_failed")
    if not evidence.base_model_unchanged:
        reasons.append("base_model_mutated")
    if evidence.base_gradients_seen:
        reasons.append("base_gradient_boundary_failed")
    return StateSpaceQualityDecision(
        status="STATE_SPACE_CORTEX_QUALITY_PASS" if not reasons else "STATE_SPACE_CORTEX_QUALITY_BLOCKED",
        reasons=reasons,
        evidence=asdict(evidence),
        thresholds=asdict(thresholds),
    )


@dataclass(slots=True)
class StateSpaceResourceEvidence:
    prompts: int
    total_layers: int
    replaced_layers: int
    baseline_median_ms: float
    l11_median_ms: float
    state_space_median_ms: float
    latency_ratio_vs_base: float
    latency_ratio_vs_l11: float
    cached_tokens_per_second: float
    replay_safe_tokens_per_second: float
    cached_speedup_vs_replay: float
    artifact_bytes: int
    cortex_parameters: int


@dataclass(slots=True)
class StateSpaceResourceThresholds:
    min_prompts: int = 4
    min_replaced_fraction: float = 0.40
    max_latency_ratio_vs_base: float = 0.85
    max_latency_ratio_vs_l11: float = 0.95
    min_cached_speedup_vs_replay: float = 1.05
    max_artifact_bytes: int = 2_000_000
    parameter_cap: int = 100_000


def decide_state_space_resources(evidence: StateSpaceResourceEvidence, thresholds=None):
    thresholds = thresholds or StateSpaceResourceThresholds()
    reasons = []
    fraction = evidence.replaced_layers / max(1, evidence.total_layers)
    if evidence.prompts < thresholds.min_prompts:
        reasons.append("insufficient_resource_prompts")
    if fraction < thresholds.min_replaced_fraction:
        reasons.append("insufficient_transformer_depth_replaced")
    if evidence.latency_ratio_vs_base > thresholds.max_latency_ratio_vs_base:
        reasons.append("latency_gain_vs_base_failed")
    if evidence.latency_ratio_vs_l11 > thresholds.max_latency_ratio_vs_l11:
        reasons.append("latency_gain_vs_l11_failed")
    if evidence.cached_speedup_vs_replay < thresholds.min_cached_speedup_vs_replay:
        reasons.append("cached_generation_speed_gate_failed")
    if evidence.artifact_bytes > thresholds.max_artifact_bytes:
        reasons.append("artifact_size_gate_failed")
    if evidence.cortex_parameters > thresholds.parameter_cap:
        reasons.append("parameter_cap_failed")
    return {
        "status": "STATE_SPACE_CORTEX_RESOURCE_PASS" if not reasons else "STATE_SPACE_CORTEX_RESOURCE_BLOCKED",
        "reasons": reasons,
        "evidence": asdict(evidence),
        "thresholds": asdict(thresholds),
    }
