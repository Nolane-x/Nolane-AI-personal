from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class DepthBridgeQualityEvidence:
    test_examples: int
    anchor_examples: int
    baseline_nll: float
    l6_nll: float
    l7_hybrid_nll: float
    depth_bridge_nll: float
    improvement_vs_base: float
    improvement_vs_l6: float
    improvement_vs_l7_hybrid: float
    anchor_baseline_nll: float
    anchor_depth_bridge_nll: float
    anchor_nll_regression: float
    bridge_parameters: int
    base_model_unchanged: bool
    base_gradients_seen: int


@dataclass(slots=True)
class DepthBridgeQualityThresholds:
    min_test_examples: int = 2
    min_anchor_examples: int = 4
    min_vs_base_nll_improvement: float = 0.01
    min_vs_l6_nll_improvement: float = 0.005
    min_vs_l7_hybrid_nll_improvement: float = 0.005
    max_anchor_nll_regression: float = 0.05
    parameter_cap: int = 100_000


@dataclass(slots=True)
class DepthBridgeQualityDecision:
    status: str
    reasons: list[str]
    evidence: dict[str, Any]
    thresholds: dict[str, Any]


def decide_depth_bridge_quality(
    evidence: DepthBridgeQualityEvidence,
    thresholds: DepthBridgeQualityThresholds | None = None,
) -> DepthBridgeQualityDecision:
    thresholds = thresholds or DepthBridgeQualityThresholds()
    reasons: list[str] = []
    if evidence.test_examples < thresholds.min_test_examples:
        reasons.append("insufficient_test_examples")
    if evidence.anchor_examples < thresholds.min_anchor_examples:
        reasons.append("insufficient_anchor_examples")
    if evidence.improvement_vs_base < thresholds.min_vs_base_nll_improvement:
        reasons.append("base_comparison_gate_failed")
    if evidence.improvement_vs_l6 < thresholds.min_vs_l6_nll_improvement:
        reasons.append("l6_comparison_gate_failed")
    if evidence.improvement_vs_l7_hybrid < thresholds.min_vs_l7_hybrid_nll_improvement:
        reasons.append("l7_hybrid_comparison_gate_failed")
    if evidence.anchor_nll_regression > thresholds.max_anchor_nll_regression:
        reasons.append("general_regression_gate_failed")
    if evidence.bridge_parameters > thresholds.parameter_cap:
        reasons.append("parameter_cap_failed")
    if not evidence.base_model_unchanged:
        reasons.append("base_model_mutated")
    if evidence.base_gradients_seen:
        reasons.append("base_gradient_boundary_failed")
    return DepthBridgeQualityDecision(
        status="DEPTH_BRIDGE_QUALITY_PASS" if not reasons else "DEPTH_BRIDGE_QUALITY_BLOCKED",
        reasons=reasons,
        evidence=asdict(evidence),
        thresholds=asdict(thresholds),
    )
