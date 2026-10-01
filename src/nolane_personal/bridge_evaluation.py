from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class BridgeQualityEvidence:
    test_examples: int
    anchor_examples: int
    baseline_nll: float
    bridge_nll: float
    nll_improvement: float
    l6_nll: float | None
    bridge_vs_l6_improvement: float | None
    anchor_baseline_nll: float
    anchor_bridge_nll: float
    anchor_nll_regression: float
    bridge_parameters: int
    base_model_unchanged: bool
    base_gradients_seen: int


@dataclass(slots=True)
class BridgeQualityThresholds:
    min_test_examples: int = 2
    min_anchor_examples: int = 4
    min_nll_improvement: float = 0.01
    min_vs_l6_nll_improvement: float = 0.005
    max_anchor_nll_regression: float = 0.05
    parameter_cap: int = 100_000
    require_l6_comparison: bool = True


def decide_bridge_quality(evidence: BridgeQualityEvidence, thresholds=None):
    thresholds = thresholds or BridgeQualityThresholds()
    reasons = []
    if evidence.test_examples < thresholds.min_test_examples:
        reasons.append("insufficient_test_examples")
    if evidence.anchor_examples < thresholds.min_anchor_examples:
        reasons.append("insufficient_anchor_examples")
    if evidence.nll_improvement < thresholds.min_nll_improvement:
        reasons.append("heldout_nll_gate_failed")
    if thresholds.require_l6_comparison:
        if evidence.l6_nll is None or evidence.bridge_vs_l6_improvement is None:
            reasons.append("missing_l6_comparison")
        elif evidence.bridge_vs_l6_improvement < thresholds.min_vs_l6_nll_improvement:
            reasons.append("l6_comparison_gate_failed")
    if evidence.anchor_nll_regression > thresholds.max_anchor_nll_regression:
        reasons.append("general_regression_gate_failed")
    if evidence.bridge_parameters > thresholds.parameter_cap:
        reasons.append("parameter_cap_failed")
    if not evidence.base_model_unchanged:
        reasons.append("base_model_mutated")
    if evidence.base_gradients_seen:
        reasons.append("base_gradient_boundary_failed")
    return {
        "status": "BRIDGE_QUALITY_PASS" if not reasons else "BRIDGE_QUALITY_BLOCKED",
        "reasons": reasons,
        "evidence": asdict(evidence),
        "thresholds": asdict(thresholds),
    }
