from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class ReplacementQualityEvidence:
    test_examples: int
    anchor_examples: int
    skipped_layers: int
    baseline_nll: float
    l6_nll: float
    l7_nll: float
    l8_nll: float
    replacement_nll: float
    improvement_vs_base: float
    best_prior_nll: float
    degradation_vs_best_prior: float
    anchor_baseline_nll: float
    anchor_replacement_nll: float
    anchor_nll_regression: float
    replacement_parameters: int
    base_model_unchanged: bool
    base_gradients_seen: int


@dataclass(slots=True)
class ReplacementQualityThresholds:
    min_test_examples: int = 2
    min_anchor_examples: int = 4
    min_skipped_layers: int = 1
    min_vs_base_improvement: float = 0.005
    max_degradation_vs_best_prior: float = 0.010
    max_anchor_nll_regression: float = 0.05
    parameter_cap: int = 100_000


@dataclass(slots=True)
class ReplacementQualityDecision:
    status: str
    reasons: list[str]
    evidence: dict[str, Any]
    thresholds: dict[str, Any]


def decide_replacement_quality(evidence: ReplacementQualityEvidence, thresholds=None):
    thresholds = thresholds or ReplacementQualityThresholds()
    reasons = []
    if evidence.test_examples < thresholds.min_test_examples:
        reasons.append("insufficient_test_examples")
    if evidence.anchor_examples < thresholds.min_anchor_examples:
        reasons.append("insufficient_anchor_examples")
    if evidence.skipped_layers < thresholds.min_skipped_layers:
        reasons.append("no_real_block_replacement")
    if evidence.improvement_vs_base < thresholds.min_vs_base_improvement:
        reasons.append("base_quality_gate_failed")
    if evidence.degradation_vs_best_prior > thresholds.max_degradation_vs_best_prior:
        reasons.append("best_prior_noninferiority_failed")
    if evidence.anchor_nll_regression > thresholds.max_anchor_nll_regression:
        reasons.append("general_regression_gate_failed")
    if evidence.replacement_parameters > thresholds.parameter_cap:
        reasons.append("parameter_cap_failed")
    if not evidence.base_model_unchanged:
        reasons.append("base_model_mutated")
    if evidence.base_gradients_seen:
        reasons.append("base_gradient_boundary_failed")
    return ReplacementQualityDecision(
        status="BLOCK_REPLACEMENT_QUALITY_PASS" if not reasons else "BLOCK_REPLACEMENT_QUALITY_BLOCKED",
        reasons=reasons,
        evidence=asdict(evidence),
        thresholds=asdict(thresholds),
    )
