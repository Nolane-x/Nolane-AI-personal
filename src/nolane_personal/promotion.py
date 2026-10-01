from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class PromotionThresholds:
    min_state_test_cases: int = 50
    min_return_test_cases: int = 20
    state_mae_max: float = 0.030
    state_max_error_max: float = 0.200
    brier_improvement_min: float = 0.020
    parameter_cap: int = 100_000


@dataclass(slots=True)
class PromotionEvidence:
    protocol_verified: bool
    checkpoint_protocol_match: bool
    state_test_cases: int
    return_test_cases: int
    state_mae: float
    state_max_error: float
    baseline_brier: float
    candidate_brier: float
    parameter_count: int


@dataclass(slots=True)
class PromotionDecision:
    status: str
    reasons: list[str]
    evidence: dict[str, Any]
    thresholds: dict[str, Any]


def decide_promotion(
    evidence: PromotionEvidence,
    thresholds: PromotionThresholds | None = None,
) -> PromotionDecision:
    thresholds = thresholds or PromotionThresholds()
    reasons: list[str] = []

    if not evidence.protocol_verified:
        reasons.append("replay_protocol_not_verified")
    if not evidence.checkpoint_protocol_match:
        reasons.append("checkpoint_protocol_mismatch")
    if evidence.state_test_cases < thresholds.min_state_test_cases:
        reasons.append("insufficient_state_test_cases")
    if evidence.return_test_cases < thresholds.min_return_test_cases:
        reasons.append("insufficient_return_test_cases")
    if evidence.state_mae > thresholds.state_mae_max:
        reasons.append("state_mae_gate_failed")
    if evidence.state_max_error > thresholds.state_max_error_max:
        reasons.append("state_max_error_gate_failed")
    improvement = evidence.baseline_brier - evidence.candidate_brier
    if improvement < thresholds.brier_improvement_min:
        reasons.append("return_forecast_gate_failed")
    if evidence.parameter_count > thresholds.parameter_cap:
        reasons.append("parameter_cap_failed")

    status = "PROMOTION_PASS" if not reasons else "PROMOTION_BLOCKED"
    return PromotionDecision(
        status=status,
        reasons=reasons,
        evidence=asdict(evidence) | {"brier_improvement": improvement},
        thresholds=asdict(thresholds),
    )
