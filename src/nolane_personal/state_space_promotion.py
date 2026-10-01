from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class StateSpacePromotionDecision:
    status: str
    reasons: list[str]
    checkpoint_sha256: str | None
    plan_sha256: str | None


def decide_state_space_promotion(
    *,
    quality_status: str,
    quality_checkpoint_sha256: str | None,
    quality_plan_sha256: str | None,
    resource_status: str,
    resource_checkpoint_sha256: str | None,
    resource_plan_sha256: str | None,
) -> StateSpacePromotionDecision:
    reasons = []
    if quality_status != "STATE_SPACE_CORTEX_QUALITY_PASS":
        reasons.append("quality_court_not_passed")
    if resource_status != "STATE_SPACE_CORTEX_RESOURCE_PASS":
        reasons.append("resource_court_not_passed")
    if not quality_checkpoint_sha256 or not resource_checkpoint_sha256:
        reasons.append("checkpoint_identity_missing")
    elif quality_checkpoint_sha256 != resource_checkpoint_sha256:
        reasons.append("court_checkpoint_mismatch")
    if not quality_plan_sha256 or not resource_plan_sha256:
        reasons.append("plan_identity_missing")
    elif quality_plan_sha256 != resource_plan_sha256:
        reasons.append("court_plan_mismatch")
    return StateSpacePromotionDecision(
        status="STATE_SPACE_CORTEX_PROMOTION_PASS" if not reasons else "STATE_SPACE_CORTEX_PROMOTION_BLOCKED",
        reasons=reasons,
        checkpoint_sha256=(
            quality_checkpoint_sha256
            if quality_checkpoint_sha256 == resource_checkpoint_sha256
            else None
        ),
        plan_sha256=(
            quality_plan_sha256
            if quality_plan_sha256 == resource_plan_sha256
            else None
        ),
    )
