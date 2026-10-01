from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ReplacementPromotionDecision:
    status: str
    reasons: list[str]
    checkpoint_sha256: str | None


def decide_replacement_promotion(
    *,
    quality_status: str,
    quality_checkpoint_sha256: str | None,
    resource_status: str,
    resource_checkpoint_sha256: str | None,
) -> ReplacementPromotionDecision:
    reasons = []
    if quality_status != "BLOCK_REPLACEMENT_QUALITY_PASS":
        reasons.append("quality_court_not_passed")
    if resource_status != "BLOCK_REPLACEMENT_RESOURCE_PASS":
        reasons.append("resource_court_not_passed")
    if not quality_checkpoint_sha256 or not resource_checkpoint_sha256:
        reasons.append("checkpoint_identity_missing")
    elif quality_checkpoint_sha256 != resource_checkpoint_sha256:
        reasons.append("court_checkpoint_mismatch")
    checkpoint = quality_checkpoint_sha256 if quality_checkpoint_sha256 == resource_checkpoint_sha256 else None
    return ReplacementPromotionDecision(
        status="BLOCK_REPLACEMENT_PROMOTION_PASS" if not reasons else "BLOCK_REPLACEMENT_PROMOTION_BLOCKED",
        reasons=reasons,
        checkpoint_sha256=checkpoint,
    )
