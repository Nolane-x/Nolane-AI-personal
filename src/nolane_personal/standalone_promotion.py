from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class StandalonePromotionDecision:
    status: str
    reasons: list[str]
    checkpoint_sha256: str | None
    source_l15_checkpoint_sha256: str | None


def decide_standalone_promotion(
    *,
    parity_status: str,
    parity_checkpoint_sha256: str | None,
    parity_source_l15_checkpoint_sha256: str | None,
    resource_status: str,
    resource_checkpoint_sha256: str | None,
    resource_source_l15_checkpoint_sha256: str | None,
) -> StandalonePromotionDecision:
    reasons = []
    if parity_status != "STANDALONE_PARITY_PASS":
        reasons.append("parity_court_not_passed")
    if resource_status != "STANDALONE_RESOURCE_PASS":
        reasons.append("resource_court_not_passed")
    if not parity_checkpoint_sha256 or not resource_checkpoint_sha256:
        reasons.append("checkpoint_identity_missing")
    elif parity_checkpoint_sha256 != resource_checkpoint_sha256:
        reasons.append("court_checkpoint_mismatch")
    if not parity_source_l15_checkpoint_sha256 or not resource_source_l15_checkpoint_sha256:
        reasons.append("source_l15_identity_missing")
    elif parity_source_l15_checkpoint_sha256 != resource_source_l15_checkpoint_sha256:
        reasons.append("source_l15_checkpoint_mismatch")
    return StandalonePromotionDecision(
        status="STANDALONE_PROMOTION_PASS" if not reasons else "STANDALONE_PROMOTION_BLOCKED",
        reasons=reasons,
        checkpoint_sha256=(
            parity_checkpoint_sha256
            if parity_checkpoint_sha256 == resource_checkpoint_sha256
            else None
        ),
        source_l15_checkpoint_sha256=(
            parity_source_l15_checkpoint_sha256
            if parity_source_l15_checkpoint_sha256 == resource_source_l15_checkpoint_sha256
            else None
        ),
    )
