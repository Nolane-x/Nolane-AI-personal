from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RankFrontierPromotionDecision:
    status: str
    reasons: list[str]
    checkpoint_sha256: str | None
    source_l16_checkpoint_sha256: str | None
    selected_rank: int | None


def decide_rank_frontier_promotion(
    *,
    frontier_selected_rank: int | None,
    frontier_checkpoint_sha256: str | None,
    frontier_source_l16_checkpoint_sha256: str | None,
    quality_status: str,
    quality_rank: int | None,
    quality_checkpoint_sha256: str | None,
    quality_source_l16_checkpoint_sha256: str | None,
    resource_status: str,
    resource_checkpoint_sha256: str | None,
    resource_source_l16_checkpoint_sha256: str | None,
) -> RankFrontierPromotionDecision:
    reasons = []
    if quality_status != "FACTORIZED_BOUNDARY_QUALITY_PASS":
        reasons.append("quality_court_not_passed")
    if resource_status != "FACTORIZED_BOUNDARY_RESOURCE_PASS":
        reasons.append("resource_court_not_passed")
    if frontier_selected_rank is None or quality_rank is None:
        reasons.append("selected_rank_missing")
    elif int(frontier_selected_rank) != int(quality_rank):
        reasons.append("selected_rank_quality_mismatch")

    checkpoints = {
        value
        for value in (
            frontier_checkpoint_sha256,
            quality_checkpoint_sha256,
            resource_checkpoint_sha256,
        )
        if value
    }
    if len(checkpoints) != 1 or not frontier_checkpoint_sha256:
        reasons.append("checkpoint_identity_mismatch")

    sources = {
        value
        for value in (
            frontier_source_l16_checkpoint_sha256,
            quality_source_l16_checkpoint_sha256,
            resource_source_l16_checkpoint_sha256,
        )
        if value
    }
    if len(sources) != 1 or not frontier_source_l16_checkpoint_sha256:
        reasons.append("source_l16_identity_mismatch")

    return RankFrontierPromotionDecision(
        status=(
            "ADAPTIVE_RANK_FRONTIER_PROMOTION_PASS"
            if not reasons
            else "ADAPTIVE_RANK_FRONTIER_PROMOTION_BLOCKED"
        ),
        reasons=reasons,
        checkpoint_sha256=frontier_checkpoint_sha256 if not reasons else None,
        source_l16_checkpoint_sha256=(
            frontier_source_l16_checkpoint_sha256 if not reasons else None
        ),
        selected_rank=int(frontier_selected_rank) if frontier_selected_rank is not None else None,
    )
