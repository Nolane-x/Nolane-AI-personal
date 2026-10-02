from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class QuantizedPromotionDecision:
    status: str
    reasons: list[str]
    checkpoint_sha256: str | None
    source_factorized_checkpoint_sha256: str | None
    source_l16_checkpoint_sha256: str | None


def decide_quantized_promotion(
    *,
    quality_status,
    quality_checkpoint_sha256,
    quality_source_factorized_checkpoint_sha256,
    quality_source_l16_checkpoint_sha256,
    resource_status,
    resource_checkpoint_sha256,
    resource_source_factorized_checkpoint_sha256,
    resource_source_l16_checkpoint_sha256,
):
    reasons=[]
    if quality_status!="QUANTIZED_FACTOR_QUALITY_PASS": reasons.append("quality_court_not_passed")
    if resource_status!="QUANTIZED_FACTOR_RESOURCE_PASS": reasons.append("resource_court_not_passed")
    if not quality_checkpoint_sha256 or quality_checkpoint_sha256!=resource_checkpoint_sha256:
        reasons.append("quantized_checkpoint_identity_mismatch")
    if (
        not quality_source_factorized_checkpoint_sha256
        or quality_source_factorized_checkpoint_sha256!=resource_source_factorized_checkpoint_sha256
    ):
        reasons.append("source_factorized_checkpoint_mismatch")
    if (
        not quality_source_l16_checkpoint_sha256
        or quality_source_l16_checkpoint_sha256!=resource_source_l16_checkpoint_sha256
    ):
        reasons.append("source_l16_checkpoint_mismatch")
    return QuantizedPromotionDecision(
        status="QUANTIZED_FACTOR_PROMOTION_PASS" if not reasons else "QUANTIZED_FACTOR_PROMOTION_BLOCKED",
        reasons=reasons,
        checkpoint_sha256=quality_checkpoint_sha256 if not reasons else None,
        source_factorized_checkpoint_sha256=quality_source_factorized_checkpoint_sha256 if not reasons else None,
        source_l16_checkpoint_sha256=quality_source_l16_checkpoint_sha256 if not reasons else None,
    )
