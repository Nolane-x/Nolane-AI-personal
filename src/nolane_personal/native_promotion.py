from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class NativePromotionDecision:
    status: str
    reasons: list[str]
    checkpoint_sha256: str | None
    spec_sha256: str | None


def decide_native_promotion(
    *,
    quality_status: str,
    quality_checkpoint_sha256: str | None,
    quality_spec_sha256: str | None,
    resource_status: str,
    resource_checkpoint_sha256: str | None,
    resource_spec_sha256: str | None,
) -> NativePromotionDecision:
    reasons = []
    if quality_status != "NATIVE_BOUNDARY_QUALITY_PASS":
        reasons.append("quality_court_not_passed")
    if resource_status != "NATIVE_BOUNDARY_RESOURCE_PASS":
        reasons.append("resource_court_not_passed")
    if not quality_checkpoint_sha256 or not resource_checkpoint_sha256:
        reasons.append("checkpoint_identity_missing")
    elif quality_checkpoint_sha256 != resource_checkpoint_sha256:
        reasons.append("court_checkpoint_mismatch")
    if not quality_spec_sha256 or not resource_spec_sha256:
        reasons.append("spec_identity_missing")
    elif quality_spec_sha256 != resource_spec_sha256:
        reasons.append("court_spec_mismatch")
    return NativePromotionDecision(
        status="NATIVE_BOUNDARY_PROMOTION_PASS" if not reasons else "NATIVE_BOUNDARY_PROMOTION_BLOCKED",
        reasons=reasons,
        checkpoint_sha256=(
            quality_checkpoint_sha256
            if quality_checkpoint_sha256 == resource_checkpoint_sha256
            else None
        ),
        spec_sha256=(
            quality_spec_sha256
            if quality_spec_sha256 == resource_spec_sha256
            else None
        ),
    )
