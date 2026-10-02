from __future__ import annotations
from dataclasses import dataclass

@dataclass(slots=True)
class FactorizedPromotionDecision:
    status:str
    reasons:list[str]
    checkpoint_sha256:str|None
    source_l16_checkpoint_sha256:str|None

def decide_factorized_promotion(*,quality_status,quality_checkpoint_sha256,quality_source_l16_checkpoint_sha256,resource_status,resource_checkpoint_sha256,resource_source_l16_checkpoint_sha256):
    reasons=[]
    if quality_status!="FACTORIZED_BOUNDARY_QUALITY_PASS": reasons.append("quality_court_not_passed")
    if resource_status!="FACTORIZED_BOUNDARY_RESOURCE_PASS": reasons.append("resource_court_not_passed")
    if not quality_checkpoint_sha256 or not resource_checkpoint_sha256: reasons.append("checkpoint_identity_missing")
    elif quality_checkpoint_sha256!=resource_checkpoint_sha256: reasons.append("court_checkpoint_mismatch")
    if not quality_source_l16_checkpoint_sha256 or not resource_source_l16_checkpoint_sha256: reasons.append("source_l16_identity_missing")
    elif quality_source_l16_checkpoint_sha256!=resource_source_l16_checkpoint_sha256: reasons.append("source_l16_checkpoint_mismatch")
    return FactorizedPromotionDecision(
        status="FACTORIZED_BOUNDARY_PROMOTION_PASS" if not reasons else "FACTORIZED_BOUNDARY_PROMOTION_BLOCKED",
        reasons=reasons,
        checkpoint_sha256=quality_checkpoint_sha256 if quality_checkpoint_sha256==resource_checkpoint_sha256 else None,
        source_l16_checkpoint_sha256=quality_source_l16_checkpoint_sha256 if quality_source_l16_checkpoint_sha256==resource_source_l16_checkpoint_sha256 else None,
    )
