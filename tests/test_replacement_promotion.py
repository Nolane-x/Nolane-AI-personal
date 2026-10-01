from nolane_personal.replacement_promotion import decide_replacement_promotion


def test_replacement_promotion_requires_both_courts_same_checkpoint():
    good = decide_replacement_promotion(
        quality_status="BLOCK_REPLACEMENT_QUALITY_PASS",
        quality_checkpoint_sha256="abc",
        resource_status="BLOCK_REPLACEMENT_RESOURCE_PASS",
        resource_checkpoint_sha256="abc",
    )
    assert good.status == "BLOCK_REPLACEMENT_PROMOTION_PASS"

    bad = decide_replacement_promotion(
        quality_status="BLOCK_REPLACEMENT_QUALITY_PASS",
        quality_checkpoint_sha256="abc",
        resource_status="BLOCK_REPLACEMENT_RESOURCE_PASS",
        resource_checkpoint_sha256="def",
    )
    assert "court_checkpoint_mismatch" in bad.reasons

    bad = decide_replacement_promotion(
        quality_status="BLOCK_REPLACEMENT_QUALITY_BLOCKED",
        quality_checkpoint_sha256="abc",
        resource_status="BLOCK_REPLACEMENT_RESOURCE_PASS",
        resource_checkpoint_sha256="abc",
    )
    assert "quality_court_not_passed" in bad.reasons
