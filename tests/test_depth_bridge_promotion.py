from nolane_personal.depth_bridge_promotion import decide_depth_bridge_promotion


def test_promotion_requires_both_courts_and_same_checkpoint():
    good = decide_depth_bridge_promotion(
        quality_status="DEPTH_BRIDGE_QUALITY_PASS",
        quality_checkpoint_sha256="abc",
        resource_status="DEPTH_BRIDGE_RESOURCE_PASS",
        resource_checkpoint_sha256="abc",
    )
    assert good.status == "DEPTH_BRIDGE_PROMOTION_PASS"

    mismatch = decide_depth_bridge_promotion(
        quality_status="DEPTH_BRIDGE_QUALITY_PASS",
        quality_checkpoint_sha256="abc",
        resource_status="DEPTH_BRIDGE_RESOURCE_PASS",
        resource_checkpoint_sha256="def",
    )
    assert mismatch.status == "DEPTH_BRIDGE_PROMOTION_BLOCKED"
    assert "court_checkpoint_mismatch" in mismatch.reasons

    weak = decide_depth_bridge_promotion(
        quality_status="DEPTH_BRIDGE_QUALITY_BLOCKED",
        quality_checkpoint_sha256="abc",
        resource_status="DEPTH_BRIDGE_RESOURCE_PASS",
        resource_checkpoint_sha256="abc",
    )
    assert "quality_court_not_passed" in weak.reasons
