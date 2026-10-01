from nolane_personal.depth_bridge_resources import (
    DepthBridgeResourceEvidence,
    decide_depth_bridge_resources,
)


def _evidence(**overrides):
    values = dict(
        prompts=8,
        baseline_median_ms=100.0,
        depth_bridge_median_ms=120.0,
        latency_ratio_vs_base=1.20,
        artifact_bytes=500_000,
        bridge_parameters=84_289,
    )
    values.update(overrides)
    return DepthBridgeResourceEvidence(**values)


def test_resource_court_blocks_expensive_depth_bridge():
    assert decide_depth_bridge_resources(_evidence()).status == "DEPTH_BRIDGE_RESOURCE_PASS"

    bad = decide_depth_bridge_resources(_evidence(latency_ratio_vs_base=1.75))
    assert bad.status == "DEPTH_BRIDGE_RESOURCE_BLOCKED"
    assert "latency_overhead_gate_failed" in bad.reasons

    bad = decide_depth_bridge_resources(_evidence(artifact_bytes=3_000_000))
    assert "artifact_size_gate_failed" in bad.reasons
