from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class DepthBridgeResourceEvidence:
    prompts: int
    baseline_median_ms: float
    depth_bridge_median_ms: float
    latency_ratio_vs_base: float
    artifact_bytes: int
    bridge_parameters: int


@dataclass(slots=True)
class DepthBridgeResourceThresholds:
    min_prompts: int = 4
    max_latency_ratio_vs_base: float = 1.50
    max_artifact_bytes: int = 2_000_000
    parameter_cap: int = 100_000


@dataclass(slots=True)
class DepthBridgeResourceDecision:
    status: str
    reasons: list[str]
    evidence: dict[str, Any]
    thresholds: dict[str, Any]


def decide_depth_bridge_resources(
    evidence: DepthBridgeResourceEvidence,
    thresholds: DepthBridgeResourceThresholds | None = None,
) -> DepthBridgeResourceDecision:
    thresholds = thresholds or DepthBridgeResourceThresholds()
    reasons: list[str] = []
    if evidence.prompts < thresholds.min_prompts:
        reasons.append("insufficient_resource_prompts")
    if evidence.baseline_median_ms <= 0 or evidence.depth_bridge_median_ms <= 0:
        reasons.append("invalid_latency_measurement")
    if evidence.latency_ratio_vs_base > thresholds.max_latency_ratio_vs_base:
        reasons.append("latency_overhead_gate_failed")
    if evidence.artifact_bytes > thresholds.max_artifact_bytes:
        reasons.append("artifact_size_gate_failed")
    if evidence.bridge_parameters > thresholds.parameter_cap:
        reasons.append("parameter_cap_failed")
    return DepthBridgeResourceDecision(
        status="DEPTH_BRIDGE_RESOURCE_PASS" if not reasons else "DEPTH_BRIDGE_RESOURCE_BLOCKED",
        reasons=reasons,
        evidence=asdict(evidence),
        thresholds=asdict(thresholds),
    )
