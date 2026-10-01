from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from .island_plan import IslandPlanConfig, IslandSensitivity, calibrate_island_sensitivity
from .state_space_region import CortexRegion
from .store import payload_digest


@dataclass(slots=True)
class StateSpacePlanConfig:
    target_fraction: float = 0.60
    stage_fractions: tuple[float, ...] = (0.20, 0.35, 0.50, 0.60)
    edge_layers_to_keep: int = 2
    min_region_width: int = 4
    max_region_width: int = 20

    def validate(self) -> None:
        if not 0.0 < self.target_fraction < 1.0:
            raise ValueError("target_fraction must be in (0,1)")
        if not self.stage_fractions:
            raise ValueError("stage_fractions cannot be empty")
        previous = 0.0
        for value in self.stage_fractions:
            if not 0.0 < float(value) <= self.target_fraction:
                raise ValueError("stage fraction out of range")
            if float(value) <= previous:
                raise ValueError("stage_fractions must be strictly increasing")
            previous = float(value)
        if abs(self.stage_fractions[-1] - self.target_fraction) > 1e-9:
            raise ValueError("final stage fraction must equal target_fraction")
        if self.edge_layers_to_keep < 1:
            raise ValueError("edge_layers_to_keep must be >=1")
        if self.min_region_width < 2:
            raise ValueError("min_region_width must be >=2")
        if self.max_region_width < self.min_region_width:
            raise ValueError("max_region_width must be >= min_region_width")


def calibrate_state_space_regions(
    model,
    encoded_inputs: list[list[int]],
    *,
    config: StateSpacePlanConfig | None = None,
) -> list[IslandSensitivity]:
    config = config or StateSpacePlanConfig()
    config.validate()
    return calibrate_island_sensitivity(
        model,
        encoded_inputs,
        config=IslandPlanConfig(
            target_fraction=config.target_fraction,
            min_width=config.min_region_width,
            max_width=config.max_region_width,
            max_islands=1,
            edge_layers_to_keep=config.edge_layers_to_keep,
            min_gap_layers=0,
        ),
    )


def _target_widths(total_layers: int, config: StateSpacePlanConfig) -> list[int]:
    interior = total_layers - 2 * config.edge_layers_to_keep
    max_width = min(config.max_region_width, interior)
    if max_width < config.min_region_width:
        raise ValueError("not enough unprotected decoder depth for state-space region")
    widths = []
    for fraction in config.stage_fractions:
        width = max(config.min_region_width, math.ceil(total_layers * float(fraction)))
        width = min(max_width, width)
        if not widths or width > widths[-1]:
            widths.append(width)
    if widths[-1] < min(max_width, math.ceil(total_layers * config.target_fraction)):
        widths.append(min(max_width, math.ceil(total_layers * config.target_fraction)))
    return widths


def _contains(outer: IslandSensitivity, inner: IslandSensitivity) -> bool:
    return outer.start <= inner.start and outer.end >= inner.end


def build_state_space_plan(
    *,
    total_layers: int,
    sensitivity: list[IslandSensitivity],
    base_model_fingerprint: str,
    dataset_fingerprint: str,
    config: StateSpacePlanConfig | None = None,
) -> dict[str, Any]:
    config = config or StateSpacePlanConfig()
    config.validate()
    total = int(total_layers)
    widths = _target_widths(total, config)
    by_width: dict[int, list[IslandSensitivity]] = {}
    for row in sensitivity:
        if row.start < config.edge_layers_to_keep:
            continue
        if row.end >= total - config.edge_layers_to_keep:
            continue
        by_width.setdefault(row.width, []).append(row)
    for rows in by_width.values():
        rows.sort(key=lambda row: (row.score_per_layer, row.transformation_score, row.start))

    stages = []
    previous = None
    for stage_number, width in enumerate(widths, start=1):
        candidates = list(by_width.get(width, []))
        if previous is not None:
            candidates = [row for row in candidates if _contains(row, previous)]
        if not candidates:
            raise ValueError(f"no calibrated nested state-space region for width {width}")
        chosen = candidates[0]
        region = CortexRegion(chosen.start, chosen.end)
        stages.append(
            {
                "stage": stage_number,
                "region": region.to_dict(),
                "width": region.width,
                "replaced_fraction": region.width / total,
                "score_per_layer": float(chosen.score_per_layer),
                "transformation_score": float(chosen.transformation_score),
            }
        )
        previous = chosen

    final_region = stages[-1]["region"]
    payload = {
        "schema": "NOLANE-L12-STATE-SPACE-REGION-PLAN-V1",
        "authority": "FROZEN_TRAINING_PLAN_ONLY",
        "base_model_fingerprint": str(base_model_fingerprint),
        "dataset_fingerprint": str(dataset_fingerprint),
        "total_layers": total,
        "config": asdict(config),
        "sensitivity": [row.to_dict() for row in sensitivity],
        "stages": stages,
        "target_region": final_region,
        "target_width": int(stages[-1]["width"]),
        "target_fraction_actual": float(stages[-1]["replaced_fraction"]),
    }
    payload["plan_sha256"] = payload_digest(payload)
    return payload


def verify_state_space_plan(
    plan: dict[str, Any],
    *,
    base_model_fingerprint: str,
    dataset_fingerprint: str,
) -> None:
    if plan.get("schema") != "NOLANE-L12-STATE-SPACE-REGION-PLAN-V1":
        raise ValueError("unsupported state-space region plan")
    supplied = plan.get("plan_sha256")
    body = dict(plan)
    body.pop("plan_sha256", None)
    if payload_digest(body) != supplied:
        raise ValueError("state-space region plan digest mismatch")
    if plan.get("base_model_fingerprint") != base_model_fingerprint:
        raise ValueError("state-space plan base-model mismatch")
    if plan.get("dataset_fingerprint") != dataset_fingerprint:
        raise ValueError("state-space plan dataset mismatch")

    total = int(plan["total_layers"])
    previous = None
    for row in plan.get("stages", []):
        region = CortexRegion(int(row["region"]["start"]), int(row["region"]["end"]))
        region.validate(total_layers=total)
        if int(row["width"]) != region.width:
            raise ValueError("state-space stage width mismatch")
        if previous is not None:
            if region.start > previous.start or region.end < previous.end:
                raise ValueError("state-space stages are not nested")
            if region.width <= previous.width:
                raise ValueError("state-space stage width did not grow")
        previous = region
    if previous is None:
        raise ValueError("state-space plan has no stages")
    target = plan.get("target_region", {})
    if previous.to_dict() != {
        "start": int(target.get("start", -1)),
        "end": int(target.get("end", -1)),
    }:
        raise ValueError("final state-space stage does not match target region")
