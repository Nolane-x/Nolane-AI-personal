from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from .island_plan import IslandPlanConfig, IslandSensitivity, calibrate_island_sensitivity
from .state_space_region import CortexRegion
from .store import payload_digest


@dataclass(slots=True)
class AnchorPlanConfig:
    stage_remaining_fractions: tuple[float, ...] = (0.20, 0.14, 0.08)
    target_remaining_layers: int = 2
    min_head_layers: int = 1
    min_tail_layers: int = 1

    def validate(self) -> None:
        if not self.stage_remaining_fractions:
            raise ValueError("stage_remaining_fractions cannot be empty")
        previous = 1.0
        for value in self.stage_remaining_fractions:
            value = float(value)
            if not 0.0 < value < 1.0:
                raise ValueError("stage remaining fraction out of range")
            if value >= previous:
                raise ValueError("stage_remaining_fractions must strictly decrease")
            previous = value
        if self.target_remaining_layers < 2:
            raise ValueError("target_remaining_layers must be >=2")
        if self.min_head_layers < 1 or self.min_tail_layers < 1:
            raise ValueError("head/tail anchors must each retain at least one layer")
        if self.target_remaining_layers < self.min_head_layers + self.min_tail_layers:
            raise ValueError("target remaining layers cannot satisfy anchor minima")


def _remaining_schedule(total: int, config: AnchorPlanConfig) -> list[int]:
    minimum = max(
        config.target_remaining_layers,
        config.min_head_layers + config.min_tail_layers,
    )
    values = []
    for fraction in config.stage_remaining_fractions:
        remaining = max(minimum, math.ceil(total * float(fraction)))
        if remaining >= total:
            continue
        if not values or remaining < values[-1]:
            values.append(remaining)
    if not values or values[-1] != minimum:
        values.append(minimum)
    return values


def _candidate_regions(total_layers: int, remaining_layers: int, config: AnchorPlanConfig):
    for head in range(config.min_head_layers, remaining_layers - config.min_tail_layers + 1):
        tail = remaining_layers - head
        if tail < config.min_tail_layers:
            continue
        start = head
        end = total_layers - tail - 1
        if end < start:
            continue
        yield head, tail, CortexRegion(start, end)


def calibrate_anchor_regions(
    model,
    encoded_inputs: list[list[int]],
    *,
    config: AnchorPlanConfig | None = None,
) -> list[IslandSensitivity]:
    config = config or AnchorPlanConfig()
    config.validate()
    total = int(model.config.num_hidden_layers)
    schedule = _remaining_schedule(total, config)
    if not schedule:
        raise ValueError("decoder is too shallow for minimal-anchor shrinking")
    min_region_width = total - schedule[0]
    max_region_width = total - schedule[-1]
    if min_region_width < 2 or max_region_width < min_region_width:
        raise ValueError("invalid minimal-anchor region widths")
    return calibrate_island_sensitivity(
        model,
        encoded_inputs,
        config=IslandPlanConfig(
            target_fraction=max_region_width / total,
            min_width=min_region_width,
            max_width=max_region_width,
            max_islands=1,
            edge_layers_to_keep=min(config.min_head_layers, config.min_tail_layers),
            min_gap_layers=0,
        ),
    )


def build_anchor_plan(
    *,
    total_layers: int,
    sensitivity: list[IslandSensitivity],
    base_model_fingerprint: str,
    dataset_fingerprint: str,
    config: AnchorPlanConfig | None = None,
) -> dict[str, Any]:
    config = config or AnchorPlanConfig()
    config.validate()
    total = int(total_layers)
    if total <= config.target_remaining_layers + 1:
        raise ValueError("decoder is too shallow for minimal-anchor shrinking")
    if not sensitivity:
        raise ValueError("anchor sensitivity evidence is empty")

    schedule = _remaining_schedule(total, config)
    scores = {(row.start, row.end): row for row in sensitivity}
    stages = []
    previous_region = None

    for stage_number, remaining in enumerate(schedule, start=1):
        candidates = []
        for head, tail, region in _candidate_regions(total, remaining, config):
            if previous_region is not None:
                if region.start > previous_region.start or region.end < previous_region.end:
                    continue
            row = scores.get((region.start, region.end))
            if row is None:
                continue
            imbalance = abs(head - tail) / max(1, remaining)
            anchor_penalty = 0.005 * imbalance
            score = float(row.score_per_layer) + anchor_penalty
            candidates.append((score, head, tail, region, row))

        if not candidates:
            raise ValueError(
                f"no calibrated nested minimal-anchor candidate for remaining={remaining}"
            )
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        score, head, tail, region, row = candidates[0]
        stages.append(
            {
                "stage": stage_number,
                "head_layers": int(head),
                "tail_layers": int(tail),
                "remaining_qwen_layers": int(remaining),
                "remaining_qwen_fraction": remaining / total,
                "region": region.to_dict(),
                "replaced_layers": region.width,
                "replaced_fraction": region.width / total,
                "region_score_per_layer": float(row.score_per_layer),
                "selection_score": float(score),
            }
        )
        previous_region = region

    payload = {
        "schema": "NOLANE-L14-MINIMAL-QWEN-ANCHOR-PLAN-V1",
        "authority": "FROZEN_TRAINING_PLAN_ONLY",
        "base_model_fingerprint": str(base_model_fingerprint),
        "dataset_fingerprint": str(dataset_fingerprint),
        "total_layers": total,
        "config": asdict(config),
        "sensitivity": [row.to_dict() for row in sensitivity],
        "stages": stages,
        "target": dict(stages[-1]),
    }
    payload["plan_sha256"] = payload_digest(payload)
    return payload


def verify_anchor_plan(
    plan: dict[str, Any],
    *,
    base_model_fingerprint: str,
    dataset_fingerprint: str,
) -> None:
    if plan.get("schema") != "NOLANE-L14-MINIMAL-QWEN-ANCHOR-PLAN-V1":
        raise ValueError("unsupported minimal-anchor plan")
    supplied = plan.get("plan_sha256")
    body = dict(plan)
    body.pop("plan_sha256", None)
    if payload_digest(body) != supplied:
        raise ValueError("minimal-anchor plan digest mismatch")
    if plan.get("base_model_fingerprint") != base_model_fingerprint:
        raise ValueError("minimal-anchor plan base-model mismatch")
    if plan.get("dataset_fingerprint") != dataset_fingerprint:
        raise ValueError("minimal-anchor plan dataset mismatch")

    total = int(plan["total_layers"])
    config = AnchorPlanConfig(**plan["config"])
    previous_region = None
    previous_remaining = total + 1
    for row in plan.get("stages", []):
        head = int(row["head_layers"])
        tail = int(row["tail_layers"])
        remaining = int(row["remaining_qwen_layers"])
        if head < config.min_head_layers or tail < config.min_tail_layers:
            raise ValueError("minimal-anchor plan violated anchor minima")
        if head + tail != remaining:
            raise ValueError("minimal-anchor layer accounting mismatch")
        region = CortexRegion(int(row["region"]["start"]), int(row["region"]["end"]))
        region.validate(total_layers=total)
        if region.width + remaining != total:
            raise ValueError("minimal-anchor region accounting mismatch")
        if remaining >= previous_remaining:
            raise ValueError("minimal-anchor Qwen shell did not shrink monotonically")
        if previous_region is not None:
            if region.start > previous_region.start or region.end < previous_region.end:
                raise ValueError("minimal-anchor replacement region did not grow")
        previous_region = region
        previous_remaining = remaining

    if previous_region is None:
        raise ValueError("minimal-anchor plan has no stages")
    target = plan.get("target", {})
    if int(target.get("remaining_qwen_layers", -1)) != config.target_remaining_layers:
        raise ValueError("minimal-anchor final target layer count mismatch")
    if int(target.get("head_layers", -1)) < config.min_head_layers:
        raise ValueError("minimal-anchor final head missing")
    if int(target.get("tail_layers", -1)) < config.min_tail_layers:
        raise ValueError("minimal-anchor final tail missing")
