from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from .island_plan import IslandPlanConfig, IslandSensitivity, calibrate_island_sensitivity
from .state_space_region import CortexRegion
from .store import payload_digest


@dataclass(slots=True)
class ScaffoldPlanConfig:
    target_remaining_fraction: float = 0.25
    stage_remaining_fractions: tuple[float, ...] = (0.50, 0.40, 0.32, 0.25)
    min_head_layers: int = 2
    min_tail_layers: int = 2

    def validate(self) -> None:
        if not 0.0 < self.target_remaining_fraction < 1.0:
            raise ValueError("target_remaining_fraction must be in (0,1)")
        if not self.stage_remaining_fractions:
            raise ValueError("stage_remaining_fractions cannot be empty")
        previous = 1.0
        for value in self.stage_remaining_fractions:
            value = float(value)
            if not self.target_remaining_fraction <= value < 1.0:
                raise ValueError("stage remaining fraction out of range")
            if value >= previous:
                raise ValueError("stage_remaining_fractions must strictly decrease")
            previous = value
        if abs(self.stage_remaining_fractions[-1] - self.target_remaining_fraction) > 1e-9:
            raise ValueError("final stage must equal target_remaining_fraction")
        if self.min_head_layers < 1 or self.min_tail_layers < 1:
            raise ValueError("head/tail scaffold must each retain at least one layer")


def _candidate_regions_for_remaining(
    total_layers: int,
    remaining_layers: int,
    config: ScaffoldPlanConfig,
):
    for head in range(
        config.min_head_layers,
        remaining_layers - config.min_tail_layers + 1,
    ):
        tail = remaining_layers - head
        if tail < config.min_tail_layers:
            continue
        start = head
        end = total_layers - tail - 1
        if end < start:
            continue
        yield head, tail, CortexRegion(start, end)


def calibrate_scaffold_regions(
    model,
    encoded_inputs: list[list[int]],
    *,
    config: ScaffoldPlanConfig | None = None,
) -> list[IslandSensitivity]:
    config = config or ScaffoldPlanConfig()
    config.validate()
    total = int(model.config.num_hidden_layers)
    minimum_remaining = config.min_head_layers + config.min_tail_layers
    max_region_width = total - minimum_remaining
    if max_region_width < 2:
        raise ValueError("decoder is too shallow for scaffold shrinking")
    min_region_width = max(
        2,
        total - math.ceil(total * config.stage_remaining_fractions[0]),
    )
    return calibrate_island_sensitivity(
        model,
        encoded_inputs,
        config=IslandPlanConfig(
            target_fraction=1.0 - config.target_remaining_fraction,
            min_width=min_region_width,
            max_width=max_region_width,
            max_islands=1,
            edge_layers_to_keep=min(
                config.min_head_layers,
                config.min_tail_layers,
            ),
            min_gap_layers=0,
        ),
    )


def build_scaffold_plan(
    *,
    total_layers: int,
    sensitivity: list[IslandSensitivity],
    base_model_fingerprint: str,
    dataset_fingerprint: str,
    config: ScaffoldPlanConfig | None = None,
) -> dict[str, Any]:
    config = config or ScaffoldPlanConfig()
    config.validate()
    total = int(total_layers)
    minimum_remaining = config.min_head_layers + config.min_tail_layers
    if total <= minimum_remaining + 1:
        raise ValueError("decoder is too shallow for scaffold shrinking")
    if not sensitivity:
        raise ValueError("scaffold sensitivity evidence is empty")

    score_by_region = {(row.start, row.end): row for row in sensitivity}
    stages = []
    previous_region = None

    for stage_number, fraction in enumerate(
        config.stage_remaining_fractions,
        start=1,
    ):
        remaining = max(minimum_remaining, math.ceil(total * float(fraction)))
        candidates = []
        for head, tail, region in _candidate_regions_for_remaining(
            total,
            remaining,
            config,
        ):
            if previous_region is not None:
                if (
                    region.start > previous_region.start
                    or region.end < previous_region.end
                ):
                    continue
            row = score_by_region.get((region.start, region.end))
            if row is None:
                continue
            imbalance = abs(head - tail) / max(1, remaining)
            selection_score = float(row.score_per_layer) + 0.01 * imbalance
            candidates.append(
                (selection_score, head, tail, region, row)
            )

        if not candidates:
            raise ValueError(
                f"no calibrated nested scaffold candidate for remaining={remaining}"
            )
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        selection_score, head, tail, region, row = candidates[0]
        stages.append(
            {
                "stage": stage_number,
                "head_layers": int(head),
                "tail_layers": int(tail),
                "remaining_qwen_layers": int(head + tail),
                "remaining_qwen_fraction": (head + tail) / total,
                "region": region.to_dict(),
                "replaced_layers": region.width,
                "replaced_fraction": region.width / total,
                "region_score_per_layer": float(row.score_per_layer),
                "selection_score": float(selection_score),
            }
        )
        previous_region = region

    payload = {
        "schema": "NOLANE-L13-SHRINKING-QWEN-SCAFFOLD-PLAN-V1",
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


def verify_scaffold_plan(
    plan: dict[str, Any],
    *,
    base_model_fingerprint: str,
    dataset_fingerprint: str,
) -> None:
    if plan.get("schema") != "NOLANE-L13-SHRINKING-QWEN-SCAFFOLD-PLAN-V1":
        raise ValueError("unsupported shrinking scaffold plan")
    supplied = plan.get("plan_sha256")
    body = dict(plan)
    body.pop("plan_sha256", None)
    if payload_digest(body) != supplied:
        raise ValueError("shrinking scaffold plan digest mismatch")
    if plan.get("base_model_fingerprint") != base_model_fingerprint:
        raise ValueError("shrinking scaffold plan base-model mismatch")
    if plan.get("dataset_fingerprint") != dataset_fingerprint:
        raise ValueError("shrinking scaffold plan dataset mismatch")

    total = int(plan["total_layers"])
    previous_region = None
    previous_remaining = total + 1
    for row in plan.get("stages", []):
        head = int(row["head_layers"])
        tail = int(row["tail_layers"])
        remaining = int(row["remaining_qwen_layers"])
        if head + tail != remaining:
            raise ValueError("scaffold remaining-layer accounting mismatch")
        region = CortexRegion(
            int(row["region"]["start"]),
            int(row["region"]["end"]),
        )
        region.validate(total_layers=total)
        if remaining + region.width != total:
            raise ValueError("scaffold region/anchor accounting mismatch")
        if remaining >= previous_remaining:
            raise ValueError("Qwen scaffold did not shrink monotonically")
        if previous_region is not None:
            if (
                region.start > previous_region.start
                or region.end < previous_region.end
            ):
                raise ValueError("replacement region did not grow monotonically")
        previous_remaining = remaining
        previous_region = region

    if previous_region is None:
        raise ValueError("shrinking scaffold plan has no stages")
    target = plan.get("target", {})
    if int(target.get("remaining_qwen_layers", -1)) != previous_remaining:
        raise ValueError("scaffold target does not match final stage")
