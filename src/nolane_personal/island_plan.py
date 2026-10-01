from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from .island_replacement import TransformerIsland, normalize_islands
from .store import payload_digest
from .surgery import resolve_transformer_layers


@dataclass(slots=True)
class IslandPlanConfig:
    target_fraction: float = 0.35
    min_width: int = 2
    max_width: int = 4
    max_islands: int = 4
    edge_layers_to_keep: int = 1
    min_gap_layers: int = 1

    def validate(self) -> None:
        if not 0.0 < self.target_fraction < 1.0:
            raise ValueError("target_fraction must be in (0,1)")
        if self.min_width < 2:
            raise ValueError("min_width must be >=2")
        if self.max_width < self.min_width:
            raise ValueError("max_width must be >= min_width")
        if self.max_islands < 1:
            raise ValueError("max_islands must be >=1")
        if self.edge_layers_to_keep < 1:
            raise ValueError("edge_layers_to_keep must be >=1")
        if self.min_gap_layers < 0:
            raise ValueError("min_gap_layers must be >=0")


@dataclass(slots=True)
class IslandSensitivity:
    start: int
    end: int
    width: int
    residual_rms_ratio: float
    cosine_change: float
    transformation_score: float
    score_per_layer: float

    @property
    def island(self) -> TransformerIsland:
        return TransformerIsland(self.start, self.end)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _capture_all_layer_io(model, input_ids):
    layers = resolve_transformer_layers(model)
    captures: dict[int, dict[str, Any]] = {}
    handles = []
    for index, layer in enumerate(layers):
        def pre_hook(_module, args, kwargs, *, _index=index):
            hidden = args[0] if args else kwargs.get("hidden_states")
            if hidden is None:
                raise RuntimeError("decoder hidden_states missing")
            captures.setdefault(_index, {})["input"] = hidden.detach()

        def post_hook(_module, _args, output, *, _index=index):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            captures.setdefault(_index, {})["output"] = hidden.detach()

        handles.append(layer.register_forward_pre_hook(pre_hook, with_kwargs=True))
        handles.append(layer.register_forward_hook(post_hook))
    try:
        model(input_ids=input_ids, use_cache=False)
    finally:
        for handle in reversed(handles):
            handle.remove()
    if len(captures) != len(layers):
        raise RuntimeError("failed to capture every decoder layer")
    return captures


def _candidate_islands(total_layers: int, config: IslandPlanConfig):
    edge = config.edge_layers_to_keep
    for width in range(config.min_width, config.max_width + 1):
        last_start = total_layers - edge - width
        for start in range(edge, last_start + 1):
            yield TransformerIsland(start, start + width - 1)


def calibrate_island_sensitivity(
    model,
    encoded_inputs: list[list[int]],
    *,
    config: IslandPlanConfig | None = None,
) -> list[IslandSensitivity]:
    config = config or IslandPlanConfig()
    config.validate()
    if not encoded_inputs:
        raise ValueError("island calibration inputs are empty")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install neural support with: pip install -e '.[neural]'") from exc

    layers = resolve_transformer_layers(model)
    candidates = list(_candidate_islands(len(layers), config))
    if not candidates:
        raise ValueError("decoder is too shallow for requested island widths")

    accum = {
        island: {"residual": 0.0, "cosine": 0.0, "count": 0}
        for island in candidates
    }
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for ids in encoded_inputs:
            tensor = torch.tensor([ids], dtype=torch.long, device=device)
            captures = _capture_all_layer_io(model, tensor)
            for island in candidates:
                before = captures[island.start]["input"].float()
                after = captures[island.end]["output"].float()
                residual = after - before
                output_rms = torch.sqrt(after.pow(2).mean()).clamp_min(1e-8)
                residual_ratio = torch.sqrt(residual.pow(2).mean()) / output_rms
                cosine = torch.nn.functional.cosine_similarity(
                    before.reshape(-1, before.shape[-1]),
                    after.reshape(-1, after.shape[-1]),
                    dim=-1,
                ).mean()
                accum[island]["residual"] += float(residual_ratio.cpu())
                accum[island]["cosine"] += float((1.0 - cosine).cpu())
                accum[island]["count"] += 1

    rows = []
    for island in candidates:
        count = max(1, int(accum[island]["count"]))
        residual = accum[island]["residual"] / count
        cosine_change = accum[island]["cosine"] / count
        transform = residual + cosine_change
        rows.append(
            IslandSensitivity(
                start=island.start,
                end=island.end,
                width=island.width,
                residual_rms_ratio=residual,
                cosine_change=cosine_change,
                transformation_score=transform,
                score_per_layer=transform / island.width,
            )
        )
    rows.sort(key=lambda row: (row.score_per_layer, row.transformation_score, -row.width, row.start))
    return rows


def _compatible(candidate: TransformerIsland, selected: list[TransformerIsland], min_gap: int) -> bool:
    for island in selected:
        if candidate.end + min_gap >= island.start and candidate.start <= island.end + min_gap:
            return False
        if island.end + min_gap >= candidate.start and island.start <= candidate.end + min_gap:
            return False
    return True


def build_island_plan(
    *,
    total_layers: int,
    sensitivity: list[IslandSensitivity],
    base_model_fingerprint: str,
    dataset_fingerprint: str,
    config: IslandPlanConfig | None = None,
) -> dict[str, Any]:
    config = config or IslandPlanConfig()
    config.validate()
    total = int(total_layers)
    if total < config.min_width + 2 * config.edge_layers_to_keep:
        raise ValueError("decoder is too shallow for island replacement")
    if not sensitivity:
        raise ValueError("island sensitivity evidence is empty")

    target_layers = max(config.min_width, math.ceil(total * config.target_fraction))
    selected: list[TransformerIsland] = []
    ranking = sorted(
        sensitivity,
        key=lambda row: (row.score_per_layer, row.transformation_score, -row.width, row.start),
    )
    for row in ranking:
        island = row.island
        if island.start < config.edge_layers_to_keep:
            continue
        if island.end >= total - config.edge_layers_to_keep:
            continue
        if not _compatible(island, selected, config.min_gap_layers):
            continue
        selected.append(island)
        if len(selected) >= config.max_islands or sum(x.width for x in selected) >= target_layers:
            break
    if not selected:
        raise ValueError("no compatible Transformer islands selected")

    selected_by_rank = list(selected)
    stages = []
    for stage_index in range(1, len(selected_by_rank) + 1):
        stage_islands = sorted(selected_by_rank[:stage_index])
        replaced = sum(island.width for island in stage_islands)
        stages.append(
            {
                "stage": stage_index,
                "islands": [island.to_dict() for island in stage_islands],
                "replaced_layers": replaced,
                "replaced_fraction": replaced / total,
            }
        )

    final_islands = sorted(selected_by_rank)
    replaced_layers = sum(island.width for island in final_islands)
    payload = {
        "schema": "NOLANE-L11-RECURRENT-ISLAND-PLAN-V1",
        "authority": "FROZEN_TRAINING_PLAN_ONLY",
        "base_model_fingerprint": str(base_model_fingerprint),
        "dataset_fingerprint": str(dataset_fingerprint),
        "total_layers": total,
        "config": asdict(config),
        "sensitivity": [row.to_dict() for row in sensitivity],
        "selection_order": [island.to_dict() for island in selected_by_rank],
        "target_islands": [island.to_dict() for island in final_islands],
        "replaced_layers": replaced_layers,
        "replaced_fraction_actual": replaced_layers / total,
        "stages": stages,
    }
    payload["plan_sha256"] = payload_digest(payload)
    return payload


def verify_island_plan(
    plan: dict[str, Any],
    *,
    base_model_fingerprint: str,
    dataset_fingerprint: str,
) -> None:
    if plan.get("schema") != "NOLANE-L11-RECURRENT-ISLAND-PLAN-V1":
        raise ValueError("unsupported recurrent island plan")
    supplied = plan.get("plan_sha256")
    body = dict(plan)
    body.pop("plan_sha256", None)
    if payload_digest(body) != supplied:
        raise ValueError("recurrent island plan digest mismatch")
    if plan.get("base_model_fingerprint") != base_model_fingerprint:
        raise ValueError("recurrent island plan base-model mismatch")
    if plan.get("dataset_fingerprint") != dataset_fingerprint:
        raise ValueError("recurrent island plan dataset mismatch")

    total = int(plan["total_layers"])
    config = IslandPlanConfig(**plan["config"])
    previous: set[tuple[int, int]] = set()
    final = set()
    for row in plan.get("stages", []):
        current_islands = normalize_islands(
            [(int(x["start"]), int(x["end"])) for x in row["islands"]],
            total_layers=total,
            min_gap_layers=config.min_gap_layers,
        )
        current = {(island.start, island.end) for island in current_islands}
        if not current or not previous.issubset(current):
            raise ValueError("recurrent island stages are not monotonic")
        previous = current
        final = current

    expected = {
        (int(row["start"]), int(row["end"]))
        for row in plan.get("target_islands", [])
    }
    if final != expected:
        raise ValueError("final recurrent island stage does not match target islands")
