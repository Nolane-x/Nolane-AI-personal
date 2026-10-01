from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .store import payload_digest
from .surgery import resolve_transformer_layers


@dataclass(slots=True)
class ProgressivePlanConfig:
    target_fraction: float = 0.50
    max_selected_layers: int = 12
    edge_layers_to_keep: int = 1
    first_stage_size: int = 1
    growth_factor: int = 2

    def validate(self) -> None:
        if not 0.0 < self.target_fraction < 1.0:
            raise ValueError("target_fraction must be in (0,1)")
        if self.max_selected_layers < 1:
            raise ValueError("max_selected_layers must be >=1")
        if self.edge_layers_to_keep < 1:
            raise ValueError("edge_layers_to_keep must be >=1")
        if self.first_stage_size < 1:
            raise ValueError("first_stage_size must be >=1")
        if self.growth_factor < 2:
            raise ValueError("growth_factor must be >=2")


@dataclass(slots=True)
class LayerSensitivity:
    layer_index: int
    residual_rms_ratio: float
    cosine_change: float
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _capture_layer_io(model, input_ids, layer_indices: Iterable[int]) -> dict[int, tuple[Any, Any]]:
    layers = resolve_transformer_layers(model)
    captures: dict[int, dict[str, Any]] = {}
    handles = []
    for index in layer_indices:
        layer = layers[int(index)]

        def pre_hook(_module, args, kwargs, *, _index=int(index)):
            hidden = args[0] if args else kwargs.get("hidden_states")
            if hidden is None:
                raise RuntimeError("decoder hidden_states missing")
            captures.setdefault(_index, {})["input"] = hidden.detach()

        def post_hook(_module, _args, output, *, _index=int(index)):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            captures.setdefault(_index, {})["output"] = hidden.detach()

        handles.append(layer.register_forward_pre_hook(pre_hook, with_kwargs=True))
        handles.append(layer.register_forward_hook(post_hook))
    try:
        model(input_ids=input_ids, use_cache=False)
    finally:
        for handle in reversed(handles):
            handle.remove()

    result = {}
    for index in layer_indices:
        row = captures.get(int(index))
        if not row or "input" not in row or "output" not in row:
            raise RuntimeError(f"failed to capture layer {index}")
        result[int(index)] = (row["input"], row["output"])
    return result


def calibrate_layer_sensitivity(
    model,
    encoded_inputs: list[list[int]],
    *,
    edge_layers_to_keep: int = 1,
) -> list[LayerSensitivity]:
    if not encoded_inputs:
        raise ValueError("calibration inputs are empty")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install neural support with: pip install -e '.[neural]'") from exc

    layers = resolve_transformer_layers(model)
    total = len(layers)
    edge = int(edge_layers_to_keep)
    candidates = list(range(edge, max(edge, total - edge)))
    if not candidates:
        raise ValueError("no replaceable layers after edge protection")

    device = next(model.parameters()).device
    accum = {
        index: {"residual": 0.0, "cosine": 0.0, "count": 0}
        for index in candidates
    }
    model.eval()
    with torch.no_grad():
        for ids in encoded_inputs:
            tensor = torch.tensor([ids], dtype=torch.long, device=device)
            captured = _capture_layer_io(model, tensor, candidates)
            for index, (before, after) in captured.items():
                before_f = before.float()
                after_f = after.float()
                residual = after_f - before_f
                output_rms = torch.sqrt(after_f.pow(2).mean()).clamp_min(1e-8)
                residual_rms = torch.sqrt(residual.pow(2).mean()) / output_rms
                cosine = torch.nn.functional.cosine_similarity(
                    before_f.reshape(-1, before_f.shape[-1]),
                    after_f.reshape(-1, after_f.shape[-1]),
                    dim=-1,
                ).mean()
                accum[index]["residual"] += float(residual_rms.cpu())
                accum[index]["cosine"] += float((1.0 - cosine).cpu())
                accum[index]["count"] += 1

    rows = []
    for index in candidates:
        count = max(1, int(accum[index]["count"]))
        residual = accum[index]["residual"] / count
        cosine_change = accum[index]["cosine"] / count
        rows.append(
            LayerSensitivity(
                layer_index=index,
                residual_rms_ratio=residual,
                cosine_change=cosine_change,
                score=residual + cosine_change,
            )
        )
    rows.sort(key=lambda row: (row.score, row.layer_index))
    return rows


def _stage_sizes(target: int, first: int, growth: int) -> list[int]:
    target = int(target)
    size = min(target, int(first))
    result = []
    while size < target:
        result.append(size)
        size = min(target, max(size + 1, size * int(growth)))
    result.append(target)
    return sorted(set(result))


def build_progressive_plan(
    *,
    total_layers: int,
    sensitivity: list[LayerSensitivity],
    base_model_fingerprint: str,
    dataset_fingerprint: str,
    config: ProgressivePlanConfig | None = None,
) -> dict[str, Any]:
    config = config or ProgressivePlanConfig()
    config.validate()
    total = int(total_layers)
    if total < 3:
        raise ValueError("progressive replacement needs at least 3 decoder layers")
    if not sensitivity:
        raise ValueError("layer sensitivity evidence is empty")

    protected = int(config.edge_layers_to_keep)
    valid = [
        row for row in sensitivity
        if protected <= row.layer_index < total - protected
    ]
    if not valid:
        raise ValueError("no valid sensitivity rows after edge protection")

    raw_target = max(1, math.ceil(total * config.target_fraction))
    target = min(raw_target, config.max_selected_layers, len(valid), total - 2 * protected)
    ranking = [row.layer_index for row in sorted(valid, key=lambda row: (row.score, row.layer_index))]
    selected = ranking[:target]

    stages = []
    for stage_number, size in enumerate(
        _stage_sizes(target, config.first_stage_size, config.growth_factor),
        start=1,
    ):
        stage_layers = sorted(selected[:size])
        stages.append(
            {
                "stage": stage_number,
                "layer_indices": stage_layers,
                "replaced_fraction": len(stage_layers) / total,
            }
        )

    payload = {
        "schema": "NOLANE-L10-PROGRESSIVE-REPLACEMENT-PLAN-V1",
        "authority": "FROZEN_TRAINING_PLAN_ONLY",
        "base_model_fingerprint": str(base_model_fingerprint),
        "dataset_fingerprint": str(dataset_fingerprint),
        "total_layers": total,
        "config": asdict(config),
        "sensitivity": [row.to_dict() for row in sensitivity],
        "ranking": ranking,
        "target_layers": sorted(selected),
        "target_fraction_actual": len(selected) / total,
        "stages": stages,
    }
    payload["plan_sha256"] = payload_digest(payload)
    return payload


def verify_progressive_plan(
    plan: dict[str, Any],
    *,
    base_model_fingerprint: str,
    dataset_fingerprint: str,
) -> None:
    if plan.get("schema") != "NOLANE-L10-PROGRESSIVE-REPLACEMENT-PLAN-V1":
        raise ValueError("unsupported progressive replacement plan")
    supplied = plan.get("plan_sha256")
    body = dict(plan)
    body.pop("plan_sha256", None)
    if payload_digest(body) != supplied:
        raise ValueError("progressive replacement plan digest mismatch")
    if plan.get("base_model_fingerprint") != base_model_fingerprint:
        raise ValueError("progressive plan base-model mismatch")
    if plan.get("dataset_fingerprint") != dataset_fingerprint:
        raise ValueError("progressive plan dataset mismatch")

    previous: set[int] = set()
    for row in plan.get("stages", []):
        current = set(int(x) for x in row.get("layer_indices", []))
        if not current or not previous.issubset(current):
            raise ValueError("progressive stages are not monotonic")
        previous = current
    if sorted(previous) != sorted(int(x) for x in plan.get("target_layers", [])):
        raise ValueError("final progressive stage does not match target layers")
