from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .ssm_distill import SSMDistillReceipt
from .ssm_replacement import SSMReplacementConfig, SelectiveStateSpaceMixer
from .surgery import module_parameter_digest


SCHEMA = "NOLANE-L8-SSM-REPLACEMENT-V1"


def save_ssm_artifact(
    path: str | Path,
    mixer: SelectiveStateSpaceMixer,
    receipt: SSMDistillReceipt,
    *,
    base_model_fingerprint: str,
    layer_index: int,
    dataset_fingerprint: str | None,
    original_layer_parameters: int,
) -> dict[str, Any]:
    torch = mixer.torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA,
        "authority": "SSM_BLOCK_REPLACEMENT_CANDIDATE_UNPROMOTED",
        "base_model_fingerprint": str(base_model_fingerprint),
        "hidden_size": mixer.hidden_size,
        "layer_index": int(layer_index),
        "mixer_config": asdict(mixer.config),
        "mixer_state": mixer.module.state_dict(),
        "dataset_fingerprint": dataset_fingerprint,
        "original_layer_parameters": int(original_layer_parameters),
        "distill_receipt": receipt.to_dict(),
    }
    torch.save(payload, path)
    checkpoint_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    replacement_parameters = mixer.parameter_count()
    return {
        "schema": SCHEMA,
        "authority": payload["authority"],
        "checkpoint_sha256": checkpoint_sha,
        "base_model_fingerprint": payload["base_model_fingerprint"],
        "hidden_size": payload["hidden_size"],
        "layer_index": payload["layer_index"],
        "dataset_fingerprint": dataset_fingerprint,
        "original_layer_parameters": int(original_layer_parameters),
        "replacement_parameters": replacement_parameters,
        "parameter_ratio": replacement_parameters / max(1, int(original_layer_parameters)),
        "mixer_digest": module_parameter_digest(mixer.module),
        "distillation": receipt.to_dict(),
    }


def load_ssm_artifact(
    path: str | Path,
    *,
    expected_base_model_fingerprint: str,
    expected_hidden_size: int,
) -> tuple[SelectiveStateSpaceMixer, dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install neural support with: pip install -e '.[neural]'") from exc

    path = Path(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported SSM replacement schema")
    if payload.get("authority") != "SSM_BLOCK_REPLACEMENT_CANDIDATE_UNPROMOTED":
        raise ValueError("SSM replacement authority mismatch")
    if payload.get("base_model_fingerprint") != expected_base_model_fingerprint:
        raise ValueError("SSM replacement base-model mismatch")
    if int(payload.get("hidden_size", -1)) != int(expected_hidden_size):
        raise ValueError("SSM replacement hidden-size mismatch")

    mixer = SelectiveStateSpaceMixer(
        int(payload["hidden_size"]),
        SSMReplacementConfig(**payload["mixer_config"]),
    )
    mixer.module.load_state_dict(payload["mixer_state"])
    metadata = {
        "schema": payload["schema"],
        "authority": payload["authority"],
        "base_model_fingerprint": payload["base_model_fingerprint"],
        "hidden_size": int(payload["hidden_size"]),
        "layer_index": int(payload["layer_index"]),
        "dataset_fingerprint": payload.get("dataset_fingerprint"),
        "original_layer_parameters": int(payload.get("original_layer_parameters", 0)),
        "replacement_parameters": mixer.parameter_count(),
        "mixer_digest": module_parameter_digest(mixer.module),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "distill_receipt": payload.get("distill_receipt"),
    }
    return mixer, metadata
