from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .hybrid_training import HybridTrainingReceipt
from .recurrent_cortex import (
    HybridCortexConfig,
    HybridRecurrentCortex,
    RecurrentCortexConfig,
    RecurrentCortexMixer,
)
from .surgery import module_parameter_digest


SCHEMA = "NOLANE-L7-HYBRID-RECURRENT-CORTEX-V1"


def save_hybrid_artifact(
    path: str | Path,
    cortex: HybridRecurrentCortex,
    receipt: HybridTrainingReceipt,
    *,
    base_model_fingerprint: str,
) -> dict[str, Any]:
    torch = cortex.mixer.torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA,
        "authority": "HYBRID_RECURRENT_CANDIDATE_UNPROMOTED",
        "base_model_fingerprint": str(base_model_fingerprint),
        "hidden_size": cortex.mixer.hidden_size,
        "mixer_config": asdict(cortex.mixer.config),
        "cortex_config": asdict(cortex.config),
        "mixer_state": cortex.mixer.module.state_dict(),
        "training_receipt": receipt.to_dict(),
    }
    torch.save(payload, path)
    checkpoint_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "schema": SCHEMA,
        "authority": payload["authority"],
        "checkpoint_sha256": checkpoint_sha,
        "base_model_fingerprint": payload["base_model_fingerprint"],
        "hidden_size": payload["hidden_size"],
        "mixer_parameters": receipt.mixer_parameters,
        "mixer_digest": module_parameter_digest(cortex.mixer.module),
        "training": receipt.to_dict(),
    }


def load_hybrid_artifact(
    path: str | Path,
    *,
    expected_base_model_fingerprint: str,
    expected_hidden_size: int,
    latent,
    model,
) -> tuple[HybridRecurrentCortex, dict[str, Any]]:
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
        raise ValueError("unsupported hybrid artifact schema")
    if payload.get("authority") != "HYBRID_RECURRENT_CANDIDATE_UNPROMOTED":
        raise ValueError("hybrid artifact authority mismatch")
    if payload.get("base_model_fingerprint") != expected_base_model_fingerprint:
        raise ValueError("hybrid artifact base-model mismatch")
    if int(payload.get("hidden_size", -1)) != int(expected_hidden_size):
        raise ValueError("hybrid artifact hidden-size mismatch")

    mixer = RecurrentCortexMixer(
        int(payload["hidden_size"]),
        RecurrentCortexConfig(**payload["mixer_config"]),
    )
    mixer.module.load_state_dict(payload["mixer_state"])
    raw = dict(payload["cortex_config"])
    if raw.get("layer_indices") is not None:
        raw["layer_indices"] = tuple(int(x) for x in raw["layer_indices"])
    config = HybridCortexConfig(**raw)
    cortex = HybridRecurrentCortex(model, mixer, latent, config=config)
    metadata = {
        "schema": payload["schema"],
        "authority": payload["authority"],
        "base_model_fingerprint": payload["base_model_fingerprint"],
        "hidden_size": int(payload["hidden_size"]),
        "mixer_digest": module_parameter_digest(mixer.module),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "training_receipt": payload.get("training_receipt"),
    }
    return cortex, metadata
