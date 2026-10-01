from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .personal_cortex import PersonalCortexConfig, TrainablePersonalCortex
from .surgery import LatentAdapterConfig, LatentResidualAdapter, module_parameter_digest


SCHEMA = "NOLANE-L6-PERSONAL-CORTEX-ADAPTER-V1"


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trained_adapter(
    checkpoint_path: str | Path,
    *,
    expected_base_model_fingerprint: str,
    expected_hidden_size: int,
) -> tuple[LatentResidualAdapter, PersonalCortexConfig, dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install neural support with: pip install -e '.[neural]'") from exc

    path = Path(checkpoint_path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported trained adapter schema")
    if payload.get("authority") != "TRAINED_CANDIDATE_UNPROMOTED":
        raise ValueError("trained adapter authority mismatch")
    if payload.get("base_model_fingerprint") != expected_base_model_fingerprint:
        raise ValueError("trained adapter base-model mismatch")
    if int(payload.get("hidden_size", -1)) != int(expected_hidden_size):
        raise ValueError("trained adapter hidden-size mismatch")

    adapter_config = LatentAdapterConfig(**payload["adapter_config"])
    adapter = LatentResidualAdapter(int(payload["hidden_size"]), adapter_config, seed=0)
    adapter.module.load_state_dict(payload["adapter_state"])

    raw_config = dict(payload.get("personal_cortex_config", {}))
    if raw_config.get("layer_indices") is not None:
        raw_config["layer_indices"] = tuple(int(x) for x in raw_config["layer_indices"])
    cortex_config = PersonalCortexConfig(**raw_config)
    cortex_config.validate()

    metadata = {
        "schema": payload["schema"],
        "authority": payload["authority"],
        "base_model_fingerprint": payload["base_model_fingerprint"],
        "hidden_size": int(payload["hidden_size"]),
        "adapter_config": payload["adapter_config"],
        "personal_cortex_config": payload.get("personal_cortex_config", {}),
        "training_receipt": payload.get("training_receipt"),
        "source_candidate_checkpoint_sha256": payload.get("source_candidate_checkpoint_sha256"),
        "dataset_fingerprint": payload.get("dataset_fingerprint"),
        "checkpoint_sha256": sha256_file(path),
        "adapter_state_digest": module_parameter_digest(adapter.module),
    }
    return adapter, cortex_config, metadata


def build_personal_cortex(
    model,
    checkpoint_path: str | Path,
    latent,
    *,
    expected_base_model_fingerprint: str,
) -> tuple[TrainablePersonalCortex, dict[str, Any]]:
    hidden_size = int(model.config.hidden_size)
    adapter, config, metadata = load_trained_adapter(
        checkpoint_path,
        expected_base_model_fingerprint=expected_base_model_fingerprint,
        expected_hidden_size=hidden_size,
    )
    device = next(model.parameters()).device
    adapter.to(str(device)).eval()
    return TrainablePersonalCortex(model, adapter, latent, config=config), metadata
