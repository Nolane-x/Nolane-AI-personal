from __future__ import annotations

import hashlib
from pathlib import Path

from .block_replacement import BlockReplacementConfig, RecurrentBlockReplacement
from .replacement_cortex import ReplacementCortexConfig, RecurrentReplacementCortex
from .surgery import module_parameter_digest


SCHEMA = "NOLANE-L9-RECURRENT-BLOCK-REPLACEMENT-V1"


def load_trained_replacement(checkpoint_path, *, expected_base_model_fingerprint, expected_hidden_size):
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
        raise ValueError("unsupported replacement schema")
    if payload.get("authority") != "TRAINED_CANDIDATE_UNPROMOTED":
        raise ValueError("replacement authority mismatch")
    if payload.get("base_model_fingerprint") != expected_base_model_fingerprint:
        raise ValueError("replacement base-model mismatch")
    if int(payload.get("hidden_size", -1)) != int(expected_hidden_size):
        raise ValueError("replacement hidden-size mismatch")

    replacement = RecurrentBlockReplacement(
        int(payload["hidden_size"]),
        BlockReplacementConfig(**payload["replacement_config"]),
        seed=0,
    )
    replacement.module.load_state_dict(payload["replacement_state"])
    raw = dict(payload["cortex_config"])
    raw["layer_indices"] = tuple(int(x) for x in raw["layer_indices"])
    config = ReplacementCortexConfig(**raw)
    metadata = {
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "replacement_state_digest": module_parameter_digest(replacement.module),
        "dataset_fingerprint": payload.get("dataset_fingerprint"),
        "training_receipt": payload.get("training_receipt"),
        "selected_layers": list(config.layer_indices),
    }
    return replacement, config, metadata


def build_replacement_cortex(model, checkpoint_path, latent, *, expected_base_model_fingerprint):
    hidden_size = int(model.config.hidden_size)
    replacement, config, metadata = load_trained_replacement(
        checkpoint_path,
        expected_base_model_fingerprint=expected_base_model_fingerprint,
        expected_hidden_size=hidden_size,
    )
    device = next(model.parameters()).device
    replacement.to(str(device)).eval()
    return RecurrentReplacementCortex(model, replacement, latent, config=config), metadata
