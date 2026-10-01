from __future__ import annotations

import hashlib
from pathlib import Path

from .bridge_cortex import LivingBridgeCortexConfig, TrainableLivingBridgeCortex
from .living_bridge import CrossLayerLivingBridge, LivingBridgeConfig
from .surgery import module_parameter_digest


def load_trained_bridge(checkpoint_path, *, expected_base_model_fingerprint, expected_hidden_size):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install neural support with: pip install -e '.[neural]'") from exc

    path = Path(checkpoint_path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("schema") != "NOLANE-L7-LIVING-BRIDGE-V1":
        raise ValueError("unsupported Living Bridge schema")
    if payload.get("authority") != "TRAINED_CANDIDATE_UNPROMOTED":
        raise ValueError("Living Bridge authority mismatch")
    if payload.get("base_model_fingerprint") != expected_base_model_fingerprint:
        raise ValueError("Living Bridge base-model mismatch")
    if int(payload.get("hidden_size", -1)) != int(expected_hidden_size):
        raise ValueError("Living Bridge hidden-size mismatch")

    bridge = CrossLayerLivingBridge(
        int(payload["hidden_size"]),
        LivingBridgeConfig(**payload["bridge_config"]),
        seed=0,
    )
    bridge.module.load_state_dict(payload["bridge_state"])
    raw = dict(payload.get("cortex_config", {}))
    if raw.get("layer_indices") is not None:
        raw["layer_indices"] = tuple(raw["layer_indices"])
    config = LivingBridgeCortexConfig(**raw)
    meta = {
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bridge_state_digest": module_parameter_digest(bridge.module),
        "dataset_fingerprint": payload.get("dataset_fingerprint"),
        "training_receipt": payload.get("training_receipt"),
    }
    return bridge, config, meta


def build_bridge_cortex(model, checkpoint_path, latent, *, expected_base_model_fingerprint):
    hidden_size = int(model.config.hidden_size)
    bridge, config, meta = load_trained_bridge(
        checkpoint_path,
        expected_base_model_fingerprint=expected_base_model_fingerprint,
        expected_hidden_size=hidden_size,
    )
    device = next(model.parameters()).device
    bridge.to(str(device)).eval()
    return TrainableLivingBridgeCortex(model, bridge, latent, config=config), meta
