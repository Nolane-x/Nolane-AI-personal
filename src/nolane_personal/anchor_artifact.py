from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

from .deep_recurrent_cortex import (
    DeepRecurrentCortexConfig,
    DeepRecurrentStateSpaceCortex,
)
from .state_space_model import StateSpaceModelConfig, StateSpacePersonalModel
from .state_space_region import CortexRegion
from .surgery import module_parameter_digest
from .store import canonical_json, payload_digest


SCHEMA = "NOLANE-L14-MINIMAL-QWEN-ANCHOR-CORTEX-V1"


def save_anchor_artifact(
    output_dir,
    model,
    training_receipt,
    plan,
    *,
    base_model_fingerprint,
    dataset_fingerprint,
):
    torch = model.cortex.torch
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "minimal-qwen-anchor-cortex.pt"
    payload = {
        "schema": SCHEMA,
        "authority": "TRAINED_CANDIDATE_UNPROMOTED",
        "base_model_fingerprint": str(base_model_fingerprint),
        "hidden_size": model.cortex.hidden_size,
        "cortex_config": asdict(model.cortex.config),
        "model_config": {
            "region": model.config.region.to_dict(),
            "carry_recurrent_state": model.config.carry_recurrent_state,
        },
        "cortex_state": model.cortex.module.state_dict(),
        "plan": plan,
        "training_receipt": training_receipt.to_dict(),
        "dataset_fingerprint": str(dataset_fingerprint),
    }
    torch.save(payload, checkpoint)
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = {
        "schema": SCHEMA,
        "authority": payload["authority"],
        "base_model_fingerprint": payload["base_model_fingerprint"],
        "hidden_size": payload["hidden_size"],
        "cortex_config": payload["cortex_config"],
        "model_config": payload["model_config"],
        "plan_sha256": plan["plan_sha256"],
        "head_layers": training_receipt.head_layers,
        "tail_layers": training_receipt.tail_layers,
        "remaining_qwen_layers": training_receipt.remaining_qwen_layers,
        "remaining_qwen_fraction": training_receipt.remaining_qwen_fraction,
        "replaced_layers": model.replaced_layer_count(),
        "cortex_parameters": model.trainable_parameter_count(),
        "dataset_fingerprint": str(dataset_fingerprint),
        "training": training_receipt.to_dict(),
        "checkpoint_sha256": checkpoint_sha,
        "cortex_state_digest": module_parameter_digest(model.cortex.module),
    }
    manifest["artifact_id"] = payload_digest(manifest)
    (output_dir / "minimal-qwen-anchor-cortex-manifest.json").write_text(
        canonical_json(manifest) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_anchor_artifact(
    checkpoint_path,
    *,
    expected_base_model_fingerprint,
    expected_hidden_size,
    expected_dataset_fingerprint=None,
):
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
        raise ValueError("unsupported minimal-anchor cortex schema")
    if payload.get("authority") != "TRAINED_CANDIDATE_UNPROMOTED":
        raise ValueError("minimal-anchor cortex authority mismatch")
    if payload.get("base_model_fingerprint") != expected_base_model_fingerprint:
        raise ValueError("minimal-anchor cortex base-model mismatch")
    if int(payload.get("hidden_size", -1)) != int(expected_hidden_size):
        raise ValueError("minimal-anchor cortex hidden-size mismatch")
    if (
        expected_dataset_fingerprint is not None
        and payload.get("dataset_fingerprint") != expected_dataset_fingerprint
    ):
        raise ValueError("minimal-anchor cortex dataset mismatch")

    cortex = DeepRecurrentStateSpaceCortex(
        int(payload["hidden_size"]),
        DeepRecurrentCortexConfig(**payload["cortex_config"]),
        seed=0,
    )
    cortex.module.load_state_dict(payload["cortex_state"])
    raw = payload["model_config"]
    config = StateSpaceModelConfig(
        region=CortexRegion(
            int(raw["region"]["start"]),
            int(raw["region"]["end"]),
        ),
        carry_recurrent_state=bool(raw.get("carry_recurrent_state", False)),
    )
    metadata = {
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "cortex_state_digest": module_parameter_digest(cortex.module),
        "dataset_fingerprint": payload.get("dataset_fingerprint"),
        "plan_sha256": payload["plan"]["plan_sha256"],
        "plan": payload["plan"],
        "training_receipt": payload.get("training_receipt"),
        "region": config.region.to_dict(),
    }
    return cortex, config, metadata


def build_anchor_model(
    qwen_model,
    checkpoint_path,
    latent,
    *,
    expected_base_model_fingerprint,
    expected_dataset_fingerprint=None,
):
    hidden_size = int(qwen_model.config.hidden_size)
    cortex, config, metadata = load_anchor_artifact(
        checkpoint_path,
        expected_base_model_fingerprint=expected_base_model_fingerprint,
        expected_hidden_size=hidden_size,
        expected_dataset_fingerprint=expected_dataset_fingerprint,
    )
    device = next(qwen_model.parameters()).device
    cortex.to(str(device)).eval()
    return StateSpacePersonalModel(
        qwen_model,
        cortex,
        latent,
        config=config,
    ), metadata
