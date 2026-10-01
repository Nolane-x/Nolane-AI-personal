from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

from .block_replacement import BlockReplacementConfig, RecurrentBlockReplacement
from .island_cortex import IslandCortexConfig, RecurrentIslandCortex
from .island_replacement import TransformerIsland
from .surgery import module_parameter_digest
from .store import canonical_json, payload_digest


SCHEMA = "NOLANE-L11-RECURRENT-TRANSFORMER-ISLANDS-V1"


def save_island_artifact(
    output_dir,
    cortex,
    training_receipt,
    plan,
    *,
    base_model_fingerprint,
    dataset_fingerprint,
):
    torch = cortex.replacement.torch
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "recurrent-transformer-islands.pt"
    payload = {
        "schema": SCHEMA,
        "authority": "TRAINED_CANDIDATE_UNPROMOTED",
        "base_model_fingerprint": str(base_model_fingerprint),
        "hidden_size": cortex.replacement.hidden_size,
        "replacement_config": asdict(cortex.replacement.config),
        "cortex_config": {
            "islands": [island.to_dict() for island in cortex.config.islands],
            "carry_recurrent_state": cortex.config.carry_recurrent_state,
        },
        "replacement_state": cortex.replacement.module.state_dict(),
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
        "replacement_config": payload["replacement_config"],
        "cortex_config": payload["cortex_config"],
        "plan_sha256": plan["plan_sha256"],
        "replaced_layers": cortex.replaced_layer_count(),
        "replacement_parameters": cortex.trainable_parameter_count(),
        "dataset_fingerprint": str(dataset_fingerprint),
        "training": training_receipt.to_dict(),
        "checkpoint_sha256": checkpoint_sha,
        "replacement_state_digest": module_parameter_digest(cortex.replacement.module),
    }
    manifest["artifact_id"] = payload_digest(manifest)
    (output_dir / "recurrent-transformer-islands-manifest.json").write_text(
        canonical_json(manifest) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_island_artifact(
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
        raise ValueError("unsupported recurrent island schema")
    if payload.get("authority") != "TRAINED_CANDIDATE_UNPROMOTED":
        raise ValueError("recurrent island authority mismatch")
    if payload.get("base_model_fingerprint") != expected_base_model_fingerprint:
        raise ValueError("recurrent island base-model mismatch")
    if int(payload.get("hidden_size", -1)) != int(expected_hidden_size):
        raise ValueError("recurrent island hidden-size mismatch")
    if (
        expected_dataset_fingerprint is not None
        and payload.get("dataset_fingerprint") != expected_dataset_fingerprint
    ):
        raise ValueError("recurrent island dataset mismatch")

    replacement = RecurrentBlockReplacement(
        int(payload["hidden_size"]),
        BlockReplacementConfig(**payload["replacement_config"]),
        seed=0,
    )
    replacement.module.load_state_dict(payload["replacement_state"])
    raw = payload["cortex_config"]
    islands = tuple(
        TransformerIsland(int(row["start"]), int(row["end"]))
        for row in raw["islands"]
    )
    config = IslandCortexConfig(
        islands=islands,
        carry_recurrent_state=bool(raw.get("carry_recurrent_state", False)),
    )
    metadata = {
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "replacement_state_digest": module_parameter_digest(replacement.module),
        "dataset_fingerprint": payload.get("dataset_fingerprint"),
        "plan_sha256": payload["plan"]["plan_sha256"],
        "plan": payload["plan"],
        "training_receipt": payload.get("training_receipt"),
        "islands": [island.to_dict() for island in islands],
    }
    return replacement, config, metadata


def build_island_cortex(
    model,
    checkpoint_path,
    latent,
    *,
    expected_base_model_fingerprint,
    expected_dataset_fingerprint=None,
):
    hidden_size = int(model.config.hidden_size)
    replacement, config, metadata = load_island_artifact(
        checkpoint_path,
        expected_base_model_fingerprint=expected_base_model_fingerprint,
        expected_hidden_size=hidden_size,
        expected_dataset_fingerprint=expected_dataset_fingerprint,
    )
    device = next(model.parameters()).device
    replacement.to(str(device)).eval()
    return RecurrentIslandCortex(model, replacement, latent, config=config), metadata
