from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

from .deep_recurrent_cortex import (
    DeepRecurrentCortexConfig,
    DeepRecurrentStateSpaceCortex,
)
from .native_artifact import load_native_boundary_artifact
from .standalone_model import (
    StandaloneBoundaryModule,
    StandaloneNolaneConfig,
    StandaloneNolaneLM,
    clone_boundary_from_qwen,
)
from .store import canonical_json, payload_digest
from .surgery import module_parameter_digest


SCHEMA = "NOLANE-L16-STANDALONE-WEIGHTS-V1"


def _tensor_bytes(tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def boundary_bytes_from_qwen(qwen_model) -> int:
    embed = qwen_model.model.embed_tokens.weight
    lm = qwen_model.lm_head.weight
    norm = qwen_model.model.norm.weight
    tied = embed.data_ptr() == lm.data_ptr()
    total = _tensor_bytes(embed) + _tensor_bytes(norm)
    if not tied:
        total += _tensor_bytes(lm)
    return total


def export_standalone_from_l15(
    qwen_model,
    l15_checkpoint,
    output_dir,
    *,
    latent,
    expected_base_model_fingerprint,
    expected_dataset_fingerprint=None,
) -> dict:
    torch = __import__("torch")
    cortex, native_config, l15_meta = load_native_boundary_artifact(
        l15_checkpoint,
        expected_base_model_fingerprint=expected_base_model_fingerprint,
        expected_hidden_size=int(qwen_model.config.hidden_size),
        expected_dataset_fingerprint=expected_dataset_fingerprint,
    )
    boundary = clone_boundary_from_qwen(qwen_model)
    model = StandaloneNolaneLM(
        boundary,
        cortex,
        latent,
        carry_recurrent_state=native_config.carry_recurrent_state,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "standalone-nolane.pt"
    payload = {
        "schema": SCHEMA,
        "authority": "STANDALONE_EXPORT_UNPROMOTED",
        "source_base_model_fingerprint": str(expected_base_model_fingerprint),
        "source_l15_checkpoint_sha256": l15_meta["checkpoint_sha256"],
        "source_l15_spec_sha256": l15_meta["spec_sha256"],
        "dataset_fingerprint": l15_meta.get("dataset_fingerprint"),
        "standalone_config": asdict(boundary.config),
        "cortex_config": asdict(cortex.config),
        "carry_recurrent_state": bool(native_config.carry_recurrent_state),
        "boundary_state": boundary.state_dict(),
        "cortex_state": cortex.module.state_dict(),
        "runtime_requires_qwen_model": False,
        "runtime_requires_transformers": False,
    }
    torch.save(payload, checkpoint)
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = {
        "schema": SCHEMA,
        "authority": payload["authority"],
        "source_base_model_fingerprint": payload["source_base_model_fingerprint"],
        "source_l15_checkpoint_sha256": payload["source_l15_checkpoint_sha256"],
        "source_l15_spec_sha256": payload["source_l15_spec_sha256"],
        "dataset_fingerprint": payload["dataset_fingerprint"],
        "standalone_config": payload["standalone_config"],
        "cortex_config": payload["cortex_config"],
        "runtime_requires_qwen_model": False,
        "runtime_requires_transformers": False,
        "boundary_parameters": boundary.parameter_count(),
        "cortex_parameters": cortex.parameter_count(),
        "total_parameters": model.total_parameter_count(),
        "source_boundary_bytes": boundary_bytes_from_qwen(qwen_model),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": checkpoint_sha,
        "boundary_state_digest": module_parameter_digest(boundary.module),
        "cortex_state_digest": module_parameter_digest(cortex.module),
    }
    manifest["artifact_id"] = payload_digest(manifest)
    (output_dir / "standalone-nolane-manifest.json").write_text(
        canonical_json(manifest) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_standalone_model(
    checkpoint_path,
    latent,
    *,
    device: str = "cpu",
    expected_source_base_model_fingerprint: str | None = None,
    expected_dataset_fingerprint: str | None = None,
) -> tuple[StandaloneNolaneLM, dict]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Standalone Nolane requires PyTorch") from exc

    path = Path(checkpoint_path)
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)

    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported standalone Nolane schema")
    if payload.get("authority") != "STANDALONE_EXPORT_UNPROMOTED":
        raise ValueError("standalone Nolane authority mismatch")
    if payload.get("runtime_requires_qwen_model") is not False:
        raise ValueError("standalone artifact declares Qwen runtime dependency")
    if payload.get("runtime_requires_transformers") is not False:
        raise ValueError("standalone artifact declares Transformers runtime dependency")
    if (
        expected_source_base_model_fingerprint is not None
        and payload.get("source_base_model_fingerprint")
        != expected_source_base_model_fingerprint
    ):
        raise ValueError("standalone source base-model mismatch")
    if (
        expected_dataset_fingerprint is not None
        and payload.get("dataset_fingerprint")
        != expected_dataset_fingerprint
    ):
        raise ValueError("standalone dataset mismatch")

    config = StandaloneNolaneConfig(**payload["standalone_config"])
    boundary = StandaloneBoundaryModule(config)
    boundary.load_state_dict(payload["boundary_state"])
    cortex = DeepRecurrentStateSpaceCortex(
        config.hidden_size,
        DeepRecurrentCortexConfig(**payload["cortex_config"]),
        seed=0,
    )
    cortex.module.load_state_dict(payload["cortex_state"])
    model = StandaloneNolaneLM(
        boundary,
        cortex,
        latent,
        carry_recurrent_state=bool(payload.get("carry_recurrent_state", False)),
    ).to(device).eval()

    metadata = {
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_base_model_fingerprint": payload["source_base_model_fingerprint"],
        "source_l15_checkpoint_sha256": payload["source_l15_checkpoint_sha256"],
        "source_l15_spec_sha256": payload["source_l15_spec_sha256"],
        "dataset_fingerprint": payload.get("dataset_fingerprint"),
        "runtime_requires_qwen_model": False,
        "runtime_requires_transformers": False,
        "boundary_state_digest": module_parameter_digest(model.boundary.module),
        "cortex_state_digest": module_parameter_digest(model.cortex.module),
    }
    return model, metadata
