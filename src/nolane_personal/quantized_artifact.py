from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

from .deep_recurrent_cortex import DeepRecurrentCortexConfig, DeepRecurrentStateSpaceCortex
from .factorized_artifact import load_factorized_model
from .quantized_boundary import QuantizedBoundaryConfig, QuantizedBoundaryModule, quantize_factorized_boundary
from .standalone_model import StandaloneNolaneLM
from .store import canonical_json, payload_digest
from .surgery import module_parameter_digest

SCHEMA = "NOLANE-L19-QUANTIZED-FACTORIZED-RUNTIME-V1"


def save_quantized_artifact(
    output_dir,
    model,
    quantization_receipt,
    *,
    source_meta,
    dataset_fingerprint=None,
):
    torch = model.cortex.torch
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "quantized-nolane.pt"

    payload = {
        "schema": SCHEMA,
        "authority": "QUANTIZED_CANDIDATE_UNPROMOTED",
        "source_factorized_checkpoint_sha256": source_meta["checkpoint_sha256"],
        "source_l16_checkpoint_sha256": source_meta.get("source_l16_checkpoint_sha256"),
        "dataset_fingerprint": dataset_fingerprint or source_meta.get("dataset_fingerprint"),
        "quantized_config": asdict(model.boundary.config),
        "cortex_config": asdict(model.cortex.config),
        "boundary_state": model.boundary.state_dict(),
        "cortex_state": model.cortex.module.state_dict(),
        "quantization_receipt": quantization_receipt,
        "carry_recurrent_state": model.carry_recurrent_state,
        "runtime_requires_qwen_model": False,
        "runtime_requires_transformers": False,
    }
    torch.save(payload, checkpoint)

    manifest = {
        "schema": SCHEMA,
        "authority": payload["authority"],
        "source_factorized_checkpoint_sha256": payload["source_factorized_checkpoint_sha256"],
        "source_l16_checkpoint_sha256": payload["source_l16_checkpoint_sha256"],
        "dataset_fingerprint": payload["dataset_fingerprint"],
        "quantized_config": payload["quantized_config"],
        "quantization_receipt": quantization_receipt,
        "boundary_storage_bytes": model.boundary.storage_bytes(),
        "boundary_tensor_count": model.boundary.tensor_count(),
        "boundary_trainable_parameters": model.boundary.parameter_count(),
        "cortex_parameters": model.cortex.parameter_count(),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "boundary_state_digest": module_parameter_digest(model.boundary.module),
        "cortex_state_digest": module_parameter_digest(model.cortex.module),
        "runtime_requires_qwen_model": False,
        "runtime_requires_transformers": False,
    }
    manifest["artifact_id"] = payload_digest(manifest)
    (output_dir / "quantized-nolane-manifest.json").write_text(
        canonical_json(manifest) + "\n",
        encoding="utf-8",
    )
    return manifest


def quantize_factorized_artifact(
    factorized_checkpoint,
    output_dir,
    *,
    latent,
    device: str = "cpu",
    logit_chunk_size: int = 4096,
    expected_source_l16_checkpoint_sha256=None,
    expected_dataset_fingerprint=None,
):
    source, source_meta = load_factorized_model(
        factorized_checkpoint,
        latent,
        device=device,
        expected_source_l16_checkpoint_sha256=expected_source_l16_checkpoint_sha256,
        expected_dataset_fingerprint=expected_dataset_fingerprint,
    )
    boundary, receipt = quantize_factorized_boundary(
        source.boundary,
        logit_chunk_size=logit_chunk_size,
    )
    cortex = DeepRecurrentStateSpaceCortex(
        source.cortex.hidden_size,
        DeepRecurrentCortexConfig(**asdict(source.cortex.config)),
        seed=0,
    )
    cortex.module.load_state_dict(source.cortex.module.state_dict())
    model = StandaloneNolaneLM(
        boundary,
        cortex,
        latent,
        carry_recurrent_state=source.carry_recurrent_state,
    ).to(device).eval()
    manifest = save_quantized_artifact(
        output_dir,
        model,
        receipt,
        source_meta=source_meta,
        dataset_fingerprint=expected_dataset_fingerprint or source_meta.get("dataset_fingerprint"),
    )
    return model, manifest, receipt


def load_quantized_model(
    checkpoint_path,
    latent,
    *,
    device: str = "cpu",
    expected_source_factorized_checkpoint_sha256=None,
    expected_source_l16_checkpoint_sha256=None,
    expected_dataset_fingerprint=None,
):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Quantized Nolane requires PyTorch") from exc

    path = Path(checkpoint_path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported quantized Nolane schema")
    if payload.get("authority") != "QUANTIZED_CANDIDATE_UNPROMOTED":
        raise ValueError("quantized Nolane authority mismatch")
    if payload.get("runtime_requires_qwen_model") is not False:
        raise ValueError("quantized artifact declares Qwen runtime dependency")
    if payload.get("runtime_requires_transformers") is not False:
        raise ValueError("quantized artifact declares Transformers runtime dependency")
    if (
        expected_source_factorized_checkpoint_sha256
        and payload.get("source_factorized_checkpoint_sha256")
        != expected_source_factorized_checkpoint_sha256
    ):
        raise ValueError("quantized source factorized checkpoint mismatch")
    if (
        expected_source_l16_checkpoint_sha256
        and payload.get("source_l16_checkpoint_sha256")
        != expected_source_l16_checkpoint_sha256
    ):
        raise ValueError("quantized source L16 checkpoint mismatch")
    if (
        expected_dataset_fingerprint
        and payload.get("dataset_fingerprint") != expected_dataset_fingerprint
    ):
        raise ValueError("quantized dataset mismatch")

    config = QuantizedBoundaryConfig(**payload["quantized_config"])
    state = payload["boundary_state"]
    prefix = "embed_tokens."
    kwargs = {
        "input_codes_q": state[prefix + "codes_q"],
        "input_code_scales": state[prefix + "code_scales"],
        "input_basis_q": state[prefix + "basis_q"],
        "input_basis_scales": state[prefix + "basis_scales"],
        "final_norm_weight": state["final_norm.weight"],
        "device": device,
    }
    if not config.tie_word_embeddings:
        kwargs.update({
            "output_codes_q": state["output.codes_q"],
            "output_code_scales": state["output.code_scales"],
            "output_basis_q": state["output.basis_q"],
            "output_basis_scales": state["output.basis_scales"],
        })
    boundary = QuantizedBoundaryModule(config, **kwargs)

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

    meta = {
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_factorized_checkpoint_sha256": payload["source_factorized_checkpoint_sha256"],
        "source_l16_checkpoint_sha256": payload.get("source_l16_checkpoint_sha256"),
        "dataset_fingerprint": payload.get("dataset_fingerprint"),
        "quantization_receipt": payload["quantization_receipt"],
        "boundary_state_digest": module_parameter_digest(boundary.module),
        "cortex_state_digest": module_parameter_digest(cortex.module),
        "runtime_requires_qwen_model": False,
        "runtime_requires_transformers": False,
    }
    return model, meta
