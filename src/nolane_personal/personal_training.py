from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .personal_cortex import TrainablePersonalCortex
from .personal_dataset import PersonalizationExample, encode_chat_example
from .store import canonical_json, payload_digest
from .surgery import module_parameter_digest, parameter_guard_snapshot


@dataclass(slots=True)
class PersonalTrainingConfig:
    epochs: int = 3
    learning_rate: float = 2e-3
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    initial_effective_gate: float = 0.02
    max_length: int = 384

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.max_length < 8:
            raise ValueError("max_length too small")


@dataclass(slots=True)
class PersonalTrainingReceipt:
    schema: str
    authority: str
    examples: int
    optimizer_steps: int
    initial_loss: float
    final_loss: float
    best_loss: float
    loss_improvement: float
    adapter_parameters: int
    frozen_base_parameters: int
    adapter_digest_before: str
    adapter_digest_after: str
    adapter_changed: bool
    base_model_unchanged: bool
    base_gradients_seen: int
    adapter_gradients_seen: int
    effective_gate_before: float
    effective_gate_after: float
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bootstrap_gate(cortex: TrainablePersonalCortex, target_gate: float) -> None:
    torch = cortex.adapter.torch
    limit = float(cortex.adapter.config.max_abs_gate)
    target = max(-0.95 * limit, min(0.95 * limit, float(target_gate)))
    if limit <= 0:
        raise ValueError("adapter gate limit must be positive")
    ratio = target / limit
    raw = 0.5 * math.log((1.0 + ratio) / (1.0 - ratio))
    with torch.no_grad():
        cortex.adapter.raw_gate.copy_(
            torch.tensor(raw, dtype=cortex.adapter.raw_gate.dtype, device=cortex.adapter.raw_gate.device)
        )


def _single_loss(cortex: TrainablePersonalCortex, input_ids, labels, weight: float = 1.0):
    torch = cortex.adapter.torch
    device = next(cortex.model.parameters()).device
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    y = torch.tensor([labels], dtype=torch.long, device=device)
    outputs = cortex.forward(input_ids=ids, labels=y, use_cache=False)
    return outputs.loss * float(weight)


def train_encoded_examples(
    cortex: TrainablePersonalCortex,
    encoded_examples: list[tuple[list[int], list[int], list[float] | None, float]],
    *,
    config: PersonalTrainingConfig | None = None,
) -> PersonalTrainingReceipt:
    config = config or PersonalTrainingConfig()
    config.validate()
    if not encoded_examples:
        raise ValueError("no encoded examples")

    torch = cortex.adapter.torch
    cortex.assert_gradient_boundary()
    cortex.model.eval()
    cortex.adapter.module.train()
    device = next(cortex.model.parameters()).device
    cortex.adapter.to(str(device))

    gate_before = float(cortex.adapter.effective_gate().detach().cpu())
    if abs(gate_before) < 1e-8:
        _bootstrap_gate(cortex, config.initial_effective_gate)
    gate_start = float(cortex.adapter.effective_gate().detach().cpu())

    base_before = parameter_guard_snapshot(cortex.model)
    adapter_before = module_parameter_digest(cortex.adapter.module)

    optimizer = torch.optim.AdamW(
        cortex.trainable_parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )

    def evaluate_loss() -> float:
        cortex.adapter.module.eval()
        losses: list[float] = []
        with torch.no_grad():
            for input_ids, labels, latent, weight in encoded_examples:
                if latent is not None:
                    cortex.set_latent(latent)
                losses.append(float(_single_loss(cortex, input_ids, labels, weight).detach().cpu()))
        cortex.adapter.module.train()
        return sum(losses) / len(losses)

    initial_loss = evaluate_loss()
    best_loss = initial_loss
    steps = 0
    adapter_gradients_seen = 0
    base_gradients_seen = 0

    for _epoch in range(config.epochs):
        for input_ids, labels, latent, weight in encoded_examples:
            if latent is not None:
                cortex.set_latent(latent)
            optimizer.zero_grad(set_to_none=True)
            loss = _single_loss(cortex, input_ids, labels, weight)
            loss.backward()

            adapter_gradients_seen += sum(
                1 for parameter in cortex.adapter.module.parameters()
                if parameter.grad is not None and torch.isfinite(parameter.grad).all()
            )
            base_gradients_seen += sum(
                1 for parameter in cortex.model.parameters()
                if parameter.grad is not None
            )
            if base_gradients_seen:
                raise RuntimeError("base Qwen received gradients")

            torch.nn.utils.clip_grad_norm_(cortex.trainable_parameters(), config.max_grad_norm)
            optimizer.step()
            steps += 1
        best_loss = min(best_loss, evaluate_loss())

    final_loss = evaluate_loss()
    base_after = parameter_guard_snapshot(cortex.model)
    adapter_after = module_parameter_digest(cortex.adapter.module)
    gate_after = float(cortex.adapter.effective_gate().detach().cpu())

    return PersonalTrainingReceipt(
        schema="NOLANE-L6-PERSONAL-CORTEX-TRAINING-V1",
        authority="TRAINED_ADAPTER_CANDIDATE_ONLY",
        examples=len(encoded_examples),
        optimizer_steps=steps,
        initial_loss=initial_loss,
        final_loss=final_loss,
        best_loss=best_loss,
        loss_improvement=initial_loss - final_loss,
        adapter_parameters=cortex.trainable_parameter_count(),
        frozen_base_parameters=cortex.frozen_base_parameter_count(),
        adapter_digest_before=adapter_before,
        adapter_digest_after=adapter_after,
        adapter_changed=adapter_before != adapter_after,
        base_model_unchanged=base_before == base_after,
        base_gradients_seen=base_gradients_seen,
        adapter_gradients_seen=adapter_gradients_seen,
        effective_gate_before=gate_start,
        effective_gate_after=gate_after,
        config=asdict(config),
    )


def train_text_examples(
    cortex: TrainablePersonalCortex,
    tokenizer,
    examples: list[PersonalizationExample],
    *,
    system_prompt: str,
    default_latent: list[float],
    config: PersonalTrainingConfig | None = None,
) -> PersonalTrainingReceipt:
    config = config or PersonalTrainingConfig()
    encoded = []
    for example in examples:
        input_ids, labels = encode_chat_example(
            tokenizer,
            example,
            system_prompt=system_prompt,
            max_length=config.max_length,
        )
        encoded.append((input_ids, labels, example.latent or default_latent, example.weight))
    return train_encoded_examples(cortex, encoded, config=config)


def save_trained_adapter(
    output_dir: str | Path,
    cortex: TrainablePersonalCortex,
    receipt: PersonalTrainingReceipt,
    *,
    base_model_fingerprint: str,
    source_candidate_checkpoint_sha256: str | None = None,
    dataset_fingerprint: str | None = None,
) -> dict[str, Any]:
    torch = cortex.adapter.torch
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "personal-cortex-adapter.pt"
    payload = {
        "schema": "NOLANE-L6-PERSONAL-CORTEX-ADAPTER-V1",
        "authority": "TRAINED_CANDIDATE_UNPROMOTED",
        "base_model_fingerprint": str(base_model_fingerprint),
        "hidden_size": cortex.adapter.hidden_size,
        "adapter_config": asdict(cortex.adapter.config),
        "personal_cortex_config": asdict(cortex.config),
        "adapter_state": cortex.adapter.module.state_dict(),
        "training_receipt": receipt.to_dict(),
        "source_candidate_checkpoint_sha256": source_candidate_checkpoint_sha256,
        "dataset_fingerprint": dataset_fingerprint,
    }
    torch.save(payload, checkpoint_path)
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    manifest = {
        "schema": payload["schema"],
        "authority": payload["authority"],
        "base_model_fingerprint": payload["base_model_fingerprint"],
        "hidden_size": payload["hidden_size"],
        "adapter_config": payload["adapter_config"],
        "personal_cortex_config": payload["personal_cortex_config"],
        "adapter_parameters": receipt.adapter_parameters,
        "training": receipt.to_dict(),
        "source_candidate_checkpoint_sha256": source_candidate_checkpoint_sha256,
        "dataset_fingerprint": dataset_fingerprint,
        "checkpoint_sha256": digest,
        "adapter_state_digest": module_parameter_digest(cortex.adapter.module),
    }
    manifest["artifact_id"] = payload_digest(manifest)
    (output_dir / "personal-cortex-manifest.json").write_text(
        canonical_json(manifest) + "\n",
        encoding="utf-8",
    )
    return manifest
