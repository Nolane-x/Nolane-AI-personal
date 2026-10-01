from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .depth_bridge_cortex import TrainableLivingBridgeCortex
from .store import canonical_json, payload_digest
from .surgery import module_parameter_digest, parameter_guard_snapshot


@dataclass(slots=True)
class BridgeTrainingConfig:
    epochs: int = 4
    learning_rate: float = 2e-3
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")


@dataclass(slots=True)
class BridgeTrainingReceipt:
    schema: str
    authority: str
    examples: int
    optimizer_steps: int
    initial_loss: float
    final_loss: float
    best_loss: float
    loss_improvement: float
    bridge_parameters: int
    frozen_base_parameters: int
    bridge_digest_before: str
    bridge_digest_after: str
    bridge_changed: bool
    base_model_unchanged: bool
    base_gradients_seen: int
    bridge_gradients_seen: int
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _loss(cortex, input_ids, labels, weight):
    torch = cortex.bridge.torch
    device = next(cortex.model.parameters()).device
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    y = torch.tensor([labels], dtype=torch.long, device=device)
    return cortex.forward(input_ids=ids, labels=y, use_cache=False).loss * float(weight)


def mean_encoded_nll(cortex, encoded_examples, *, personalized: bool) -> float:
    torch = cortex.bridge.torch
    device = next(cortex.model.parameters()).device
    losses = []
    cortex.model.eval()
    cortex.bridge.eval()
    with torch.no_grad():
        for input_ids, labels, latent, weight in encoded_examples:
            ids = torch.tensor([input_ids], dtype=torch.long, device=device)
            y = torch.tensor([labels], dtype=torch.long, device=device)
            if latent is not None:
                cortex.set_latent(latent)
            if personalized:
                out = cortex.forward(input_ids=ids, labels=y, use_cache=False)
            else:
                out = cortex.model(input_ids=ids, labels=y, use_cache=False)
            losses.append(float(out.loss.detach().cpu()) * float(weight))
    return sum(losses) / max(1, len(losses))


def train_encoded_examples(cortex, encoded_examples, *, config=None) -> BridgeTrainingReceipt:
    config = config or BridgeTrainingConfig()
    config.validate()
    if not encoded_examples:
        raise ValueError("no encoded examples")

    torch = cortex.bridge.torch
    cortex.assert_gradient_boundary()
    cortex.model.eval()
    device = next(cortex.model.parameters()).device
    cortex.bridge.to(str(device)).train()

    base_before = parameter_guard_snapshot(cortex.model)
    bridge_before = module_parameter_digest(cortex.bridge.module)
    optimizer = torch.optim.AdamW(
        cortex.trainable_parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    initial_loss = mean_encoded_nll(cortex, encoded_examples, personalized=True)
    best_loss = initial_loss
    steps = 0
    bridge_gradients_seen = 0
    base_gradients_seen = 0

    for _ in range(config.epochs):
        cortex.bridge.train()
        for input_ids, labels, latent, weight in encoded_examples:
            if latent is not None:
                cortex.set_latent(latent)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(cortex, input_ids, labels, weight)
            loss.backward()
            bridge_gradients_seen += sum(
                1 for p in cortex.bridge.module.parameters()
                if p.grad is not None and torch.isfinite(p.grad).all()
            )
            base_gradients_seen += sum(1 for p in cortex.model.parameters() if p.grad is not None)
            if base_gradients_seen:
                raise RuntimeError("base Qwen received gradients")
            torch.nn.utils.clip_grad_norm_(cortex.trainable_parameters(), config.max_grad_norm)
            optimizer.step()
            steps += 1
        best_loss = min(best_loss, mean_encoded_nll(cortex, encoded_examples, personalized=True))

    final_loss = mean_encoded_nll(cortex, encoded_examples, personalized=True)
    base_after = parameter_guard_snapshot(cortex.model)
    bridge_after = module_parameter_digest(cortex.bridge.module)
    return BridgeTrainingReceipt(
        schema="NOLANE-L8-DEPTH-BRIDGE-TRAINING-V1",
        authority="TRAINED_BRIDGE_CANDIDATE_ONLY",
        examples=len(encoded_examples),
        optimizer_steps=steps,
        initial_loss=initial_loss,
        final_loss=final_loss,
        best_loss=best_loss,
        loss_improvement=initial_loss-final_loss,
        bridge_parameters=cortex.trainable_parameter_count(),
        frozen_base_parameters=cortex.frozen_base_parameter_count(),
        bridge_digest_before=bridge_before,
        bridge_digest_after=bridge_after,
        bridge_changed=bridge_before != bridge_after,
        base_model_unchanged=base_before == base_after,
        base_gradients_seen=base_gradients_seen,
        bridge_gradients_seen=bridge_gradients_seen,
        config=asdict(config),
    )


def save_trained_bridge(output_dir, cortex, receipt, *, base_model_fingerprint, dataset_fingerprint=None):
    torch = cortex.bridge.torch
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "depth-bridge.pt"
    payload = {
        "schema": "NOLANE-L8-DEPTH-BRIDGE-V1",
        "authority": "TRAINED_CANDIDATE_UNPROMOTED",
        "base_model_fingerprint": str(base_model_fingerprint),
        "hidden_size": cortex.bridge.hidden_size,
        "bridge_config": asdict(cortex.bridge.config),
        "cortex_config": asdict(cortex.config),
        "bridge_state": cortex.bridge.module.state_dict(),
        "training_receipt": receipt.to_dict(),
        "dataset_fingerprint": dataset_fingerprint,
    }
    torch.save(payload, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = {
        "schema": payload["schema"],
        "authority": payload["authority"],
        "base_model_fingerprint": payload["base_model_fingerprint"],
        "hidden_size": payload["hidden_size"],
        "bridge_config": payload["bridge_config"],
        "cortex_config": payload["cortex_config"],
        "bridge_parameters": receipt.bridge_parameters,
        "training": receipt.to_dict(),
        "dataset_fingerprint": dataset_fingerprint,
        "checkpoint_sha256": digest,
        "bridge_state_digest": module_parameter_digest(cortex.bridge.module),
    }
    manifest["artifact_id"] = payload_digest(manifest)
    (output_dir / "depth-bridge-manifest.json").write_text(canonical_json(manifest)+"\n", encoding="utf-8")
    return manifest
