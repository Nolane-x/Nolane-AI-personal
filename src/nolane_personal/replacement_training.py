from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .replacement_cortex import RecurrentReplacementCortex
from .store import canonical_json, payload_digest
from .surgery import module_parameter_digest, parameter_guard_snapshot, resolve_transformer_layers


@dataclass(slots=True)
class ReplacementTrainingConfig:
    distill_epochs: int = 2
    task_epochs: int = 4
    learning_rate: float = 2e-3
    distill_learning_rate: float = 3e-3
    max_grad_norm: float = 1.0
    cosine_weight: float = 0.10

    def validate(self) -> None:
        if self.distill_epochs < 0 or self.task_epochs < 1:
            raise ValueError("invalid epoch counts")
        if self.learning_rate <= 0 or self.distill_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if not 0 <= self.cosine_weight <= 1:
            raise ValueError("cosine_weight must be in [0,1]")


@dataclass(slots=True)
class ReplacementTrainingReceipt:
    schema: str
    authority: str
    examples: int
    selected_layers: list[int]
    teacher_pairs_seen: int
    distill_initial_loss: float | None
    distill_final_loss: float | None
    task_initial_loss: float
    task_final_loss: float
    task_loss_improvement: float
    optimizer_steps: int
    replacement_parameters: int
    frozen_base_parameters: int
    replacement_digest_before: str
    replacement_digest_after: str
    replacement_changed: bool
    base_model_unchanged: bool
    base_gradients_seen: int
    replacement_gradients_seen: int
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _capture_teacher_pairs(model, input_ids, layer_indices):
    """Capture input/output hidden tensors from original frozen decoder blocks."""
    layers = resolve_transformer_layers(model)
    captures = {}
    handles = []

    for index in layer_indices:
        layer = layers[index]

        def pre_hook(_module, args, kwargs, *, _index=index):
            hidden = args[0] if args else kwargs.get("hidden_states")
            captures.setdefault(_index, {})["input"] = hidden.detach()

        def post_hook(_module, _args, output, *, _index=index):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            captures.setdefault(_index, {})["output"] = hidden.detach()

        handles.append(layer.register_forward_pre_hook(pre_hook, with_kwargs=True))
        handles.append(layer.register_forward_hook(post_hook))

    try:
        with model.device if False else _nullcontext():
            model(input_ids=input_ids, use_cache=False)
    finally:
        for handle in reversed(handles):
            handle.remove()

    result = []
    for index in layer_indices:
        pair = captures.get(index)
        if not pair or "input" not in pair or "output" not in pair:
            raise RuntimeError(f"failed to capture teacher layer {index}")
        result.append((index, pair["input"], pair["output"]))
    return result


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _distill_loss(replacement, pairs, latent, cosine_weight):
    torch = replacement.torch
    terms = []
    for layer_index, teacher_input, teacher_output in pairs:
        predicted, _state, _gate = replacement.replace(
            teacher_input,
            latent,
            layer_index=layer_index,
            state=None,
        )
        mse = torch.nn.functional.smooth_l1_loss(predicted.float(), teacher_output.float())
        cosine = 1.0 - torch.nn.functional.cosine_similarity(
            predicted.float(),
            teacher_output.float(),
            dim=-1,
        ).mean()
        terms.append(mse + float(cosine_weight) * cosine)
    return torch.stack(terms).mean()


def _task_loss(cortex, input_ids, labels, weight):
    torch = cortex.replacement.torch
    device = next(cortex.model.parameters()).device
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    y = torch.tensor([labels], dtype=torch.long, device=device)
    cortex.reset_state()
    return cortex.forward(input_ids=ids, labels=y, use_cache=False).loss * float(weight)


def mean_encoded_nll(cortex, encoded_examples, *, replacement_enabled: bool) -> float:
    torch = cortex.replacement.torch
    device = next(cortex.model.parameters()).device
    losses = []
    cortex.model.eval()
    cortex.replacement.eval()
    with torch.no_grad():
        for input_ids, labels, latent, weight in encoded_examples:
            ids = torch.tensor([input_ids], dtype=torch.long, device=device)
            y = torch.tensor([labels], dtype=torch.long, device=device)
            if latent is not None:
                cortex.set_latent(latent)
            cortex.reset_state()
            if replacement_enabled:
                out = cortex.forward(input_ids=ids, labels=y, use_cache=False)
            else:
                out = cortex.model(input_ids=ids, labels=y, use_cache=False)
            losses.append(float(out.loss.detach().cpu()) * float(weight))
    return sum(losses) / max(1, len(losses))


def train_encoded_examples(cortex, encoded_examples, *, config=None) -> ReplacementTrainingReceipt:
    config = config or ReplacementTrainingConfig()
    config.validate()
    if not encoded_examples:
        raise ValueError("no encoded examples")
    torch = cortex.replacement.torch
    cortex.assert_gradient_boundary()
    cortex.model.eval()
    device = next(cortex.model.parameters()).device
    cortex.replacement.to(str(device)).train()

    base_before = parameter_guard_snapshot(cortex.model)
    replacement_before = module_parameter_digest(cortex.replacement.module)
    teacher_pairs_seen = 0
    distill_first = None
    distill_last = None
    steps = 0
    replacement_gradients_seen = 0
    base_gradients_seen = 0

    if config.distill_epochs:
        optimizer = torch.optim.AdamW(
            cortex.trainable_parameters(),
            lr=config.distill_learning_rate,
            weight_decay=0.0,
        )
        for _epoch in range(config.distill_epochs):
            epoch_losses = []
            for input_ids, _labels, latent, _weight in encoded_examples:
                ids = torch.tensor([input_ids], dtype=torch.long, device=device)
                pairs = _capture_teacher_pairs(cortex.model, ids, cortex.config.layer_indices)
                teacher_pairs_seen += len(pairs)
                optimizer.zero_grad(set_to_none=True)
                loss = _distill_loss(
                    cortex.replacement,
                    pairs,
                    latent if latent is not None else cortex.latent,
                    config.cosine_weight,
                )
                if distill_first is None:
                    distill_first = float(loss.detach().cpu())
                loss.backward()
                replacement_gradients_seen += sum(
                    1 for p in cortex.replacement.module.parameters()
                    if p.grad is not None and torch.isfinite(p.grad).all()
                )
                base_gradients_seen += sum(1 for p in cortex.model.parameters() if p.grad is not None)
                if base_gradients_seen:
                    raise RuntimeError("base Qwen received gradients during distillation")
                torch.nn.utils.clip_grad_norm_(cortex.trainable_parameters(), config.max_grad_norm)
                optimizer.step()
                steps += 1
                epoch_losses.append(float(loss.detach().cpu()))
            distill_last = sum(epoch_losses) / len(epoch_losses)

    task_initial = mean_encoded_nll(cortex, encoded_examples, replacement_enabled=True)
    optimizer = torch.optim.AdamW(
        cortex.trainable_parameters(),
        lr=config.learning_rate,
        weight_decay=0.0,
    )
    for _epoch in range(config.task_epochs):
        cortex.replacement.train()
        for input_ids, labels, latent, weight in encoded_examples:
            if latent is not None:
                cortex.set_latent(latent)
            optimizer.zero_grad(set_to_none=True)
            loss = _task_loss(cortex, input_ids, labels, weight)
            loss.backward()
            replacement_gradients_seen += sum(
                1 for p in cortex.replacement.module.parameters()
                if p.grad is not None and torch.isfinite(p.grad).all()
            )
            base_gradients_seen += sum(1 for p in cortex.model.parameters() if p.grad is not None)
            if base_gradients_seen:
                raise RuntimeError("base Qwen received gradients during task training")
            torch.nn.utils.clip_grad_norm_(cortex.trainable_parameters(), config.max_grad_norm)
            optimizer.step()
            steps += 1

    task_final = mean_encoded_nll(cortex, encoded_examples, replacement_enabled=True)
    base_after = parameter_guard_snapshot(cortex.model)
    replacement_after = module_parameter_digest(cortex.replacement.module)
    return ReplacementTrainingReceipt(
        schema="NOLANE-L9-BLOCK-REPLACEMENT-TRAINING-V1",
        authority="TRAINED_REPLACEMENT_CANDIDATE_ONLY",
        examples=len(encoded_examples),
        selected_layers=list(cortex.config.layer_indices),
        teacher_pairs_seen=teacher_pairs_seen,
        distill_initial_loss=distill_first,
        distill_final_loss=distill_last,
        task_initial_loss=task_initial,
        task_final_loss=task_final,
        task_loss_improvement=task_initial-task_final,
        optimizer_steps=steps,
        replacement_parameters=cortex.trainable_parameter_count(),
        frozen_base_parameters=cortex.frozen_base_parameter_count(),
        replacement_digest_before=replacement_before,
        replacement_digest_after=replacement_after,
        replacement_changed=replacement_before != replacement_after,
        base_model_unchanged=base_before == base_after,
        base_gradients_seen=base_gradients_seen,
        replacement_gradients_seen=replacement_gradients_seen,
        config=asdict(config),
    )


def save_trained_replacement(output_dir, cortex, receipt, *, base_model_fingerprint, dataset_fingerprint=None):
    torch = cortex.replacement.torch
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "recurrent-block-replacement.pt"
    payload = {
        "schema": "NOLANE-L9-RECURRENT-BLOCK-REPLACEMENT-V1",
        "authority": "TRAINED_CANDIDATE_UNPROMOTED",
        "base_model_fingerprint": str(base_model_fingerprint),
        "hidden_size": cortex.replacement.hidden_size,
        "replacement_config": asdict(cortex.replacement.config),
        "cortex_config": asdict(cortex.config),
        "replacement_state": cortex.replacement.module.state_dict(),
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
        "replacement_config": payload["replacement_config"],
        "cortex_config": payload["cortex_config"],
        "replacement_parameters": receipt.replacement_parameters,
        "training": receipt.to_dict(),
        "dataset_fingerprint": dataset_fingerprint,
        "checkpoint_sha256": digest,
        "replacement_state_digest": module_parameter_digest(cortex.replacement.module),
    }
    manifest["artifact_id"] = payload_digest(manifest)
    (output_dir / "recurrent-block-replacement-manifest.json").write_text(
        canonical_json(manifest)+"\n",
        encoding="utf-8",
    )
    return manifest
