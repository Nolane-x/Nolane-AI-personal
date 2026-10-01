from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .surgery import module_parameter_digest, parameter_guard_snapshot


@dataclass(slots=True)
class NativeTrainingConfig:
    epochs: int = 6
    learning_rate: float = 2e-3
    distill_weight: float = 1.0
    distill_temperature: float = 2.0
    max_grad_norm: float = 1.0

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be >=1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.distill_weight < 0:
            raise ValueError("distill_weight must be non-negative")
        if self.distill_temperature <= 0:
            raise ValueError("distill_temperature must be positive")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")


@dataclass(slots=True)
class NativeTrainingReceipt:
    schema: str
    authority: str
    spec_sha256: str
    examples: int
    epochs: int
    optimizer_steps: int
    task_initial_loss: float
    task_final_loss: float
    distill_initial_loss: float
    distill_final_loss: float
    cortex_parameters: int
    cortex_digest_before: str
    cortex_digest_after: str
    cortex_changed: bool
    qwen_model_unchanged: bool
    qwen_gradients_seen: int
    cortex_gradients_seen: int
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _task_nll(model, encoded_examples) -> float:
    torch = model.cortex.torch
    device = next(model.qwen_model.parameters()).device
    losses = []
    model.qwen_model.eval()
    model.cortex.eval()
    with torch.no_grad():
        for input_ids, labels, latent, weight in encoded_examples:
            ids = torch.tensor([input_ids], dtype=torch.long, device=device)
            y = torch.tensor([labels], dtype=torch.long, device=device)
            if latent is not None:
                model.set_latent(latent)
            out = model.forward(input_ids=ids, labels=y, state=None)
            losses.append(float(out.loss.detach().cpu()) * float(weight))
    return sum(losses) / max(1, len(losses))


def _distill_kl(student_logits, teacher_logits, *, temperature: float):
    torch = __import__("torch")
    t = float(temperature)
    student = torch.nn.functional.log_softmax(student_logits.float() / t, dim=-1)
    teacher = torch.nn.functional.softmax(teacher_logits.float() / t, dim=-1)
    return torch.nn.functional.kl_div(
        student,
        teacher,
        reduction="batchmean",
    ) * (t * t)


def mean_encoded_nll(model, encoded_examples, *, native_enabled: bool) -> float:
    torch = model.cortex.torch
    device = next(model.qwen_model.parameters()).device
    losses = []
    model.qwen_model.eval()
    model.cortex.eval()
    with torch.no_grad():
        for input_ids, labels, latent, weight in encoded_examples:
            ids = torch.tensor([input_ids], dtype=torch.long, device=device)
            y = torch.tensor([labels], dtype=torch.long, device=device)
            if latent is not None:
                model.set_latent(latent)
            if native_enabled:
                out = model.forward(input_ids=ids, labels=y, state=None)
                loss = out.loss
            else:
                out = model.qwen_model(input_ids=ids, labels=y, use_cache=False)
                loss = out.loss
            losses.append(float(loss.detach().cpu()) * float(weight))
    return sum(losses) / max(1, len(losses))


def train_native_boundary(
    model,
    encoded_examples,
    spec: dict,
    *,
    config: NativeTrainingConfig | None = None,
) -> NativeTrainingReceipt:
    config = config or NativeTrainingConfig()
    config.validate()
    if not encoded_examples:
        raise ValueError("native-boundary training examples are empty")

    torch = model.cortex.torch
    device = next(model.qwen_model.parameters()).device
    model.assert_gradient_boundary()
    model.qwen_model.eval()
    model.cortex.to(str(device)).train()

    base_before = parameter_guard_snapshot(model.qwen_model)
    digest_before = module_parameter_digest(model.cortex.module)
    task_initial = _task_nll(model, encoded_examples)
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=config.learning_rate,
        weight_decay=0.0,
    )
    steps = 0
    qwen_gradients_seen = 0
    cortex_gradients_seen = 0
    distill_initial = None
    last_distill_losses = []

    for _epoch in range(config.epochs):
        epoch_distill = []
        model.cortex.train()
        for input_ids, labels, latent, weight in encoded_examples:
            ids = torch.tensor([input_ids], dtype=torch.long, device=device)
            y = torch.tensor([labels], dtype=torch.long, device=device)
            if latent is not None:
                model.set_latent(latent)

            with torch.no_grad():
                teacher_logits = model.qwen_model(
                    input_ids=ids,
                    use_cache=False,
                ).logits.detach()

            optimizer.zero_grad(set_to_none=True)
            student = model.forward(
                input_ids=ids,
                labels=y,
                state=None,
            )
            distill = _distill_kl(
                student.logits,
                teacher_logits,
                temperature=config.distill_temperature,
            )
            if distill_initial is None:
                distill_initial = float(distill.detach().cpu())
            total = (
                student.loss * float(weight)
                + config.distill_weight * distill
            )
            total.backward()

            qwen_now = sum(
                1 for parameter in model.qwen_model.parameters()
                if parameter.grad is not None
            )
            qwen_gradients_seen += qwen_now
            if qwen_now:
                raise RuntimeError("Qwen received gradients during native-boundary training")
            cortex_gradients_seen += sum(
                1 for parameter in model.cortex.module.parameters()
                if parameter.grad is not None
                and torch.isfinite(parameter.grad).all()
            )
            torch.nn.utils.clip_grad_norm_(
                model.trainable_parameters(),
                config.max_grad_norm,
            )
            optimizer.step()
            steps += 1
            epoch_distill.append(float(distill.detach().cpu()))
        last_distill_losses = epoch_distill

    task_final = _task_nll(model, encoded_examples)
    distill_final = sum(last_distill_losses) / max(1, len(last_distill_losses))
    base_after = parameter_guard_snapshot(model.qwen_model)
    digest_after = module_parameter_digest(model.cortex.module)

    return NativeTrainingReceipt(
        schema="NOLANE-L15-NATIVE-BOUNDARY-TRAINING-V1",
        authority="TRAINED_NATIVE_BOUNDARY_CANDIDATE_ONLY",
        spec_sha256=str(spec["spec_sha256"]),
        examples=len(encoded_examples),
        epochs=config.epochs,
        optimizer_steps=steps,
        task_initial_loss=float(task_initial),
        task_final_loss=float(task_final),
        distill_initial_loss=float(distill_initial if distill_initial is not None else 0.0),
        distill_final_loss=float(distill_final),
        cortex_parameters=model.trainable_parameter_count(),
        cortex_digest_before=digest_before,
        cortex_digest_after=digest_after,
        cortex_changed=digest_before != digest_after,
        qwen_model_unchanged=base_before == base_after,
        qwen_gradients_seen=qwen_gradients_seen,
        cortex_gradients_seen=cortex_gradients_seen,
        config=asdict(config),
    )
