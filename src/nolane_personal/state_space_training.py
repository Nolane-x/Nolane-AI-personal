from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .state_space_region import CortexRegion
from .surgery import module_parameter_digest, parameter_guard_snapshot, resolve_transformer_layers


@dataclass(slots=True)
class StateSpaceTrainingConfig:
    distill_epochs_per_stage: int = 2
    task_epochs_per_stage: int = 3
    learning_rate: float = 2e-3
    distill_learning_rate: float = 3e-3
    max_grad_norm: float = 1.0
    cosine_weight: float = 0.10
    max_dev_regression: float = 0.05

    def validate(self) -> None:
        if self.distill_epochs_per_stage < 0:
            raise ValueError("distill_epochs_per_stage must be >=0")
        if self.task_epochs_per_stage < 1:
            raise ValueError("task_epochs_per_stage must be >=1")
        if self.learning_rate <= 0 or self.distill_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if not 0.0 <= self.cosine_weight <= 1.0:
            raise ValueError("cosine_weight must be in [0,1]")
        if self.max_dev_regression < 0:
            raise ValueError("max_dev_regression must be >=0")


@dataclass(slots=True)
class StateSpaceStageReceipt:
    stage: int
    region: dict[str, int]
    width: int
    replaced_fraction: float
    teacher_examples_seen: int
    distill_initial_loss: float | None
    distill_final_loss: float | None
    task_initial_loss: float
    task_final_loss: float
    dev_nll_reference: float
    dev_nll_after: float
    dev_regression: float
    accepted: bool
    rollback_reason: str | None
    optimizer_steps: int
    base_model_unchanged: bool
    base_gradients_seen: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StateSpaceTrainingReceipt:
    schema: str
    authority: str
    plan_sha256: str
    stages_attempted: int
    stages_accepted: int
    final_region: dict[str, int]
    replaced_layers: int
    replaced_fraction: float
    cortex_parameters: int
    cortex_digest_before: str
    cortex_digest_after: str
    cortex_changed: bool
    base_model_unchanged: bool
    base_gradients_seen: int
    cortex_gradients_seen: int
    stage_receipts: list[dict[str, Any]]
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _capture_teacher_region(model, input_ids, region: CortexRegion):
    layers = resolve_transformer_layers(model)
    capture: dict[str, Any] = {}
    handles = []

    def pre_hook(_module, args, kwargs):
        hidden = args[0] if args else kwargs.get("hidden_states")
        if hidden is None:
            raise RuntimeError("decoder hidden_states missing")
        capture["input"] = hidden.detach()

    def post_hook(_module, _args, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        capture["output"] = hidden.detach()

    handles.append(layers[region.start].register_forward_pre_hook(pre_hook, with_kwargs=True))
    handles.append(layers[region.end].register_forward_hook(post_hook))
    try:
        model(input_ids=input_ids, use_cache=False)
    finally:
        for handle in reversed(handles):
            handle.remove()
    if "input" not in capture or "output" not in capture:
        raise RuntimeError("failed to capture state-space teacher region")
    return capture["input"], capture["output"]


def _distill_loss(cortex, teacher_input, teacher_output, latent, cosine_weight):
    torch = cortex.torch
    predicted, _state, _trace = cortex.scan(
        teacher_input,
        latent,
        state=None,
    )
    mse = torch.nn.functional.smooth_l1_loss(
        predicted.float(),
        teacher_output.float(),
    )
    cosine = 1.0 - torch.nn.functional.cosine_similarity(
        predicted.float(),
        teacher_output.float(),
        dim=-1,
    ).mean()
    return mse + float(cosine_weight) * cosine


def _task_loss(model, input_ids, labels, weight):
    torch = model.cortex.torch
    device = next(model.model.parameters()).device
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    y = torch.tensor([labels], dtype=torch.long, device=device)
    model.reset_state()
    return model.forward(input_ids=ids, labels=y, use_cache=False).loss * float(weight)


def mean_encoded_nll(model, encoded_examples, *, cortex_enabled: bool) -> float:
    torch = model.cortex.torch
    device = next(model.model.parameters()).device
    losses = []
    model.model.eval()
    model.cortex.eval()
    with torch.no_grad():
        for input_ids, labels, latent, weight in encoded_examples:
            ids = torch.tensor([input_ids], dtype=torch.long, device=device)
            y = torch.tensor([labels], dtype=torch.long, device=device)
            if latent is not None:
                model.set_latent(latent)
            model.reset_state()
            if cortex_enabled:
                output = model.forward(input_ids=ids, labels=y, use_cache=False)
            else:
                output = model.model(input_ids=ids, labels=y, use_cache=False)
            losses.append(float(output.loss.detach().cpu()) * float(weight))
    return sum(losses) / max(1, len(losses))


def _clone_state(module):
    return {
        name: tensor.detach().clone()
        for name, tensor in module.state_dict().items()
    }


def _train_stage(model, train_examples, config):
    torch = model.cortex.torch
    device = next(model.model.parameters()).device
    steps = 0
    cortex_gradients_seen = 0
    base_gradients_seen = 0
    teacher_seen = 0
    distill_initial = None
    distill_final = None

    if config.distill_epochs_per_stage:
        optimizer = torch.optim.AdamW(
            model.trainable_parameters(),
            lr=config.distill_learning_rate,
            weight_decay=0.0,
        )
        for _epoch in range(config.distill_epochs_per_stage):
            epoch_losses = []
            for input_ids, _labels, latent, _weight in train_examples:
                ids = torch.tensor([input_ids], dtype=torch.long, device=device)
                teacher_input, teacher_output = _capture_teacher_region(
                    model.model,
                    ids,
                    model.config.region,
                )
                optimizer.zero_grad(set_to_none=True)
                loss = _distill_loss(
                    model.cortex,
                    teacher_input,
                    teacher_output,
                    latent if latent is not None else model.latent,
                    config.cosine_weight,
                )
                if distill_initial is None:
                    distill_initial = float(loss.detach().cpu())
                loss.backward()
                cortex_gradients_seen += sum(
                    1 for p in model.cortex.module.parameters()
                    if p.grad is not None and torch.isfinite(p.grad).all()
                )
                base_gradients_seen += sum(
                    1 for p in model.model.parameters()
                    if p.grad is not None
                )
                if base_gradients_seen:
                    raise RuntimeError("base Qwen received gradients during state-space distillation")
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), config.max_grad_norm)
                optimizer.step()
                steps += 1
                teacher_seen += 1
                epoch_losses.append(float(loss.detach().cpu()))
            distill_final = sum(epoch_losses) / len(epoch_losses)

    task_initial = mean_encoded_nll(model, train_examples, cortex_enabled=True)
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=config.learning_rate,
        weight_decay=0.0,
    )
    for _epoch in range(config.task_epochs_per_stage):
        model.cortex.train()
        for input_ids, labels, latent, weight in train_examples:
            if latent is not None:
                model.set_latent(latent)
            optimizer.zero_grad(set_to_none=True)
            loss = _task_loss(model, input_ids, labels, weight)
            loss.backward()
            cortex_gradients_seen += sum(
                1 for p in model.cortex.module.parameters()
                if p.grad is not None and torch.isfinite(p.grad).all()
            )
            base_gradients_seen += sum(
                1 for p in model.model.parameters()
                if p.grad is not None
            )
            if base_gradients_seen:
                raise RuntimeError("base Qwen received gradients during state-space task training")
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), config.max_grad_norm)
            optimizer.step()
            steps += 1

    task_final = mean_encoded_nll(model, train_examples, cortex_enabled=True)
    return {
        "teacher_examples_seen": teacher_seen,
        "distill_initial_loss": distill_initial,
        "distill_final_loss": distill_final,
        "task_initial_loss": task_initial,
        "task_final_loss": task_final,
        "optimizer_steps": steps,
        "cortex_gradients_seen": cortex_gradients_seen,
        "base_gradients_seen": base_gradients_seen,
    }


def train_state_space_stages(model, train_examples, dev_examples, plan, *, config=None):
    config = config or StateSpaceTrainingConfig()
    config.validate()
    if not train_examples or not dev_examples:
        raise ValueError("state-space train/dev examples must be non-empty")

    model.assert_gradient_boundary()
    model.model.eval()
    device = next(model.model.parameters()).device
    model.cortex.to(str(device)).train()

    total_layers = int(plan["total_layers"])
    base_before = parameter_guard_snapshot(model.model)
    digest_before = module_parameter_digest(model.cortex.module)
    accepted_region = None
    accepted_dev = mean_encoded_nll(model, dev_examples, cortex_enabled=False)
    stage_receipts = []
    base_gradients_seen = 0
    cortex_gradients_seen = 0

    for row in plan["stages"]:
        region = CortexRegion(
            int(row["region"]["start"]),
            int(row["region"]["end"]),
        )
        region.validate(total_layers=total_layers)
        if accepted_region is not None:
            if region.start > accepted_region.start or region.end < accepted_region.end:
                raise ValueError("state-space plan attempted to shrink accepted region")

        snapshot = _clone_state(model.cortex.module)
        prior_region = model.config.region
        dev_reference = accepted_dev
        model.config.region = region
        stage_base = parameter_guard_snapshot(model.model)
        result = _train_stage(model, train_examples, config)
        dev_after = mean_encoded_nll(model, dev_examples, cortex_enabled=True)
        regression = dev_after - dev_reference
        accepted = bool(
            result["base_gradients_seen"] == 0
            and stage_base == parameter_guard_snapshot(model.model)
            and regression <= config.max_dev_regression
        )
        reason = None
        if regression > config.max_dev_regression:
            reason = "dev_regression_gate_failed"
        elif result["base_gradients_seen"]:
            reason = "base_gradient_boundary_failed"

        if accepted:
            accepted_region = region
            accepted_dev = dev_after
        else:
            model.cortex.module.load_state_dict(snapshot)
            model.config.region = accepted_region or prior_region

        base_gradients_seen += int(result["base_gradients_seen"])
        cortex_gradients_seen += int(result["cortex_gradients_seen"])
        stage_receipts.append(
            StateSpaceStageReceipt(
                stage=int(row["stage"]),
                region=region.to_dict(),
                width=region.width,
                replaced_fraction=region.width / total_layers,
                teacher_examples_seen=int(result["teacher_examples_seen"]),
                distill_initial_loss=result["distill_initial_loss"],
                distill_final_loss=result["distill_final_loss"],
                task_initial_loss=float(result["task_initial_loss"]),
                task_final_loss=float(result["task_final_loss"]),
                dev_nll_reference=float(dev_reference),
                dev_nll_after=float(dev_after),
                dev_regression=float(regression),
                accepted=accepted,
                rollback_reason=reason,
                optimizer_steps=int(result["optimizer_steps"]),
                base_model_unchanged=stage_base == parameter_guard_snapshot(model.model),
                base_gradients_seen=int(result["base_gradients_seen"]),
            ).to_dict()
        )
        if not accepted:
            break

    if accepted_region is None:
        raise RuntimeError("no state-space cortex stage passed the dev gate")

    model.config.region = accepted_region
    base_after = parameter_guard_snapshot(model.model)
    digest_after = module_parameter_digest(model.cortex.module)
    return StateSpaceTrainingReceipt(
        schema="NOLANE-L12-STATE-SPACE-CORTEX-TRAINING-V1",
        authority="TRAINED_STATE_SPACE_CANDIDATE_ONLY",
        plan_sha256=str(plan["plan_sha256"]),
        stages_attempted=len(stage_receipts),
        stages_accepted=sum(1 for row in stage_receipts if row["accepted"]),
        final_region=accepted_region.to_dict(),
        replaced_layers=accepted_region.width,
        replaced_fraction=accepted_region.width / total_layers,
        cortex_parameters=model.trainable_parameter_count(),
        cortex_digest_before=digest_before,
        cortex_digest_after=digest_after,
        cortex_changed=digest_before != digest_after,
        base_model_unchanged=base_before == base_after,
        base_gradients_seen=base_gradients_seen,
        cortex_gradients_seen=cortex_gradients_seen,
        stage_receipts=stage_receipts,
        config=asdict(config),
    )
