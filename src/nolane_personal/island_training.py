from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .island_replacement import TransformerIsland, normalize_islands
from .surgery import module_parameter_digest, parameter_guard_snapshot, resolve_transformer_layers


@dataclass(slots=True)
class IslandTrainingConfig:
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
class IslandStageReceipt:
    stage: int
    islands: list[dict[str, int]]
    replaced_layers: int
    replaced_fraction: float
    teacher_regions_seen: int
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
class IslandTrainingReceipt:
    schema: str
    authority: str
    plan_sha256: str
    stages_attempted: int
    stages_accepted: int
    final_islands: list[dict[str, int]]
    replaced_layers: int
    replaced_fraction: float
    replacement_parameters: int
    replacement_digest_before: str
    replacement_digest_after: str
    replacement_changed: bool
    base_model_unchanged: bool
    base_gradients_seen: int
    replacement_gradients_seen: int
    stage_receipts: list[dict[str, Any]]
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _capture_region_teacher_pairs(model, input_ids, islands):
    layers = resolve_transformer_layers(model)
    captures: dict[int, dict[str, Any]] = {}
    handles = []
    for island in islands:
        start_layer = layers[island.start]
        end_layer = layers[island.end]

        def pre_hook(_module, args, kwargs, *, _start=island.start):
            hidden = args[0] if args else kwargs.get("hidden_states")
            if hidden is None:
                raise RuntimeError("decoder hidden_states missing")
            captures.setdefault(_start, {})["input"] = hidden.detach()

        def post_hook(_module, _args, output, *, _start=island.start):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            captures.setdefault(_start, {})["output"] = hidden.detach()

        handles.append(start_layer.register_forward_pre_hook(pre_hook, with_kwargs=True))
        handles.append(end_layer.register_forward_hook(post_hook))
    try:
        model(input_ids=input_ids, use_cache=False)
    finally:
        for handle in reversed(handles):
            handle.remove()

    result = []
    for island in islands:
        row = captures.get(island.start)
        if not row or "input" not in row or "output" not in row:
            raise RuntimeError(f"failed to capture teacher island {island.start}:{island.end}")
        result.append((island, row["input"], row["output"]))
    return result


def _distill_loss(replacement, pairs, latent, cosine_weight):
    torch = replacement.torch
    terms = []
    for island, teacher_input, teacher_output in pairs:
        predicted, _state, _gate = replacement.replace(
            teacher_input,
            latent,
            layer_index=island.start,
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


def mean_encoded_nll(cortex, encoded_examples, *, islands_enabled: bool) -> float:
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
            if islands_enabled:
                output = cortex.forward(input_ids=ids, labels=y, use_cache=False)
            else:
                output = cortex.model(input_ids=ids, labels=y, use_cache=False)
            losses.append(float(output.loss.detach().cpu()) * float(weight))
    return sum(losses) / max(1, len(losses))


def _clone_state(module):
    return {
        name: tensor.detach().clone()
        for name, tensor in module.state_dict().items()
    }


def _train_current_stage(cortex, encoded_examples, config):
    torch = cortex.replacement.torch
    device = next(cortex.model.parameters()).device
    teacher_regions_seen = 0
    distill_initial = None
    distill_final = None
    steps = 0
    replacement_gradients_seen = 0
    base_gradients_seen = 0

    if config.distill_epochs_per_stage:
        optimizer = torch.optim.AdamW(
            cortex.trainable_parameters(),
            lr=config.distill_learning_rate,
            weight_decay=0.0,
        )
        for _epoch in range(config.distill_epochs_per_stage):
            epoch_losses = []
            for input_ids, _labels, latent, _weight in encoded_examples:
                ids = torch.tensor([input_ids], dtype=torch.long, device=device)
                pairs = _capture_region_teacher_pairs(cortex.model, ids, cortex.config.islands)
                teacher_regions_seen += len(pairs)
                optimizer.zero_grad(set_to_none=True)
                loss = _distill_loss(
                    cortex.replacement,
                    pairs,
                    latent if latent is not None else cortex.latent,
                    config.cosine_weight,
                )
                if distill_initial is None:
                    distill_initial = float(loss.detach().cpu())
                loss.backward()
                replacement_gradients_seen += sum(
                    1 for parameter in cortex.replacement.module.parameters()
                    if parameter.grad is not None and torch.isfinite(parameter.grad).all()
                )
                base_gradients_seen += sum(
                    1 for parameter in cortex.model.parameters()
                    if parameter.grad is not None
                )
                if base_gradients_seen:
                    raise RuntimeError("base Qwen received gradients during island distillation")
                torch.nn.utils.clip_grad_norm_(cortex.trainable_parameters(), config.max_grad_norm)
                optimizer.step()
                steps += 1
                epoch_losses.append(float(loss.detach().cpu()))
            distill_final = sum(epoch_losses) / len(epoch_losses)

    task_initial = mean_encoded_nll(cortex, encoded_examples, islands_enabled=True)
    optimizer = torch.optim.AdamW(
        cortex.trainable_parameters(),
        lr=config.learning_rate,
        weight_decay=0.0,
    )
    for _epoch in range(config.task_epochs_per_stage):
        cortex.replacement.train()
        for input_ids, labels, latent, weight in encoded_examples:
            if latent is not None:
                cortex.set_latent(latent)
            optimizer.zero_grad(set_to_none=True)
            loss = _task_loss(cortex, input_ids, labels, weight)
            loss.backward()
            replacement_gradients_seen += sum(
                1 for parameter in cortex.replacement.module.parameters()
                if parameter.grad is not None and torch.isfinite(parameter.grad).all()
            )
            base_gradients_seen += sum(
                1 for parameter in cortex.model.parameters()
                if parameter.grad is not None
            )
            if base_gradients_seen:
                raise RuntimeError("base Qwen received gradients during island task training")
            torch.nn.utils.clip_grad_norm_(cortex.trainable_parameters(), config.max_grad_norm)
            optimizer.step()
            steps += 1

    task_final = mean_encoded_nll(cortex, encoded_examples, islands_enabled=True)
    return {
        "teacher_regions_seen": teacher_regions_seen,
        "distill_initial_loss": distill_initial,
        "distill_final_loss": distill_final,
        "task_initial_loss": task_initial,
        "task_final_loss": task_final,
        "optimizer_steps": steps,
        "replacement_gradients_seen": replacement_gradients_seen,
        "base_gradients_seen": base_gradients_seen,
    }


def train_island_stages(cortex, train_examples, dev_examples, plan, *, config=None) -> IslandTrainingReceipt:
    config = config or IslandTrainingConfig()
    config.validate()
    if not train_examples or not dev_examples:
        raise ValueError("island train/dev examples must be non-empty")

    torch = cortex.replacement.torch
    cortex.assert_gradient_boundary()
    cortex.model.eval()
    device = next(cortex.model.parameters()).device
    cortex.replacement.to(str(device)).train()

    total_layers = int(plan["total_layers"])
    base_before = parameter_guard_snapshot(cortex.model)
    digest_before = module_parameter_digest(cortex.replacement.module)
    base_gradients_seen = 0
    replacement_gradients_seen = 0
    stage_receipts = []
    accepted_islands: tuple[TransformerIsland, ...] = tuple()
    accepted_dev = mean_encoded_nll(cortex, dev_examples, islands_enabled=False)

    for row in plan["stages"]:
        islands = normalize_islands(
            [(int(item["start"]), int(item["end"])) for item in row["islands"]],
            total_layers=total_layers,
            min_gap_layers=int(plan["config"]["min_gap_layers"]),
        )
        if accepted_islands and not set(accepted_islands).issubset(set(islands)):
            raise ValueError("island plan attempted to remove an accepted island")

        snapshot = _clone_state(cortex.replacement.module)
        prior_islands = cortex.config.islands
        dev_reference = accepted_dev
        cortex.config.islands = islands
        stage_base_guard = parameter_guard_snapshot(cortex.model)
        train_result = _train_current_stage(cortex, train_examples, config)
        dev_after = mean_encoded_nll(cortex, dev_examples, islands_enabled=True)
        regression = dev_after - dev_reference
        accepted = bool(
            train_result["base_gradients_seen"] == 0
            and stage_base_guard == parameter_guard_snapshot(cortex.model)
            and regression <= config.max_dev_regression
        )
        reason = None
        if regression > config.max_dev_regression:
            reason = "dev_regression_gate_failed"
        elif train_result["base_gradients_seen"]:
            reason = "base_gradient_boundary_failed"

        if accepted:
            accepted_islands = islands
            accepted_dev = dev_after
        else:
            cortex.replacement.module.load_state_dict(snapshot)
            cortex.config.islands = accepted_islands or prior_islands

        base_gradients_seen += int(train_result["base_gradients_seen"])
        replacement_gradients_seen += int(train_result["replacement_gradients_seen"])
        stage_receipts.append(
            IslandStageReceipt(
                stage=int(row["stage"]),
                islands=[island.to_dict() for island in islands],
                replaced_layers=sum(island.width for island in islands),
                replaced_fraction=sum(island.width for island in islands) / total_layers,
                teacher_regions_seen=int(train_result["teacher_regions_seen"]),
                distill_initial_loss=train_result["distill_initial_loss"],
                distill_final_loss=train_result["distill_final_loss"],
                task_initial_loss=float(train_result["task_initial_loss"]),
                task_final_loss=float(train_result["task_final_loss"]),
                dev_nll_reference=float(dev_reference),
                dev_nll_after=float(dev_after),
                dev_regression=float(regression),
                accepted=accepted,
                rollback_reason=reason,
                optimizer_steps=int(train_result["optimizer_steps"]),
                base_model_unchanged=stage_base_guard == parameter_guard_snapshot(cortex.model),
                base_gradients_seen=int(train_result["base_gradients_seen"]),
            ).to_dict()
        )
        if not accepted:
            break

    if not accepted_islands:
        raise RuntimeError("no recurrent island stage passed the dev gate")

    cortex.config.islands = accepted_islands
    base_after = parameter_guard_snapshot(cortex.model)
    digest_after = module_parameter_digest(cortex.replacement.module)
    replaced = sum(island.width for island in accepted_islands)
    return IslandTrainingReceipt(
        schema="NOLANE-L11-RECURRENT-ISLAND-TRAINING-V1",
        authority="TRAINED_ISLAND_CANDIDATE_ONLY",
        plan_sha256=str(plan["plan_sha256"]),
        stages_attempted=len(stage_receipts),
        stages_accepted=sum(1 for row in stage_receipts if row["accepted"]),
        final_islands=[island.to_dict() for island in accepted_islands],
        replaced_layers=replaced,
        replaced_fraction=replaced / total_layers,
        replacement_parameters=cortex.trainable_parameter_count(),
        replacement_digest_before=digest_before,
        replacement_digest_after=digest_after,
        replacement_changed=digest_before != digest_after,
        base_model_unchanged=base_before == base_after,
        base_gradients_seen=base_gradients_seen,
        replacement_gradients_seen=replacement_gradients_seen,
        stage_receipts=stage_receipts,
        config=asdict(config),
    )
