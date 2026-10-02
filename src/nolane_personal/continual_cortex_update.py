from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .continual_learning_court import (
    ContinualLearningPolicy,
    assess_continual_learning,
    verify_continual_learning_digest,
)
from .store import payload_digest
from .surgery import module_parameter_digest


@dataclass(slots=True)
class ContinualCortexUpdateConfig:
    epochs: int = 2
    learning_rate: float = 2e-4
    adaptation_task_weight: float = 1.0
    retention_task_weight: float = 0.5
    retention_distill_weight: float = 1.0
    cortex_anchor_weight: float = 0.01
    distill_temperature: float = 2.0
    max_grad_norm: float = 1.0

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be >=1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.adaptation_task_weight <= 0:
            raise ValueError("adaptation_task_weight must be positive")
        if self.retention_task_weight < 0:
            raise ValueError("retention_task_weight must be non-negative")
        if self.retention_distill_weight < 0:
            raise ValueError("retention_distill_weight must be non-negative")
        if self.cortex_anchor_weight < 0:
            raise ValueError("cortex_anchor_weight must be non-negative")
        if self.distill_temperature <= 0:
            raise ValueError("distill_temperature must be positive")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")


@dataclass(slots=True)
class ContinualCortexUpdateReceipt:
    schema: str
    authority: str
    adaptation_train_examples: int
    retention_rehearsal_examples: int
    retention_eval_examples: int
    adaptation_eval_examples: int
    epochs: int
    optimizer_steps: int
    adaptation_nll_before: float
    adaptation_nll_after: float
    retention_nll_before: float
    retention_nll_after: float
    candidate_model_state_sha256_before: str
    candidate_model_state_sha256_after: str
    candidate_boundary_digest_before: str
    candidate_boundary_digest_after: str
    candidate_boundary_unchanged: bool
    reference_boundary_digest_before: str
    reference_boundary_digest_after: str
    reference_boundary_unchanged: bool
    candidate_cortex_digest_before: str
    candidate_cortex_digest_after: str
    candidate_cortex_changed: bool
    reference_cortex_digest_before: str
    reference_cortex_digest_after: str
    reference_cortex_unchanged: bool
    candidate_boundary_gradients_seen: int
    candidate_cortex_gradients_seen: int
    config: dict[str, Any]
    continual_learning: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["update_sha256"] = payload_digest(data)
        return data


def verify_continual_cortex_update_receipt(
    receipt: dict[str, Any],
) -> dict[str, Any]:
    if receipt.get("schema") != "NOLANE-L38-RECURRENT-CORTEX-CONTINUAL-UPDATE-V1":
        raise ValueError("unsupported recurrent-cortex update schema")
    if receipt.get("authority") != "RECURRENT_CORTEX_UPDATE_CANDIDATE_ONLY":
        raise ValueError("recurrent-cortex update authority mismatch")
    supplied = receipt.get("update_sha256")
    body = dict(receipt)
    body.pop("update_sha256", None)
    if payload_digest(body) != supplied:
        raise ValueError("recurrent-cortex update receipt digest mismatch")
    if not receipt.get("candidate_cortex_changed"):
        raise ValueError("recurrent-cortex candidate did not change")
    if not receipt.get("candidate_boundary_unchanged"):
        raise ValueError("recurrent-cortex update changed language boundary")
    if not receipt.get("reference_boundary_unchanged"):
        raise ValueError("recurrent-cortex update changed reference boundary")
    if not receipt.get("reference_cortex_unchanged"):
        raise ValueError("recurrent-cortex update changed reference cortex")
    if int(receipt.get("candidate_boundary_gradients_seen", -1)) != 0:
        raise ValueError("recurrent-cortex update leaked gradients to boundary")
    if int(receipt.get("candidate_cortex_gradients_seen", 0)) <= 0:
        raise ValueError("recurrent-cortex update has no cortex gradients")
    continual = receipt.get("continual_learning")
    if not isinstance(continual, dict):
        raise ValueError("recurrent-cortex L30 receipt missing")
    verify_continual_learning_digest(continual)
    if (
        continual.get("pre_update_checkpoint_sha256")
        != receipt.get("candidate_model_state_sha256_before")
    ):
        raise ValueError("recurrent-cortex L30 pre-update state mismatch")
    if (
        continual.get("post_update_checkpoint_sha256")
        != receipt.get("candidate_model_state_sha256_after")
    ):
        raise ValueError("recurrent-cortex L30 post-update state mismatch")
    return receipt


def _validate_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a sha256 hex string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a sha256 hex string") from exc
    return value


def verify_continual_cortex_run_receipt(
    run: dict[str, Any],
) -> dict[str, Any]:
    if run.get("schema") != "NOLANE-L38-RECURRENT-CORTEX-UPDATE-RUN-V1":
        raise ValueError("unsupported recurrent-cortex run schema")
    if (
        run.get("authority")
        != "RECURRENT_CORTEX_UPDATE_EVIDENCE_ONLY_UNPROMOTED"
    ):
        raise ValueError("recurrent-cortex run authority mismatch")

    parent = _validate_sha256(
        run.get("parent_factorized_checkpoint_sha256"),
        name="parent_factorized_checkpoint_sha256",
    )
    training = run.get("training")
    lineage = run.get("lineage")
    artifact = run.get("artifact")
    if not isinstance(training, dict):
        raise ValueError("recurrent-cortex training receipt missing")
    if not isinstance(lineage, dict):
        raise ValueError("recurrent-cortex lineage missing")
    if not isinstance(artifact, dict):
        raise ValueError("recurrent-cortex artifact manifest missing")

    verify_continual_cortex_update_receipt(training)

    if lineage.get("schema") != "NOLANE-L38-CORTEX-UPDATE-LINEAGE-V1":
        raise ValueError("unsupported recurrent-cortex lineage schema")
    supplied_lineage = lineage.get("lineage_sha256")
    lineage_body = dict(lineage)
    lineage_body.pop("lineage_sha256", None)
    if payload_digest(lineage_body) != supplied_lineage:
        raise ValueError("recurrent-cortex lineage digest mismatch")
    for key in (
        "parent_factorized_checkpoint_sha256",
        "retention_dataset_sha256",
        "retention_protocol_sha256",
        "retention_quality_court_sha256",
        "adaptation_dataset_sha256",
        "adaptation_protocol_sha256",
        "adaptation_quality_court_sha256",
        "lineage_sha256",
    ):
        _validate_sha256(lineage.get(key), name=key)
    if lineage["parent_factorized_checkpoint_sha256"] != parent:
        raise ValueError("recurrent-cortex lineage parent mismatch")

    if artifact.get("schema") != "NOLANE-L17-FACTORIZED-LANGUAGE-BOUNDARY-V1":
        raise ValueError("recurrent-cortex output artifact schema mismatch")
    if artifact.get("authority") != "FACTORIZED_CANDIDATE_UNPROMOTED":
        raise ValueError("recurrent-cortex output artifact authority mismatch")
    _validate_sha256(
        artifact.get("checkpoint_sha256"),
        name="artifact checkpoint_sha256",
    )
    boundary_after = _validate_sha256(
        training.get("candidate_boundary_digest_after"),
        name="candidate_boundary_digest_after",
    )
    cortex_after = _validate_sha256(
        training.get("candidate_cortex_digest_after"),
        name="candidate_cortex_digest_after",
    )
    if artifact.get("boundary_state_digest") != boundary_after:
        raise ValueError("recurrent-cortex saved boundary digest mismatch")
    if artifact.get("cortex_state_digest") != cortex_after:
        raise ValueError("recurrent-cortex saved cortex digest mismatch")
    if artifact.get("dataset_fingerprint") != lineage["lineage_sha256"]:
        raise ValueError("recurrent-cortex artifact lineage fingerprint mismatch")
    if artifact.get("training_receipt") != training:
        raise ValueError("recurrent-cortex artifact training receipt mismatch")

    artifact_state = payload_digest(
        {
            "boundary_state_digest": artifact["boundary_state_digest"],
            "cortex_state_digest": artifact["cortex_state_digest"],
        }
    )
    if artifact_state != training["candidate_model_state_sha256_after"]:
        raise ValueError("recurrent-cortex artifact model-state digest mismatch")
    return run


def _model_state_digest(model) -> str:
    return payload_digest(
        {
            "boundary_state_digest": module_parameter_digest(
                model.boundary.module
            ),
            "cortex_state_digest": module_parameter_digest(
                model.cortex.module
            ),
        }
    )


def _per_example_nll(model, examples) -> list[float]:
    torch = model.cortex.torch
    device = next(model.cortex.module.parameters()).device
    values: list[float] = []
    model.eval()
    with torch.no_grad():
        for ids, labels, latent, weight in examples:
            x = torch.tensor([ids], dtype=torch.long, device=device)
            y = torch.tensor([labels], dtype=torch.long, device=device)
            if latent is not None:
                model.set_latent(latent)
            loss = model.forward(
                input_ids=x,
                labels=y,
                state=None,
            ).loss
            values.append(float(loss.detach().cpu()) * float(weight))
    return values


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _kl(student_logits, teacher_logits, temperature: float):
    torch = __import__("torch")
    t = float(temperature)
    student = torch.nn.functional.log_softmax(
        student_logits.float() / t,
        dim=-1,
    )
    teacher = torch.nn.functional.softmax(
        teacher_logits.float() / t,
        dim=-1,
    )
    return torch.nn.functional.kl_div(
        student,
        teacher,
        reduction="batchmean",
    ) * (t * t)


def _anchor_loss(candidate, reference):
    torch = candidate.cortex.torch
    total = None
    count = 0
    reference_by_name = dict(reference.cortex.module.named_parameters())
    for name, parameter in candidate.cortex.module.named_parameters():
        if name not in reference_by_name:
            raise RuntimeError("candidate/reference cortex parameter mismatch")
        target = reference_by_name[name].detach().to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        term = (parameter - target).float().pow(2).mean()
        total = term if total is None else total + term
        count += 1
    if total is None or count == 0:
        raise RuntimeError("candidate cortex has no trainable parameters")
    return total / float(count)


def train_continual_cortex_update(
    candidate,
    reference,
    adaptation_train_examples,
    retention_rehearsal_examples,
    *,
    retention_eval_examples,
    adaptation_eval_examples,
    adaptation_group_sha256: list[str],
    retention_group_sha256: list[str],
    config: ContinualCortexUpdateConfig | None = None,
    court_policy: ContinualLearningPolicy | None = None,
) -> ContinualCortexUpdateReceipt:
    """
    Train only the candidate deep recurrent cortex.

    The factorized language boundary is frozen exactly. Training sees only
    adaptation-train and retention-rehearsal rows. L30 is computed exclusively
    from disjoint old/new held-out rows after the update.
    """
    config = config or ContinualCortexUpdateConfig()
    config.validate()
    court_policy = court_policy or ContinualLearningPolicy()

    if candidate is reference:
        raise ValueError("candidate and reference must be distinct model objects")
    if not adaptation_train_examples:
        raise ValueError("adaptation training examples are empty")
    if not retention_rehearsal_examples:
        raise ValueError("retention rehearsal examples are empty")
    if not retention_eval_examples:
        raise ValueError("retention held-out examples are empty")
    if not adaptation_eval_examples:
        raise ValueError("adaptation held-out examples are empty")
    if len(adaptation_group_sha256) != len(adaptation_eval_examples):
        raise ValueError("adaptation source-group lineage length mismatch")
    if len(retention_group_sha256) != len(retention_eval_examples):
        raise ValueError("retention source-group lineage length mismatch")

    torch = candidate.cortex.torch
    device = next(candidate.cortex.module.parameters()).device

    candidate_boundary_before = module_parameter_digest(
        candidate.boundary.module
    )
    reference_boundary_before = module_parameter_digest(
        reference.boundary.module
    )
    candidate_cortex_before = module_parameter_digest(
        candidate.cortex.module
    )
    reference_cortex_before = module_parameter_digest(
        reference.cortex.module
    )
    candidate_model_before = _model_state_digest(candidate)
    reference_model_before = _model_state_digest(reference)

    if candidate_boundary_before != reference_boundary_before:
        raise ValueError(
            "candidate must start from the exact reference boundary state"
        )
    if candidate_cortex_before != reference_cortex_before:
        raise ValueError(
            "candidate must start from the exact reference cortex state"
        )
    if candidate_model_before != reference_model_before:
        raise ValueError(
            "candidate must start from the exact reference model state"
        )

    for parameter in reference.boundary.module.parameters():
        parameter.requires_grad_(False)
    for parameter in reference.cortex.module.parameters():
        parameter.requires_grad_(False)
    for parameter in candidate.boundary.module.parameters():
        parameter.requires_grad_(False)
    for parameter in candidate.cortex.module.parameters():
        parameter.requires_grad_(True)

    retention_before_values = _per_example_nll(
        reference,
        retention_eval_examples,
    )
    adaptation_before_values = _per_example_nll(
        reference,
        adaptation_eval_examples,
    )

    params = [
        parameter
        for parameter in candidate.cortex.module.parameters()
        if parameter.requires_grad
    ]
    if not params:
        raise RuntimeError("candidate cortex has no trainable parameters")
    optimizer = torch.optim.AdamW(
        params,
        lr=config.learning_rate,
        weight_decay=0.0,
    )

    steps = 0
    boundary_gradients_seen = 0
    cortex_gradients_seen = 0
    reference.eval()
    candidate.eval()

    for _ in range(config.epochs):
        for index, adaptation in enumerate(adaptation_train_examples):
            a_ids, a_labels, a_latent, a_weight = adaptation
            retention = retention_rehearsal_examples[
                index % len(retention_rehearsal_examples)
            ]
            r_ids, r_labels, r_latent, r_weight = retention

            ax = torch.tensor([a_ids], dtype=torch.long, device=device)
            ay = torch.tensor([a_labels], dtype=torch.long, device=device)
            rx = torch.tensor([r_ids], dtype=torch.long, device=device)
            ry = torch.tensor([r_labels], dtype=torch.long, device=device)

            if a_latent is not None:
                candidate.set_latent(a_latent)
            adaptation_out = candidate.forward(
                input_ids=ax,
                labels=ay,
                state=None,
            )

            if r_latent is not None:
                candidate.set_latent(r_latent)
                reference.set_latent(r_latent)
            with torch.no_grad():
                reference_logits = reference.forward(
                    input_ids=rx,
                    state=None,
                ).logits.detach()
            retention_out = candidate.forward(
                input_ids=rx,
                labels=ry,
                state=None,
            )
            retention_kl = _kl(
                retention_out.logits,
                reference_logits,
                config.distill_temperature,
            )
            anchor = _anchor_loss(candidate, reference)

            loss = (
                config.adaptation_task_weight
                * adaptation_out.loss
                * float(a_weight)
                + config.retention_task_weight
                * retention_out.loss
                * float(r_weight)
                + config.retention_distill_weight
                * retention_kl
                + config.cortex_anchor_weight
                * anchor
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            boundary_gradients_seen += sum(
                1
                for parameter in candidate.boundary.module.parameters()
                if parameter.grad is not None
            )
            if boundary_gradients_seen:
                raise RuntimeError(
                    "candidate boundary received gradients during cortex update"
                )
            finite_cortex = [
                parameter
                for parameter in params
                if parameter.grad is not None
                and torch.isfinite(parameter.grad).all()
            ]
            cortex_gradients_seen += len(finite_cortex)
            if not finite_cortex:
                raise RuntimeError(
                    "candidate cortex received no finite gradients"
                )

            torch.nn.utils.clip_grad_norm_(
                params,
                config.max_grad_norm,
            )
            optimizer.step()
            steps += 1

    retention_after_values = _per_example_nll(
        candidate,
        retention_eval_examples,
    )
    adaptation_after_values = _per_example_nll(
        candidate,
        adaptation_eval_examples,
    )

    candidate_boundary_after = module_parameter_digest(
        candidate.boundary.module
    )
    reference_boundary_after = module_parameter_digest(
        reference.boundary.module
    )
    candidate_cortex_after = module_parameter_digest(
        candidate.cortex.module
    )
    reference_cortex_after = module_parameter_digest(
        reference.cortex.module
    )
    candidate_model_after = _model_state_digest(candidate)
    reference_model_after = _model_state_digest(reference)

    if reference_model_after != reference_model_before:
        raise RuntimeError("frozen reference model mutated during cortex update")
    if candidate_boundary_after != candidate_boundary_before:
        raise RuntimeError("candidate boundary mutated during cortex update")

    continual = assess_continual_learning(
        pre_update_checkpoint_sha256=candidate_model_before,
        post_update_checkpoint_sha256=candidate_model_after,
        retention_group_sha256=retention_group_sha256,
        retention_before_values=retention_before_values,
        retention_after_values=retention_after_values,
        adaptation_group_sha256=adaptation_group_sha256,
        adaptation_before_values=adaptation_before_values,
        adaptation_after_values=adaptation_after_values,
        policy=court_policy,
    )

    return ContinualCortexUpdateReceipt(
        schema="NOLANE-L38-RECURRENT-CORTEX-CONTINUAL-UPDATE-V1",
        authority="RECURRENT_CORTEX_UPDATE_CANDIDATE_ONLY",
        adaptation_train_examples=len(adaptation_train_examples),
        retention_rehearsal_examples=len(retention_rehearsal_examples),
        retention_eval_examples=len(retention_eval_examples),
        adaptation_eval_examples=len(adaptation_eval_examples),
        epochs=config.epochs,
        optimizer_steps=steps,
        adaptation_nll_before=_mean(adaptation_before_values),
        adaptation_nll_after=_mean(adaptation_after_values),
        retention_nll_before=_mean(retention_before_values),
        retention_nll_after=_mean(retention_after_values),
        candidate_model_state_sha256_before=candidate_model_before,
        candidate_model_state_sha256_after=candidate_model_after,
        candidate_boundary_digest_before=candidate_boundary_before,
        candidate_boundary_digest_after=candidate_boundary_after,
        candidate_boundary_unchanged=(
            candidate_boundary_before == candidate_boundary_after
        ),
        reference_boundary_digest_before=reference_boundary_before,
        reference_boundary_digest_after=reference_boundary_after,
        reference_boundary_unchanged=(
            reference_boundary_before == reference_boundary_after
        ),
        candidate_cortex_digest_before=candidate_cortex_before,
        candidate_cortex_digest_after=candidate_cortex_after,
        candidate_cortex_changed=(
            candidate_cortex_before != candidate_cortex_after
        ),
        reference_cortex_digest_before=reference_cortex_before,
        reference_cortex_digest_after=reference_cortex_after,
        reference_cortex_unchanged=(
            reference_cortex_before == reference_cortex_after
        ),
        candidate_boundary_gradients_seen=boundary_gradients_seen,
        candidate_cortex_gradients_seen=cortex_gradients_seen,
        config=asdict(config),
        continual_learning=continual,
    )
