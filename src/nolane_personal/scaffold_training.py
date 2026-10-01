from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .state_space_training import (
    StateSpaceTrainingConfig,
    train_state_space_stages,
)


@dataclass(slots=True)
class ScaffoldTrainingReceipt:
    schema: str
    authority: str
    plan_sha256: str
    stages_attempted: int
    stages_accepted: int
    final_region: dict[str, int]
    head_layers: int
    tail_layers: int
    remaining_qwen_layers: int
    remaining_qwen_fraction: float
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


def train_scaffold_stages(
    model,
    train_examples,
    dev_examples,
    plan,
    *,
    config: StateSpaceTrainingConfig | None = None,
) -> ScaffoldTrainingReceipt:
    receipt = train_state_space_stages(
        model,
        train_examples,
        dev_examples,
        plan,
        config=config,
    )
    final = None
    for row in plan["stages"]:
        if {
            "start": int(row["region"]["start"]),
            "end": int(row["region"]["end"]),
        } == receipt.final_region:
            final = row
            break
    if final is None:
        raise RuntimeError("accepted scaffold region is not present in frozen plan")

    return ScaffoldTrainingReceipt(
        schema="NOLANE-L13-SHRINKING-SCAFFOLD-TRAINING-V1",
        authority="TRAINED_SCAFFOLD_CANDIDATE_ONLY",
        plan_sha256=receipt.plan_sha256,
        stages_attempted=receipt.stages_attempted,
        stages_accepted=receipt.stages_accepted,
        final_region=dict(receipt.final_region),
        head_layers=int(final["head_layers"]),
        tail_layers=int(final["tail_layers"]),
        remaining_qwen_layers=int(final["remaining_qwen_layers"]),
        remaining_qwen_fraction=float(final["remaining_qwen_fraction"]),
        replaced_layers=receipt.replaced_layers,
        replaced_fraction=receipt.replaced_fraction,
        cortex_parameters=receipt.cortex_parameters,
        cortex_digest_before=receipt.cortex_digest_before,
        cortex_digest_after=receipt.cortex_digest_after,
        cortex_changed=receipt.cortex_changed,
        base_model_unchanged=receipt.base_model_unchanged,
        base_gradients_seen=receipt.base_gradients_seen,
        cortex_gradients_seen=receipt.cortex_gradients_seen,
        stage_receipts=list(receipt.stage_receipts),
        config=dict(receipt.config),
    )
