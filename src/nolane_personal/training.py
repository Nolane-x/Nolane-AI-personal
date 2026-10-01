from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .continuity_targets import return_targets
from .dynamics import seconds_between
from .living_core import EventFeaturizer, LivingCoreConfig, TinyLivingCore, state_vector, time_features
from .replay_protocol import records_for_split, verify_replay_protocol
from .store import LivingStore


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_protocol(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def train_from_store(
    db_path: str | Path,
    output_dir: str | Path,
    *,
    protocol_path: str | Path | None = None,
    epochs: int = 4,
    learning_rate: float = 1e-3,
    truncation: int = 16,
    return_horizon_seconds: float = 3600.0,
    config: LivingCoreConfig | None = None,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install neural support with: pip install -e '.[neural]'") from exc

    db_path = Path(db_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    store = LivingStore(db_path)
    try:
        all_records = store.replay_records()
        protocol = _load_protocol(protocol_path) if protocol_path is not None else None
        if protocol is not None:
            verify_replay_protocol(store, protocol)
            records = records_for_split(store, protocol, "train")
            protocol_sha = str(protocol["protocol_sha256"])
            evidence_status = "FROZEN_TRAIN_DEVELOPMENT"
        else:
            records = all_records
            protocol_sha = None
            evidence_status = "UNFROZEN_DEVELOPMENT"
    finally:
        store.close()

    if not records:
        raise ValueError("no replay transitions available for training")

    targets = return_targets(records, horizon_seconds=return_horizon_seconds)
    observed_return_targets = sum(1 for target in targets.values() if target.observed)

    core = TinyLivingCore(config)
    model = core.module
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=0.01)
    featurizer = EventFeaturizer(core.config.event_dim)

    epoch_losses: list[float] = []
    for _epoch in range(max(1, int(epochs))):
        latent = core.initial_latent()
        optimizer.zero_grad(set_to_none=True)
        chunk_losses = []
        scalar_losses: list[float] = []

        for index, record in enumerate(records):
            before = record["before"]
            after = record["after"]
            event = record["event"]
            dt = seconds_between(before.last_event_at or before.updated_at, event.at)

            before_v = torch.tensor([state_vector(before)], dtype=torch.float32)
            after_v = torch.tensor([state_vector(after)], dtype=torch.float32)
            event_v = torch.tensor([featurizer.encode(event)], dtype=torch.float32)
            dt_v = torch.tensor([time_features(dt)], dtype=torch.float32)

            latent, predicted_delta, _action_logits, return_logit = core(before_v, event_v, dt_v, latent)
            target_delta = torch.clamp(after_v - before_v, -0.12, 0.12)
            state_loss = torch.nn.functional.smooth_l1_loss(predicted_delta, target_delta)
            loss = state_loss

            return_target = targets[event.event_id]
            if return_target.observed:
                label = torch.tensor([[float(return_target.returned_within_horizon)]], dtype=torch.float32)
                return_loss = torch.nn.functional.binary_cross_entropy_with_logits(return_logit, label)
                loss = loss + 0.15 * return_loss

            chunk_losses.append(loss)
            scalar_losses.append(float(loss.detach().cpu()))

            boundary = (index + 1) % max(1, int(truncation)) == 0 or index + 1 == len(records)
            if boundary:
                torch.stack(chunk_losses).mean().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                latent = latent.detach()
                chunk_losses = []

        epoch_losses.append(sum(scalar_losses) / max(1, len(scalar_losses)))

    checkpoint_path = output_dir / "living-core.pt"
    torch.save(
        {
            "schema": "NOLANE-LIVING-CORE-CHECKPOINT-V2",
            "model_state": model.state_dict(),
            "config": asdict(core.config),
            "protocol_sha256": protocol_sha,
            "return_horizon_seconds": float(return_horizon_seconds),
            "evidence_status": evidence_status,
        },
        checkpoint_path,
    )
    manifest = {
        "schema": "NOLANE-LIVING-CORE-DEV-V2",
        "evidence_status": evidence_status,
        "parameter_count": core.parameter_count(),
        "replay_transitions": len(records),
        "observed_return_targets": observed_return_targets,
        "epochs": max(1, int(epochs)),
        "learning_rate": float(learning_rate),
        "truncation": max(1, int(truncation)),
        "return_horizon_seconds": float(return_horizon_seconds),
        "epoch_losses": epoch_losses,
        "protocol_sha256": protocol_sha,
        "source_db_sha256": _file_sha256(db_path),
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "config": asdict(core.config),
    }
    manifest_path = output_dir / "living-core-manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
