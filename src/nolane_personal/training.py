from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .dynamics import seconds_between
from .living_core import EventFeaturizer, LivingCoreConfig, TinyLivingCore, state_vector, time_features
from .store import LivingStore


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_from_store(
    db_path: str | Path,
    output_dir: str | Path,
    *,
    epochs: int = 4,
    learning_rate: float = 1e-3,
    truncation: int = 16,
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
        records = store.replay_records()
    finally:
        store.close()
    if not records:
        raise ValueError("no replay transitions available")

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

            latent, predicted_delta, _action_logits, confidence = core(before_v, event_v, dt_v, latent)
            target_delta = torch.clamp(after_v - before_v, -0.12, 0.12)
            state_loss = torch.nn.functional.smooth_l1_loss(predicted_delta, target_delta)
            target_confidence = torch.exp(-8.0 * torch.mean(torch.abs(target_delta), dim=-1, keepdim=True))
            confidence_loss = torch.nn.functional.mse_loss(confidence, target_confidence)
            loss = state_loss + 0.05 * confidence_loss
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
            "model_state": model.state_dict(),
            "config": asdict(core.config),
            "evidence_status": "DEVELOPMENT_UNPROMOTED",
        },
        checkpoint_path,
    )
    manifest = {
        "schema": "NOLANE-LIVING-CORE-DEV-V1",
        "evidence_status": "DEVELOPMENT_UNPROMOTED",
        "parameter_count": core.parameter_count(),
        "replay_transitions": len(records),
        "epochs": max(1, int(epochs)),
        "learning_rate": float(learning_rate),
        "truncation": max(1, int(truncation)),
        "epoch_losses": epoch_losses,
        "source_db_sha256": _file_sha256(db_path),
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "config": asdict(core.config),
    }
    manifest_path = output_dir / "living-core-manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
