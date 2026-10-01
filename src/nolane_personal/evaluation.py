from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .continuity_targets import return_targets
from .dynamics import seconds_between
from .living_core import EventFeaturizer, LivingCoreConfig, TinyLivingCore, state_vector, time_features
from .promotion import PromotionEvidence, PromotionThresholds, decide_promotion
from .replay_protocol import records_for_split, verify_replay_protocol
from .store import LivingStore


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_checkpoint(torch, path: str | Path):
    try:
        return torch.load(Path(path), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(Path(path), map_location="cpu")


def evaluate_checkpoint(
    db_path: str | Path,
    protocol_path: str | Path,
    checkpoint_path: str | Path,
    *,
    thresholds: PromotionThresholds | None = None,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install neural support with: pip install -e '.[neural]'") from exc

    protocol = _load_json(protocol_path)
    store = LivingStore(db_path)
    try:
        verify_replay_protocol(store, protocol)
        train_records = records_for_split(store, protocol, "train")
        test_records = records_for_split(store, protocol, "test")
        all_records = store.replay_records()
    finally:
        store.close()

    checkpoint = _load_checkpoint(torch, checkpoint_path)
    config = LivingCoreConfig(**checkpoint["config"])
    core = TinyLivingCore(config)
    core.module.load_state_dict(checkpoint["model_state"])
    core.module.eval()

    checkpoint_protocol_match = checkpoint.get("protocol_sha256") == protocol.get("protocol_sha256")
    horizon = float(checkpoint.get("return_horizon_seconds", 3600.0))
    train_targets = return_targets(train_records, horizon_seconds=horizon)
    test_targets = return_targets(test_records, horizon_seconds=horizon)

    observed_train = [t.returned_within_horizon for t in train_targets.values() if t.observed]
    base_rate = sum(observed_train) / len(observed_train) if observed_train else 0.5

    frozen_ids = {
        entry["event_id"]
        for split in ("train", "dev", "test")
        for entry in protocol["splits"][split]
    }
    test_ids = {entry["event_id"] for entry in protocol["splits"]["test"]}
    frozen_records = [record for record in all_records if record["event"].event_id in frozen_ids]

    featurizer = EventFeaturizer(config.event_dim)
    latent = core.initial_latent()
    abs_errors: list[float] = []
    candidate_brier_terms: list[float] = []
    baseline_brier_terms: list[float] = []

    with torch.no_grad():
        for record in frozen_records:
            before = record["before"]
            after = record["after"]
            event = record["event"]
            dt = seconds_between(before.last_event_at or before.updated_at, event.at)
            before_v = torch.tensor([state_vector(before)], dtype=torch.float32)
            after_v = torch.tensor([state_vector(after)], dtype=torch.float32)
            event_v = torch.tensor([featurizer.encode(event)], dtype=torch.float32)
            dt_v = torch.tensor([time_features(dt)], dtype=torch.float32)

            latent, predicted_delta, _action_logits, return_logit = core(before_v, event_v, dt_v, latent)
            if event.event_id not in test_ids:
                continue

            target_delta = torch.clamp(after_v - before_v, -0.12, 0.12)
            abs_errors.extend(torch.abs(predicted_delta - target_delta).view(-1).cpu().tolist())

            target = test_targets[event.event_id]
            if target.observed:
                label = float(target.returned_within_horizon)
                probability = float(torch.sigmoid(return_logit).item())
                candidate_brier_terms.append((probability - label) ** 2)
                baseline_brier_terms.append((base_rate - label) ** 2)

    state_mae = sum(abs_errors) / len(abs_errors) if abs_errors else float("inf")
    state_max_error = max(abs_errors) if abs_errors else float("inf")
    candidate_brier = (
        sum(candidate_brier_terms) / len(candidate_brier_terms)
        if candidate_brier_terms
        else 1.0
    )
    baseline_brier = (
        sum(baseline_brier_terms) / len(baseline_brier_terms)
        if baseline_brier_terms
        else 1.0
    )

    evidence = PromotionEvidence(
        protocol_verified=True,
        checkpoint_protocol_match=checkpoint_protocol_match,
        state_test_cases=len(test_records),
        return_test_cases=len(candidate_brier_terms),
        state_mae=state_mae,
        state_max_error=state_max_error,
        baseline_brier=baseline_brier,
        candidate_brier=candidate_brier,
        parameter_count=core.parameter_count(),
    )
    decision = decide_promotion(evidence, thresholds)
    return {
        "schema": "NOLANE-LIVING-CORE-PROMOTION-EVAL-V1",
        "protocol_sha256": protocol["protocol_sha256"],
        "checkpoint_protocol_sha256": checkpoint.get("protocol_sha256"),
        "checkpoint_evidence_status": checkpoint.get("evidence_status"),
        "return_horizon_seconds": horizon,
        "train_return_base_rate": base_rate,
        "promotion": asdict(decision),
    }
