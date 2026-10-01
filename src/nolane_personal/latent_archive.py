from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .dynamics import seconds_between
from .living_core import EventFeaturizer, LivingCoreConfig, TinyLivingCore, state_vector, time_features
from .replay_protocol import verify_replay_protocol
from .store import LivingStore, canonical_json, payload_digest


SCHEMA = "NOLANE-LATENT-ARCHIVE-V1"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_checkpoint(torch, path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(Path(path), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(Path(path), map_location="cpu")


def _split_map(protocol: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for split in ("train", "dev", "test"):
        for entry in protocol["splits"][split]:
            result[str(entry["event_id"])] = split
    return result


def build_latent_archive(
    db_path: str | Path,
    protocol_path: str | Path,
    living_core_checkpoint_path: str | Path,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install neural support with: pip install -e '.[neural]'") from exc

    protocol = load_json(protocol_path)
    store = LivingStore(db_path)
    try:
        verify_replay_protocol(store, protocol)
        records = store.replay_records()
    finally:
        store.close()

    checkpoint = _load_checkpoint(torch, living_core_checkpoint_path)
    checkpoint_protocol = checkpoint.get("protocol_sha256")
    if checkpoint_protocol != protocol.get("protocol_sha256"):
        raise ValueError("Living Core checkpoint does not match frozen replay protocol")

    config = LivingCoreConfig(**checkpoint["config"])
    core = TinyLivingCore(config)
    core.module.load_state_dict(checkpoint["model_state"])
    core.module.eval()
    featurizer = EventFeaturizer(config.event_dim)
    latent = core.initial_latent()
    split_by_event = _split_map(protocol)
    frozen_ids = set(split_by_event)
    frozen_records = [r for r in records if r["event"].event_id in frozen_ids]
    if len(frozen_records) != int(protocol["counts"]["total"]):
        raise ValueError("frozen replay record count mismatch")

    entries: list[dict[str, Any]] = []
    with torch.no_grad():
        for record in frozen_records:
            before = record["before"]
            event = record["event"]
            dt = seconds_between(before.last_event_at or before.updated_at, event.at)
            latent_before = latent[0].cpu().tolist()

            observed = torch.tensor([state_vector(before)], dtype=torch.float32)
            event_features = torch.tensor([featurizer.encode(event)], dtype=torch.float32)
            dt_features = torch.tensor([time_features(dt)], dtype=torch.float32)
            latent, _state_delta, _action_logits, _return_logit = core(
                observed,
                event_features,
                dt_features,
                latent,
            )
            entries.append(
                {
                    "version": int(record["version"]),
                    "event_id": event.event_id,
                    "event_kind": event.kind,
                    "split": split_by_event[event.event_id],
                    "latent_before": latent_before,
                    "latent_after": latent[0].cpu().tolist(),
                }
            )

    archive: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": "HISTORICAL_SHADOW_LATENT_ONLY",
        "identity_id": protocol["identity_id"],
        "protocol_sha256": protocol["protocol_sha256"],
        "living_core_checkpoint_sha256": file_sha256(living_core_checkpoint_path),
        "living_core_checkpoint_protocol_sha256": checkpoint_protocol,
        "latent_dim": int(config.latent_dim),
        "entry_count": len(entries),
        "entries": entries,
    }
    archive["archive_sha256"] = payload_digest(archive)
    return archive


def verify_latent_archive(
    archive: dict[str, Any],
    protocol: dict[str, Any],
    *,
    living_core_checkpoint_path: str | Path | None = None,
) -> None:
    if archive.get("schema") != SCHEMA:
        raise ValueError("unsupported latent archive schema")
    if archive.get("authority") != "HISTORICAL_SHADOW_LATENT_ONLY":
        raise ValueError("latent archive authority mismatch")
    supplied = archive.get("archive_sha256")
    body = dict(archive)
    body.pop("archive_sha256", None)
    if supplied != payload_digest(body):
        raise ValueError("latent archive digest mismatch")
    if archive.get("protocol_sha256") != protocol.get("protocol_sha256"):
        raise ValueError("latent archive protocol mismatch")
    if archive.get("identity_id") != protocol.get("identity_id"):
        raise ValueError("latent archive identity mismatch")
    if living_core_checkpoint_path is not None:
        if archive.get("living_core_checkpoint_sha256") != file_sha256(living_core_checkpoint_path):
            raise ValueError("latent archive Living Core checkpoint mismatch")

    split_by_event = _split_map(protocol)
    entries = archive.get("entries", [])
    if len(entries) != int(protocol["counts"]["total"]):
        raise ValueError("latent archive entry count mismatch")
    if int(archive.get("entry_count", -1)) != len(entries):
        raise ValueError("latent archive declared count mismatch")
    latent_dim = int(archive.get("latent_dim", -1))
    seen: set[str] = set()
    previous_version = 0
    for entry in entries:
        event_id = str(entry["event_id"])
        if event_id in seen:
            raise ValueError("duplicate latent archive event")
        seen.add(event_id)
        if event_id not in split_by_event or entry.get("split") != split_by_event[event_id]:
            raise ValueError("latent archive split mismatch")
        version = int(entry["version"])
        if version <= previous_version:
            raise ValueError("latent archive is not chronological")
        previous_version = version
        if len(entry.get("latent_before", [])) != latent_dim or len(entry.get("latent_after", [])) != latent_dim:
            raise ValueError("latent archive dimension mismatch")


def write_latent_archive(path: str | Path, archive: dict[str, Any]) -> None:
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite latent archive: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(archive) + "\n", encoding="utf-8")
