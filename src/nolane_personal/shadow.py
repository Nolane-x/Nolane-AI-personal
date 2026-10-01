from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .dynamics import seconds_between
from .latent import LatentBindingError, LatentStore
from .living_core import EventFeaturizer, LivingCoreConfig, TinyLivingCore, state_vector, time_features
from .replay_protocol import verify_replay_protocol
from .store import LivingStore, canonical_json, payload_digest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(torch, path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class ShadowLivingCoreRunner:
    """Advance a persistent neural latent in shadow mode without state authority."""

    def __init__(
        self,
        db_path: str | Path,
        checkpoint_path: str | Path,
        latent_path: str | Path,
        *,
        protocol_path: str | Path | None = None,
        receipt_path: str | Path | None = None,
        reset_latent: bool = False,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Install neural support with: pip install -e '.[neural]'") from exc

        self.torch = torch
        self.db_path = Path(db_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.latent_store = LatentStore(latent_path)
        self.receipt_path = Path(receipt_path) if receipt_path else Path(str(latent_path) + ".receipts.jsonl")

        checkpoint = _load_checkpoint(torch, self.checkpoint_path)
        config = LivingCoreConfig(**checkpoint["config"])
        self.core = TinyLivingCore(config)
        self.core.module.load_state_dict(checkpoint["model_state"])
        self.core.module.eval()
        self.featurizer = EventFeaturizer(config.event_dim)
        self.checkpoint_sha256 = _sha256_file(self.checkpoint_path)
        self.protocol_sha256 = checkpoint.get("protocol_sha256")

        store = LivingStore(self.db_path)
        try:
            state = store.load_state()
            if state is None:
                raise ValueError("living runtime is not initialized")
            if protocol_path is not None:
                protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
                verify_replay_protocol(store, protocol)
                if checkpoint.get("protocol_sha256") != protocol.get("protocol_sha256"):
                    raise LatentBindingError("checkpoint does not match replay protocol")
            self.identity_id = state.identity_id
        finally:
            store.close()

        if reset_latent and self.latent_store.path.exists():
            self.latent_store.path.unlink()

        latent = self.latent_store.load_bound(
            identity_id=self.identity_id,
            checkpoint_sha256=self.checkpoint_sha256,
            latent_dim=config.latent_dim,
            protocol_sha256=self.protocol_sha256,
        )
        self.latent = latent or self.latent_store.initialize(
            identity_id=self.identity_id,
            checkpoint_sha256=self.checkpoint_sha256,
            latent_dim=config.latent_dim,
            protocol_sha256=self.protocol_sha256,
        )

    def process_pending(self, limit: int = 10000) -> dict[str, Any]:
        torch = self.torch
        store = LivingStore(self.db_path)
        try:
            records = store.replay_records(
                limit,
                after_version=self.latent.source_state_version,
            )
        finally:
            store.close()

        processed = 0
        last_receipt: dict[str, Any] | None = None
        latent_tensor = torch.tensor([self.latent.values], dtype=torch.float32)

        with torch.no_grad():
            for record in records:
                before = record["before"]
                event = record["event"]
                dt = seconds_between(before.last_event_at or before.updated_at, event.at)
                observed = torch.tensor([state_vector(before)], dtype=torch.float32)
                event_features = torch.tensor([self.featurizer.encode(event)], dtype=torch.float32)
                dt_features = torch.tensor([time_features(dt)], dtype=torch.float32)
                latent_tensor, predicted_delta, action_logits, return_logit = self.core(
                    observed, event_features, dt_features, latent_tensor
                )

                self.latent.values = latent_tensor[0].cpu().tolist()
                self.latent.source_state_version = int(record["version"])
                self.latent.sequence += 1
                self.latent_store.save(self.latent)

                receipt = {
                    "schema": "NOLANE-LIVING-CORE-SHADOW-RECEIPT-V1",
                    "identity_id": self.identity_id,
                    "state_version": int(record["version"]),
                    "event_id": event.event_id,
                    "event_kind": event.kind,
                    "checkpoint_sha256": self.checkpoint_sha256,
                    "protocol_sha256": self.protocol_sha256,
                    "latent_digest": self.latent.digest,
                    "predicted_state_delta": predicted_delta[0].cpu().tolist(),
                    "action_logits": action_logits[0].cpu().tolist(),
                    "return_probability": float(torch.sigmoid(return_logit).item()),
                    "authority": "SHADOW_ONLY_NO_STATE_MUTATION",
                }
                receipt["receipt_sha256"] = payload_digest(receipt)
                self._append_receipt(receipt)
                last_receipt = receipt
                processed += 1

        return {
            "processed": processed,
            "cursor_version": self.latent.source_state_version,
            "latent_sequence": self.latent.sequence,
            "latent_digest": self.latent.digest,
            "last_receipt": last_receipt,
            "authority": "SHADOW_ONLY_NO_STATE_MUTATION",
        }

    def _append_receipt(self, receipt: dict[str, Any]) -> None:
        self.receipt_path.parent.mkdir(parents=True, exist_ok=True)
        with self.receipt_path.open("a", encoding="utf-8") as fh:
            fh.write(canonical_json(receipt) + "\n")
