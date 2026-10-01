from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .state import utc_now_iso
from .store import canonical_json, payload_digest


SCHEMA = "NOLANE-HYBRID-RECURRENT-STATE-V1"


@dataclass(slots=True)
class RecurrentStateBundle:
    identity_id: str
    base_model_fingerprint: str
    mixer_digest: str
    latent_digest: str
    recurrent_dim: int
    layer_states: dict[str, list[float]]
    sequence: int = 0
    updated_at: str = ""
    schema: str = SCHEMA
    digest: str = ""

    def body(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("digest", None)
        return payload

    def seal(self) -> "RecurrentStateBundle":
        if self.schema != SCHEMA:
            raise ValueError("unsupported recurrent-state schema")
        if self.recurrent_dim <= 0:
            raise ValueError("recurrent_dim must be positive")
        normalized: dict[str, list[float]] = {}
        for key, values in self.layer_states.items():
            if len(values) != self.recurrent_dim:
                raise ValueError("recurrent layer-state dimension mismatch")
            row = [float(x) for x in values]
            if not all(math.isfinite(x) for x in row):
                raise ValueError("non-finite recurrent state")
            normalized[str(int(key))] = row
        self.layer_states = normalized
        self.sequence = max(0, int(self.sequence))
        self.updated_at = self.updated_at or utc_now_iso()
        self.digest = payload_digest(self.body())
        return self

    def verify(self) -> None:
        supplied = self.digest
        if not supplied or payload_digest(self.body()) != supplied:
            raise ValueError("recurrent-state digest mismatch")
        if self.schema != SCHEMA:
            raise ValueError("unsupported recurrent-state schema")
        if self.recurrent_dim <= 0:
            raise ValueError("recurrent_dim must be positive")
        for values in self.layer_states.values():
            if len(values) != self.recurrent_dim:
                raise ValueError("recurrent layer-state dimension mismatch")
            if not all(math.isfinite(float(x)) for x in values):
                raise ValueError("non-finite recurrent state")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RecurrentStateBundle":
        value = cls(**payload)
        value.verify()
        return value


class RecurrentStateStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> RecurrentStateBundle | None:
        if not self.path.exists():
            return None
        return RecurrentStateBundle.from_dict(
            json.loads(self.path.read_text(encoding="utf-8"))
        )

    def save(self, bundle: RecurrentStateBundle) -> RecurrentStateBundle:
        bundle.updated_at = utc_now_iso()
        bundle.seal()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(canonical_json(asdict(bundle)) + "\n", encoding="utf-8")
        temp.replace(self.path)
        return bundle

    def load_bound(
        self,
        *,
        identity_id: str,
        base_model_fingerprint: str,
        mixer_digest: str,
        latent_digest: str,
        recurrent_dim: int,
    ) -> RecurrentStateBundle | None:
        value = self.load()
        if value is None:
            return None
        expected = {
            "identity_id": identity_id,
            "base_model_fingerprint": base_model_fingerprint,
            "mixer_digest": mixer_digest,
            "latent_digest": latent_digest,
            "recurrent_dim": int(recurrent_dim),
        }
        actual = {
            "identity_id": value.identity_id,
            "base_model_fingerprint": value.base_model_fingerprint,
            "mixer_digest": value.mixer_digest,
            "latent_digest": value.latent_digest,
            "recurrent_dim": value.recurrent_dim,
        }
        for key in expected:
            if actual[key] != expected[key]:
                raise ValueError(f"recurrent-state binding mismatch: {key}")
        return value
