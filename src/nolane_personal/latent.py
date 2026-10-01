from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .state import utc_now_iso
from .store import canonical_json, payload_digest


SCHEMA = "NOLANE-PERSISTENT-LATENT-V1"


class LatentBindingError(ValueError):
    pass


@dataclass(slots=True)
class PersistentLatent:
    identity_id: str
    checkpoint_sha256: str
    latent_dim: int
    values: list[float]
    source_state_version: int = 0
    sequence: int = 0
    protocol_sha256: str | None = None
    updated_at: str = ""
    schema: str = SCHEMA
    digest: str = ""

    def body(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("digest", None)
        return data

    def seal(self) -> "PersistentLatent":
        if self.schema != SCHEMA:
            raise ValueError("unsupported latent schema")
        if self.latent_dim <= 0 or len(self.values) != self.latent_dim:
            raise ValueError("latent dimension mismatch")
        if not all(math.isfinite(float(v)) for v in self.values):
            raise ValueError("latent contains non-finite value")
        self.values = [float(v) for v in self.values]
        self.source_state_version = max(0, int(self.source_state_version))
        self.sequence = max(0, int(self.sequence))
        self.updated_at = self.updated_at or utc_now_iso()
        self.digest = payload_digest(self.body())
        return self

    def verify(self) -> None:
        supplied = self.digest
        if not supplied:
            raise ValueError("latent digest missing")
        body = self.body()
        if payload_digest(body) != supplied:
            raise ValueError("latent digest mismatch")
        if self.schema != SCHEMA:
            raise ValueError("unsupported latent schema")
        if self.latent_dim <= 0 or len(self.values) != self.latent_dim:
            raise ValueError("latent dimension mismatch")
        if not all(math.isfinite(float(v)) for v in self.values):
            raise ValueError("latent contains non-finite value")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PersistentLatent":
        latent = cls(**payload)
        latent.verify()
        return latent


class LatentStore:
    """Atomic file-backed neural latent checkpoint, independent from token context."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> PersistentLatent | None:
        if not self.path.exists():
            return None
        import json

        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return PersistentLatent.from_dict(payload)

    def save(self, latent: PersistentLatent) -> PersistentLatent:
        latent.updated_at = utc_now_iso()
        latent.seal()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(canonical_json(asdict(latent)) + "\n", encoding="utf-8")
        temp.replace(self.path)
        return latent

    def load_bound(
        self,
        *,
        identity_id: str,
        checkpoint_sha256: str,
        latent_dim: int,
        protocol_sha256: str | None,
    ) -> PersistentLatent | None:
        latent = self.load()
        if latent is None:
            return None
        if latent.identity_id != identity_id:
            raise LatentBindingError("latent identity mismatch")
        if latent.checkpoint_sha256 != checkpoint_sha256:
            raise LatentBindingError("latent checkpoint mismatch")
        if latent.latent_dim != int(latent_dim):
            raise LatentBindingError("latent dimension mismatch")
        if latent.protocol_sha256 != protocol_sha256:
            raise LatentBindingError("latent protocol mismatch")
        return latent

    def initialize(
        self,
        *,
        identity_id: str,
        checkpoint_sha256: str,
        latent_dim: int,
        protocol_sha256: str | None,
    ) -> PersistentLatent:
        return self.save(
            PersistentLatent(
                identity_id=identity_id,
                checkpoint_sha256=checkpoint_sha256,
                protocol_sha256=protocol_sha256,
                latent_dim=int(latent_dim),
                values=[0.0] * int(latent_dim),
            )
        )
