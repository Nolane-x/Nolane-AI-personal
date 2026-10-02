from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .store import canonical_json, payload_digest


PROFILE_SCHEMA = "NOLANE-PRODUCT-PERSONALIZATION-V1"


@dataclass(slots=True)
class ProductProfile:
    preferred_name: str = ""
    language: str = "auto"
    response_length: str = "balanced"
    conversation_style: str = "natural"
    initiative: str = "gentle"
    memory_enabled: bool = True
    personal_instruction: str = ""
    schema: str = PROFILE_SCHEMA

    def normalize(self) -> None:
        self.preferred_name = str(self.preferred_name).strip()[:80]
        if self.language not in {"auto", "vi", "en"}:
            self.language = "auto"
        if self.response_length not in {"compact", "balanced", "expansive"}:
            self.response_length = "balanced"
        if self.conversation_style not in {"natural", "warm", "direct", "playful"}:
            self.conversation_style = "natural"
        if self.initiative not in {"off", "gentle", "active"}:
            self.initiative = "gentle"
        self.memory_enabled = bool(self.memory_enabled)
        self.personal_instruction = str(self.personal_instruction).strip()[:1200]
        self.schema = PROFILE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        self.normalize()
        data = asdict(self)
        body = dict(data)
        body.pop("digest", None)
        data["digest"] = payload_digest(body)
        return data

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProductProfile":
        body = dict(payload)
        supplied = body.pop("digest", None)
        profile = cls(
            preferred_name=str(body.get("preferred_name", "")),
            language=str(body.get("language", "auto")),
            response_length=str(body.get("response_length", "balanced")),
            conversation_style=str(body.get("conversation_style", "natural")),
            initiative=str(body.get("initiative", "gentle")),
            memory_enabled=bool(body.get("memory_enabled", True)),
            personal_instruction=str(body.get("personal_instruction", "")),
            schema=str(body.get("schema", PROFILE_SCHEMA)),
        )
        profile.normalize()
        expected_body = asdict(profile)
        if supplied is not None and payload_digest(expected_body) != supplied:
            raise ValueError("product personalization digest mismatch")
        return profile


class ProductProfileStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> ProductProfile:
        if not self.path.exists():
            return ProductProfile()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("product personalization payload must be an object")
        return ProductProfile.from_dict(payload)

    def save(self, profile: ProductProfile) -> ProductProfile:
        profile.normalize()
        payload = profile.to_dict()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self.path.parent,
            prefix=self.path.name + ".",
            suffix=".tmp",
            delete=False,
        )
        tmp = Path(handle.name)
        try:
            with handle:
                handle.write(canonical_json(payload) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
        finally:
            if tmp.exists():
                tmp.unlink()
        return profile

    def update(self, patch: dict[str, Any]) -> ProductProfile:
        profile = self.load()
        allowed = {
            "preferred_name",
            "language",
            "response_length",
            "conversation_style",
            "initiative",
            "memory_enabled",
            "personal_instruction",
        }
        for key, value in patch.items():
            if key in allowed:
                setattr(profile, key, value)
        return self.save(profile)
