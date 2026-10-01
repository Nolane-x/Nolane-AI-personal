from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4

from .state import utc_now_iso


@dataclass(slots=True)
class LivingEvent:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    source: str = "runtime"
    salience: float = 0.5
    at: str = field(default_factory=utc_now_iso)
    event_id: str = field(default_factory=lambda: str(uuid4()))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
