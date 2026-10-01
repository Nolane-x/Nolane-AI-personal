from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable
from uuid import uuid4

from .state import utc_now_iso

_TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)


@dataclass(slots=True)
class MemoryRecord:
    text: str
    kind: str = "episodic"
    salience: float = 0.5
    confidence: float = 1.0
    source_event_id: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    memory_id: str = field(default_factory=lambda: str(uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def lexical_tokens(text: str) -> set[str]:
    return {token.casefold() for token in _TOKEN_RE.findall(text) if len(token) > 1}


def score_memory(memory: MemoryRecord, query: str, now: datetime | None = None) -> float:
    now = now or datetime.now(timezone.utc)
    q = lexical_tokens(query)
    m = lexical_tokens(memory.text)
    overlap = (len(q & m) / max(1, len(q))) if q else 0.0
    try:
        created = datetime.fromisoformat(memory.created_at)
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (now - created).total_seconds() / 86400.0)
    except ValueError:
        age_days = 3650.0
    recency = math.exp(-age_days / 30.0)
    return 0.52 * overlap + 0.30 * float(memory.salience) + 0.18 * recency


def rank_memories(memories: Iterable[MemoryRecord], query: str, limit: int = 6) -> list[MemoryRecord]:
    scored = sorted(memories, key=lambda m: score_memory(m, query), reverse=True)
    return scored[: max(0, limit)]
