from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(slots=True)
class ReturnTarget:
    event_id: str
    observed: bool
    returned_within_horizon: int


def _dt(ts: str) -> datetime:
    value = datetime.fromisoformat(ts)
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def return_targets(
    records: list[dict[str, object]],
    *,
    horizon_seconds: float = 3600.0,
) -> dict[str, ReturnTarget]:
    """Create non-leaky future-return labels inside one chronological split.

    Tail examples whose full horizon is not observed are marked censored and are
    excluded from return-loss / Brier evaluation.
    """
    if not records:
        return {}
    horizon = max(1.0, float(horizon_seconds))
    end_time = _dt(records[-1]["event"].at)
    result: dict[str, ReturnTarget] = {}

    next_user_at: datetime | None = None
    for record in reversed(records):
        event = record["event"]
        at = _dt(event.at)
        horizon_observed = (end_time - at).total_seconds() >= horizon
        returned = 0
        if next_user_at is not None:
            gap = (next_user_at - at).total_seconds()
            returned = int(0.0 < gap <= horizon)
        result[event.event_id] = ReturnTarget(
            event_id=event.event_id,
            observed=horizon_observed,
            returned_within_horizon=returned,
        )
        if event.kind == "user_message":
            next_user_at = at
    return result
