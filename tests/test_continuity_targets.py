from datetime import datetime, timedelta, timezone

from nolane_personal.continuity_targets import return_targets
from nolane_personal.events import LivingEvent


def _record(kind, at):
    return {"event": LivingEvent(kind=kind, source="test", at=at.isoformat())}


def test_return_targets_use_future_user_event_and_censor_tail():
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    records = [
        _record("clock_tick", base),
        _record("assistant_speech", base + timedelta(seconds=30)),
        _record("user_message", base + timedelta(seconds=50)),
        _record("clock_tick", base + timedelta(seconds=130)),
    ]
    targets = return_targets(records, horizon_seconds=60)
    first = records[0]["event"].event_id
    second = records[1]["event"].event_id
    tail = records[-1]["event"].event_id
    assert targets[first].observed and targets[first].returned_within_horizon == 1
    assert targets[second].observed and targets[second].returned_within_horizon == 1
    assert not targets[tail].observed
