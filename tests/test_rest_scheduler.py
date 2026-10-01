from datetime import datetime, timedelta, timezone

from nolane_personal.rest import RestPolicy, RestScheduler
from nolane_personal.state import LivingState


def test_rest_scheduler_requires_user_history_idle_window_and_cooldown():
    policy = RestPolicy(min_idle_seconds=1800, min_cycle_interval_seconds=2700)
    scheduler = RestScheduler(policy)
    state = LivingState()
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    due, reasons = scheduler.due(state, now)
    assert not due
    assert reasons == ["no_user_history"]

    state.last_user_event_at = (now - timedelta(minutes=20)).isoformat()
    due, reasons = scheduler.due(state, now)
    assert not due
    assert reasons == ["user_not_idle_enough"]

    state.last_user_event_at = (now - timedelta(minutes=40)).isoformat()
    due, reasons = scheduler.due(state, now)
    assert due
    assert "idle_window" in reasons

    state.rest.last_cycle_at = (now - timedelta(minutes=10)).isoformat()
    due, reasons = scheduler.due(state, now)
    assert not due
    assert reasons == ["rest_cycle_cooldown"]
