from datetime import datetime, timedelta, timezone

from nolane_personal.initiative import InitiativeEngine, InitiativePolicy
from nolane_personal.state import LivingState, OpenThread


def test_recent_user_activity_forces_silence():
    now = datetime.now(timezone.utc)
    state = LivingState(last_user_event_at=(now - timedelta(minutes=2)).isoformat())
    state.affect.social_drive = 1.0
    decision = InitiativeEngine().decide(state, now)
    assert not decision.speak
    assert "user_recently_active" in decision.reasons


def test_important_open_thread_can_trigger_followup():
    now = datetime.now(timezone.utc)
    state = LivingState(last_user_event_at=(now - timedelta(hours=2)).isoformat())
    state.relationship.closeness = 0.8
    state.affect.social_drive = 0.9
    state.working.curiosity = 0.9
    state.open_threads.append(OpenThread(thread_id="t", topic="math exam", importance=1.0))
    policy = InitiativePolicy(threshold=0.5)
    decision = InitiativeEngine(policy).decide(state, now)
    assert decision.speak
    assert decision.intent.startswith("follow_up:math exam")


def test_long_silence_without_thread_does_not_chase_user():
    now = datetime.now(timezone.utc)
    state = LivingState(last_user_event_at=(now - timedelta(days=3)).isoformat())
    state.affect.social_drive = 1.0
    decision = InitiativeEngine().decide(state, now)
    assert not decision.speak
    assert "long_silence_without_open_thread" in decision.reasons
