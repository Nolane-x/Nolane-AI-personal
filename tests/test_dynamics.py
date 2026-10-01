from nolane_personal.dynamics import advance_time, apply_event
from nolane_personal.events import LivingEvent
from nolane_personal.state import LivingState


def test_negative_cue_changes_bounded_control_state():
    state = LivingState()
    event = LivingEvent(kind="user_message", payload={"text": "Hôm nay tôi hơi buồn và mệt"}, source="user")
    state = apply_event(state, event)
    assert state.affect.concern > 0.0
    assert 0.0 <= state.affect.concern <= 1.0


def test_time_decay_reduces_irritation_and_advances_tick():
    state = LivingState()
    state.affect.irritation = 0.9
    before = state.tick
    advance_time(state, 3600)
    assert state.affect.irritation < 0.9
    assert state.tick == before + 1
