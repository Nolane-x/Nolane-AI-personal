from nolane_personal.court import build_baseline_cases, deterministic_predict, evaluate_predictor
from nolane_personal.events import LivingEvent
from nolane_personal.living_core import EventFeaturizer, LivingCoreConfig, analytical_parameter_count
from nolane_personal.state import LivingState


def test_default_living_core_is_only_14515_parameters():
    assert analytical_parameter_count(LivingCoreConfig()) == 14515


def test_event_features_are_deterministic_and_bounded():
    event = LivingEvent(kind="user_message", payload={"text": "xin chao hello"}, source="user")
    f = EventFeaturizer(48)
    a = f.encode(event)
    b = f.encode(event)
    assert a == b
    assert len(a) == 48
    assert abs(sum(x * x for x in a) - 1.0) < 1e-9


def test_replay_court_accepts_exact_deterministic_baseline():
    state = LivingState()
    event = LivingEvent(kind="user_message", payload={"text": "hello"}, source="user")
    cases = build_baseline_cases([(state, event, 60.0)])
    result = evaluate_predictor(cases, deterministic_predict, mae_gate=1e-12, max_error_gate=1e-12)
    assert result.decision == "COURT_PASS"
    assert result.mean_absolute_error == 0.0
