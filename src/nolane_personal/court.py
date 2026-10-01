from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Iterable

from .dynamics import advance_time, apply_event
from .events import LivingEvent
from .living_core import state_vector
from .state import LivingState


@dataclass(slots=True)
class ReplayCase:
    before: LivingState
    event: LivingEvent
    dt_seconds: float
    target: LivingState


@dataclass(slots=True)
class CourtMetrics:
    cases: int
    mean_absolute_error: float
    max_absolute_error: float
    identity_violations: int
    bound_violations: int
    decision: str


def deterministic_predict(before: LivingState, event: LivingEvent, dt_seconds: float) -> LivingState:
    predicted = advance_time(deepcopy(before), dt_seconds)
    return apply_event(predicted, event)


def build_baseline_cases(
    states_and_events: Iterable[tuple[LivingState, LivingEvent, float]],
) -> list[ReplayCase]:
    cases: list[ReplayCase] = []
    for before, event, dt in states_and_events:
        target = deterministic_predict(before, event, dt)
        cases.append(ReplayCase(deepcopy(before), event, dt, deepcopy(target)))
    return cases


def evaluate_predictor(
    cases: Iterable[ReplayCase],
    predictor: Callable[[LivingState, LivingEvent, float], LivingState],
    *,
    mae_gate: float = 0.03,
    max_error_gate: float = 0.20,
) -> CourtMetrics:
    count = 0
    abs_sum = 0.0
    max_error = 0.0
    identity_violations = 0
    bound_violations = 0

    for case in cases:
        predicted = predictor(deepcopy(case.before), case.event, case.dt_seconds)
        predicted.normalize()
        pv = state_vector(predicted)
        tv = state_vector(case.target)
        errors = [abs(a - b) for a, b in zip(pv, tv)]
        abs_sum += sum(errors)
        max_error = max(max_error, max(errors, default=0.0))
        count += len(errors)
        if predicted.identity_id != case.before.identity_id:
            identity_violations += 1
        if any(not -1.000001 <= x <= 1.000001 for x in pv):
            bound_violations += 1

    mae = abs_sum / max(1, count)
    cases_count = count // 15
    decision = (
        "COURT_PASS"
        if cases_count > 0
        and mae <= mae_gate
        and max_error <= max_error_gate
        and identity_violations == 0
        and bound_violations == 0
        else "COURT_FAIL"
    )
    return CourtMetrics(
        cases=cases_count,
        mean_absolute_error=mae,
        max_absolute_error=max_error,
        identity_violations=identity_violations,
        bound_violations=bound_violations,
        decision=decision,
    )
