from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from .state import LivingState


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    value = datetime.fromisoformat(ts)
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


@dataclass(slots=True)
class InitiativePolicy:
    threshold: float = 0.66
    min_user_silence_seconds: float = 15 * 60.0
    speech_cooldown_seconds: float = 30 * 60.0
    hard_max_without_user_seconds: float = 24 * 3600.0
    max_thread_component: float = 0.34


@dataclass(slots=True)
class InitiativeDecision:
    speak: bool
    score: float
    intent: str = "remain_silent"
    reasons: list[str] = field(default_factory=list)


class InitiativeEngine:
    def __init__(self, policy: InitiativePolicy | None = None) -> None:
        self.policy = policy or InitiativePolicy()

    def decide(self, state: LivingState, now: datetime | None = None) -> InitiativeDecision:
        now = now or datetime.now(timezone.utc)
        last_user = _parse(state.last_user_event_at)
        last_speech = _parse(state.last_ai_speech_at)
        silence = (now - last_user).total_seconds() if last_user else float("inf")
        cooldown = (now - last_speech).total_seconds() if last_speech else float("inf")

        if silence < self.policy.min_user_silence_seconds:
            return InitiativeDecision(False, 0.0, reasons=["user_recently_active"])
        if cooldown < self.policy.speech_cooldown_seconds:
            return InitiativeDecision(False, 0.0, reasons=["speech_cooldown"])
        if silence > self.policy.hard_max_without_user_seconds and not any(t.unresolved for t in state.open_threads):
            return InitiativeDecision(False, 0.0, reasons=["long_silence_without_open_thread"])

        unresolved = [t for t in state.open_threads if t.unresolved]
        thread_component = min(
            self.policy.max_thread_component,
            sum(0.12 + 0.14 * t.importance for t in unresolved[:3]),
        )
        concern_component = 0.20 * state.affect.concern
        social_component = 0.26 * state.affect.social_drive
        curiosity_component = 0.16 * state.working.curiosity
        closeness_component = 0.08 * state.relationship.closeness
        score = max(0.0, min(1.0, thread_component + concern_component + social_component + curiosity_component + closeness_component))

        reasons: list[str] = []
        if thread_component:
            reasons.append("unresolved_thread")
        if state.affect.concern > 0.35:
            reasons.append("concern")
        if state.affect.social_drive > 0.45:
            reasons.append("social_drive")
        if state.working.curiosity > 0.55:
            reasons.append("curiosity")

        if score < self.policy.threshold:
            return InitiativeDecision(False, score, reasons=reasons or ["below_threshold"])
        if unresolved:
            top = max(unresolved, key=lambda t: t.importance)
            return InitiativeDecision(True, score, intent=f"follow_up:{top.topic}", reasons=reasons)
        if state.affect.concern > 0.45:
            return InitiativeDecision(True, score, intent="gentle_check_in", reasons=reasons)
        return InitiativeDecision(True, score, intent="casual_reconnect", reasons=reasons)
