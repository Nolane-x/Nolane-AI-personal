from __future__ import annotations

import math
from datetime import datetime, timezone

from .events import LivingEvent
from .state import LivingState, utc_now_iso

NEGATIVE_CUES = {
    "buồn", "mệt", "chán", "khóc", "tệ", "cô đơn", "stress", "áp lực",
    "sad", "tired", "upset", "awful", "cry", "lonely", "stressed",
}
POSITIVE_CUES = {
    "vui", "tuyệt", "haha", "hehe", "hihi", "đỉnh", "thích",
    "happy", "great", "awesome", "lol", "nice", "love",
}
ANGER_CUES = {"tức", "bực", "ghét", "điên", "angry", "mad", "furious", "annoyed"}


def _relax(value: float, target: float, dt_seconds: float, half_life_seconds: float) -> float:
    if dt_seconds <= 0:
        return value
    retention = 0.5 ** (dt_seconds / half_life_seconds)
    return target + (value - target) * retention


def advance_time(state: LivingState, dt_seconds: float) -> LivingState:
    """Deterministic slow dynamics between events."""
    dt = max(0.0, min(float(dt_seconds), 7 * 86400.0))
    a = state.affect
    a.valence = _relax(a.valence, 0.0, dt, 6 * 3600.0)
    a.arousal = _relax(a.arousal, 0.35, dt, 2 * 3600.0)
    a.energy = _relax(a.energy, 0.62, dt, 8 * 3600.0)
    a.playfulness = _relax(a.playfulness, 0.45, dt, 10 * 3600.0)
    a.irritation = _relax(a.irritation, 0.0, dt, 45 * 60.0)
    a.concern = _relax(a.concern, 0.0, dt, 4 * 3600.0)
    rise = 1.0 - math.exp(-dt / (6 * 3600.0))
    a.social_drive = min(1.0, a.social_drive + 0.22 * rise * (1.0 - a.social_drive))
    state.tick += 1
    state.updated_at = utc_now_iso()
    state.normalize()
    return state


def _contains_any(text: str, cues: set[str]) -> bool:
    folded = text.casefold()
    return any(cue in folded for cue in cues)


def apply_event(state: LivingState, event: LivingEvent) -> LivingState:
    state.last_event_at = event.at
    state.updated_at = event.at

    if event.kind == "user_message":
        text = str(event.payload.get("text", ""))
        state.last_user_event_at = event.at
        r = state.relationship
        r.interaction_count += 1
        r.familiarity += 0.012 * (1.0 - r.familiarity)
        r.closeness += 0.004 * (1.0 - r.closeness)
        state.affect.social_drive *= 0.45
        state.affect.energy += 0.02
        state.working.curiosity += 0.04
        if text.strip():
            state.working.recent_topics.append(text.strip()[:300])

        if _contains_any(text, NEGATIVE_CUES):
            state.affect.concern += 0.22
            state.affect.playfulness -= 0.10
            state.affect.valence -= 0.06
        if _contains_any(text, POSITIVE_CUES):
            state.affect.playfulness += 0.11
            state.affect.valence += 0.08
        if _contains_any(text, ANGER_CUES):
            state.affect.concern += 0.10
            state.affect.arousal += 0.12

    elif event.kind == "assistant_speech":
        state.last_ai_speech_at = event.at
        state.affect.social_drive *= 0.30
        state.working.active_intent = event.payload.get("intent") or state.working.active_intent

    elif event.kind == "social_signal":
        deltas = event.payload.get("affect_delta", {})
        for name in ("valence", "arousal", "energy", "playfulness", "irritation", "concern", "social_drive"):
            if name in deltas:
                current = getattr(state.affect, name)
                delta = max(-0.20, min(0.20, float(deltas[name])))
                setattr(state.affect, name, current + delta)

    state.normalize()
    return state


def seconds_between(earlier: str | None, later: str | None) -> float:
    if not earlier or not later:
        return 0.0
    a = datetime.fromisoformat(earlier)
    b = datetime.fromisoformat(later)
    if a.tzinfo is None:
        a = a.replace(tzinfo=timezone.utc)
    if b.tzinfo is None:
        b = b.replace(tzinfo=timezone.utc)
    return max(0.0, (b - a).total_seconds())
