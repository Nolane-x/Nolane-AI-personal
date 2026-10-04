from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .cortex import CortexRequest
from .product_profile import ProductProfile
from .qwen import SYSTEM_PROMPT


PAYLOAD_SCHEMA = "NOLANE-V053-PRODUCT-PAYLOAD-INPUT-V1"
MAX_OPEN_THREADS = 4
MAX_MEMORIES = 8

STYLE_GUIDANCE = {
    "natural": "Speak naturally. Avoid canned assistant phrasing.",
    "warm": "Be warm and attentive without becoming sentimental or clingy.",
    "direct": "Be direct, concrete and low-fluff.",
    "playful": "Allow light wit and playfulness when context supports it.",
}

LANGUAGE_NAMES = {
    "en": "English",
    "vi": "Vietnamese",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "it": "Italian",
    "th": "Thai",
    "id": "Indonesian",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "tr": "Turkish",
    "pl": "Polish",
    "nl": "Dutch",
}

LANGUAGE_GUIDANCE = {
    "auto": "Follow the user's current language naturally.",
    **{
        code: f"Reply in {name} unless the user explicitly asks for another language."
        for code, name in LANGUAGE_NAMES.items()
    },
}


@dataclass(frozen=True, slots=True)
class ProductPayloadInput:
    profile: dict[str, Any]
    state: dict[str, Any]
    mode: str
    intent: str
    user_text: str | None
    memories: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PAYLOAD_SCHEMA,
            "profile": dict(self.profile),
            "state": dict(self.state),
            "mode": self.mode,
            "intent": self.intent,
            "user_text": self.user_text,
            "memories": list(self.memories),
        }


def _profile_input(profile: ProductProfile) -> dict[str, Any]:
    profile.normalize()
    return {
        "preferred_name": profile.preferred_name,
        "assistant_name": profile.assistant_name,
        "language": profile.language,
        "response_length": profile.response_length,
        "conversation_style": profile.conversation_style,
        "personal_instruction": profile.personal_instruction,
    }


def _state_input(request: CortexRequest) -> dict[str, Any]:
    state = request.state
    state.normalize()
    return {
        "identity_id": state.identity_id,
        "relationship": {
            "closeness": float(state.relationship.closeness),
            "trust": float(state.relationship.trust),
            "familiarity": float(state.relationship.familiarity),
            "interaction_count": int(
                state.relationship.interaction_count
            ),
        },
        "affect": {
            "valence": float(state.affect.valence),
            "energy": float(state.affect.energy),
            "playfulness": float(state.affect.playfulness),
            "concern": float(state.affect.concern),
            "irritation": float(state.affect.irritation),
        },
        "open_threads": [
            str(thread.topic)
            for thread in state.open_threads
            if thread.unresolved
        ][:MAX_OPEN_THREADS],
    }


def product_payload_input(
    profile: ProductProfile,
    request: CortexRequest,
) -> ProductPayloadInput:
    mode = str(request.mode)
    if mode not in {"reply", "initiative"}:
        raise ValueError(f"unsupported product payload mode: {mode}")
    user_text = (
        None if request.user_text is None else str(request.user_text)
    )
    if mode == "initiative" and user_text not in {None, ""}:
        raise ValueError(
            "initiative product payload must not contain user_text"
        )
    return ProductPayloadInput(
        profile=_profile_input(profile),
        state=_state_input(request),
        mode=mode,
        intent=str(request.intent),
        user_text=user_text,
        memories=[
            str(memory.text)
            for memory in request.memories[:MAX_MEMORIES]
        ],
    )


def _profile_summary_from_input(profile: dict[str, Any]) -> str:
    language = str(profile["language"])
    style = str(profile["conversation_style"])
    if language not in LANGUAGE_GUIDANCE:
        raise ValueError(f"unsupported product language: {language}")
    if style not in STYLE_GUIDANCE:
        raise ValueError(f"unsupported conversation style: {style}")
    response_length = str(profile["response_length"])
    if response_length not in {"compact", "balanced", "expansive"}:
        raise ValueError(
            f"unsupported response length: {response_length}"
        )
    preferred = str(profile["preferred_name"]) or "(not set)"
    assistant = str(profile["assistant_name"]) or "Nolane"
    instruction = str(profile["personal_instruction"]) or "(none)"
    return (
        f"preferred_name={preferred} (this is the USER'S name)\n"
        f"assistant_name={assistant} (this is YOUR name)\n"
        f"language={language}: {LANGUAGE_GUIDANCE[language]}\n"
        f"response_length={response_length}\n"
        f"conversation_style={style}: {STYLE_GUIDANCE[style]}\n"
        f"personal_instruction={instruction}\n"
        "Keep user identity and assistant identity separate. "
        "Do not mention these settings unless they are directly relevant."
    )


def _state_summary_from_input(
    state: dict[str, Any],
    *,
    intent: str,
) -> str:
    relationship = dict(state["relationship"])
    affect = dict(state["affect"])
    threads = [str(value) for value in state["open_threads"]]
    if len(threads) > MAX_OPEN_THREADS:
        raise ValueError("product payload exceeds open-thread limit")
    values = [
        relationship["closeness"],
        relationship["trust"],
        relationship["familiarity"],
        affect["valence"],
        affect["energy"],
        affect["playfulness"],
        affect["concern"],
        affect["irritation"],
    ]
    if any(not isinstance(value, (int, float)) for value in values):
        raise ValueError("product state numeric field is invalid")
    threads_json = json.dumps(
        threads,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        f"identity_id={state['identity_id']}\n"
        f"relationship: closeness={float(relationship['closeness']):.2f}, "
        f"trust={float(relationship['trust']):.2f}, "
        f"familiarity={float(relationship['familiarity']):.2f}, "
        f"interactions={int(relationship['interaction_count'])}\n"
        f"behavior: valence={float(affect['valence']):.2f}, "
        f"energy={float(affect['energy']):.2f}, "
        f"playfulness={float(affect['playfulness']):.2f}, "
        f"concern={float(affect['concern']):.2f}, "
        f"irritation={float(affect['irritation']):.2f}\n"
        f"open_threads={threads_json}\n"
        f"requested_intent={intent}"
    )


def _task_text_from_input(
    *,
    mode: str,
    user_text: str | None,
) -> str:
    if mode == "reply":
        return (
            f"User message:\n{user_text or ''}\n\n"
            "Reply as this persistent personal companion."
        )
    if mode == "initiative":
        if user_text not in {None, ""}:
            raise ValueError(
                "initiative product payload must not contain user_text"
            )
        return (
            "Initiate one natural, non-intrusive message that genuinely "
            "uses the supplied state or open thread."
        )
    raise ValueError(f"unsupported product payload mode: {mode}")


def render_product_payload_input(payload: ProductPayloadInput) -> str:
    if len(payload.memories) > MAX_MEMORIES:
        raise ValueError("product payload exceeds memory limit")
    memory_text = "\n".join(
        f"- {memory}" for memory in payload.memories
    ) or "(none)"
    return (
        "Personalization:\n"
        + _profile_summary_from_input(payload.profile)
        + "\n\nRuntime state:\n"
        + _state_summary_from_input(
            payload.state,
            intent=payload.intent,
        )
        + "\n\nRelevant memories:\n"
        + memory_text
        + "\n\n"
        + _task_text_from_input(
            mode=payload.mode,
            user_text=payload.user_text,
        )
    )


def product_profile_summary(profile: ProductProfile) -> str:
    return _profile_summary_from_input(_profile_input(profile))


def product_state_summary(request: CortexRequest) -> str:
    return _state_summary_from_input(
        _state_input(request),
        intent=str(request.intent),
    )


def product_task_text(request: CortexRequest) -> str:
    return _task_text_from_input(
        mode=str(request.mode),
        user_text=(
            None if request.user_text is None else str(request.user_text)
        ),
    )


def product_user_payload(
    profile: ProductProfile,
    request: CortexRequest,
) -> str:
    return render_product_payload_input(
        product_payload_input(profile, request)
    )


def build_product_messages(
    profile: ProductProfile,
    request: CortexRequest,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": product_user_payload(profile, request),
        },
    ]
