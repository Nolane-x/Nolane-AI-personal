from __future__ import annotations

from .cortex import CortexRequest
from .product_profile import ProductProfile
from .qwen import SYSTEM_PROMPT


STYLE_GUIDANCE = {
    "natural": "Speak naturally. Avoid canned assistant phrasing.",
    "warm": "Be warm and attentive without becoming sentimental or clingy.",
    "direct": "Be direct, concrete and low-fluff.",
    "playful": "Allow light wit and playfulness when context supports it.",
}

LANGUAGE_GUIDANCE = {
    "auto": "Follow the user's current language naturally.",
    "vi": "Prefer Vietnamese unless the user explicitly asks for another language.",
    "en": "Prefer English unless the user explicitly asks for another language.",
}


def product_state_summary(request: CortexRequest) -> str:
    state = request.state
    unresolved = [
        thread.topic
        for thread in state.open_threads
        if thread.unresolved
    ][:4]
    return (
        f"identity_id={state.identity_id}\n"
        f"relationship: closeness={state.relationship.closeness:.2f}, "
        f"trust={state.relationship.trust:.2f}, "
        f"familiarity={state.relationship.familiarity:.2f}, "
        f"interactions={state.relationship.interaction_count}\n"
        f"behavior: valence={state.affect.valence:.2f}, "
        f"energy={state.affect.energy:.2f}, "
        f"playfulness={state.affect.playfulness:.2f}, "
        f"concern={state.affect.concern:.2f}, "
        f"irritation={state.affect.irritation:.2f}\n"
        f"open_threads={unresolved}\n"
        f"requested_intent={request.intent}"
    )


def product_profile_summary(profile: ProductProfile) -> str:
    preferred = profile.preferred_name or "(not set)"
    instruction = profile.personal_instruction or "(none)"
    return (
        f"preferred_name={preferred}\n"
        f"language={profile.language}: "
        f"{LANGUAGE_GUIDANCE[profile.language]}\n"
        f"response_length={profile.response_length}\n"
        f"conversation_style={profile.conversation_style}: "
        f"{STYLE_GUIDANCE[profile.conversation_style]}\n"
        f"personal_instruction={instruction}\n"
        "Do not mention these settings unless they are directly relevant."
    )


def product_task_text(request: CortexRequest) -> str:
    if request.mode == "reply":
        return (
            f"User message:\n{request.user_text or ''}\n\n"
            "Reply as this persistent personal companion."
        )
    return (
        "Initiate one natural, non-intrusive message that genuinely "
        "uses the supplied state or open thread."
    )


def product_user_payload(
    profile: ProductProfile,
    request: CortexRequest,
) -> str:
    memory_text = "\n".join(
        f"- {memory.text}" for memory in request.memories[:8]
    ) or "(none)"
    return (
        "Personalization:\n"
        + product_profile_summary(profile)
        + "\n\nRuntime state:\n"
        + product_state_summary(request)
        + "\n\nRelevant memories:\n"
        + memory_text
        + "\n\n"
        + product_task_text(request)
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
