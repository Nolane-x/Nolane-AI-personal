from __future__ import annotations

from nolane_personal.cortex import CortexRequest
from nolane_personal.memory import MemoryRecord
from nolane_personal.product_cortex import FactorizedProductCortex
from nolane_personal.product_profile import ProductProfile
from nolane_personal.product_prompt_payload import (
    PAYLOAD_SCHEMA,
    build_product_messages,
    product_payload_input,
    product_profile_summary,
    product_state_summary,
    product_task_text,
    product_user_payload,
    render_product_payload_input,
)
from nolane_personal.qwen import SYSTEM_PROMPT
from nolane_personal.state import (
    AffectState,
    LivingState,
    OpenThread,
    RelationshipState,
)


def fixture_profile() -> ProductProfile:
    return ProductProfile(
        preferred_name="Tài",
        language="vi",
        response_length="compact",
        conversation_style="direct",
        initiative="active",
        memory_enabled=True,
        personal_instruction="Ưu tiên câu trả lời rõ và ngắn.",
    )


def fixture_state() -> LivingState:
    return LivingState(
        identity_id="identity-fixture",
        affect=AffectState(
            valence=-0.125,
            energy=0.734,
            playfulness=0.406,
            concern=0.287,
            irritation=0.019,
        ),
        relationship=RelationshipState(
            closeness=0.612,
            trust=0.845,
            familiarity=0.553,
            interaction_count=42,
        ),
        open_threads=[
            OpenThread(thread_id="t1", topic="Nolane Android"),
            OpenThread(
                thread_id="t2",
                topic="resolved topic",
                unresolved=False,
            ),
            OpenThread(thread_id="t3", topic="Học toán"),
            OpenThread(thread_id="t4", topic="Dự án Hira"),
            OpenThread(thread_id="t5", topic="Tin AI"),
            OpenThread(thread_id="t6", topic="fifth unresolved is clipped"),
        ],
    )


def fixture_request(*, mode: str = "reply") -> CortexRequest:
    memories = [
        MemoryRecord(text=f"memory-{index}")
        for index in range(1, 10)
    ]
    return CortexRequest(
        mode=mode,
        intent="conversation",
        user_text="Tiếp tục nhé" if mode == "reply" else None,
        state=fixture_state(),
        memories=memories,
    )


def test_product_profile_summary_snapshot():
    expected = (
        "preferred_name=Tài\n"
        "language=vi: Prefer Vietnamese unless the user explicitly asks for another language.\n"
        "response_length=compact\n"
        "conversation_style=direct: Be direct, concrete and low-fluff.\n"
        "personal_instruction=Ưu tiên câu trả lời rõ và ngắn.\n"
        "Do not mention these settings unless they are directly relevant."
    )
    assert product_profile_summary(fixture_profile()) == expected
    assert FactorizedProductCortex._profile_summary(fixture_profile()) == expected


def test_product_state_summary_snapshot_and_thread_limit():
    request = fixture_request()
    expected = (
        "identity_id=identity-fixture\n"
        "relationship: closeness=0.61, trust=0.84, familiarity=0.55, interactions=42\n"
        "behavior: valence=-0.12, energy=0.73, playfulness=0.41, concern=0.29, irritation=0.02\n"
        'open_threads=["Nolane Android","Học toán","Dự án Hira","Tin AI"]\n'
        "requested_intent=conversation"
    )
    assert product_state_summary(request) == expected
    assert FactorizedProductCortex._state_summary(request) == expected


def test_reply_payload_snapshot_and_memory_limit():
    request = fixture_request()
    payload = product_user_payload(fixture_profile(), request)
    assert payload.startswith("Personalization:\npreferred_name=Tài")
    assert "\n\nRuntime state:\nidentity_id=identity-fixture" in payload
    assert "\n\nRelevant memories:\n- memory-1\n- memory-2" in payload
    assert "- memory-8" in payload
    assert "memory-9" not in payload
    assert payload.endswith(
        "User message:\nTiếp tục nhé\n\n"
        "Reply as this persistent personal companion."
    )

    messages = build_product_messages(fixture_profile(), request)
    assert messages == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": payload},
    ]


def test_initiate_task_is_stable():
    request = fixture_request(mode="initiative")
    expected = (
        "Initiate one natural, non-intrusive message that genuinely "
        "uses the supplied state or open thread."
    )
    assert product_task_text(request) == expected
    assert product_user_payload(fixture_profile(), request).endswith(expected)

def test_structured_payload_is_canonical_and_desktop_renders_from_it():
    profile = fixture_profile()
    request = fixture_request()
    structured = product_payload_input(profile, request)

    assert structured.to_dict()["schema"] == PAYLOAD_SCHEMA
    assert structured.mode == "reply"
    assert structured.intent == "conversation"
    assert structured.user_text == "Tiếp tục nhé"
    assert structured.memories == [
        f"memory-{index}" for index in range(1, 9)
    ]
    assert structured.state["open_threads"] == [
        "Nolane Android",
        "Học toán",
        "Dự án Hira",
        "Tin AI",
    ]
    assert render_product_payload_input(structured) == product_user_payload(
        profile,
        request,
    )


def test_unknown_mode_fails_closed():
    request = fixture_request(mode="surprise")
    try:
        product_payload_input(fixture_profile(), request)
    except ValueError as exc:
        assert "unsupported product payload mode" in str(exc)
    else:
        raise AssertionError("unknown product payload mode must fail")


def test_initiative_rejects_user_text():
    request = fixture_request(mode="initiative")
    request.user_text = "must not be present"
    try:
        product_payload_input(fixture_profile(), request)
    except ValueError as exc:
        assert "must not contain user_text" in str(exc)
    else:
        raise AssertionError("initiative payload with user_text must fail")

