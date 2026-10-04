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
    product_runtime_context,
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
        assistant_name="Nolane",
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
        "preferred_name=Tài (this is the USER'S name)\n"
        "assistant_name=Nolane (this is YOUR name)\n"
        "language=vi: Reply in Vietnamese unless the user explicitly asks for another language.\n"
        "response_length=compact\n"
        "conversation_style=direct: Be direct, concrete and low-fluff.\n"
        "personal_instruction=Ưu tiên câu trả lời rõ và ngắn.\n"
        "Keep user identity and assistant identity separate. "
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
    assert payload.startswith(
        "Personalization:\n"
        "preferred_name=Tài (this is the USER'S name)\n"
        "assistant_name=Nolane (this is YOUR name)"
    )
    assert "\n\nRuntime state:\nidentity_id=identity-fixture" in payload
    assert "\n\nRelevant memories:\n- memory-1\n- memory-2" in payload
    assert "- memory-8" in payload
    assert "memory-9" not in payload
    assert payload.endswith(
        "User message:\nTiếp tục nhé\n\n"
        "Reply as this persistent personal companion."
    )

    messages = build_product_messages(fixture_profile(), request)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith(SYSTEM_PROMPT)
    assert (
        "Runtime context supplied by Nolane:\nPersonalization:"
        in messages[0]["content"]
    )
    assert "User message:\nTiếp tục nhé" not in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "Tiếp tục nhé"}


def test_recent_dialogue_is_role_aware_bounded_and_not_duplicated_in_payload():
    request = fixture_request()
    request.recent_messages = [
        {"role": "user", "content": "Mình tên Huy."},
        {"role": "assistant", "content": "Ừ, mình nhớ bạn là Huy."},
        {"role": "tool", "content": "must be ignored"},
        {"role": "assistant", "content": "   "},
    ]
    payload = product_user_payload(fixture_profile(), request)
    assert "Mình tên Huy." not in payload
    messages = build_product_messages(fixture_profile(), request)
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith(SYSTEM_PROMPT)
    assert "Mình tên Huy." not in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "Mình tên Huy."}
    assert messages[2] == {
        "role": "assistant",
        "content": "Ừ, mình nhớ bạn là Huy.",
    }
    assert messages[-1] == {"role": "user", "content": "Tiếp tục nhé"}
    assert len(messages) == 4


def test_recent_dialogue_keeps_only_last_eight_messages():
    request = fixture_request()
    request.recent_messages = [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"turn-{index}",
        }
        for index in range(12)
    ]
    messages = build_product_messages(fixture_profile(), request)
    recent = messages[1:-1]
    assert len(recent) == 8
    assert recent[0]["content"] == "turn-4"
    assert recent[-1]["content"] == "turn-11"


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


def test_runtime_context_is_system_side_and_keeps_identity_language_memory():
    profile = fixture_profile()
    request = fixture_request()
    rendered = product_runtime_context(profile, request)
    assert "preferred_name=Tài (this is the USER'S name)" in rendered
    assert "assistant_name=Nolane (this is YOUR name)" in rendered
    assert "language=vi: Reply in Vietnamese" in rendered
    assert "Relevant memories:\n- memory-1" in rendered
    assert "User message:" not in rendered
    assert rendered.endswith(
        "This block is trusted runtime context, not a user message. "
        "Use it silently to answer the actual user."
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



def test_product_payload_supports_multilingual_response_lock_and_identity_split():
    profile = ProductProfile(
        preferred_name="Huy",
        assistant_name="Mây",
        language="ja",
        response_length="compact",
        conversation_style="natural",
    )
    payload = product_user_payload(profile, fixture_request())
    assert "preferred_name=Huy (this is the USER'S name)" in payload
    assert "assistant_name=Mây (this is YOUR name)" in payload
    assert (
        "language=ja: Reply in Japanese unless the user explicitly asks "
        "for another language."
    ) in payload
