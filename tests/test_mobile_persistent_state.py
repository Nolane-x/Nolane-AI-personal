from __future__ import annotations

import json

import pytest

from nolane_personal.cortex import CortexRequest
from nolane_personal.memory import MemoryRecord
from nolane_personal.mobile_persistent_state import (
    PERSISTENT_MOBILE_STATE_INTEGRITY_TYPED_V1,
    PERSISTENT_MOBILE_STATE_INTEGRITY_TYPED_V2,
    PERSISTENT_MOBILE_STATE_SCHEMA,
    build_persistent_mobile_state,
    read_persistent_mobile_state,
    write_persistent_mobile_state,
)
from nolane_personal.product_profile import ProductProfile
from nolane_personal.product_prompt_payload import product_payload_input
from nolane_personal.state import LivingState, OpenThread
from nolane_personal.store import payload_digest


def fixture_state() -> tuple[ProductProfile, LivingState]:
    profile = ProductProfile(
        preferred_name="Thuận",
        assistant_name="Nolane",
        language="vi",
        response_length="compact",
        conversation_style="natural",
        personal_instruction="Nói ngắn gọn.",
    )
    state = LivingState(identity_id="nolane-mobile-identity")
    state.relationship.closeness = 0.71
    state.relationship.trust = 0.82
    state.relationship.familiarity = 0.63
    state.relationship.interaction_count = 17
    state.affect.valence = 0.2
    state.affect.energy = 0.4
    state.affect.playfulness = 0.1
    state.affect.concern = 0.05
    state.affect.irritation = 0.0
    state.open_threads = [
        OpenThread(
            thread_id="v055",
            topic="Hoàn thành v0.55",
        )
    ]
    return profile, state


def test_python_persistent_mobile_state_roundtrip(tmp_path):
    profile, state = fixture_state()
    payload = build_persistent_mobile_state(
        source_checkpoint_sha256="a" * 64,
        latent=[0.125, -0.25, 0.5, 1.0],
        profile=profile,
        state=state,
        memories=["Người dùng đang xây Nolane."],
    )
    path = write_persistent_mobile_state(
        tmp_path / "mobile" / "state.json",
        payload,
        expected_source_checkpoint_sha256="a" * 64,
        expected_latent_dim=4,
    )
    loaded = read_persistent_mobile_state(
        path,
        expected_source_checkpoint_sha256="a" * 64,
        expected_latent_dim=4,
    )
    assert loaded == payload
    envelope = json.loads(path.read_text(encoding="utf-8"))
    assert envelope["schema"] == PERSISTENT_MOBILE_STATE_SCHEMA
    assert (
        envelope["integrity"]
        == PERSISTENT_MOBILE_STATE_INTEGRITY_TYPED_V2
    )
    assert len(envelope["state_sha256"]) == 64
    assert not list(path.parent.glob("*.tmp"))


def test_persistent_mobile_state_reads_legacy_v1_identity_default(tmp_path):
    profile, state = fixture_state()
    payload = build_persistent_mobile_state(
        source_checkpoint_sha256="a" * 64,
        latent=[0.125, -0.25, 0.5, 1.0],
        profile=profile,
        state=state,
        memories=[],
    )
    legacy_payload = json.loads(json.dumps(payload))
    legacy_payload["profile"].pop("assistant_name", None)
    from nolane_personal.mobile_persistent_state import _typed_integrity_digest
    envelope = {
        "schema": PERSISTENT_MOBILE_STATE_SCHEMA,
        "integrity": PERSISTENT_MOBILE_STATE_INTEGRITY_TYPED_V1,
        "state_sha256": _typed_integrity_digest(
            legacy_payload,
            include_assistant_name=False,
        ),
        "state": legacy_payload,
    }
    path = tmp_path / "legacy-state.json"
    path.write_text(
        json.dumps(envelope, ensure_ascii=False),
        encoding="utf-8",
    )
    loaded = read_persistent_mobile_state(path)
    assert loaded["profile"]["assistant_name"] == "Nolane"


def test_persistent_state_projection_matches_product_payload_contract():
    profile, state = fixture_state()
    memory = MemoryRecord(text="Người dùng đang xây Nolane.")
    request = CortexRequest(
        mode="reply",
        intent="conversation",
        user_text="Tiếp tục nhé",
        state=state,
        memories=[memory],
    )
    product = product_payload_input(profile, request).to_dict()
    persistent = build_persistent_mobile_state(
        source_checkpoint_sha256="a" * 64,
        latent=[0.125, -0.25, 0.5, 1.0],
        profile=profile,
        state=state,
        memories=[memory.text],
    )
    assert persistent["profile"] == product["profile"]
    assert persistent["state"] == product["state"]
    assert persistent["memories"] == product["memories"]


def test_persistent_mobile_state_fails_closed_on_tamper(tmp_path):
    profile, state = fixture_state()
    payload = build_persistent_mobile_state(
        source_checkpoint_sha256="a" * 64,
        latent=[0.125, -0.25, 0.5, 1.0],
        profile=profile,
        state=state,
        memories=[],
    )
    path = write_persistent_mobile_state(tmp_path / "state.json", payload)

    with pytest.raises(ValueError, match="checkpoint mismatch"):
        read_persistent_mobile_state(
            path,
            expected_source_checkpoint_sha256="b" * 64,
        )

    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["state"]["profile"]["preferred_name"] = "tampered"
    path.write_text(
        json.dumps(envelope, ensure_ascii=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="integrity mismatch"):
        read_persistent_mobile_state(path)



def test_persistent_mobile_state_reads_legacy_v055_integrity(tmp_path):
    profile, state = fixture_state()
    payload = build_persistent_mobile_state(
        source_checkpoint_sha256="a" * 64,
        latent=[0.125, -0.25, 0.5, 1.0],
        profile=profile,
        state=state,
        memories=["legacy state remains readable"],
    )
    path = tmp_path / "legacy-state.json"
    path.write_text(
        json.dumps(
            {
                "schema": PERSISTENT_MOBILE_STATE_SCHEMA,
                "state_sha256": payload_digest(payload),
                "state": payload,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    assert read_persistent_mobile_state(
        path,
        expected_source_checkpoint_sha256="a" * 64,
        expected_latent_dim=4,
    ) == payload
