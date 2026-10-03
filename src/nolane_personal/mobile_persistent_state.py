from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Iterable

from .product_profile import ProductProfile
from .state import LivingState
from .store import canonical_json, payload_digest


PERSISTENT_MOBILE_STATE_SCHEMA = "NOLANE-V055-MOBILE-PERSISTENT-STATE-V1"
MAX_PERSISTENT_MOBILE_STATE_BYTES = 4 * 1024 * 1024
MAX_OPEN_THREADS = 4
MAX_MEMORIES = 8


def _is_lower_hex_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def build_persistent_mobile_state(
    *,
    source_checkpoint_sha256: str,
    latent: Iterable[float],
    profile: ProductProfile,
    state: LivingState,
    memories: Iterable[str],
) -> dict[str, object]:
    profile.normalize()
    state.normalize()
    payload: dict[str, object] = {
        "source_checkpoint_sha256": str(source_checkpoint_sha256),
        "latent": [float(value) for value in latent],
        "profile": {
            "preferred_name": profile.preferred_name,
            "language": profile.language,
            "response_length": profile.response_length,
            "conversation_style": profile.conversation_style,
            "personal_instruction": profile.personal_instruction,
        },
        "state": {
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
        },
        "memories": [
            str(memory) for memory in list(memories)[:MAX_MEMORIES]
        ],
    }
    validate_persistent_mobile_state(payload)
    return payload


def validate_persistent_mobile_state(
    payload: dict[str, object],
    *,
    expected_source_checkpoint_sha256: str | None = None,
    expected_latent_dim: int | None = None,
) -> None:
    source = str(payload.get("source_checkpoint_sha256", ""))
    if not _is_lower_hex_sha256(source):
        raise ValueError(
            "persistent mobile state checkpoint digest is malformed"
        )
    if (
        expected_source_checkpoint_sha256 is not None
        and source != str(expected_source_checkpoint_sha256).lower()
    ):
        raise ValueError("persistent mobile state checkpoint mismatch")

    latent = payload.get("latent")
    if not isinstance(latent, list) or not latent:
        raise ValueError("persistent mobile latent must not be empty")
    if (
        expected_latent_dim is not None
        and len(latent) != int(expected_latent_dim)
    ):
        raise ValueError(
            "persistent mobile latent shape mismatch: "
            f"expected {int(expected_latent_dim)}, got {len(latent)}"
        )
    if any(
        not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in latent
    ):
        raise ValueError(
            "persistent mobile latent contains non-finite value"
        )

    profile = payload.get("profile")
    if not isinstance(profile, dict):
        raise ValueError("persistent mobile profile must be an object")
    if str(profile.get("language", "")) not in {"auto", "vi", "en"}:
        raise ValueError("unsupported product language")
    if str(profile.get("response_length", "")) not in {
        "compact",
        "balanced",
        "expansive",
    }:
        raise ValueError("unsupported response length")
    if str(profile.get("conversation_style", "")) not in {
        "natural",
        "warm",
        "direct",
        "playful",
    }:
        raise ValueError("unsupported conversation style")

    runtime_state = payload.get("state")
    if not isinstance(runtime_state, dict):
        raise ValueError("persistent mobile runtime state must be an object")
    if not str(runtime_state.get("identity_id", "")).strip():
        raise ValueError(
            "persistent mobile identity_id must not be empty"
        )
    relationship = runtime_state.get("relationship")
    affect = runtime_state.get("affect")
    if not isinstance(relationship, dict) or not isinstance(affect, dict):
        raise ValueError(
            "persistent mobile relationship/affect state is invalid"
        )
    numeric = [
        relationship.get("closeness"),
        relationship.get("trust"),
        relationship.get("familiarity"),
        affect.get("valence"),
        affect.get("energy"),
        affect.get("playfulness"),
        affect.get("concern"),
        affect.get("irritation"),
    ]
    if any(
        not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in numeric
    ):
        raise ValueError(
            "persistent mobile state contains non-finite value"
        )
    interaction_count = relationship.get("interaction_count")
    if (
        not isinstance(interaction_count, int)
        or interaction_count < 0
    ):
        raise ValueError(
            "persistent mobile interaction_count must be non-negative"
        )

    open_threads = runtime_state.get("open_threads")
    if not isinstance(open_threads, list):
        raise ValueError("persistent mobile open_threads must be a list")
    if len(open_threads) > MAX_OPEN_THREADS:
        raise ValueError(
            "persistent mobile state exceeds open-thread limit"
        )

    memories = payload.get("memories")
    if not isinstance(memories, list):
        raise ValueError("persistent mobile memories must be a list")
    if len(memories) > MAX_MEMORIES:
        raise ValueError("persistent mobile state exceeds memory limit")


def write_persistent_mobile_state(
    path: str | Path,
    payload: dict[str, object],
    *,
    expected_source_checkpoint_sha256: str | None = None,
    expected_latent_dim: int | None = None,
) -> Path:
    validate_persistent_mobile_state(
        payload,
        expected_source_checkpoint_sha256=(
            expected_source_checkpoint_sha256
        ),
        expected_latent_dim=expected_latent_dim,
    )
    envelope = {
        "schema": PERSISTENT_MOBILE_STATE_SCHEMA,
        "state_sha256": payload_digest(payload),
        "state": payload,
    }
    encoded = (canonical_json(envelope) + "\n").encode("utf-8")
    if len(encoded) > MAX_PERSISTENT_MOBILE_STATE_BYTES:
        raise ValueError(
            "persistent mobile state exceeds size limit: "
            f"{len(encoded)} > {MAX_PERSISTENT_MOBILE_STATE_BYTES}"
        )

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "wb",
        dir=destination.parent,
        prefix=destination.name + ".",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def read_persistent_mobile_state(
    path: str | Path,
    *,
    expected_source_checkpoint_sha256: str | None = None,
    expected_latent_dim: int | None = None,
) -> dict[str, object]:
    source = Path(path)
    size = source.stat().st_size
    if size > MAX_PERSISTENT_MOBILE_STATE_BYTES:
        raise ValueError(
            "persistent mobile state exceeds size limit: "
            f"{size} > {MAX_PERSISTENT_MOBILE_STATE_BYTES}"
        )
    envelope = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(envelope, dict):
        raise ValueError(
            "persistent mobile state envelope must be an object"
        )
    if envelope.get("schema") != PERSISTENT_MOBILE_STATE_SCHEMA:
        raise ValueError("persistent mobile state schema mismatch")
    digest = str(envelope.get("state_sha256", ""))
    if not _is_lower_hex_sha256(digest):
        raise ValueError(
            "persistent mobile state integrity digest is malformed"
        )
    payload = envelope.get("state")
    if not isinstance(payload, dict):
        raise ValueError("persistent mobile state payload is missing")
    if payload_digest(payload) != digest:
        raise ValueError("persistent mobile state integrity mismatch")
    validate_persistent_mobile_state(
        payload,
        expected_source_checkpoint_sha256=(
            expected_source_checkpoint_sha256
        ),
        expected_latent_dim=expected_latent_dim,
    )
    return payload
