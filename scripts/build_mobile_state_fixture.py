from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from nolane_personal.cortex import CortexRequest
from nolane_personal.product_profile import ProductProfile
from nolane_personal.product_prompt_payload import product_payload_input
from nolane_personal.state import LivingState


SCHEMA = "NOLANE-V055-MOBILE-STATE-PARITY-FIXTURE-V1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    identity_id = "identity-v055-default"
    checkpoint_sha256 = "4" * 64
    latent_dim = 4

    profile = ProductProfile()
    state = LivingState(identity_id=identity_id)
    request = CortexRequest(
        mode="reply",
        intent="conversation",
        user_text="xin chào",
        state=state,
        memories=[],
    )
    payload = product_payload_input(profile, request).to_dict()

    full_profile = {
        "preferred_name": profile.preferred_name,
        "language": profile.language,
        "response_length": profile.response_length,
        "conversation_style": profile.conversation_style,
        "initiative": profile.initiative,
        "memory_enabled": profile.memory_enabled,
        "personal_instruction": profile.personal_instruction,
    }

    fixture = {
        "schema": SCHEMA,
        "identity_id": identity_id,
        "checkpoint_sha256": checkpoint_sha256,
        "latent_dim": latent_dim,
        "full_profile": full_profile,
        "payload": payload,
        "latent_values": [0.0] * latent_dim,
        "living_state_defaults": {
            "affect": asdict(state.affect),
            "relationship": asdict(state.relationship),
        },
    }
    (output / "fixture.json").write_text(
        json.dumps(
            fixture,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
