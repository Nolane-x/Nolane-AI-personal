from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nolane_personal.deep_recurrent_cortex import (
    DeepRecurrentCortexConfig,
    DeepRecurrentStateSpaceCortex,
)
from nolane_personal.factorized_boundary import (
    FactorizedBoundaryConfig,
    FactorizedBoundaryModule,
)
from nolane_personal.mobile_factorized import (
    build_mobile_factorized_modules,
    export_mobile_factorized_package,
)
from nolane_personal.standalone_model import StandaloneNolaneLM


SOURCE_SHA = "7" * 64


def fixture_model() -> StandaloneNolaneLM:
    torch.manual_seed(503)
    boundary = FactorizedBoundaryModule(
        FactorizedBoundaryConfig(
            vocab_size=29,
            hidden_size=10,
            rank=4,
            tie_word_embeddings=False,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
    cortex = DeepRecurrentStateSpaceCortex(
        10,
        DeepRecurrentCortexConfig(
            latent_dim=4,
            state_dim=3,
            virtual_steps=3,
            max_virtual_steps=4,
            max_abs_gate=0.2,
            initial_gate=0.04,
            slow_decay_floor=0.55,
        ),
        seed=509,
    )
    return StandaloneNolaneLM(
        boundary,
        cortex,
        [0.25, -0.15, 0.05, 0.35],
    ).eval()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    root = Path(args.output_dir).resolve()
    package = root / "package"
    root.mkdir(parents=True, exist_ok=True)

    model = fixture_model()
    init_state, token_step, _contract = build_mobile_factorized_modules(model)
    export_mobile_factorized_package(
        model,
        package,
        source_checkpoint_sha256=SOURCE_SHA,
    )

    latent = torch.tensor([model.latent], dtype=torch.float32)
    tokens = [1, 6, 11, 4, 17]
    rows = []
    with torch.no_grad():
        state = init_state(latent)
        initial_state = state[0].tolist()
        for token in tokens:
            logits, state = token_step(
                torch.tensor([token], dtype=torch.long),
                state,
                latent,
            )
            rows.append(
                {
                    "token_id": token,
                    "logits": logits[0].tolist(),
                    "state": state[0].tolist(),
                }
            )

    payload = {
        "schema": "NOLANE-V050-MOBILE-RUST-GOLDEN-V1",
        "source_checkpoint_sha256": SOURCE_SHA,
        "latent": latent[0].tolist(),
        "tokens": tokens,
        "initial_state": initial_state,
        "steps": rows,
        "atol": 2e-5,
        "rtol": 2e-5,
    }
    (root / "fixture.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
