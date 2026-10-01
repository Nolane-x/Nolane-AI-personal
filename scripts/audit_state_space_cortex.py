from __future__ import annotations

import argparse
import json

from nolane_personal.state_space_core import (
    StateSpaceCortexConfig,
    state_space_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--state-dim", type=int, default=32)
    args = parser.parse_args()

    result = state_space_audit(
        args.hidden_size,
        StateSpaceCortexConfig(
            latent_dim=32,
            state_dim=args.state_dim,
            max_abs_gate=0.20,
            initial_gate=0.04,
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["parameter_count"] > 100_000:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
