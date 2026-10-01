from __future__ import annotations

import argparse
import json

from nolane_personal.multiscale_cortex import (
    MultiTimescaleCortexConfig,
    multiscale_cortex_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--state-dim", type=int, default=24)
    parser.add_argument("--slow-decay-floor", type=float, default=0.50)
    args = parser.parse_args()

    result = multiscale_cortex_audit(
        args.hidden_size,
        MultiTimescaleCortexConfig(
            latent_dim=32,
            state_dim=args.state_dim,
            max_abs_gate=0.20,
            initial_gate=0.04,
            slow_decay_floor=args.slow_decay_floor,
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["parameter_count"] <= 100_000 else 2


if __name__ == "__main__":
    raise SystemExit(main())
