from __future__ import annotations

import argparse
import json
from pathlib import Path

from nolane_personal.unified_promotion_authority import (
    UnifiedPromotionAuthorizationPolicy,
    create_unified_operator_promotion_request,
    decide_unified_promotion_authorization,
)


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cycle",
        action="append",
        required=True,
        help="Ordered L31/L38 run receipt. Repeat once per cycle.",
    )
    parser.add_argument("--long-horizon", required=True)
    parser.add_argument("--unified-chain", required=True)
    parser.add_argument("--active-parent-checkpoint-sha256", required=True)
    parser.add_argument("--candidate-checkpoint-sha256", required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--approve", action="store_true")
    parser.add_argument("--min-cycles", type=int, default=2)
    parser.add_argument("--min-cortex-cycles", type=int, default=1)
    parser.add_argument("--ttl-seconds", type=int, default=3600)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise SystemExit(
            f"refusing to overwrite L40 authorization: {output}"
        )

    cycles = [load_json(path) for path in args.cycle]
    horizon = load_json(args.long_horizon)
    chain = load_json(args.unified_chain)

    request = create_unified_operator_promotion_request(
        active_parent_checkpoint_sha256=(
            args.active_parent_checkpoint_sha256
        ),
        candidate_checkpoint_sha256=args.candidate_checkpoint_sha256,
        unified_chain_sha256=chain["chain_sha256"],
        long_horizon_retention_court_sha256=horizon["court_sha256"],
        approved=bool(args.approve),
        nonce=args.nonce,
    )
    authorization = decide_unified_promotion_authorization(
        cycles=cycles,
        long_horizon_retention=horizon,
        unified_chain=chain,
        operator_request=request,
        policy=UnifiedPromotionAuthorizationPolicy(
            min_cycles=args.min_cycles,
            min_cortex_cycles=args.min_cortex_cycles,
            authorization_ttl_seconds=args.ttl_seconds,
        ),
    )
    result = {
        "schema": "NOLANE-L40-UNIFIED-PROMOTION-DECISION-V1",
        "operator_request": request,
        "authorization": authorization,
    }
    rendered = json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if authorization["status"] == "AUTHORIZED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
