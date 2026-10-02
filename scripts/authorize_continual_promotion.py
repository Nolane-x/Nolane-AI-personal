from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path

from nolane_personal.promotion_authority import (
    PromotionAuthorizationPolicy,
    create_operator_promotion_request,
    decide_promotion_authorization,
)


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_once(path: str | Path, payload) -> None:
    output = Path(path)
    if output.exists():
        raise SystemExit(f"refusing to overwrite promotion artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    request = sub.add_parser("request")
    request.add_argument("--active-parent", required=True)
    request.add_argument("--candidate", required=True)
    request.add_argument("--chain", required=True)
    request.add_argument("--long-horizon", required=True)
    request.add_argument("--approve", action="store_true")
    request.add_argument("--nonce", default=None)
    request.add_argument("--output", required=True)

    authorize = sub.add_parser("authorize")
    authorize.add_argument("--cycle", action="append", required=True)
    authorize.add_argument("--long-horizon", required=True)
    authorize.add_argument("--chain", required=True)
    authorize.add_argument("--request", required=True)
    authorize.add_argument("--min-cycles", type=int, default=2)
    authorize.add_argument("--ttl-seconds", type=int, default=3600)
    authorize.add_argument("--output", required=True)

    args = parser.parse_args()

    if args.command == "request":
        chain = load_json(args.chain)
        horizon = load_json(args.long_horizon)
        nonce = args.nonce or secrets.token_hex(24)
        payload = create_operator_promotion_request(
            active_parent_checkpoint_sha256=args.active_parent,
            candidate_checkpoint_sha256=args.candidate,
            multicycle_chain_sha256=chain["chain_sha256"],
            long_horizon_retention_court_sha256=horizon["court_sha256"],
            approved=args.approve,
            nonce=nonce,
        )
        write_once(args.output, payload)
        return 0 if args.approve else 2

    if args.command == "authorize":
        cycles = [load_json(path) for path in args.cycle]
        horizon = load_json(args.long_horizon)
        chain = load_json(args.chain)
        request_payload = load_json(args.request)
        authorization = decide_promotion_authorization(
            cycles=cycles,
            long_horizon_retention=horizon,
            multicycle_chain=chain,
            operator_request=request_payload,
            policy=PromotionAuthorizationPolicy(
                min_cycles=args.min_cycles,
                authorization_ttl_seconds=args.ttl_seconds,
            ),
        )
        write_once(args.output, authorization)
        return 0 if authorization["status"] == "AUTHORIZED" else 2

    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
