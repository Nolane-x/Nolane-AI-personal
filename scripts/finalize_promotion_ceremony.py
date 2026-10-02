from __future__ import annotations

import argparse
import json
from pathlib import Path

from nolane_personal.promotion_ceremony import (
    audit_promotion_ceremonies,
    finalize_promotion_ceremony,
    load_promotion_ceremony,
)
from nolane_personal.transactional_registry import CheckpointRegistry


def emit(payload) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        default="runtime-data/continual-checkpoint-registry",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    finalize = sub.add_parser("finalize")
    finalize.add_argument("--transaction", required=True)
    finalize.add_argument("--convergence", required=True)
    finalize.add_argument("--no-persist", action="store_true")

    verify = sub.add_parser("verify")
    verify.add_argument("--generation", type=int, required=True)

    sub.add_parser("audit")

    args = parser.parse_args()
    registry = CheckpointRegistry(args.registry)

    if args.command == "finalize":
        convergence = json.loads(
            Path(args.convergence).read_text(encoding="utf-8")
        )
        receipt = finalize_promotion_ceremony(
            registry,
            transaction_id=args.transaction,
            convergence_receipt=convergence,
            persist=not args.no_persist,
        )
        emit(receipt)
        return 0 if receipt["status"] == "COMPLETE" else 2
    if args.command == "verify":
        emit(load_promotion_ceremony(registry, args.generation))
        return 0
    if args.command == "audit":
        emit(audit_promotion_ceremonies(registry))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
