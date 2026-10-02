from __future__ import annotations

import argparse
import json

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

    init = sub.add_parser("init")
    init.add_argument("--bundle", required=True)

    begin = sub.add_parser("begin")
    begin.add_argument("--candidate", required=True)

    verify = sub.add_parser("verify")
    verify.add_argument("--transaction", required=True)

    commit = sub.add_parser("commit")
    commit.add_argument("--transaction", required=True)

    abort = sub.add_parser("abort")
    abort.add_argument("--transaction", required=True)
    abort.add_argument("--reason", required=True)

    recover = sub.add_parser("recover")
    recover.add_argument("--break-stale-lock", action="store_true")

    rollback = sub.add_parser("rollback")
    rollback.add_argument("--generation", type=int, required=True)

    sub.add_parser("audit")
    sub.add_parser("active")

    args = parser.parse_args()
    registry = CheckpointRegistry(args.registry)

    if args.command == "init":
        emit(registry.initialize(args.bundle))
    elif args.command == "begin":
        emit({"transaction_id": registry.begin_l31_update(args.candidate)})
    elif args.command == "verify":
        emit(registry.verify_update(args.transaction))
    elif args.command == "commit":
        emit(registry.commit_update(args.transaction))
    elif args.command == "abort":
        emit(
            registry.abort_update(
                args.transaction,
                reason=args.reason,
            )
        )
    elif args.command == "recover":
        emit(
            {
                "recovered": registry.recover(
                    break_stale_lock=args.break_stale_lock,
                )
            }
        )
    elif args.command == "rollback":
        emit(registry.rollback_to_generation(args.generation))
    elif args.command == "audit":
        emit(registry.verify_registry())
    elif args.command == "active":
        emit(registry.active_pointer())
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
