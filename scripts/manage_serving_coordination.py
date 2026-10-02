from __future__ import annotations

import argparse
import json
from pathlib import Path

from nolane_personal.serving_coordination import (
    ServingCoordinator,
    ServingLeasePolicy,
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
    parser.add_argument("--lease-seconds", type=float, default=30.0)
    parser.add_argument("--min-live-processes", type=int, default=1)
    sub = parser.add_subparsers(dest="command", required=True)

    register = sub.add_parser("register-active")
    register.add_argument("--process-id", required=True)

    heartbeat = sub.add_parser("heartbeat")
    heartbeat.add_argument("--process-id", required=True)

    plan = sub.add_parser("plan")
    plan.add_argument("--process-id", required=True)

    ack = sub.add_parser("ack")
    ack.add_argument("--process-id", required=True)
    ack.add_argument("--plan", required=True)
    ack.add_argument("--loaded-checkpoint-sha256", required=True)

    gate = sub.add_parser("gate")
    gate.add_argument("--process-id", required=True)

    sub.add_parser("convergence")

    args = parser.parse_args()
    registry = CheckpointRegistry(args.registry)
    coordinator = ServingCoordinator(
        registry,
        policy=ServingLeasePolicy(
            lease_seconds=args.lease_seconds,
            min_live_processes=args.min_live_processes,
        ),
    )

    if args.command == "register-active":
        emit(
            coordinator.register_loaded(
                args.process_id,
                pointer=registry.active_pointer(),
            )
        )
    elif args.command == "heartbeat":
        emit(coordinator.heartbeat(args.process_id))
    elif args.command == "plan":
        emit(coordinator.reload_plan(args.process_id))
    elif args.command == "ack":
        plan_payload = json.loads(
            Path(args.plan).read_text(encoding="utf-8")
        )
        emit(
            coordinator.ack_reload(
                args.process_id,
                plan=plan_payload,
                loaded_checkpoint_sha256=args.loaded_checkpoint_sha256,
            )
        )
    elif args.command == "gate":
        emit(coordinator.serving_gate(args.process_id))
    elif args.command == "convergence":
        emit(coordinator.assess_convergence())
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
