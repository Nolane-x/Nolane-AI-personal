from __future__ import annotations

import argparse
import os

from nolane_personal.serving_coordination import ServingCoordinator
from nolane_personal.transactional_registry import CheckpointRegistry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--process-id", required=True)
    parser.add_argument(
        "--crash-at",
        choices=["TARGET_VERIFIED_BEFORE_ACK", "LEASE_ACKED"],
        required=True,
    )
    parser.add_argument("--exit-code", type=int, default=92)
    args = parser.parse_args()

    registry = CheckpointRegistry(args.registry)
    coordinator = ServingCoordinator(registry)
    plan = coordinator.reload_plan(args.process_id)
    if plan is None:
        raise SystemExit("serving process does not require reload")
    target = registry.pointer_for_generation(
        int(plan["target_generation"])
    )
    registry.artifact_path_for_pointer(target)

    if args.crash_at == "TARGET_VERIFIED_BEFORE_ACK":
        os._exit(int(args.exit_code))

    coordinator.ack_reload(
        args.process_id,
        plan=plan,
        loaded_checkpoint_sha256=target["checkpoint_sha256"],
    )
    if args.crash_at == "LEASE_ACKED":
        os._exit(int(args.exit_code))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
