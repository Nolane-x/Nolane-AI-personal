from __future__ import annotations

import argparse
import os

from nolane_personal.transactional_registry import CheckpointRegistry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--transaction", required=True)
    parser.add_argument(
        "--crash-at",
        required=True,
        choices=[
            "ARTIFACT_INSTALLED",
            "POINTER_SNAPSHOT_WRITTEN",
            "ACTIVE_POINTER_SWAPPED",
            "POINTER_EVENT_RECORDED",
            "COMMIT_EVENT_RECORDED",
        ],
    )
    parser.add_argument("--exit-code", type=int, default=91)
    args = parser.parse_args()

    registry = CheckpointRegistry(args.registry)

    def crash_hook(phase, details):
        if phase == args.crash_at:
            os._exit(int(args.exit_code))

    registry.commit_update(
        args.transaction,
        fault_hook=crash_hook,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
