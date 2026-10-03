from __future__ import annotations

import argparse
import json
from pathlib import Path

from nolane_personal.v1_closure import (
    build_android_performance_receipt,
    verify_android_device_receipt,
    verify_android_performance_policy,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-receipt", required=True)
    parser.add_argument("--measurements", required=True)
    parser.add_argument(
        "--policy",
        default="config/v1-android-performance-policy.json",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = json.loads(
        Path(args.device_receipt).read_text(encoding="utf-8")
    )
    verify_android_device_receipt(device, require_real=True)
    policy = json.loads(Path(args.policy).read_text(encoding="utf-8"))
    verify_android_performance_policy(policy)
    measurements = json.loads(
        Path(args.measurements).read_text(encoding="utf-8")
    )

    receipt = build_android_performance_receipt(
        evidence_class=str(
            measurements.get("evidence_class", "REAL_PHYSICAL_DEVICE")
        ),
        apk_sha256=device["apk_sha256"],
        candidate_checkpoint_sha256=device[
            "candidate_checkpoint_sha256"
        ],
        device_fingerprint_sha256=device[
            "device_fingerprint_sha256"
        ],
        policy=policy,
        sample_count=int(measurements["sample_count"]),
        metrics=dict(measurements["metrics"]),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
