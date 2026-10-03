from __future__ import annotations

import argparse
import json
from pathlib import Path

from nolane_personal.v1_closure import (
    READY_STATUS,
    evaluate_v1_closure,
    verify_v1_closure_receipt,
)


def maybe_path(value: str | None) -> Path | None:
    if not value:
        return None
    return Path(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--longitudinal-report")
    parser.add_argument("--promotion-ceremony")
    parser.add_argument("--windows-clean-install")
    parser.add_argument("--android-device")
    parser.add_argument("--android-performance")
    parser.add_argument(
        "--android-performance-policy",
        default="config/v1-android-performance-policy.json",
    )
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    receipt = evaluate_v1_closure(
        longitudinal_report_path=maybe_path(args.longitudinal_report),
        promotion_ceremony_path=maybe_path(args.promotion_ceremony),
        windows_receipt_path=maybe_path(args.windows_clean_install),
        android_device_receipt_path=maybe_path(args.android_device),
        android_performance_receipt_path=maybe_path(args.android_performance),
        android_performance_policy_path=maybe_path(
            args.android_performance_policy
        ),
        expected_product_version=args.expected_version,
    )
    verify_v1_closure_receipt(receipt)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"] == READY_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
