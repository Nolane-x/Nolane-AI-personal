from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from nolane_personal.promotion_ceremony import (
    verify_promotion_ceremony_receipt,
)
from nolane_personal.v1_closure import build_android_device_receipt


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apk", required=True)
    parser.add_argument("--ceremony", required=True)
    parser.add_argument("--observation", required=True)
    parser.add_argument("--product-version", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ceremony_path = Path(args.ceremony)
    ceremony = json.loads(ceremony_path.read_text(encoding="utf-8"))
    verify_promotion_ceremony_receipt(ceremony, require_complete=True)

    observation = json.loads(
        Path(args.observation).read_text(encoding="utf-8")
    )
    fingerprint = str(observation.get("device_fingerprint", ""))
    if not fingerprint:
        raise SystemExit("device_fingerprint is required in observation")
    fingerprint_sha256 = hashlib.sha256(
        fingerprint.encode("utf-8")
    ).hexdigest()

    receipt = build_android_device_receipt(
        evidence_class=str(
            observation.get("evidence_class", "REAL_PHYSICAL_DEVICE")
        ),
        product_version=args.product_version,
        apk_sha256=sha256_file(Path(args.apk)),
        candidate_checkpoint_sha256=ceremony[
            "candidate_checkpoint_sha256"
        ],
        promotion_ceremony_sha256=ceremony["ceremony_sha256"],
        device_fingerprint_sha256=fingerprint_sha256,
        android_sdk=int(observation["android_sdk"]),
        physical_device=bool(observation["physical_device"]),
        installed_version_pass=bool(
            observation["installed_version_pass"]
        ),
        first_boot_pass=bool(observation["first_boot_pass"]),
        local_chat_pass=bool(observation["local_chat_pass"]),
        force_stop_restart_pass=bool(
            observation["force_stop_restart_pass"]
        ),
        identity_continuity_pass=bool(
            observation["identity_continuity_pass"]
        ),
        history_continuity_pass=bool(
            observation["history_continuity_pass"]
        ),
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
