from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--secret-file", required=True)
    parser.add_argument("--output", action="append", required=True)
    parser.add_argument("--fail", action="store_true")
    args = parser.parse_args()

    secret_path = Path(args.secret_file)
    if not secret_path.exists():
        raise SystemExit("synthetic fixture secret file missing")
    secret_sha256 = sha256_file(secret_path)

    if args.fail:
        print(f"synthetic fixture intentionally failed at {args.stage}")
        return 7

    for raw in args.output:
        path = Path(raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "NOLANE-L22-SYNTHETIC-STAGE-ARTIFACT-V1",
            "authority": "SYNTHETIC_NON_AUTHORITY_NEVER_PROMOTABLE",
            "stage": args.stage,
            "secret_sha256": secret_sha256,
        }
        path.write_text(
            json.dumps(payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
