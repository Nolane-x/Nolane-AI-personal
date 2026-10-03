from __future__ import annotations

import argparse
import json
from pathlib import Path

from nolane_personal.promotion_ceremony import (
    AUTHORITY,
    SCHEMA,
    verify_promotion_ceremony_receipt,
)
from nolane_personal.store import canonical_json, payload_digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mobile-package-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    package = Path(args.mobile_package_dir)
    manifest = json.loads(
        (package / "manifest.json").read_text(encoding="utf-8")
    )
    checkpoint = str(manifest["source_checkpoint_sha256"])
    at = "2026-01-01T00:00:00+00:00"
    body = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "status": "COMPLETE",
        "reasons": [],
        "authorization_sha256": "1" * 64,
        "multicycle_chain_sha256": "2" * 64,
        "long_horizon_retention_court_sha256": "3" * 64,
        "candidate_checkpoint_sha256": checkpoint,
        "pointer_sha256": "4" * 64,
        "serving_convergence_sha256": "5" * 64,
        "transaction_id": "synthetic-mobile-release-court",
        "pointer_generation": 1,
        "authorization_issued_at": at,
        "authorization_expires_at": "2027-01-01T00:00:00+00:00",
        "transaction_prepared_at": at,
        "transaction_committed_at": at,
        "pointer_created_at": at,
        "serving_convergence_assessed_at": at,
        "ceremony_at": at,
    }
    body["ceremony_sha256"] = payload_digest(body)
    verify_promotion_ceremony_receipt(body, require_complete=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        canonical_json(body) + "\n",
        encoding="utf-8",
    )
    print(body["ceremony_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
