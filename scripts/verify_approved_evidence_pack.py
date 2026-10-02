from __future__ import annotations

import argparse
import json

from nolane_personal.approved_evidence import verify_approved_evidence_pack


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",default="runtime-data/approved-evidence/approved-evidence-manifest.json")
    a=p.parse_args()
    manifest=verify_approved_evidence_pack(a.manifest)
    print(json.dumps({
        "status":"APPROVED_EVIDENCE_PACK_VERIFIED",
        "manifest_sha256":manifest["manifest_sha256"],
        "dataset_sha256":manifest["dataset_sha256"],
        "protocol_sha256":manifest["protocol_sha256"],
        "stats":manifest["stats"],
    },indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
