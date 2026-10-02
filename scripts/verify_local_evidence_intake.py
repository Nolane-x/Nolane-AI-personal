from __future__ import annotations

import argparse
import json

from nolane_personal.local_evidence_intake import verify_local_evidence_intake


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--queue-manifest",required=True)
    parser.add_argument("--decisions",required=True)
    args=parser.parse_args()

    manifest=verify_local_evidence_intake(
        args.manifest,
        queue_manifest_path=args.queue_manifest,
        decisions_path=args.decisions,
    )
    print(json.dumps({
        "status":"LOCAL_EVIDENCE_INTAKE_VERIFIED",
        "authority":manifest["authority"],
        "manifest_sha256":manifest["manifest_sha256"],
        "approved_manifest_sha256":manifest["approved_manifest_sha256"],
        "dataset_sha256":manifest["dataset_sha256"],
        "protocol_sha256":manifest["protocol_sha256"],
        "stats":manifest["stats"],
    },ensure_ascii=False,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
