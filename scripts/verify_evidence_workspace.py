from __future__ import annotations

import argparse
import json

from nolane_personal.evidence_workspace import verify_evidence_workspace


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument(
        "--spec",
        default="runtime-data/local-evidence-workspace/evidence-workspace.json",
    )
    a=p.parse_args()
    spec=verify_evidence_workspace(a.spec)
    print(json.dumps({
        "status":"LOCAL_EVIDENCE_WORKSPACE_VERIFIED",
        "workspace_sha256":spec["workspace_sha256"],
        "evidence_manifest_sha256":spec["binding"]["evidence_manifest_sha256"],
        "latent_digest":spec["binding"]["latent_digest"],
        "requested_model_revision":spec["binding"]["requested_model_revision"],
        "resolved_model_revision":spec["binding"]["resolved_model_revision"],
        "readiness_status":spec["readiness"]["status"],
    },indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
