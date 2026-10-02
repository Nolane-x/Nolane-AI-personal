from __future__ import annotations

import argparse
import json

from nolane_personal.evidence_workspace import (
    prepare_evidence_workspace,
    verify_evidence_workspace,
)


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--evidence-pack",required=True)
    p.add_argument("--anchor",default="research/personalization-general-anchor.jsonl")
    p.add_argument("--latent",default="runtime-data/living-core-shadow/latent.json")
    p.add_argument("--model-lock",default="model.lock.json")
    p.add_argument("--model",default="models/Qwen3-0.6B")
    p.add_argument("--l14-anchor",required=True)
    p.add_argument("--output-dir",default="runtime-data/local-evidence-workspace")
    a=p.parse_args()

    result=prepare_evidence_workspace(
        evidence_pack_manifest=a.evidence_pack,
        anchor=a.anchor,
        latent=a.latent,
        model_lock=a.model_lock,
        model_dir=a.model,
        l14_anchor=a.l14_anchor,
        output_dir=a.output_dir,
    )
    verified=verify_evidence_workspace(result["spec_path"])
    print(json.dumps({
        "status":verified["status"],
        "authority":verified["authority"],
        "workspace_sha256":verified["workspace_sha256"],
        "spec":str(result["spec_path"]),
        "readiness":verified["readiness"],
    },indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
