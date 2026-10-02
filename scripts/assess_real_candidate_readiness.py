from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from nolane_personal.approved_evidence import resolve_approved_evidence_pack
from nolane_personal.real_candidate_readiness import assess_real_candidate_readiness


ROOT=Path(__file__).resolve().parents[1]


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--evidence-pack",default=None)
    p.add_argument("--dataset",default="runtime-data/personalization.jsonl")
    p.add_argument("--protocol",default="runtime-data/personalization-protocol-v1.json")
    p.add_argument("--anchor",default=str(ROOT/"research/personalization-general-anchor.jsonl"))
    p.add_argument("--latent",default="runtime-data/living-core-shadow/latent.json")
    p.add_argument("--model-lock",default=str(ROOT/"model.lock.json"))
    p.add_argument("--model",default=str(ROOT/"models/Qwen3-0.6B"))
    p.add_argument("--l14-anchor",default="runtime-data/l14-minimal-anchor/minimal-qwen-anchor-cortex.pt")
    p.add_argument("--output",default=None)
    a=p.parse_args()

    approved_manifest=None
    if a.evidence_pack:
        approved_manifest,dataset_path,protocol_path=resolve_approved_evidence_pack(a.evidence_pack)
        a.dataset=str(dataset_path)
        a.protocol=str(protocol_path)

    decision=assess_real_candidate_readiness(
        dataset=a.dataset,
        protocol=a.protocol,
        anchor=a.anchor,
        latent=a.latent,
        model_lock=a.model_lock,
        model_dir=a.model,
        l14_anchor=a.l14_anchor,
    )
    result={
        "schema":"NOLANE-L21-REAL-CANDIDATE-READINESS-V1",
        "authority":"READINESS_ONLY_NO_TRAINING_AUTHORITY",
        "approved_evidence_manifest_sha256":(
            approved_manifest.get("manifest_sha256") if approved_manifest else None
        ),
        "decision":asdict(decision),
    }
    rendered=json.dumps(result,indent=2,sort_keys=True)
    if a.output:
        path=Path(a.output)
        if path.exists():
            raise SystemExit(f"refusing to overwrite readiness receipt: {path}")
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(rendered+"\n",encoding="utf-8")
    print(rendered)
    return 0 if decision.status=="REAL_CANDIDATE_INPUTS_READY" else 2


if __name__=="__main__":
    raise SystemExit(main())
