from __future__ import annotations

import argparse
import json

from nolane_personal.approved_evidence import (
    ApprovedEvidencePolicy,
    build_approved_evidence_pack,
    verify_approved_evidence_pack,
)
from nolane_personal.personal_protocol import PersonalizationSplitPolicy


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    p.add_argument("--output-dir",default="runtime-data/approved-evidence")
    p.add_argument("--languages",default="vi,en")
    p.add_argument("--min-approved",type=int,default=7)
    p.add_argument("--train-fraction",type=float,default=0.70)
    p.add_argument("--dev-fraction",type=float,default=0.15)
    a=p.parse_args()

    languages=tuple(x.strip().lower() for x in a.languages.split(",") if x.strip())
    result=build_approved_evidence_pack(
        a.source,
        a.output_dir,
        policy=ApprovedEvidencePolicy(
            allowed_languages=languages,
            min_approved_examples=a.min_approved,
        ),
        split_policy=PersonalizationSplitPolicy(
            train_fraction=a.train_fraction,
            dev_fraction=a.dev_fraction,
            min_examples=max(7,a.min_approved),
        ),
    )
    verify_approved_evidence_pack(result.manifest_path)
    print(json.dumps({
        "status":"APPROVED_EVIDENCE_PACK_READY",
        "authority":"USER_APPROVED_LOCAL_EVIDENCE_UNPROMOTED",
        "manifest":result.manifest,
        "dataset":str(result.dataset_path),
        "protocol":str(result.protocol_path),
    },ensure_ascii=False,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
