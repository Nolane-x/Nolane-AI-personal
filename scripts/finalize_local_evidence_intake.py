from __future__ import annotations

import argparse
import json

from nolane_personal.local_evidence_intake import finalize_local_evidence_intake


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--queue-manifest",required=True)
    parser.add_argument("--decisions",required=True)
    parser.add_argument("--output-dir",default="runtime-data/local-evidence-intake")
    args=parser.parse_args()

    result=finalize_local_evidence_intake(
        args.queue_manifest,
        args.decisions,
        args.output_dir,
    )
    print(json.dumps({
        "status":"LOCAL_EVIDENCE_INTAKE_FINALIZED",
        "authority":result.manifest["authority"],
        "manifest":result.manifest,
        "approved_evidence_manifest":str(result.approved_manifest_path),
    },ensure_ascii=False,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
