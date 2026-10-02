from __future__ import annotations

import argparse
import json
from pathlib import Path

from nolane_personal.evidence_quality import assess_evidence_quality_files


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",required=True)
    parser.add_argument("--protocol",required=True)
    parser.add_argument("--output",default=None)
    args=parser.parse_args()

    receipt=assess_evidence_quality_files(args.dataset,args.protocol)
    rendered=json.dumps(receipt,ensure_ascii=False,indent=2,sort_keys=True)

    if args.output:
        path=Path(args.output)
        if path.exists():
            raise SystemExit(f"refusing to overwrite evidence-quality receipt: {path}")
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(rendered+"\n",encoding="utf-8")

    print(rendered)
    return 0 if receipt["status"]=="PASS" else 2


if __name__=="__main__":
    raise SystemExit(main())
