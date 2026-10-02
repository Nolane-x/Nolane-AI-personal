from __future__ import annotations

import argparse
import json

from nolane_personal.review_queue import apply_review_decisions


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--queue-manifest",required=True)
    p.add_argument("--decisions",required=True)
    p.add_argument("--output-dir",default="runtime-data/reviewed-evidence")
    a=p.parse_args()
    result=apply_review_decisions(
        a.queue_manifest,
        a.decisions,
        a.output_dir,
    )
    print(json.dumps({
        "status":"REVIEW_DECISIONS_APPLIED",
        "authority":"EXPLICIT_LOCAL_REVIEW_DECISIONS_UNPROMOTED",
        "manifest":result["manifest"],
        "reviewed_source":str(result["reviewed_source_path"]),
    },ensure_ascii=False,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
