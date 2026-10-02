from __future__ import annotations

import argparse
import json

from nolane_personal.review_queue import build_review_queue, verify_review_queue


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    p.add_argument("--output-dir",default="runtime-data/review-queue")
    p.add_argument("--default-language",choices=["vi","en"],default=None)
    a=p.parse_args()
    result=build_review_queue(
        a.source,
        a.output_dir,
        default_language=a.default_language,
    )
    verify_review_queue(result["manifest_path"])
    print(json.dumps({
        "status":"LOCAL_REVIEW_QUEUE_READY",
        "authority":"LOCAL_REVIEW_QUEUE_NEVER_APPROVED_BY_IMPORT",
        "manifest":result["manifest"],
        "queue":str(result["queue_path"]),
    },ensure_ascii=False,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
