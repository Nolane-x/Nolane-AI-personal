from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from nolane_personal.quantized_promotion import decide_quantized_promotion
from nolane_personal.heldout_group_robustness import verify_group_robustness_digest


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--quality",required=True)
    p.add_argument("--resources",required=True)
    p.add_argument("--output",default=None)
    a=p.parse_args()
    q=json.loads(Path(a.quality).read_text(encoding="utf-8"))
    r=json.loads(Path(a.resources).read_text(encoding="utf-8"))
    group=q.get("group_robustness")
    try:
        verify_group_robustness_digest(group if isinstance(group,dict) else {})
        group_pass=group.get("status")=="PASS"
    except ValueError:
        group_pass=False
    effective_quality_status=q.get("decision",{}).get("status","") if group_pass else ""
    d=decide_quantized_promotion(
        quality_status=effective_quality_status,
        quality_checkpoint_sha256=q.get("quantized_checkpoint_sha256"),
        quality_source_factorized_checkpoint_sha256=q.get("source_factorized_checkpoint_sha256"),
        quality_source_l16_checkpoint_sha256=q.get("source_l16_checkpoint_sha256"),
        resource_status=r.get("decision",{}).get("status",""),
        resource_checkpoint_sha256=r.get("quantized_checkpoint_sha256"),
        resource_source_factorized_checkpoint_sha256=r.get("source_factorized_checkpoint_sha256"),
        resource_source_l16_checkpoint_sha256=r.get("source_l16_checkpoint_sha256"),
    )
    result={"schema":"NOLANE-L19-QUANTIZED-PROMOTION-V1","group_robustness_status":group.get("status") if isinstance(group,dict) else None,"group_robustness_court_sha256":group.get("court_sha256") if isinstance(group,dict) else None,"decision":asdict(d)}
    rendered=json.dumps(result,indent=2,sort_keys=True)
    if a.output:
        path=Path(a.output)
        if path.exists():
            raise SystemExit(f"refusing to overwrite promotion decision: {path}")
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(rendered+"\n",encoding="utf-8")
    print(rendered)
    return 0 if d.status=="QUANTIZED_FACTOR_PROMOTION_PASS" else 2


if __name__=="__main__":
    raise SystemExit(main())
