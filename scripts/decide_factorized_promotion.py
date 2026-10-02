from __future__ import annotations
import argparse,json
from dataclasses import asdict
from pathlib import Path
from nolane_personal.factorized_promotion import decide_factorized_promotion
from nolane_personal.heldout_group_robustness import effective_quality_status

def main():
    p=argparse.ArgumentParser(); p.add_argument("--quality",required=True); p.add_argument("--resources",required=True); p.add_argument("--output",default=None); a=p.parse_args()
    q=json.loads(Path(a.quality).read_text()); r=json.loads(Path(a.resources).read_text())
    group=q.get("group_robustness")
    effective_status=effective_quality_status(q)
    d=decide_factorized_promotion(
        quality_status=effective_status,quality_checkpoint_sha256=q.get("factorized_checkpoint_sha256"),quality_source_l16_checkpoint_sha256=q.get("source_l16_checkpoint_sha256"),
        resource_status=r.get("decision",{}).get("status",""),resource_checkpoint_sha256=r.get("factorized_checkpoint_sha256"),resource_source_l16_checkpoint_sha256=r.get("source_l16_checkpoint_sha256"),
    )
    result={"schema":"NOLANE-L17-FACTORIZED-BOUNDARY-PROMOTION-V1","group_robustness_status":group.get("status") if isinstance(group,dict) else None,"group_robustness_court_sha256":group.get("court_sha256") if isinstance(group,dict) else None,"decision":asdict(d)}; s=json.dumps(result,indent=2,sort_keys=True)
    if a.output:
        path=Path(a.output)
        if path.exists(): raise SystemExit(f"refusing to overwrite promotion decision: {path}")
        path.parent.mkdir(parents=True,exist_ok=True); path.write_text(s+"\n")
    print(s); return 0 if d.status=="FACTORIZED_BOUNDARY_PROMOTION_PASS" else 2
if __name__=="__main__": raise SystemExit(main())
