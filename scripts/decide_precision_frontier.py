from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from nolane_personal.precision_frontier import decide_precision_frontier


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--int8-quality",required=True)
    p.add_argument("--int8-resources",required=True)
    p.add_argument("--int4-quality",required=True)
    p.add_argument("--int4-resources",required=True)
    p.add_argument("--output",default=None)
    a=p.parse_args()
    i8q=json.loads(Path(a.int8_quality).read_text(encoding="utf-8"))
    i8r=json.loads(Path(a.int8_resources).read_text(encoding="utf-8"))
    i4q=json.loads(Path(a.int4_quality).read_text(encoding="utf-8"))
    i4r=json.loads(Path(a.int4_resources).read_text(encoding="utf-8"))

    int8_checkpoint=i8q.get("quantized_checkpoint_sha256")
    if int8_checkpoint!=i8r.get("quantized_checkpoint_sha256"):
        raise SystemExit("INT8 quality/resource checkpoint mismatch")
    int8_source=i8q.get("source_factorized_checkpoint_sha256")
    if int8_source!=i8r.get("source_factorized_checkpoint_sha256"):
        raise SystemExit("INT8 quality/resource source mismatch")

    int4_checkpoint=i4q.get("packed_int4_checkpoint_sha256")
    d=decide_precision_frontier(
        int8_quality_status=i8q.get("decision",{}).get("status",""),
        int8_resource_status=i8r.get("decision",{}).get("status",""),
        int8_checkpoint_sha256=int8_checkpoint,
        int8_source_factorized_checkpoint_sha256=int8_source,
        int4_quality_status=i4q.get("decision",{}).get("status",""),
        int4_resource_status=i4r.get("decision",{}).get("status",""),
        int4_checkpoint_sha256=int4_checkpoint,
        int4_source_factorized_checkpoint_sha256=i4q.get("source_factorized_checkpoint_sha256"),
        int4_quality_reference_int8_checkpoint_sha256=i4q.get("reference_int8_checkpoint_sha256"),
        int4_resource_reference_int8_checkpoint_sha256=i4r.get("reference_int8_checkpoint_sha256"),
        int4_resource_checkpoint_sha256=i4r.get("packed_int4_checkpoint_sha256"),
    )
    result={
        "schema":"NOLANE-L20-PRECISION-FRONTIER-DECISION-V1",
        "decision":asdict(d),
    }
    rendered=json.dumps(result,indent=2,sort_keys=True)
    if a.output:
        path=Path(a.output)
        if path.exists():
            raise SystemExit(f"refusing to overwrite precision decision: {path}")
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(rendered+"\n",encoding="utf-8")
    print(rendered)
    return 0 if d.status in {"PRECISION_FRONTIER_SELECT_INT4","PRECISION_FRONTIER_KEEP_INT8"} else 2


if __name__=="__main__":
    raise SystemExit(main())
