from __future__ import annotations

import json

from nolane_personal.evidence_chain import (
    REAL_CANDIDATE_STAGE_ORDER,
    evidence_chain_contract_sha256,
)


def main() -> int:
    stages = list(REAL_CANDIDATE_STAGE_ORDER)
    evidence = {
        "schema": "NOLANE-L22-EVIDENCE-CHAIN-CONTRACT-AUDIT-V1",
        "stage_count": len(stages),
        "stages": stages,
        "contract_sha256": evidence_chain_contract_sha256(),
    }
    print(json.dumps(evidence, indent=2, sort_keys=True))
    if len(stages) != 17:
        raise SystemExit("real candidate evidence chain must contain exactly 17 stages")
    if len(set(stages)) != len(stages):
        raise SystemExit("real candidate evidence chain contains duplicate stage names")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
