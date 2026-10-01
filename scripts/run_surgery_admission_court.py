from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from nolane_personal.surgery import CounterfactualReceipt
from nolane_personal.surgery_court import evaluate_shadow_admission


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipts", nargs="+", help="JSON files emitted by probe_latent_adapter.py")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rows = []
    for item in args.receipts:
        payload = json.loads(Path(item).read_text(encoding="utf-8"))
        raw = payload.get("receipt", payload)
        rows.append(CounterfactualReceipt(**raw))

    decision = evaluate_shadow_admission(rows)
    rendered = json.dumps(asdict(decision), ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        if path.exists():
            raise SystemExit(f"refusing to overwrite court result: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if decision.status == "SHADOW_ADMISSION_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
