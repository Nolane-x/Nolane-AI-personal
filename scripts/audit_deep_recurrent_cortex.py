from __future__ import annotations

import json

from nolane_personal.deep_recurrent_cortex import deep_recurrent_cortex_audit


def main() -> int:
    evidence = deep_recurrent_cortex_audit(1024)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    if evidence["parameter_count"] > 100_000:
        raise SystemExit("deep recurrent cortex exceeds 100K parameter cap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
