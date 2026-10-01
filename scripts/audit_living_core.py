from __future__ import annotations

import json

from nolane_personal.living_core import LivingCoreConfig, analytical_parameter_count, core_audit


def main() -> int:
    config = LivingCoreConfig()
    print(json.dumps({
        "analytical_parameter_count_without_torch": analytical_parameter_count(config),
        "runtime_audit": core_audit(config),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
