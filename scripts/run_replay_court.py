from __future__ import annotations

import argparse
import json

from nolane_personal.court import ReplayCase, deterministic_predict, evaluate_predictor
from nolane_personal.dynamics import seconds_between
from nolane_personal.store import LivingStore


SUPPORTED = {"user_message", "clock_tick", "assistant_speech"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="runtime-data/living.db")
    parser.add_argument("--limit", type=int, default=10000)
    args = parser.parse_args()

    store = LivingStore(args.db)
    try:
        records = store.replay_records(args.limit)
    finally:
        store.close()

    cases = []
    for record in records:
        event = record["event"]
        if event.kind not in SUPPORTED:
            continue
        before = record["before"]
        after = record["after"]
        dt = seconds_between(before.last_event_at or before.updated_at, event.at)
        cases.append(ReplayCase(before=before, event=event, dt_seconds=dt, target=after))

    metrics = evaluate_predictor(cases, deterministic_predict, mae_gate=1e-9, max_error_gate=1e-9)
    print(json.dumps(asdict(metrics), indent=2, sort_keys=True))
    return 0 if metrics.decision == "COURT_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
