from __future__ import annotations

import argparse
import json

from nolane_personal.store import LivingStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="runtime-data/living.db")
    args = parser.parse_args()

    store = LivingStore(args.db)
    try:
        state = store.load_state()
        if state is None:
            raise SystemExit("runtime is not initialized")
        memories = store.memories(limit=100000)
        links = store.memory_links(limit=100000)
        memory_ids = {m.memory_id for m in memories}
        broken = [
            link
            for link in links
            if link["parent_memory_id"] not in memory_ids or link["child_memory_id"] not in memory_ids
        ]
        consolidated_children = {link["child_memory_id"] for link in links}
        result = {
            "schema": "NOLANE-REST-AUDIT-V1",
            "identity_id": state.identity_id,
            "state_schema_version": state.schema_version,
            "rest_cycles": state.rest.cycles,
            "last_rest_at": state.rest.last_cycle_at,
            "memory_count": len(memories),
            "memory_link_count": len(links),
            "consolidated_memory_count": len(consolidated_children),
            "broken_memory_links": len(broken),
            "status": "PASS" if not broken else "FAIL",
        }
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if not broken else 2
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
