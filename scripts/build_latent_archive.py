from __future__ import annotations

import argparse
import json

from nolane_personal.latent_archive import build_latent_archive, write_latent_archive


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="runtime-data/living.db")
    parser.add_argument("--protocol", default="runtime-data/replay-protocol-v1.json")
    parser.add_argument("--living-core", default="runtime-data/living-core-dev/living-core.pt")
    parser.add_argument("--output", default="runtime-data/latent-archive-v1.json")
    args = parser.parse_args()

    archive = build_latent_archive(args.db, args.protocol, args.living_core)
    write_latent_archive(args.output, archive)
    print(json.dumps({
        "schema": archive["schema"],
        "entry_count": archive["entry_count"],
        "latent_dim": archive["latent_dim"],
        "protocol_sha256": archive["protocol_sha256"],
        "archive_sha256": archive["archive_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
