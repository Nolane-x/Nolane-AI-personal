from __future__ import annotations

import argparse
import json
from pathlib import Path

from nolane_personal.latent import LatentStore
from nolane_personal.state import utc_now_iso


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--latent", default="runtime-data/living-core-shadow/latent.json")
    parser.add_argument("--dataset", default="runtime-data/personalization.jsonl")
    parser.add_argument("--language", default=None)
    parser.add_argument("--weight", type=float, default=1.0)
    args = parser.parse_args()

    if args.weight <= 0:
        raise SystemExit("--weight must be positive")
    latent = LatentStore(args.latent).load()
    if latent is None:
        raise SystemExit(f"persistent latent not found: {args.latent}")

    record = {
        "prompt": args.prompt,
        "target": args.target,
        "latent": latent.values,
        "weight": args.weight,
        "language": args.language,
        "metadata": {
            "captured_at": utc_now_iso(),
            "identity_id": latent.identity_id,
            "latent_digest": latent.digest,
            "source_state_version": latent.source_state_version,
            "latent_sequence": latent.sequence,
            "checkpoint_sha256": latent.checkpoint_sha256,
            "protocol_sha256": latent.protocol_sha256,
        },
    }
    path = Path(args.dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"appended personalization example -> {path}")
    print(f"latent_digest={latent.digest}")
    print(f"source_state_version={latent.source_state_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
