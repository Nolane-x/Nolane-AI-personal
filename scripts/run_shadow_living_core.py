from __future__ import annotations

import argparse
import json
import time

from nolane_personal.shadow import ShadowLivingCoreRunner


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="runtime-data/living.db")
    parser.add_argument("--checkpoint", default="runtime-data/living-core-dev/living-core.pt")
    parser.add_argument("--latent", default="runtime-data/living-core-shadow/latent.json")
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--receipts", default=None)
    parser.add_argument("--reset-latent", action="store_true")
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()

    runner = ShadowLivingCoreRunner(
        args.db,
        args.checkpoint,
        args.latent,
        protocol_path=args.protocol,
        receipt_path=args.receipts,
        reset_latent=args.reset_latent,
    )
    while True:
        result = runner.process_pending()
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        if not args.follow:
            return 0
        time.sleep(max(0.5, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
