from __future__ import annotations

import argparse
import json

from nolane_personal.learning_campaign import (
    build_real_learning_campaign,
    verify_real_learning_campaign,
)


def main() -> int:
    parser = argparse.ArgumentParser(prog="real-learning-campaign")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("--spec", required=True)
    build.add_argument("--output-dir", required=True)

    verify = sub.add_parser("verify")
    verify.add_argument("--output-dir", required=True)

    args = parser.parse_args()

    if args.command == "build":
        result = build_real_learning_campaign(
            args.spec,
            args.output_dir,
        )
    else:
        result = verify_real_learning_campaign(args.output_dir)

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
