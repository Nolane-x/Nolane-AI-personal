from __future__ import annotations

import argparse
from pathlib import Path

from nolane_personal.interactive_review import LocalReviewSession


def _language_for_approval(candidate: dict) -> str:
    inherited = candidate.get("language")
    if inherited in {"vi", "en"}:
        return str(inherited)
    while True:
        value = input("Language [vi/en]: ").strip().lower()
        if value in {"vi", "en"}:
            return value
        print("Please enter vi or en.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-manifest", required=True)
    parser.add_argument(
        "--decisions",
        default="runtime-data/review-decisions.jsonl",
    )
    parser.add_argument(
        "--progress-manifest",
        default="runtime-data/review-progress-manifest.json",
    )
    args = parser.parse_args()

    session = LocalReviewSession(
        args.queue_manifest,
        args.decisions,
        args.progress_manifest,
    )
    pending = session.pending_candidates()
    if not pending:
        progress = session.progress()
        print(
            f"Review complete: {progress.decided}/{progress.total} decided; "
            f"{progress.approved_non_sensitive} approved non-sensitive."
        )
        return 0

    print(
        "Local reviewer. No model/network calls. "
        "Commands: [a]pprove, [r]eject, [s]ensitive reject, [k]skip, [q]uit."
    )
    for candidate in pending:
        progress = session.progress()
        print("\n" + "=" * 72)
        print(
            f"Progress {progress.decided}/{progress.total} "
            f"(remaining {progress.remaining})"
        )
        print(f"Language: {candidate.get('language') or 'unknown'}")
        print("\nUSER:\n" + str(candidate["prompt"]))
        print("\nASSISTANT:\n" + str(candidate["target"]))

        while True:
            command = input("\nDecision [a/r/s/k/q]: ").strip().lower()
            if command in {"a", "r", "s", "k", "q"}:
                break
            print("Choose a, r, s, k, or q.")

        if command == "q":
            print("Review session stopped. Existing decisions remain saved.")
            return 0
        if command == "k":
            continue
        if command == "a":
            session.record_decision(
                candidate["candidate_id"],
                approved=True,
                sensitive=False,
                language=_language_for_approval(candidate),
            )
        elif command == "r":
            session.record_decision(
                candidate["candidate_id"],
                approved=False,
                sensitive=False,
                language=candidate.get("language"),
            )
        elif command == "s":
            session.record_decision(
                candidate["candidate_id"],
                approved=False,
                sensitive=True,
                language=candidate.get("language"),
            )

    progress = session.progress()
    print(
        f"Review pass finished: {progress.decided}/{progress.total} decided; "
        f"{progress.approved_non_sensitive} approved non-sensitive; "
        f"{progress.remaining} remain."
    )
    print(f"Decisions: {Path(args.decisions)}")
    print(f"Progress manifest: {Path(args.progress_manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
