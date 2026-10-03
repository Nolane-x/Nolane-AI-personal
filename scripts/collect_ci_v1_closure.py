from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from nolane_personal.v1_closure import (
    CI_READY_STATUS,
    CI_REQUIRED_PRODUCT_JOBS,
    CI_REQUIRED_WORKFLOWS,
    build_ci_v1_closure_receipt,
    verify_ci_v1_closure_receipt,
)


API_ROOT = "https://api.github.com"


def github_get(path: str, token: str) -> Any:
    request = urllib.request.Request(
        API_ROOT + path,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "nolane-v1-ci-closure",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def choose_run(
    runs: list[dict[str, Any]],
    *,
    name: str,
    sha: str,
) -> dict[str, Any] | None:
    candidates = [
        run
        for run in runs
        if run.get("name") == name
        and run.get("head_sha") == sha
        and run.get("event") == "push"
        and run.get("head_branch") == "main"
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda run: (
            str(run.get("created_at", "")),
            int(run.get("id", 0)),
        ),
        reverse=True,
    )
    return candidates[0]


def fetch_runs(repository: str, sha: str, token: str) -> list[dict[str, Any]]:
    encoded = urllib.parse.quote(sha, safe="")
    payload = github_get(
        f"/repos/{repository}/actions/runs?head_sha={encoded}&per_page=100",
        token,
    )
    return list(payload.get("workflow_runs", []))


def fetch_jobs(
    repository: str,
    run_id: int,
    token: str,
) -> list[dict[str, Any]]:
    payload = github_get(
        f"/repos/{repository}/actions/runs/{run_id}/jobs?per_page=100",
        token,
    )
    return list(payload.get("jobs", []))


def collect(
    *,
    repository: str,
    sha: str,
    product_run_id: int | None,
    token: str,
    wait_seconds: int,
    poll_seconds: int,
) -> tuple[dict[str, str], dict[str, str], int]:
    deadline = time.monotonic() + max(0, wait_seconds)
    resolved_product_run_id = product_run_id

    while True:
        runs = fetch_runs(repository, sha, token)
        selected: dict[str, dict[str, Any] | None] = {
            name: choose_run(runs, name=name, sha=sha)
            for name in CI_REQUIRED_WORKFLOWS
        }

        if resolved_product_run_id is None:
            product = selected.get("Product Client Court")
            if product is not None:
                resolved_product_run_id = int(product["id"])

        workflows: dict[str, str] = {}
        pending = False
        for name in CI_REQUIRED_WORKFLOWS:
            run = selected.get(name)
            if run is None:
                workflows[name] = "missing"
                pending = True
                continue
            status = str(run.get("status", "missing"))
            conclusion = run.get("conclusion")
            if status != "completed":
                workflows[name] = status
                pending = True
            else:
                workflows[name] = str(conclusion or "missing")

        product_jobs: dict[str, str] = {
            name: "missing" for name in CI_REQUIRED_PRODUCT_JOBS
        }
        if resolved_product_run_id is None:
            pending = True
        else:
            jobs = fetch_jobs(
                repository,
                resolved_product_run_id,
                token,
            )
            by_name = {str(job.get("name")): job for job in jobs}
            for name in CI_REQUIRED_PRODUCT_JOBS:
                job = by_name.get(name)
                if job is None:
                    pending = True
                    continue
                status = str(job.get("status", "missing"))
                if status != "completed":
                    product_jobs[name] = status
                    pending = True
                else:
                    product_jobs[name] = str(
                        job.get("conclusion") or "missing"
                    )

        if not pending or time.monotonic() >= deadline:
            if resolved_product_run_id is None:
                resolved_product_run_id = 0
            return workflows, product_jobs, resolved_product_run_id

        time.sleep(max(1, poll_seconds))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--product-run-id", type=int)
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN"))
    parser.add_argument("--product-version", required=True)
    parser.add_argument("--wait-seconds", type=int, default=1200)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not args.token:
        raise SystemExit("GitHub token is required")

    workflows, jobs, product_run_id = collect(
        repository=args.repository,
        sha=args.sha.lower(),
        product_run_id=args.product_run_id,
        token=args.token,
        wait_seconds=args.wait_seconds,
        poll_seconds=args.poll_seconds,
    )

    receipt = build_ci_v1_closure_receipt(
        repository=args.repository,
        branch="main",
        commit_sha=args.sha.lower(),
        product_version=args.product_version,
        workflows=workflows,
        product_jobs=jobs,
    )
    receipt["product_client_run_id"] = product_run_id

    # Recompute the self-digest after attaching the run identifier.
    from nolane_personal.store import payload_digest

    receipt.pop("closure_sha256", None)
    receipt["closure_sha256"] = payload_digest(receipt)
    verify_ci_v1_closure_receipt(receipt)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"] == CI_READY_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
