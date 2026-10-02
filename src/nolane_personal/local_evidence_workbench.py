from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .approved_evidence import resolve_approved_evidence_pack
from .interactive_review import (
    LocalReviewSession,
    ReviewProgress,
    load_review_decisions,
)
from .local_evidence_intake import (
    finalize_local_evidence_intake,
    verify_local_evidence_intake,
)
from .real_candidate_readiness import assess_real_candidate_readiness
from .review_queue import build_review_queue, sha256_file, verify_review_queue
from .store import canonical_json, payload_digest


SCHEMA = "NOLANE-L27-LOCAL-EVIDENCE-WORKBENCH-V1"


@dataclass(slots=True)
class WorkbenchPaths:
    root: Path
    queue_dir: Path
    queue_manifest: Path
    decisions: Path
    review_progress: Path
    intake_dir: Path
    intake_manifest: Path
    approved_manifest: Path
    workbench_manifest: Path


def paths_for(root: str | Path) -> WorkbenchPaths:
    root = Path(root)
    queue_dir = root / "review-queue"
    intake_dir = root / "intake"
    return WorkbenchPaths(
        root=root,
        queue_dir=queue_dir,
        queue_manifest=queue_dir / "review-queue-manifest.json",
        decisions=root / "review-decisions.jsonl",
        review_progress=root / "review-progress-manifest.json",
        intake_dir=intake_dir,
        intake_manifest=intake_dir / "local-evidence-intake-manifest.json",
        approved_manifest=(
            intake_dir / "approved-pack" / "approved-evidence-manifest.json"
        ),
        workbench_manifest=root / "workbench-manifest.json",
    )


def _review_progress(
    queue_manifest_path: Path,
    decisions_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], ReviewProgress]:
    queue_manifest, candidates = verify_review_queue(queue_manifest_path)
    known = {str(row["candidate_id"]) for row in candidates}
    decisions = load_review_decisions(
        decisions_path,
        known_candidate_ids=known,
    )
    approved = rejected = sensitive = 0
    for decision in decisions.values():
        if decision["approved"] and not decision["sensitive"]:
            approved += 1
        else:
            rejected += 1
        if decision["sensitive"]:
            sensitive += 1
    progress = ReviewProgress(
        total=len(candidates),
        decided=len(decisions),
        approved_non_sensitive=approved,
        rejected=rejected,
        sensitive=sensitive,
        remaining=len(candidates)-len(decisions),
    )
    return queue_manifest, candidates, progress


def _manifest_digest(payload: dict[str, Any]) -> str:
    body = dict(payload)
    body.pop("manifest_sha256", None)
    return payload_digest(body)


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    payload["manifest_sha256"] = _manifest_digest(payload)
    path.write_text(canonical_json(payload)+"\n", encoding="utf-8")


def initialize_workbench(
    source_export: str | Path,
    root: str | Path,
) -> dict[str, Any]:
    paths = paths_for(root)
    if paths.root.exists() and any(paths.root.iterdir()):
        raise FileExistsError(
            f"refusing to initialize non-empty workbench: {paths.root}"
        )
    paths.root.mkdir(parents=True, exist_ok=True)
    queue = build_review_queue(source_export, paths.queue_dir)
    LocalReviewSession(
        queue["manifest_path"],
        paths.decisions,
        paths.review_progress,
    )
    status = workbench_status(paths.root)
    return status


def workbench_status(root: str | Path) -> dict[str, Any]:
    paths = paths_for(root)
    if not paths.queue_manifest.exists():
        raise ValueError("workbench review queue missing")

    queue_manifest, candidates, progress = _review_progress(
        paths.queue_manifest,
        paths.decisions,
    )
    decisions_sha = (
        sha256_file(paths.decisions)
        if paths.decisions.exists()
        else None
    )

    intake_ready = False
    intake_sha = None
    approved_sha = None
    dataset_sha = None
    protocol_sha = None
    if paths.intake_manifest.exists():
        if not paths.decisions.exists():
            raise ValueError("intake exists without review decisions")
        intake = verify_local_evidence_intake(
            paths.intake_manifest,
            queue_manifest_path=paths.queue_manifest,
            decisions_path=paths.decisions,
        )
        intake_ready = True
        intake_sha = str(intake["manifest_sha256"])
        approved_sha = str(intake["approved_manifest_sha256"])
        dataset_sha = str(intake["dataset_sha256"])
        protocol_sha = str(intake["protocol_sha256"])

    if intake_ready:
        phase = "INTAKE_READY"
    elif progress.remaining == 0:
        phase = "REVIEW_COMPLETE"
    elif progress.decided > 0:
        phase = "REVIEW_IN_PROGRESS"
    else:
        phase = "QUEUE_READY"

    manifest = {
        "schema": SCHEMA,
        "authority": "LOCAL_WORKBENCH_NO_AUTO_APPROVAL_NO_MODEL_AUTHORITY",
        "phase": phase,
        "privacy": {
            "manifest_contains_raw_prompt_target": False,
            "manifest_contains_candidate_ids": False,
            "local_only_recommended": True,
        },
        "source_sha256": str(queue_manifest["source_sha256"]),
        "queue_manifest_sha256": str(queue_manifest["manifest_sha256"]),
        "queue_sha256": str(queue_manifest["queue_sha256"]),
        "decisions_sha256": decisions_sha,
        "review_progress": progress.to_dict(),
        "intake_ready": intake_ready,
        "intake_manifest_sha256": intake_sha,
        "approved_manifest_sha256": approved_sha,
        "dataset_sha256": dataset_sha,
        "protocol_sha256": protocol_sha,
        "paths": {
            "queue_manifest": str(paths.queue_manifest.relative_to(paths.root)),
            "decisions": str(paths.decisions.relative_to(paths.root)),
            "review_progress": str(paths.review_progress.relative_to(paths.root)),
            "intake_manifest": str(paths.intake_manifest.relative_to(paths.root)),
            "approved_manifest": str(paths.approved_manifest.relative_to(paths.root)),
        },
    }
    _write_manifest(paths.workbench_manifest, manifest)
    return manifest


def finalize_workbench(
    root: str | Path,
    *,
    allow_undecided: bool = False,
) -> dict[str, Any]:
    paths = paths_for(root)
    _queue_manifest, _candidates, progress = _review_progress(
        paths.queue_manifest,
        paths.decisions,
    )
    if not paths.decisions.exists():
        raise ValueError("cannot finalize without explicit review decisions")
    if progress.remaining and not allow_undecided:
        raise ValueError(
            "cannot finalize while review candidates remain undecided"
        )
    if progress.approved_non_sensitive < 7:
        raise ValueError(
            "cannot finalize: need at least 7 approved non-sensitive examples"
        )
    if paths.intake_dir.exists() and any(paths.intake_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite existing intake: {paths.intake_dir}"
        )
    finalize_local_evidence_intake(
        paths.queue_manifest,
        paths.decisions,
        paths.intake_dir,
    )
    return workbench_status(paths.root)


def assess_workbench_readiness(
    root: str | Path,
    *,
    anchor: str | Path,
    latent: str | Path,
    model_lock: str | Path,
    model_dir: str | Path,
    l14_anchor: str | Path,
) -> dict[str, Any]:
    paths = paths_for(root)
    status = workbench_status(paths.root)
    if status["phase"] != "INTAKE_READY":
        raise ValueError("workbench intake is not finalized")
    approved_manifest, dataset, protocol = resolve_approved_evidence_pack(
        paths.approved_manifest
    )
    decision = assess_real_candidate_readiness(
        dataset=dataset,
        protocol=protocol,
        anchor=anchor,
        latent=latent,
        model_lock=model_lock,
        model_dir=model_dir,
        l14_anchor=l14_anchor,
    )
    result = {
        "schema": "NOLANE-L27-WORKBENCH-READINESS-V1",
        "authority": "READINESS_ONLY_NO_TRAINING_AUTHORITY",
        "workbench_manifest_sha256": status["manifest_sha256"],
        "approved_evidence_manifest_sha256": approved_manifest["manifest_sha256"],
        "readiness": asdict(decision),
    }
    result["receipt_sha256"] = payload_digest(result)
    return result


def verify_workbench_manifest(root: str | Path) -> dict[str, Any]:
    paths = paths_for(root)
    stored = json.loads(
        paths.workbench_manifest.read_text(encoding="utf-8")
    )
    if stored.get("schema") != SCHEMA:
        raise ValueError("unsupported workbench manifest schema")
    if _manifest_digest(stored) != stored.get("manifest_sha256"):
        raise ValueError("workbench manifest digest mismatch")
    current = workbench_status(paths.root)
    if stored != current:
        raise ValueError("workbench manifest does not match current local state")
    return stored
