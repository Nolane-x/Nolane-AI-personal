from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .review_queue import sha256_file, verify_review_queue
from .store import canonical_json, payload_digest


SCHEMA = "NOLANE-L26-INTERACTIVE-LOCAL-REVIEW-V1"
MAX_CORRECTED_TARGET_CHARS = 8000


@dataclass(slots=True)
class ReviewProgress:
    total: int
    decided: int
    approved_non_sensitive: int
    rejected: int
    sensitive: int
    remaining: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _normalize_language(value: Any) -> str | None:
    if value is None:
        return None
    language = str(value).strip().lower()
    if not language:
        return None
    if language not in {"vi", "en"}:
        raise ValueError("language must be vi or en")
    return language


def _validate_decision_payload(payload: dict[str, Any], *, source: str) -> dict[str, Any]:
    candidate_id = str(payload.get("candidate_id", "")).strip()
    if not candidate_id:
        raise ValueError(f"{source}: candidate_id missing")
    if not isinstance(payload.get("approved"), bool):
        raise ValueError(f"{source}: approved must be explicit boolean")
    if not isinstance(payload.get("sensitive"), bool):
        raise ValueError(f"{source}: sensitive must be explicit boolean")
    language = _normalize_language(payload.get("language"))
    weight = float(payload.get("weight", 1.0))
    if not 0.25 <= weight <= 4.0:
        raise ValueError(f"{source}: weight outside allowed range")
    if payload["approved"] and not payload["sensitive"] and language not in {"vi", "en"}:
        raise ValueError(f"{source}: approved non-sensitive decision requires vi/en language")

    corrected_raw = payload.get("corrected_target")
    corrected_target: str | None = None
    if corrected_raw is not None:
        corrected_target = str(corrected_raw).strip()
        if not corrected_target:
            raise ValueError(f"{source}: corrected_target cannot be empty")
        if len(corrected_target) > MAX_CORRECTED_TARGET_CHARS:
            raise ValueError(f"{source}: corrected_target exceeds character limit")
        if not payload["approved"] or payload["sensitive"]:
            raise ValueError(
                f"{source}: corrected_target is only valid for approved non-sensitive decisions"
            )

    return {
        "candidate_id": candidate_id,
        "approved": bool(payload["approved"]),
        "sensitive": bool(payload["sensitive"]),
        "language": language,
        "weight": weight,
        "corrected_target": corrected_target,
    }


def load_review_decisions(
    path: str | Path,
    *,
    known_candidate_ids: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return {}
    decisions: dict[str, dict[str, Any]] = {}
    with source.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{lineno}: invalid JSON") from exc
            decision = _validate_decision_payload(raw, source=f"{source}:{lineno}")
            candidate_id = decision["candidate_id"]
            if candidate_id in decisions:
                raise ValueError(f"{source}:{lineno}: duplicate candidate decision")
            if known_candidate_ids is not None and candidate_id not in known_candidate_ids:
                raise ValueError(f"{source}:{lineno}: unknown candidate_id")
            decisions[candidate_id] = decision
    return decisions


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    )
    tmp = Path(handle.name)
    try:
        with handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


class LocalReviewSession:
    """Local-only explicit review state.

    No model or network dependency exists in this module. Importing or opening
    a queue never creates approval. Existing decisions are immutable through
    this interface; a reviewer must deliberately start a new decision file to
    reconsider prior decisions.
    """

    def __init__(
        self,
        queue_manifest_path: str | Path,
        decisions_path: str | Path,
        progress_manifest_path: str | Path,
    ) -> None:
        self.queue_manifest_path = Path(queue_manifest_path)
        self.decisions_path = Path(decisions_path)
        self.progress_manifest_path = Path(progress_manifest_path)
        self.queue_manifest, self.candidates = verify_review_queue(
            self.queue_manifest_path
        )
        self.by_id = {
            str(candidate["candidate_id"]): candidate
            for candidate in self.candidates
        }
        self.decisions = load_review_decisions(
            self.decisions_path,
            known_candidate_ids=set(self.by_id),
        )
        self._write_progress_manifest()

    def pending_candidates(self) -> list[dict[str, Any]]:
        return [
            candidate
            for candidate in self.candidates
            if str(candidate["candidate_id"]) not in self.decisions
        ]

    def progress(self) -> ReviewProgress:
        approved = rejected = sensitive = 0
        for decision in self.decisions.values():
            if decision["approved"] and not decision["sensitive"]:
                approved += 1
            else:
                rejected += 1
            if decision["sensitive"]:
                sensitive += 1
        decided = len(self.decisions)
        return ReviewProgress(
            total=len(self.candidates),
            decided=decided,
            approved_non_sensitive=approved,
            rejected=rejected,
            sensitive=sensitive,
            remaining=len(self.candidates) - decided,
        )

    def record_decision(
        self,
        candidate_id: str,
        *,
        approved: bool,
        sensitive: bool,
        language: str | None = None,
        weight: float = 1.0,
        corrected_target: str | None = None,
    ) -> dict[str, Any]:
        candidate_id = str(candidate_id).strip()
        if candidate_id not in self.by_id:
            raise ValueError("unknown candidate_id")
        if candidate_id in self.decisions:
            raise ValueError(
                "candidate already has a frozen decision; use a new decisions file to reconsider"
            )

        candidate = self.by_id[candidate_id]
        inherited_language = candidate.get("language")
        final_language = _normalize_language(
            language if language is not None else inherited_language
        )
        decision = _validate_decision_payload(
            {
                "candidate_id": candidate_id,
                "approved": approved,
                "sensitive": sensitive,
                "language": final_language,
                "weight": weight,
                "corrected_target": corrected_target,
            },
            source="interactive review",
        )
        self.decisions[candidate_id] = decision
        self._persist_decisions()
        self._write_progress_manifest()
        return decision

    def _persist_decisions(self) -> None:
        rows = [
            self.decisions[str(candidate["candidate_id"])]
            for candidate in self.candidates
            if str(candidate["candidate_id"]) in self.decisions
        ]
        _atomic_write_jsonl(self.decisions_path, rows)

    def _progress_manifest(self) -> dict[str, Any]:
        progress = self.progress()
        decisions_sha = (
            sha256_file(self.decisions_path)
            if self.decisions_path.exists()
            else None
        )
        manifest = {
            "schema": SCHEMA,
            "authority": "HUMAN_REVIEW_SESSION_NO_AUTO_APPROVAL",
            "queue_manifest_sha256": str(
                self.queue_manifest["manifest_sha256"]
            ),
            "queue_sha256": str(self.queue_manifest["queue_sha256"]),
            "decisions_sha256": decisions_sha,
            "progress": progress.to_dict(),
            "privacy": {
                "manifest_contains_raw_prompt_target": False,
                "manifest_contains_candidate_ids": False,
                "review_happens_locally": True,
                "decision_file_may_contain_user_corrected_target": True,
            },
        }
        manifest["manifest_sha256"] = payload_digest(manifest)
        return manifest

    def _write_progress_manifest(self) -> None:
        self.progress_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.progress_manifest_path.write_text(
            canonical_json(self._progress_manifest()) + "\n",
            encoding="utf-8",
        )

    def verify_progress_manifest(self) -> dict[str, Any]:
        manifest = json.loads(
            self.progress_manifest_path.read_text(encoding="utf-8")
        )
        if manifest.get("schema") != SCHEMA:
            raise ValueError("unsupported interactive review manifest schema")
        supplied = manifest.get("manifest_sha256")
        body = dict(manifest)
        body.pop("manifest_sha256", None)
        if payload_digest(body) != supplied:
            raise ValueError("interactive review manifest digest mismatch")
        if str(manifest.get("queue_manifest_sha256")) != str(
            self.queue_manifest["manifest_sha256"]
        ):
            raise ValueError("interactive review queue lineage mismatch")
        decisions_sha = (
            sha256_file(self.decisions_path)
            if self.decisions_path.exists()
            else None
        )
        if manifest.get("decisions_sha256") != decisions_sha:
            raise ValueError("interactive review decisions digest mismatch")
        if manifest.get("progress") != self.progress().to_dict():
            raise ValueError("interactive review progress mismatch")
        return manifest
