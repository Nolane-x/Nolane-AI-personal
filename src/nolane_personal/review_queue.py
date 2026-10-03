from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .store import canonical_json, payload_digest


QUEUE_SCHEMA = "NOLANE-L24-LOCAL-REVIEW-QUEUE-V1"
DECISION_SCHEMA = "NOLANE-L24-REVIEW-DECISIONS-V1"
FINAL_SCHEMA = "NOLANE-L24-REVIEWED-EVIDENCE-SOURCE-V1"
MAX_CORRECTED_TARGET_CHARS = 8000


@dataclass(slots=True)
class ReviewQueuePolicy:
    max_prompt_chars: int = 8000
    max_target_chars: int = 8000
    allowed_roles: tuple[str, ...] = ("user", "assistant")
    reject_exact_duplicates: bool = True

    def validate(self) -> None:
        if self.max_prompt_chars < 1 or self.max_target_chars < 1:
            raise ValueError("review queue character limits must be positive")
        if tuple(self.allowed_roles) != ("user", "assistant"):
            raise ValueError("review queue roles must be exactly user,assistant")


@dataclass(slots=True)
class ReviewQueueStats:
    conversations: int
    messages_seen: int
    candidates: int
    duplicate_pairs: int
    skipped_empty: int
    skipped_non_dialogue_roles: int


@dataclass(slots=True)
class ReviewApplyStats:
    queue_candidates: int
    decisions_supplied: int
    approved: int
    rejected: int
    sensitive: int
    undecided: int


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_text(value: Any) -> str:
    return unicodedata.normalize("NFC", str(value)).strip()


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return _normalize_text(value)
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                text = _normalize_text(item)
            elif isinstance(item, dict):
                raw = item.get("text", item.get("content", ""))
                text = _normalize_text(raw) if raw is not None else ""
            else:
                text = ""
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    if isinstance(value, dict):
        raw = value.get("text", value.get("content", ""))
        return _normalize_text(raw) if raw is not None else ""
    return ""


def _load_conversation_objects(source: str | Path) -> list[dict[str, Any]]:
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{lineno}: invalid JSON") from exc
                if not isinstance(payload, dict):
                    raise ValueError(f"{path}:{lineno}: expected JSON object")
                rows.append(payload)
        return rows

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("conversations"), list):
        rows = payload["conversations"]
    else:
        raise ValueError(
            "conversation export must be a JSON array, {conversations:[...]}, or JSONL"
        )
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("conversation export contains non-object entries")
    return list(rows)


def _candidate_id(
    *,
    source_sha256: str,
    conversation_index: int,
    pair_index: int,
    prompt: str,
    target: str,
) -> str:
    return payload_digest({
        "source_sha256": source_sha256,
        "conversation_index": int(conversation_index),
        "pair_index": int(pair_index),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "target_sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
    })


def build_review_queue(
    source: str | Path,
    output_dir: str | Path,
    *,
    policy: ReviewQueuePolicy | None = None,
    default_language: str | None = None,
) -> dict[str, Any]:
    policy = policy or ReviewQueuePolicy()
    policy.validate()
    if default_language is not None:
        default_language = _normalize_text(default_language).lower() or None
        if default_language not in {"vi", "en"}:
            raise ValueError("default_language must be vi or en")

    source_path = Path(source)
    source_sha = sha256_file(source_path)
    conversations = _load_conversation_objects(source_path)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty review queue directory: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)

    queue_path = output / "review-queue.jsonl"
    manifest_path = output / "review-queue-manifest.json"
    candidates: list[dict[str, Any]] = []
    pair_hashes: set[str] = set()
    messages_seen = 0
    duplicate_pairs = 0
    skipped_empty = 0
    skipped_non_dialogue = 0

    for conversation_index, conversation in enumerate(conversations):
        messages = conversation.get("messages")
        if not isinstance(messages, list):
            raise ValueError(
                f"conversation {conversation_index}: messages must be a list"
            )
        conversation_id = conversation.get(
            "conversation_id",
            conversation.get("id", f"conversation-{conversation_index}"),
        )
        language = conversation.get("language", default_language)
        if language is not None:
            language = _normalize_text(language).lower() or None
            if language not in {"vi", "en"}:
                language = None

        pending_user: str | None = None
        pair_index = 0
        for message in messages:
            messages_seen += 1
            if not isinstance(message, dict):
                raise ValueError(
                    f"conversation {conversation_index}: message must be object"
                )
            role = _normalize_text(message.get("role", "")).lower()
            if role not in policy.allowed_roles:
                skipped_non_dialogue += 1
                continue
            text = _content_text(message.get("content", ""))
            if not text:
                skipped_empty += 1
                continue

            if role == "user":
                pending_user = text
                continue

            if pending_user is None:
                continue

            prompt = pending_user
            target = text
            pending_user = None
            if len(prompt) > policy.max_prompt_chars:
                raise ValueError(
                    f"conversation {conversation_index}: prompt exceeds character limit"
                )
            if len(target) > policy.max_target_chars:
                raise ValueError(
                    f"conversation {conversation_index}: target exceeds character limit"
                )

            pair_sha = payload_digest({"prompt": prompt, "target": target})
            if pair_sha in pair_hashes:
                duplicate_pairs += 1
                if policy.reject_exact_duplicates:
                    raise ValueError(
                        f"conversation {conversation_index}: duplicate prompt/target pair"
                    )
                continue
            pair_hashes.add(pair_sha)

            candidate_id = _candidate_id(
                source_sha256=source_sha,
                conversation_index=conversation_index,
                pair_index=pair_index,
                prompt=prompt,
                target=target,
            )
            source_id = (
                f"{conversation_id}:pair:{pair_index}"
                if conversation_id is not None
                else f"conversation-{conversation_index}:pair:{pair_index}"
            )
            candidates.append({
                "schema": "NOLANE-L24-REVIEW-CANDIDATE-V1",
                "candidate_id": candidate_id,
                "prompt": prompt,
                "target": target,
                "language": language,
                "weight": 1.0,
                "approved": False,
                "sensitive": None,
                "reviewed": False,
                "source_id": str(source_id),
            })
            pair_index += 1

    if not candidates:
        raise ValueError("conversation export produced no review candidates")

    with queue_path.open("w", encoding="utf-8") as fh:
        for candidate in candidates:
            fh.write(json.dumps(candidate, ensure_ascii=False, sort_keys=True) + "\n")

    stats = ReviewQueueStats(
        conversations=len(conversations),
        messages_seen=messages_seen,
        candidates=len(candidates),
        duplicate_pairs=duplicate_pairs,
        skipped_empty=skipped_empty,
        skipped_non_dialogue_roles=skipped_non_dialogue,
    )
    manifest = {
        "schema": QUEUE_SCHEMA,
        "authority": "LOCAL_REVIEW_QUEUE_NEVER_APPROVED_BY_IMPORT",
        "source_sha256": source_sha,
        "queue_filename": queue_path.name,
        "queue_sha256": sha256_file(queue_path),
        "candidate_count": len(candidates),
        "policy": asdict(policy),
        "default_language": default_language,
        "stats": asdict(stats),
        "privacy": {
            "queue_contains_raw_text": True,
            "manifest_contains_raw_text": False,
            "local_only_recommended": True,
        },
    }
    manifest["manifest_sha256"] = payload_digest(manifest)
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return {
        "manifest": manifest,
        "queue_path": queue_path,
        "manifest_path": manifest_path,
    }


def verify_review_queue(manifest_path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != QUEUE_SCHEMA:
        raise ValueError("unsupported review queue manifest schema")
    supplied = manifest.get("manifest_sha256")
    body = dict(manifest)
    body.pop("manifest_sha256", None)
    if payload_digest(body) != supplied:
        raise ValueError("review queue manifest digest mismatch")

    queue_path = path.parent / str(manifest["queue_filename"])
    if sha256_file(queue_path) != manifest.get("queue_sha256"):
        raise ValueError("review queue digest mismatch")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with queue_path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            candidate_id = str(row.get("candidate_id", ""))
            if not candidate_id or candidate_id in seen:
                raise ValueError(f"{queue_path}:{lineno}: invalid/duplicate candidate_id")
            seen.add(candidate_id)
            if row.get("approved") is not False:
                raise ValueError(
                    f"{queue_path}:{lineno}: imported review candidate must default approved=false"
                )
            if row.get("reviewed") is not False:
                raise ValueError(
                    f"{queue_path}:{lineno}: imported review candidate must default reviewed=false"
                )
            rows.append(row)
    if len(rows) != int(manifest.get("candidate_count", -1)):
        raise ValueError("review queue candidate count mismatch")
    return manifest, rows


def _load_decisions(path: str | Path) -> dict[str, dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = {}
    source = Path(path)
    with source.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            candidate_id = str(row.get("candidate_id", "")).strip()
            if not candidate_id:
                raise ValueError(f"{source}:{lineno}: candidate_id missing")
            if candidate_id in decisions:
                raise ValueError(f"{source}:{lineno}: duplicate candidate decision")
            if not isinstance(row.get("approved"), bool):
                raise ValueError(f"{source}:{lineno}: approved must be explicit boolean")
            if not isinstance(row.get("sensitive"), bool):
                raise ValueError(f"{source}:{lineno}: sensitive must be explicit boolean")
            language = row.get("language")
            if language is not None:
                language = _normalize_text(language).lower()
                if language not in {"vi", "en"}:
                    raise ValueError(f"{source}:{lineno}: language must be vi or en")
            weight = float(row.get("weight", 1.0))
            if not 0.25 <= weight <= 4.0:
                raise ValueError(f"{source}:{lineno}: weight outside allowed range")
            corrected_raw = row.get("corrected_target")
            corrected_target = None
            if corrected_raw is not None:
                corrected_target = _normalize_text(corrected_raw)
                if not corrected_target:
                    raise ValueError(
                        f"{source}:{lineno}: corrected_target cannot be empty"
                    )
                if len(corrected_target) > MAX_CORRECTED_TARGET_CHARS:
                    raise ValueError(
                        f"{source}:{lineno}: corrected_target exceeds character limit"
                    )
                if not row["approved"] or row["sensitive"]:
                    raise ValueError(
                        f"{source}:{lineno}: corrected_target is only valid for approved non-sensitive decisions"
                    )
            decisions[candidate_id] = {
                "approved": bool(row["approved"]),
                "sensitive": bool(row["sensitive"]),
                "language": language,
                "weight": weight,
                "corrected_target": corrected_target,
            }
    if not decisions:
        raise ValueError("review decisions file is empty")
    return decisions


def apply_review_decisions(
    queue_manifest_path: str | Path,
    decisions_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    queue_manifest, candidates = verify_review_queue(queue_manifest_path)
    decisions = _load_decisions(decisions_path)
    known = {str(row["candidate_id"]) for row in candidates}
    unknown = sorted(set(decisions) - known)
    if unknown:
        raise ValueError(f"review decisions reference unknown candidate IDs: {unknown}")

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty reviewed evidence directory: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    reviewed_path = output / "reviewed-evidence-source.jsonl"
    manifest_path = output / "reviewed-evidence-manifest.json"

    approved = rejected = sensitive = undecided = 0
    with reviewed_path.open("w", encoding="utf-8") as fh:
        for candidate in candidates:
            decision = decisions.get(str(candidate["candidate_id"]))
            if decision is None:
                row = {
                    "prompt": candidate["prompt"],
                    "target": candidate["target"],
                    "language": candidate.get("language"),
                    "weight": float(candidate.get("weight", 1.0)),
                    "approved": False,
                    "sensitive": None,
                    "reviewed": False,
                    "source_id": candidate.get("source_id"),
                }
                undecided += 1
            else:
                language = decision["language"] or candidate.get("language")
                if decision["approved"] and not decision["sensitive"]:
                    if language not in {"vi", "en"}:
                        raise ValueError(
                            f"approved candidate {candidate['candidate_id']} requires explicit vi/en language"
                        )
                    approved += 1
                else:
                    rejected += 1
                if decision["sensitive"]:
                    sensitive += 1
                row = {
                    "prompt": candidate["prompt"],
                    "target": (
                        decision["corrected_target"]
                        if decision["corrected_target"] is not None
                        else candidate["target"]
                    ),
                    "language": language,
                    "weight": decision["weight"],
                    "approved": bool(decision["approved"]),
                    "sensitive": bool(decision["sensitive"]),
                    "reviewed": True,
                    "source_id": candidate.get("source_id"),
                }
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    stats = ReviewApplyStats(
        queue_candidates=len(candidates),
        decisions_supplied=len(decisions),
        approved=approved,
        rejected=rejected,
        sensitive=sensitive,
        undecided=undecided,
    )
    manifest = {
        "schema": FINAL_SCHEMA,
        "authority": "EXPLICIT_LOCAL_REVIEW_DECISIONS_UNPROMOTED",
        "queue_manifest_sha256": queue_manifest["manifest_sha256"],
        "decisions_sha256": sha256_file(decisions_path),
        "reviewed_source_filename": reviewed_path.name,
        "reviewed_source_sha256": sha256_file(reviewed_path),
        "stats": asdict(stats),
        "privacy": {
            "reviewed_source_contains_raw_text": True,
            "decisions_may_contain_user_corrected_target": True,
            "manifest_contains_raw_text": False,
            "local_only_recommended": True,
        },
    }
    manifest["manifest_sha256"] = payload_digest(manifest)
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return {
        "manifest": manifest,
        "reviewed_source_path": reviewed_path,
        "manifest_path": manifest_path,
    }
