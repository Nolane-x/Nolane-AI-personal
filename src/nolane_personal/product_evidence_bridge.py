from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .local_evidence_workbench import (
    initialize_workbench,
    paths_for,
    verify_workbench_manifest,
)
from .longitudinal_execution import PLAN_SCHEMA, validate_longitudinal_plan
from .personal_dataset import load_jsonl
from .store import canonical_json, payload_digest


EXPORT_SCHEMA = "NOLANE-L44-PRODUCT-EVIDENCE-EXPORT-V1"
WINDOW_SCHEMA = "NOLANE-L44-PRODUCT-EVIDENCE-WINDOW-V1"
PLAN_SPEC_SCHEMA = "NOLANE-L44-PRODUCT-LONGITUDINAL-SPEC-V1"
PLAN_BRIDGE_SCHEMA = "NOLANE-L44-PRODUCT-LONGITUDINAL-BRIDGE-V1"


@dataclass(slots=True)
class ProductEvidenceExportPolicy:
    session_gap_minutes: int = 45
    max_pairs_per_group: int = 4
    max_prompt_chars: int = 8000
    max_target_chars: int = 8000

    def validate(self) -> None:
        if not 1 <= self.session_gap_minutes <= 24 * 60:
            raise ValueError("session_gap_minutes must be in [1,1440]")
        if not 1 <= self.max_pairs_per_group <= 12:
            raise ValueError("max_pairs_per_group must be in [1,12]")
        if self.max_prompt_chars < 1 or self.max_target_chars < 1:
            raise ValueError("product evidence character limits must be positive")


@dataclass(slots=True)
class ProductEvidenceExportResult:
    manifest: dict[str, Any]
    export_path: Path
    manifest_path: Path


@dataclass(slots=True)
class _Turn:
    user_rowid: int
    assistant_rowid: int
    user_event_id: str
    assistant_event_id: str
    user_at: str
    assistant_at: str
    prompt: str
    target: str


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_text(value: Any) -> str:
    return unicodedata.normalize("NFC", str(value)).strip()


def _leakage_key(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"\s+", " ", text).strip()


def _parse_time(value: str) -> datetime | None:
    rendered = str(value).strip()
    if not rendered:
        return None
    try:
        return datetime.fromisoformat(rendered.replace("Z", "+00:00"))
    except ValueError:
        return None


def _load_profile_language(profile_path: str | Path | None) -> str | None:
    if profile_path is None:
        return None
    path = Path(profile_path)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    language = str(payload.get("language", "")).strip().lower()
    return language if language in {"vi", "en"} else None


def _read_product_turns(
    db_path: str | Path,
    *,
    after_rowid: int,
    through_rowid: int | None,
    policy: ProductEvidenceExportPolicy,
) -> tuple[list[_Turn], dict[str, Any]]:
    path = Path(db_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if after_rowid < 0:
        raise ValueError("after_rowid must be non-negative")
    if through_rowid is not None and through_rowid <= after_rowid:
        raise ValueError("through_rowid must be greater than after_rowid")

    uri = path.as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("BEGIN")
        clauses = [
            "rowid > ?",
            "kind IN ('user_message','assistant_speech')",
        ]
        values: list[Any] = [int(after_rowid)]
        if through_rowid is not None:
            clauses.append("rowid <= ?")
            values.append(int(through_rowid))
        rows = connection.execute(
            f"""
            SELECT rowid,event_id,at,kind,payload_json
            FROM events
            WHERE {' AND '.join(clauses)}
            ORDER BY rowid ASC
            """,
            values,
        ).fetchall()
    finally:
        connection.close()

    selected_digest_rows: list[dict[str, Any]] = []
    turns: list[_Turn] = []
    pending: sqlite3.Row | None = None
    skipped_orphan_assistant = 0
    skipped_superseded_user = 0
    skipped_empty = 0

    for row in rows:
        raw_payload = str(row["payload_json"])
        selected_digest_rows.append(
            {
                "rowid": int(row["rowid"]),
                "event_id": str(row["event_id"]),
                "at": str(row["at"]),
                "kind": str(row["kind"]),
                "payload_sha256": hashlib.sha256(
                    raw_payload.encode("utf-8")
                ).hexdigest(),
            }
        )
        payload = json.loads(raw_payload)
        text = _normalize_text(payload.get("text", ""))
        if not text:
            skipped_empty += 1
            continue

        if row["kind"] == "user_message":
            if pending is not None:
                skipped_superseded_user += 1
            pending = row
            continue

        if pending is None:
            skipped_orphan_assistant += 1
            continue

        user_payload = json.loads(str(pending["payload_json"]))
        prompt = _normalize_text(user_payload.get("text", ""))
        target = text
        if not prompt:
            skipped_empty += 1
            pending = None
            continue
        if len(prompt) > policy.max_prompt_chars:
            raise ValueError(
                f"user event {pending['event_id']} exceeds prompt character limit"
            )
        if len(target) > policy.max_target_chars:
            raise ValueError(
                f"assistant event {row['event_id']} exceeds target character limit"
            )

        turns.append(
            _Turn(
                user_rowid=int(pending["rowid"]),
                assistant_rowid=int(row["rowid"]),
                user_event_id=str(pending["event_id"]),
                assistant_event_id=str(row["event_id"]),
                user_at=str(pending["at"]),
                assistant_at=str(row["at"]),
                prompt=prompt,
                target=target,
            )
        )
        pending = None

    snapshot = {
        "selected_event_rows": len(rows),
        "paired_turns": len(turns),
        "skipped_orphan_assistant": skipped_orphan_assistant,
        "skipped_superseded_user": skipped_superseded_user,
        "skipped_empty": skipped_empty,
        "first_selected_rowid": (
            int(rows[0]["rowid"]) if rows else None
        ),
        "last_selected_rowid": (
            int(rows[-1]["rowid"]) if rows else int(after_rowid)
        ),
        "selected_event_snapshot_sha256": payload_digest(
            selected_digest_rows
        ),
    }
    return turns, snapshot


def _sessionize(
    turns: list[_Turn],
    *,
    policy: ProductEvidenceExportPolicy,
    language: str | None,
) -> list[dict[str, Any]]:
    conversations: list[dict[str, Any]] = []
    current: list[_Turn] = []
    last_time: datetime | None = None

    def flush() -> None:
        nonlocal current, last_time
        if not current:
            return
        identity = {
            "first_user_event_id": current[0].user_event_id,
            "last_assistant_event_id": current[-1].assistant_event_id,
            "first_rowid": current[0].user_rowid,
            "last_rowid": current[-1].assistant_rowid,
        }
        conversation_id = (
            "product-session-" + payload_digest(identity)[:24]
        )
        messages: list[dict[str, str]] = []
        for turn in current:
            messages.append({"role": "user", "content": turn.prompt})
            messages.append({"role": "assistant", "content": turn.target})
        row: dict[str, Any] = {
            "conversation_id": conversation_id,
            "messages": messages,
        }
        if language is not None:
            row["language"] = language
        conversations.append(row)
        current = []
        last_time = None

    for turn in turns:
        user_time = _parse_time(turn.user_at)
        gap_minutes: float | None = None
        if user_time is not None and last_time is not None:
            try:
                gap_minutes = (user_time - last_time).total_seconds() / 60.0
            except TypeError:
                gap_minutes = None
        should_split = bool(current) and (
            len(current) >= policy.max_pairs_per_group
            or (
                gap_minutes is not None
                and gap_minutes > policy.session_gap_minutes
            )
        )
        if should_split:
            flush()
        current.append(turn)
        last_time = _parse_time(turn.assistant_at) or user_time
    flush()
    return conversations


def export_product_evidence(
    db_path: str | Path,
    output_dir: str | Path,
    *,
    after_rowid: int = 0,
    through_rowid: int | None = None,
    profile_path: str | Path | None = None,
    language: str | None = None,
    policy: ProductEvidenceExportPolicy | None = None,
) -> ProductEvidenceExportResult:
    policy = policy or ProductEvidenceExportPolicy()
    policy.validate()
    if language is not None:
        language = str(language).strip().lower()
        if language not in {"vi", "en"}:
            raise ValueError("language must be vi or en")
    else:
        language = _load_profile_language(profile_path)

    turns, snapshot = _read_product_turns(
        db_path,
        after_rowid=after_rowid,
        through_rowid=through_rowid,
        policy=policy,
    )
    if not turns:
        raise ValueError("product database range contains no complete user/assistant turns")

    conversations = _sessionize(
        turns,
        policy=policy,
        language=language,
    )
    if len(conversations) < 3:
        raise ValueError(
            "product evidence export needs at least 3 leakage-safe source groups; "
            "collect more sessions/turns or lower max_pairs_per_group"
        )

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty product evidence export: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    export_path = output / "product-conversations.jsonl"
    with export_path.open("w", encoding="utf-8") as handle:
        for conversation in conversations:
            handle.write(
                json.dumps(
                    conversation,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    manifest = {
        "schema": EXPORT_SCHEMA,
        "authority": "PRODUCT_TRANSCRIPT_EXPORT_REVIEW_REQUIRED",
        "source": {
            "after_rowid_exclusive": int(after_rowid),
            "through_rowid_inclusive": (
                None if through_rowid is None else int(through_rowid)
            ),
            **snapshot,
        },
        "policy": asdict(policy),
        "language_hint": language,
        "conversation_groups": len(conversations),
        "paired_turns": len(turns),
        "export_filename": export_path.name,
        "export_sha256": _sha256_file(export_path),
        "privacy": {
            "export_contains_raw_prompt_target": True,
            "manifest_contains_raw_prompt_target": False,
            "manifest_contains_raw_event_id": False,
            "local_only": True,
            "approval_created_by_export": False,
        },
    }
    manifest["manifest_sha256"] = payload_digest(manifest)
    manifest_path = output / "product-evidence-export-manifest.json"
    manifest_path.write_text(
        canonical_json(manifest) + "\n",
        encoding="utf-8",
    )
    return ProductEvidenceExportResult(
        manifest=manifest,
        export_path=export_path,
        manifest_path=manifest_path,
    )


def verify_product_evidence_export(
    manifest_path: str | Path,
) -> tuple[dict[str, Any], Path]:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != EXPORT_SCHEMA:
        raise ValueError("unsupported product evidence export schema")
    supplied = manifest.get("manifest_sha256")
    body = dict(manifest)
    body.pop("manifest_sha256", None)
    if payload_digest(body) != supplied:
        raise ValueError("product evidence export manifest digest mismatch")

    export_path = path.parent / str(manifest["export_filename"])
    if _sha256_file(export_path) != manifest.get("export_sha256"):
        raise ValueError("product evidence export digest mismatch")

    groups = turns = 0
    with export_path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(
                    f"{export_path}:{lineno}: invalid conversation messages"
                )
            if len(messages) % 2:
                raise ValueError(
                    f"{export_path}:{lineno}: unpaired conversation messages"
                )
            for index in range(0, len(messages), 2):
                if messages[index].get("role") != "user":
                    raise ValueError("product export pair must start with user")
                if messages[index + 1].get("role") != "assistant":
                    raise ValueError("product export pair must end with assistant")
                turns += 1
            groups += 1

    if groups != int(manifest.get("conversation_groups", -1)):
        raise ValueError("product export conversation-group count mismatch")
    if turns != int(manifest.get("paired_turns", -1)):
        raise ValueError("product export paired-turn count mismatch")
    if groups < 3:
        raise ValueError("product export lacks leakage-safe source groups")
    return manifest, export_path


def prepare_product_evidence_window(
    db_path: str | Path,
    workspace: str | Path,
    *,
    after_rowid: int = 0,
    through_rowid: int | None = None,
    profile_path: str | Path | None = None,
    language: str | None = None,
    policy: ProductEvidenceExportPolicy | None = None,
) -> dict[str, Any]:
    root = Path(workspace)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty product evidence window: {root}"
        )
    root.mkdir(parents=True, exist_ok=True)
    export = export_product_evidence(
        db_path,
        root / "export",
        after_rowid=after_rowid,
        through_rowid=through_rowid,
        profile_path=profile_path,
        language=language,
        policy=policy,
    )
    workbench_root = root / "workbench"
    workbench = initialize_workbench(
        export.export_path,
        workbench_root,
    )
    manifest = {
        "schema": WINDOW_SCHEMA,
        "authority": "PRODUCT_EVIDENCE_WINDOW_REVIEW_REQUIRED_NO_TRAINING_AUTHORITY",
        "export_manifest_sha256": export.manifest["manifest_sha256"],
        "review_queue_manifest_sha256": workbench["queue_manifest_sha256"],
        "review_progress": workbench["review_progress"],
        "source_event_range": {
            "after_rowid_exclusive": int(after_rowid),
            "through_rowid_inclusive": export.manifest["source"][
                "last_selected_rowid"
            ],
        },
        "paths": {
            "export_manifest": "export/product-evidence-export-manifest.json",
            "workbench": "workbench",
        },
        "privacy": {
            "manifest_contains_raw_prompt_target": False,
            "local_only": True,
            "auto_approval": False,
            "auto_training": False,
        },
    }
    manifest["manifest_sha256"] = payload_digest(manifest)
    manifest_path = root / "product-evidence-window-manifest.json"
    manifest_path.write_text(
        canonical_json(manifest) + "\n",
        encoding="utf-8",
    )
    return manifest


def verify_product_evidence_window(
    workspace: str | Path,
) -> dict[str, Any]:
    root = Path(workspace)
    path = root / "product-evidence-window-manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != WINDOW_SCHEMA:
        raise ValueError("unsupported product evidence window schema")
    supplied = manifest.get("manifest_sha256")
    body = dict(manifest)
    body.pop("manifest_sha256", None)
    if payload_digest(body) != supplied:
        raise ValueError("product evidence window manifest digest mismatch")

    export_path = root / str(manifest["paths"]["export_manifest"])
    export, _ = verify_product_evidence_export(export_path)
    if export["manifest_sha256"] != manifest["export_manifest_sha256"]:
        raise ValueError("product evidence window export lineage mismatch")

    workbench_root = root / str(manifest["paths"]["workbench"])
    workbench = verify_workbench_manifest(workbench_root)
    if (
        workbench["queue_manifest_sha256"]
        != manifest["review_queue_manifest_sha256"]
    ):
        raise ValueError("product evidence window review-queue lineage mismatch")
    return manifest


def _resolved_workbench(root: Path) -> dict[str, Any]:
    status = verify_workbench_manifest(root)
    if status.get("phase") != "INTAKE_READY":
        raise ValueError(
            f"workbench is not finalized/INTAKE_READY: {root}"
        )
    paths = paths_for(root)
    return {
        "root": root,
        "status": status,
        "approved_manifest": paths.approved_manifest,
    }


def _dataset_fingerprints(
    approved_manifest_path: Path,
) -> dict[str, Any]:
    manifest = json.loads(
        approved_manifest_path.read_text(encoding="utf-8")
    )
    dataset = approved_manifest_path.parent / str(
        manifest["dataset_filename"]
    )
    examples = load_jsonl(dataset)
    prompts: set[str] = set()
    pairs: set[str] = set()
    for example in examples:
        prompt = _leakage_key(example.prompt)
        target = _leakage_key(example.target)
        prompts.add(hashlib.sha256(prompt.encode("utf-8")).hexdigest())
        pairs.add(
            hashlib.sha256(
                (prompt + "\n" + target).encode("utf-8")
            ).hexdigest()
        )
    return {
        "prompts": prompts,
        "pairs": pairs,
        "prompt_set_sha256": payload_digest(sorted(prompts)),
        "pair_set_sha256": payload_digest(sorted(pairs)),
        "examples": len(examples),
    }


def _assert_no_overlap(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    label: str,
) -> None:
    prompt_overlap = left["prompts"] & right["prompts"]
    pair_overlap = left["pairs"] & right["pairs"]
    if prompt_overlap or pair_overlap:
        raise ValueError(
            f"{label} content leakage: "
            f"prompt_overlap={len(prompt_overlap)} "
            f"pair_overlap={len(pair_overlap)}"
        )


def build_product_longitudinal_plan(
    spec_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    spec_path = Path(spec_path).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("schema") != PLAN_SPEC_SCHEMA:
        raise ValueError("unsupported product longitudinal spec schema")
    base = spec_path.parent

    def resolve(value: Any, field: str) -> Path:
        rendered = str(value or "").strip()
        if not rendered:
            raise ValueError(f"{field} is required")
        path = Path(rendered)
        return path if path.is_absolute() else (base / path).resolve()

    fixed = _resolved_workbench(
        resolve(spec.get("fixed_workbench"), "fixed_workbench")
    )
    cycles_raw = spec.get("cycles")
    if not isinstance(cycles_raw, list) or len(cycles_raw) < 5:
        raise ValueError("product longitudinal spec requires at least 5 cycles")

    cycles: list[dict[str, Any]] = []
    bridge_cycles: list[dict[str, Any]] = []
    adaptation_fingerprints: list[dict[str, Any]] = []
    fixed_fp = _dataset_fingerprints(fixed["approved_manifest"])
    training_fingerprints: list[dict[str, Any]] = []

    for index, raw in enumerate(cycles_raw, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"cycle {index} must be an object")
        retention = _resolved_workbench(
            resolve(
                raw.get("retention_workbench"),
                f"cycles[{index}].retention_workbench",
            )
        )
        adaptation = _resolved_workbench(
            resolve(
                raw.get("adaptation_workbench"),
                f"cycles[{index}].adaptation_workbench",
            )
        )
        retention_fp = _dataset_fingerprints(
            retention["approved_manifest"]
        )
        adaptation_fp = _dataset_fingerprints(
            adaptation["approved_manifest"]
        )
        _assert_no_overlap(
            retention_fp,
            adaptation_fp,
            label=f"cycle {index} retention/adaptation",
        )
        for previous_index, previous in enumerate(
            adaptation_fingerprints,
            start=1,
        ):
            _assert_no_overlap(
                previous,
                adaptation_fp,
                label=(
                    f"adaptation cycle {previous_index}/cycle {index}"
                ),
            )
        adaptation_fingerprints.append(adaptation_fp)
        training_fingerprints.extend([retention_fp, adaptation_fp])

        cycles.append(
            {
                "retention_manifest": str(
                    retention["approved_manifest"].resolve()
                ),
                "adaptation_manifest": str(
                    adaptation["approved_manifest"].resolve()
                ),
                "training": dict(raw.get("training", {})),
            }
        )
        bridge_cycles.append(
            {
                "cycle": index,
                "retention_workbench_manifest_sha256": retention[
                    "status"
                ]["manifest_sha256"],
                "adaptation_workbench_manifest_sha256": adaptation[
                    "status"
                ]["manifest_sha256"],
                "retention_content": {
                    "examples": retention_fp["examples"],
                    "prompt_set_sha256": retention_fp[
                        "prompt_set_sha256"
                    ],
                    "pair_set_sha256": retention_fp["pair_set_sha256"],
                },
                "adaptation_content": {
                    "examples": adaptation_fp["examples"],
                    "prompt_set_sha256": adaptation_fp[
                        "prompt_set_sha256"
                    ],
                    "pair_set_sha256": adaptation_fp["pair_set_sha256"],
                },
            }
        )

    for training_fp in training_fingerprints:
        _assert_no_overlap(
            fixed_fp,
            training_fp,
            label="fixed-panel/training",
        )

    plan = {
        "schema": PLAN_SCHEMA,
        "initial_factorized": str(
            resolve(spec.get("initial_factorized"), "initial_factorized")
        ),
        "latent": str(resolve(spec.get("latent"), "latent")),
        "tokenizer": str(resolve(spec.get("tokenizer"), "tokenizer")),
        "device": str(spec.get("device", "cpu")),
        "fixed_panel_manifest": str(
            fixed["approved_manifest"].resolve()
        ),
        "cycles": cycles,
    }
    if "policy" in spec:
        plan["policy"] = dict(spec["policy"])

    output = Path(output_path)
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite longitudinal plan: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=output.parent,
        prefix=output.name + ".",
        suffix=".tmp",
        delete=False,
    )
    temp = Path(handle.name)
    try:
        with handle:
            handle.write(canonical_json(plan) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        validated = validate_longitudinal_plan(temp)
        os.replace(temp, output)
    finally:
        if temp.exists():
            temp.unlink()

    bridge = {
        "schema": PLAN_BRIDGE_SCHEMA,
        "authority": "PRODUCT_REVIEWED_EVIDENCE_TO_L43_PLAN_NO_TRAINING_AUTHORITY",
        "l43_plan_sha256": validated.receipt["plan_sha256"],
        "fixed_workbench_manifest_sha256": fixed["status"][
            "manifest_sha256"
        ],
        "fixed_content": {
            "examples": fixed_fp["examples"],
            "prompt_set_sha256": fixed_fp["prompt_set_sha256"],
            "pair_set_sha256": fixed_fp["pair_set_sha256"],
        },
        "cycles": bridge_cycles,
        "privacy": {
            "contains_raw_prompt_target": False,
            "contains_local_paths": False,
            "contains_individual_prompt_hashes": False,
            "auto_training": False,
            "auto_promotion": False,
        },
    }
    bridge["receipt_sha256"] = payload_digest(bridge)
    bridge_path = output.with_name(
        output.stem + "-product-bridge-receipt.json"
    )
    bridge_path.write_text(
        canonical_json(bridge) + "\n",
        encoding="utf-8",
    )
    return bridge
