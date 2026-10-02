from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .approved_evidence import (
    ApprovedEvidencePolicy,
    build_approved_evidence_pack,
    verify_approved_evidence_pack,
)
from .review_queue import (
    apply_review_decisions,
    sha256_file,
    verify_review_queue,
)
from .store import canonical_json, payload_digest


SCHEMA = "NOLANE-L25-LOCAL-EVIDENCE-INTAKE-V1"


@dataclass(slots=True)
class LocalEvidenceIntakeStats:
    queue_candidates: int
    decisions_supplied: int
    approved: int
    rejected: int
    sensitive: int
    undecided: int
    output_examples: int
    train_examples: int
    dev_examples: int
    test_examples: int
    language_counts: dict[str, int]


@dataclass(slots=True)
class LocalEvidenceIntakeResult:
    manifest: dict[str, Any]
    manifest_path: Path
    reviewed_manifest_path: Path
    approved_manifest_path: Path


def _verify_manifest_digest(payload: dict[str, Any], *, field: str) -> None:
    supplied = payload.get(field)
    body = dict(payload)
    body.pop(field, None)
    if payload_digest(body) != supplied:
        raise ValueError(f"{field} digest mismatch")


def verify_reviewed_evidence_source(
    manifest_path: str | Path,
    *,
    queue_manifest_path: str | Path | None = None,
    decisions_path: str | Path | None = None,
) -> tuple[dict[str, Any], Path]:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "NOLANE-L24-REVIEWED-EVIDENCE-SOURCE-V1":
        raise ValueError("unsupported reviewed evidence manifest schema")
    _verify_manifest_digest(manifest, field="manifest_sha256")

    reviewed_path = path.parent / str(manifest["reviewed_source_filename"])
    if not reviewed_path.exists():
        raise ValueError("reviewed evidence source missing")
    if sha256_file(reviewed_path) != str(manifest.get("reviewed_source_sha256")):
        raise ValueError("reviewed evidence source digest mismatch")

    if queue_manifest_path is not None:
        queue_manifest, _rows = verify_review_queue(queue_manifest_path)
        if str(queue_manifest["manifest_sha256"]) != str(
            manifest.get("queue_manifest_sha256")
        ):
            raise ValueError("reviewed source queue-manifest lineage mismatch")

    if decisions_path is not None:
        if sha256_file(decisions_path) != str(manifest.get("decisions_sha256")):
            raise ValueError("reviewed source decisions digest mismatch")

    counts = {
        "queue_candidates": 0,
        "approved": 0,
        "rejected": 0,
        "sensitive": 0,
        "undecided": 0,
    }
    with reviewed_path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            counts["queue_candidates"] += 1
            reviewed = row.get("reviewed") is True
            approved = row.get("approved") is True
            sensitive = row.get("sensitive") is True
            if not reviewed:
                counts["undecided"] += 1
            else:
                if approved and not sensitive:
                    counts["approved"] += 1
                else:
                    counts["rejected"] += 1
                if sensitive:
                    counts["sensitive"] += 1
            if approved and not sensitive and row.get("language") not in {"vi", "en"}:
                raise ValueError(
                    f"{reviewed_path}:{lineno}: eligible reviewed row missing vi/en language"
                )

    stats = manifest.get("stats", {})
    for key in ("queue_candidates", "approved", "rejected", "sensitive", "undecided"):
        if int(stats.get(key, -1)) != counts[key]:
            raise ValueError(f"reviewed evidence stats mismatch: {key}")

    if int(stats.get("decisions_supplied", -1)) < 0:
        raise ValueError("reviewed evidence decisions_supplied invalid")
    return manifest, reviewed_path


def finalize_local_evidence_intake(
    queue_manifest_path: str | Path,
    decisions_path: str | Path,
    output_dir: str | Path,
    *,
    policy: ApprovedEvidencePolicy | None = None,
) -> LocalEvidenceIntakeResult:
    queue_manifest_path = Path(queue_manifest_path)
    decisions_path = Path(decisions_path)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty local intake directory: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)

    queue_manifest, queue_rows = verify_review_queue(queue_manifest_path)
    reviewed_dir = output / "reviewed"
    reviewed = apply_review_decisions(
        queue_manifest_path,
        decisions_path,
        reviewed_dir,
    )
    reviewed_manifest, reviewed_source = verify_reviewed_evidence_source(
        reviewed["manifest_path"],
        queue_manifest_path=queue_manifest_path,
        decisions_path=decisions_path,
    )

    approved_dir = output / "approved-pack"
    approved = build_approved_evidence_pack(
        reviewed_source,
        approved_dir,
        policy=policy,
    )
    approved_manifest = verify_approved_evidence_pack(approved.manifest_path)

    review_stats = reviewed_manifest["stats"]
    approved_stats = approved_manifest["stats"]
    stats = LocalEvidenceIntakeStats(
        queue_candidates=int(review_stats["queue_candidates"]),
        decisions_supplied=int(review_stats["decisions_supplied"]),
        approved=int(review_stats["approved"]),
        rejected=int(review_stats["rejected"]),
        sensitive=int(review_stats["sensitive"]),
        undecided=int(review_stats["undecided"]),
        output_examples=int(approved_stats["output_examples"]),
        train_examples=int(approved_stats["train_examples"]),
        dev_examples=int(approved_stats["dev_examples"]),
        test_examples=int(approved_stats["test_examples"]),
        language_counts=dict(approved_stats["language_counts"]),
    )

    manifest = {
        "schema": SCHEMA,
        "authority": "EXPLICIT_REVIEWED_LOCAL_EVIDENCE_UNPROMOTED",
        "privacy": {
            "raw_prompt_target_in_manifest": False,
            "raw_source_id_in_manifest": False,
            "local_only_recommended": True,
        },
        "queue_manifest_sha256": str(queue_manifest["manifest_sha256"]),
        "queue_file_sha256": str(queue_manifest["queue_sha256"]),
        "decisions_sha256": sha256_file(decisions_path),
        "reviewed_manifest_sha256": str(reviewed_manifest["manifest_sha256"]),
        "reviewed_source_sha256": str(reviewed_manifest["reviewed_source_sha256"]),
        "approved_manifest_sha256": str(approved_manifest["manifest_sha256"]),
        "dataset_sha256": str(approved_manifest["dataset_sha256"]),
        "protocol_sha256": str(approved_manifest["protocol_sha256"]),
        "reviewed_manifest_relpath": str(
            reviewed["manifest_path"].relative_to(output)
        ),
        "approved_manifest_relpath": str(
            approved.manifest_path.relative_to(output)
        ),
        "stats": asdict(stats),
    }
    manifest["manifest_sha256"] = payload_digest(manifest)
    manifest_path = output / "local-evidence-intake-manifest.json"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")

    return LocalEvidenceIntakeResult(
        manifest=manifest,
        manifest_path=manifest_path,
        reviewed_manifest_path=reviewed["manifest_path"],
        approved_manifest_path=approved.manifest_path,
    )


def verify_local_evidence_intake(
    manifest_path: str | Path,
    *,
    queue_manifest_path: str | Path,
    decisions_path: str | Path,
) -> dict[str, Any]:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported local evidence intake manifest schema")
    _verify_manifest_digest(manifest, field="manifest_sha256")

    root = path.parent
    reviewed_manifest_path = root / str(manifest["reviewed_manifest_relpath"])
    approved_manifest_path = root / str(manifest["approved_manifest_relpath"])

    queue_manifest, _ = verify_review_queue(queue_manifest_path)
    reviewed_manifest, _ = verify_reviewed_evidence_source(
        reviewed_manifest_path,
        queue_manifest_path=queue_manifest_path,
        decisions_path=decisions_path,
    )
    approved_manifest = verify_approved_evidence_pack(approved_manifest_path)

    bindings = {
        "queue_manifest_sha256": queue_manifest["manifest_sha256"],
        "decisions_sha256": sha256_file(decisions_path),
        "reviewed_manifest_sha256": reviewed_manifest["manifest_sha256"],
        "reviewed_source_sha256": reviewed_manifest["reviewed_source_sha256"],
        "approved_manifest_sha256": approved_manifest["manifest_sha256"],
        "dataset_sha256": approved_manifest["dataset_sha256"],
        "protocol_sha256": approved_manifest["protocol_sha256"],
    }
    for key, expected in bindings.items():
        if str(manifest.get(key)) != str(expected):
            raise ValueError(f"local evidence intake lineage mismatch: {key}")

    approved_stats = approved_manifest["stats"]
    if int(manifest["stats"]["output_examples"]) != int(
        approved_stats["output_examples"]
    ):
        raise ValueError("local evidence intake output-example count mismatch")
    if int(manifest["stats"]["test_examples"]) < 2:
        raise ValueError("local evidence intake held-out test reserve invalid")

    return manifest
