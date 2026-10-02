from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .personal_dataset import PersonalizationExample, load_jsonl
from .personal_protocol import PersonalizationSplitPolicy, build_personalization_protocol, verify_personalization_protocol
from .store import canonical_json, payload_digest


SCHEMA = "NOLANE-L23-APPROVED-EVIDENCE-PACK-V1"


@dataclass(slots=True)
class ApprovedEvidencePolicy:
    allowed_languages: tuple[str, ...] = ("vi", "en")
    min_approved_examples: int = 7
    max_prompt_chars: int = 8000
    max_target_chars: int = 8000
    min_weight: float = 0.25
    max_weight: float = 4.0
    reject_exact_duplicates: bool = True

    def validate(self) -> None:
        if not self.allowed_languages:
            raise ValueError("allowed_languages cannot be empty")
        if len(set(self.allowed_languages)) != len(self.allowed_languages):
            raise ValueError("allowed_languages contains duplicates")
        if self.min_approved_examples < 7:
            raise ValueError("min_approved_examples must be >=7 for L21 held-out readiness")
        if self.max_prompt_chars < 1 or self.max_target_chars < 1:
            raise ValueError("max character limits must be positive")
        if not 0 < self.min_weight <= self.max_weight:
            raise ValueError("invalid weight bounds")


@dataclass(slots=True)
class ApprovedEvidenceStats:
    source_rows: int
    approved_rows: int
    excluded_unapproved: int
    excluded_sensitive: int
    output_examples: int
    duplicate_pairs: int
    language_counts: dict[str, int]
    train_examples: int
    dev_examples: int
    test_examples: int


@dataclass(slots=True)
class ApprovedEvidenceBuildResult:
    manifest: dict[str, Any]
    dataset_path: Path
    protocol_path: Path
    manifest_path: Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_text(value: Any) -> str:
    return unicodedata.normalize("NFC", str(value)).strip()


def _pair_digest(prompt: str, target: str) -> str:
    return payload_digest({"prompt": prompt, "target": target})


def _source_id_digest(value: Any) -> str | None:
    if value is None:
        return None
    rendered = str(value).strip()
    if not rendered:
        return None
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _load_approved_rows(
    source: str | Path,
    *,
    policy: ApprovedEvidencePolicy,
) -> tuple[list[dict[str, Any]], ApprovedEvidenceStats, list[str]]:
    policy.validate()
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(path)

    rows: list[dict[str, Any]] = []
    source_rows = 0
    excluded_unapproved = 0
    excluded_sensitive = 0
    duplicate_pairs = 0
    seen_pairs: set[str] = set()
    language_counts: dict[str, int] = {}
    source_digests: list[str] = []

    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            source_rows += 1
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON") from exc

            if payload.get("approved") is not True:
                excluded_unapproved += 1
                continue
            if payload.get("sensitive") is True:
                excluded_sensitive += 1
                continue

            prompt = _normalize_text(payload.get("prompt", ""))
            target = _normalize_text(payload.get("target", ""))
            if not prompt:
                raise ValueError(f"{path}:{lineno}: approved prompt is empty")
            if not target:
                raise ValueError(f"{path}:{lineno}: approved target is empty")
            if len(prompt) > policy.max_prompt_chars:
                raise ValueError(f"{path}:{lineno}: prompt exceeds character limit")
            if len(target) > policy.max_target_chars:
                raise ValueError(f"{path}:{lineno}: target exceeds character limit")

            language = _normalize_text(payload.get("language", "")).lower()
            if language not in policy.allowed_languages:
                raise ValueError(
                    f"{path}:{lineno}: language {language!r} not allowed; "
                    f"allowed={list(policy.allowed_languages)}"
                )
            weight = float(payload.get("weight", 1.0))
            if not policy.min_weight <= weight <= policy.max_weight:
                raise ValueError(f"{path}:{lineno}: weight outside approved bounds")

            pair_sha = _pair_digest(prompt, target)
            if pair_sha in seen_pairs:
                duplicate_pairs += 1
                if policy.reject_exact_duplicates:
                    raise ValueError(f"{path}:{lineno}: duplicate approved prompt/target pair")
                continue
            seen_pairs.add(pair_sha)
            language_counts[language] = language_counts.get(language, 0) + 1
            source_id_sha = _source_id_digest(payload.get("source_id"))
            if source_id_sha is not None:
                source_digests.append(source_id_sha)

            rows.append({
                "prompt": prompt,
                "target": target,
                "language": language,
                "weight": weight,
            })

    if len(rows) < policy.min_approved_examples:
        raise ValueError(
            f"need at least {policy.min_approved_examples} approved non-sensitive examples; "
            f"got {len(rows)}"
        )

    stats = ApprovedEvidenceStats(
        source_rows=source_rows,
        approved_rows=len(rows),
        excluded_unapproved=excluded_unapproved,
        excluded_sensitive=excluded_sensitive,
        output_examples=len(rows),
        duplicate_pairs=duplicate_pairs,
        language_counts=dict(sorted(language_counts.items())),
        train_examples=0,
        dev_examples=0,
        test_examples=0,
    )
    return rows, stats, sorted(source_digests)


def build_approved_evidence_pack(
    source: str | Path,
    output_dir: str | Path,
    *,
    policy: ApprovedEvidencePolicy | None = None,
    split_policy: PersonalizationSplitPolicy | None = None,
) -> ApprovedEvidenceBuildResult:
    policy = policy or ApprovedEvidencePolicy()
    split_policy = split_policy or PersonalizationSplitPolicy(min_examples=7)
    if split_policy.min_examples < 7:
        raise ValueError("split policy min_examples must be >=7 for L21 readiness")

    rows, stats, source_id_sha256 = _load_approved_rows(source, policy=policy)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty evidence pack directory: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)

    dataset_path = output / "personalization.jsonl"
    protocol_path = output / "personalization-protocol-v1.json"
    manifest_path = output / "approved-evidence-manifest.json"

    with dataset_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    dataset_sha256 = sha256_file(dataset_path)
    examples = load_jsonl(dataset_path)
    protocol = build_personalization_protocol(
        examples,
        dataset_sha256=dataset_sha256,
        policy=split_policy,
    )
    verify_personalization_protocol(protocol, dataset_sha256=dataset_sha256)
    protocol_path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    counts = protocol["counts"]
    stats.train_examples = int(counts["train"])
    stats.dev_examples = int(counts["dev"])
    stats.test_examples = int(counts["test"])
    if stats.test_examples < 2:
        raise RuntimeError("approved evidence pack must reserve at least two held-out test examples")

    manifest = {
        "schema": SCHEMA,
        "authority": "USER_APPROVED_LOCAL_EVIDENCE_UNPROMOTED",
        "privacy": {
            "raw_prompt_target_in_manifest": False,
            "raw_source_id_in_manifest": False,
            "local_only_recommended": True,
        },
        "source_sha256": sha256_file(source),
        "source_id_sha256": source_id_sha256,
        "dataset_sha256": dataset_sha256,
        "protocol_sha256": str(protocol["protocol_sha256"]),
        "dataset_filename": dataset_path.name,
        "protocol_filename": protocol_path.name,
        "stats": asdict(stats),
        "policy": asdict(policy),
        "split_policy": asdict(split_policy),
    }
    manifest["manifest_sha256"] = payload_digest(manifest)
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")

    return ApprovedEvidenceBuildResult(
        manifest=manifest,
        dataset_path=dataset_path,
        protocol_path=protocol_path,
        manifest_path=manifest_path,
    )


def verify_approved_evidence_pack(
    manifest_path: str | Path,
    *,
    root: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported approved evidence manifest schema")
    supplied = manifest.get("manifest_sha256")
    body = dict(manifest)
    body.pop("manifest_sha256", None)
    if payload_digest(body) != supplied:
        raise ValueError("approved evidence manifest digest mismatch")
    base = Path(root) if root is not None else path.parent
    dataset = base / str(manifest["dataset_filename"])
    protocol = base / str(manifest["protocol_filename"])
    if sha256_file(dataset) != manifest.get("dataset_sha256"):
        raise ValueError("approved evidence dataset digest mismatch")
    frozen = json.loads(protocol.read_text(encoding="utf-8"))
    verify_personalization_protocol(
        frozen,
        dataset_sha256=str(manifest["dataset_sha256"]),
    )
    if str(frozen["protocol_sha256"]) != str(manifest.get("protocol_sha256")):
        raise ValueError("approved evidence protocol digest mismatch")
    examples = load_jsonl(dataset)
    if len(examples) != int(manifest["stats"]["output_examples"]):
        raise ValueError("approved evidence example count mismatch")
    return manifest


def resolve_approved_evidence_pack(
    manifest_path: str | Path,
) -> tuple[dict[str, Any], Path, Path]:
    path = Path(manifest_path)
    manifest = verify_approved_evidence_pack(path)
    root = path.parent
    dataset = root / str(manifest["dataset_filename"])
    protocol = root / str(manifest["protocol_filename"])
    return manifest, dataset, protocol
