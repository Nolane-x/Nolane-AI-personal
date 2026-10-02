from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .evidence_quality import assess_evidence_quality
from .latent import LatentStore
from .personal_dataset import load_jsonl
from .personal_protocol import load_protocol, verify_personalization_protocol


@dataclass(slots=True)
class RealCandidateReadinessThresholds:
    min_train_examples: int = 1
    min_dev_examples: int = 1
    min_test_examples: int = 2
    min_test_source_groups: int = 2
    min_anchor_examples: int = 4
    latent_dim: int = 32


@dataclass(slots=True)
class RealCandidateReadinessEvidence:
    dataset_present: bool
    protocol_present: bool
    dataset_sha256: str | None
    protocol_sha256: str | None
    dataset_examples: int
    train_examples: int
    dev_examples: int
    test_examples: int
    test_source_groups: int
    evidence_quality_status: str | None
    evidence_quality_court_sha256: str | None
    evidence_quality_reasons: list[str]
    anchor_examples: int
    latent_present: bool
    latent_valid: bool
    latent_dim: int | None
    latent_digest: str | None
    model_lock_present: bool
    model_present: bool
    requested_model_revision: str | None
    resolved_model_revision: str | None
    model_revision_matches: bool
    l14_anchor_present: bool


@dataclass(slots=True)
class RealCandidateReadinessDecision:
    status: str
    reasons: list[str]
    evidence: dict[str, Any]
    thresholds: dict[str, Any]


def sha256_file(path: str | Path) -> str:
    digest=hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda:fh.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_weights_present(model_dir: Path) -> bool:
    if not (model_dir/"config.json").exists():
        return False
    if (model_dir/"model.safetensors").exists():
        return True
    return any(model_dir.glob("model-*.safetensors"))


def assess_real_candidate_readiness(
    *,
    dataset,
    protocol,
    anchor,
    latent,
    model_lock,
    model_dir,
    l14_anchor,
    thresholds: RealCandidateReadinessThresholds | None = None,
) -> RealCandidateReadinessDecision:
    t=thresholds or RealCandidateReadinessThresholds()
    reasons=[]

    dataset_path=Path(dataset)
    protocol_path=Path(protocol)
    anchor_path=Path(anchor)
    latent_path=Path(latent)
    lock_path=Path(model_lock)
    model_path=Path(model_dir)
    l14_path=Path(l14_anchor)

    dataset_sha=None
    protocol_sha=None
    dataset_examples=train_examples=dev_examples=test_examples=0
    test_source_groups=0
    evidence_quality_status=None
    evidence_quality_court_sha256=None
    evidence_quality_reasons: list[str]=[]

    if not dataset_path.exists():
        reasons.append("personalization_dataset_missing")
    if not protocol_path.exists():
        reasons.append("personalization_protocol_missing")

    if dataset_path.exists() and protocol_path.exists():
        try:
            dataset_sha=sha256_file(dataset_path)
            examples=load_jsonl(dataset_path)
            frozen=load_protocol(protocol_path)
            verify_personalization_protocol(frozen,dataset_sha256=dataset_sha)
            protocol_sha=str(frozen["protocol_sha256"])
            counts=frozen.get("counts",{})
            dataset_examples=len(examples)
            train_examples=int(counts.get("train",0))
            dev_examples=int(counts.get("dev",0))
            test_examples=int(counts.get("test",0))
            test_source_groups=len({
                str(row.get("source_group_sha256"))
                for row in frozen.get("splits",{}).get("test",[])
                if isinstance(row.get("source_group_sha256"),str)
                and row.get("source_group_sha256")
            })
            if int(counts.get("total",0))!=dataset_examples:
                reasons.append("protocol_dataset_count_mismatch")
            quality=assess_evidence_quality(examples,frozen)
            evidence_quality_status=str(quality["status"])
            evidence_quality_court_sha256=str(quality["court_sha256"])
            evidence_quality_reasons=list(quality["reasons"])
            if evidence_quality_status!="PASS":
                reasons.extend(
                    f"evidence_quality:{reason}"
                    for reason in evidence_quality_reasons
                )
        except Exception as exc:
            reasons.append(f"personalization_protocol_invalid:{type(exc).__name__}")

    anchor_examples=0
    if not anchor_path.exists():
        reasons.append("general_anchor_missing")
    else:
        try:
            anchor_examples=len(load_jsonl(anchor_path))
        except Exception as exc:
            reasons.append(f"general_anchor_invalid:{type(exc).__name__}")

    latent_present=latent_path.exists()
    latent_valid=False
    latent_dim=None
    latent_digest=None
    if not latent_present:
        reasons.append("persistent_latent_missing")
    else:
        try:
            value=LatentStore(latent_path).load()
            if value is None:
                raise ValueError("latent missing")
            latent_valid=True
            latent_dim=int(value.latent_dim)
            latent_digest=str(value.digest)
        except Exception as exc:
            reasons.append(f"persistent_latent_invalid:{type(exc).__name__}")

    requested_revision=None
    resolved_revision=None
    lock_present=lock_path.exists()
    if not lock_present:
        reasons.append("model_lock_missing")
    else:
        try:
            lock=json.loads(lock_path.read_text(encoding="utf-8"))
            requested_revision=str(lock["upstream"]["revision"])
        except Exception as exc:
            reasons.append(f"model_lock_invalid:{type(exc).__name__}")

    marker=model_path/".nolane-model-revision"
    model_present=_model_weights_present(model_path) and marker.exists()
    if not model_present:
        reasons.append("pinned_model_missing")
    elif marker.exists():
        resolved_revision=marker.read_text(encoding="utf-8").strip()

    revision_matches=bool(
        requested_revision
        and resolved_revision
        and requested_revision==resolved_revision
    )
    if model_present and requested_revision and not revision_matches:
        reasons.append("pinned_model_revision_mismatch")

    l14_present=l14_path.exists()
    if not l14_present:
        reasons.append("l14_anchor_candidate_missing")

    if train_examples<t.min_train_examples:
        reasons.append("insufficient_train_examples")
    if dev_examples<t.min_dev_examples:
        reasons.append("insufficient_dev_examples")
    if test_examples<t.min_test_examples:
        reasons.append("insufficient_test_examples")
    if test_source_groups<t.min_test_source_groups:
        reasons.append("insufficient_test_source_groups")
    if anchor_examples<t.min_anchor_examples:
        reasons.append("insufficient_anchor_examples")
    if latent_valid and latent_dim!=t.latent_dim:
        reasons.append("latent_dimension_mismatch")

    evidence=RealCandidateReadinessEvidence(
        dataset_present=dataset_path.exists(),
        protocol_present=protocol_path.exists(),
        dataset_sha256=dataset_sha,
        protocol_sha256=protocol_sha,
        dataset_examples=dataset_examples,
        train_examples=train_examples,
        dev_examples=dev_examples,
        test_examples=test_examples,
        test_source_groups=test_source_groups,
        evidence_quality_status=evidence_quality_status,
        evidence_quality_court_sha256=evidence_quality_court_sha256,
        evidence_quality_reasons=evidence_quality_reasons,
        anchor_examples=anchor_examples,
        latent_present=latent_present,
        latent_valid=latent_valid,
        latent_dim=latent_dim,
        latent_digest=latent_digest,
        model_lock_present=lock_present,
        model_present=model_present,
        requested_model_revision=requested_revision,
        resolved_model_revision=resolved_revision,
        model_revision_matches=revision_matches,
        l14_anchor_present=l14_present,
    )
    return RealCandidateReadinessDecision(
        status="REAL_CANDIDATE_INPUTS_READY" if not reasons else "REAL_CANDIDATE_INPUTS_BLOCKED",
        reasons=reasons,
        evidence=asdict(evidence),
        thresholds=asdict(t),
    )
