from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .approved_evidence import resolve_approved_evidence_pack
from .latent import LatentStore
from .real_candidate_readiness import assess_real_candidate_readiness, sha256_file
from .store import canonical_json, payload_digest


SCHEMA = "NOLANE-L25-LOCAL-EVIDENCE-WORKSPACE-V1"


@dataclass(slots=True)
class EvidenceWorkspacePaths:
    evidence_pack_manifest: str
    dataset: str
    protocol: str
    anchor: str
    latent: str
    model_lock: str
    model_dir: str
    l14_anchor: str


@dataclass(slots=True)
class EvidenceWorkspaceBinding:
    evidence_manifest_sha256: str
    evidence_manifest_file_sha256: str
    dataset_sha256: str
    protocol_sha256: str
    anchor_sha256: str
    latent_file_sha256: str
    latent_digest: str
    model_lock_sha256: str
    requested_model_revision: str
    resolved_model_revision: str
    model_config_sha256: str
    model_weight_files: list[dict[str, Any]]
    l14_anchor_sha256: str


def _resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _weight_inventory(model_dir: Path) -> list[dict[str, Any]]:
    paths: list[Path] = []
    single = model_dir / "model.safetensors"
    if single.exists():
        paths.append(single)
    paths.extend(sorted(model_dir.glob("model-*.safetensors")))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    if not unique:
        raise ValueError("pinned model has no safetensors weight files")
    return [
        {
            "filename": path.name,
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for path in unique
    ]


def _workspace_body(
    *,
    paths: EvidenceWorkspacePaths,
    binding: EvidenceWorkspaceBinding,
    readiness: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "authority": "LOCAL_EXECUTION_WORKSPACE_UNPROMOTED",
        "status": (
            "LOCAL_EVIDENCE_WORKSPACE_READY"
            if readiness.get("status") == "REAL_CANDIDATE_INPUTS_READY"
            else "LOCAL_EVIDENCE_WORKSPACE_BLOCKED"
        ),
        "paths": asdict(paths),
        "binding": asdict(binding),
        "readiness": readiness,
        "privacy": {
            "copies_private_dataset": False,
            "contains_prompt_target_text": False,
            "contains_local_paths": True,
            "local_only_required": True,
        },
    }


def prepare_evidence_workspace(
    *,
    evidence_pack_manifest: str | Path,
    anchor: str | Path,
    latent: str | Path,
    model_lock: str | Path,
    model_dir: str | Path,
    l14_anchor: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    output = _resolve(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty evidence workspace: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)

    manifest_path = _resolve(evidence_pack_manifest)
    approved_manifest, dataset_path, protocol_path = resolve_approved_evidence_pack(
        manifest_path
    )
    anchor_path = _resolve(anchor)
    latent_path = _resolve(latent)
    lock_path = _resolve(model_lock)
    model_path = _resolve(model_dir)
    l14_path = _resolve(l14_anchor)

    readiness = assess_real_candidate_readiness(
        dataset=dataset_path,
        protocol=protocol_path,
        anchor=anchor_path,
        latent=latent_path,
        model_lock=lock_path,
        model_dir=model_path,
        l14_anchor=l14_path,
    )

    # These are required to build a trustworthy binding even when readiness is blocked.
    if not anchor_path.exists():
        raise FileNotFoundError(anchor_path)
    if not latent_path.exists():
        raise FileNotFoundError(latent_path)
    if not lock_path.exists():
        raise FileNotFoundError(lock_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not l14_path.exists():
        raise FileNotFoundError(l14_path)
    config_path = model_path / "config.json"
    marker_path = model_path / ".nolane-model-revision"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not marker_path.exists():
        raise FileNotFoundError(marker_path)

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    requested_revision = str(lock["upstream"]["revision"])
    resolved_revision = marker_path.read_text(encoding="utf-8").strip()

    latent_value = LatentStore(latent_path).load()
    if latent_value is None:
        raise ValueError("persistent latent missing")

    paths = EvidenceWorkspacePaths(
        evidence_pack_manifest=str(manifest_path),
        dataset=str(dataset_path.resolve()),
        protocol=str(protocol_path.resolve()),
        anchor=str(anchor_path),
        latent=str(latent_path),
        model_lock=str(lock_path),
        model_dir=str(model_path),
        l14_anchor=str(l14_path),
    )
    binding = EvidenceWorkspaceBinding(
        evidence_manifest_sha256=str(approved_manifest["manifest_sha256"]),
        evidence_manifest_file_sha256=sha256_file(manifest_path),
        dataset_sha256=sha256_file(dataset_path),
        protocol_sha256=sha256_file(protocol_path),
        anchor_sha256=sha256_file(anchor_path),
        latent_file_sha256=sha256_file(latent_path),
        latent_digest=str(latent_value.digest),
        model_lock_sha256=sha256_file(lock_path),
        requested_model_revision=requested_revision,
        resolved_model_revision=resolved_revision,
        model_config_sha256=sha256_file(config_path),
        model_weight_files=_weight_inventory(model_path),
        l14_anchor_sha256=sha256_file(l14_path),
    )
    receipt = _workspace_body(
        paths=paths,
        binding=binding,
        readiness={
            "status": readiness.status,
            "reasons": list(readiness.reasons),
            "evidence": dict(readiness.evidence),
            "thresholds": dict(readiness.thresholds),
        },
    )
    receipt["workspace_sha256"] = payload_digest(receipt)
    spec_path = output / "evidence-workspace.json"
    spec_path.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return {
        "spec": receipt,
        "spec_path": spec_path,
    }


def _verify_weight_inventory(model_dir: Path, expected: list[dict[str, Any]]) -> None:
    actual = _weight_inventory(model_dir)
    if actual != expected:
        raise ValueError(
            f"model weight inventory mismatch: expected={expected} actual={actual}"
        )


def verify_evidence_workspace(spec_path: str | Path) -> dict[str, Any]:
    path = _resolve(spec_path)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("schema") != SCHEMA:
        raise ValueError("unsupported evidence workspace schema")
    supplied = receipt.get("workspace_sha256")
    body = dict(receipt)
    body.pop("workspace_sha256", None)
    if payload_digest(body) != supplied:
        raise ValueError("evidence workspace digest mismatch")
    if receipt.get("authority") != "LOCAL_EXECUTION_WORKSPACE_UNPROMOTED":
        raise ValueError("evidence workspace authority mismatch")

    paths = receipt["paths"]
    binding = receipt["binding"]
    approved_manifest, dataset_path, protocol_path = resolve_approved_evidence_pack(
        paths["evidence_pack_manifest"]
    )
    if str(approved_manifest["manifest_sha256"]) != binding["evidence_manifest_sha256"]:
        raise ValueError("approved evidence manifest lineage mismatch")
    if sha256_file(paths["evidence_pack_manifest"]) != binding["evidence_manifest_file_sha256"]:
        raise ValueError("approved evidence manifest file digest mismatch")
    if str(dataset_path.resolve()) != str(_resolve(paths["dataset"])):
        raise ValueError("approved evidence dataset path mismatch")
    if str(protocol_path.resolve()) != str(_resolve(paths["protocol"])):
        raise ValueError("approved evidence protocol path mismatch")

    checks = (
        ("dataset", "dataset_sha256"),
        ("protocol", "protocol_sha256"),
        ("anchor", "anchor_sha256"),
        ("latent", "latent_file_sha256"),
        ("model_lock", "model_lock_sha256"),
        ("l14_anchor", "l14_anchor_sha256"),
    )
    for path_key, digest_key in checks:
        if sha256_file(paths[path_key]) != binding[digest_key]:
            raise ValueError(f"{path_key} digest mismatch")

    latent_value = LatentStore(paths["latent"]).load()
    if latent_value is None or str(latent_value.digest) != binding["latent_digest"]:
        raise ValueError("persistent latent digest mismatch")

    model_dir = _resolve(paths["model_dir"])
    config_path = model_dir / "config.json"
    marker_path = model_dir / ".nolane-model-revision"
    if sha256_file(config_path) != binding["model_config_sha256"]:
        raise ValueError("model config digest mismatch")
    resolved_revision = marker_path.read_text(encoding="utf-8").strip()
    if resolved_revision != binding["resolved_model_revision"]:
        raise ValueError("model resolved revision mismatch")
    lock = json.loads(_resolve(paths["model_lock"]).read_text(encoding="utf-8"))
    requested_revision = str(lock["upstream"]["revision"])
    if requested_revision != binding["requested_model_revision"]:
        raise ValueError("model requested revision mismatch")
    if requested_revision != resolved_revision:
        raise ValueError("pinned model revision mismatch")
    _verify_weight_inventory(model_dir, binding["model_weight_files"])

    readiness = assess_real_candidate_readiness(
        dataset=paths["dataset"],
        protocol=paths["protocol"],
        anchor=paths["anchor"],
        latent=paths["latent"],
        model_lock=paths["model_lock"],
        model_dir=paths["model_dir"],
        l14_anchor=paths["l14_anchor"],
    )
    if readiness.status != "REAL_CANDIDATE_INPUTS_READY":
        raise ValueError(
            f"workspace no longer readiness-valid: reasons={readiness.reasons}"
        )
    if receipt.get("status") != "LOCAL_EVIDENCE_WORKSPACE_READY":
        raise ValueError("workspace spec was not prepared from ready inputs")
    return receipt
