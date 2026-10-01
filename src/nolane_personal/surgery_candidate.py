from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .store import canonical_json, payload_digest
from .surgery import LatentAdapterConfig, LatentResidualAdapter, analytical_adapter_parameter_count, module_parameter_digest


SCHEMA = "NOLANE-L5-LATENT-ADAPTER-CANDIDATE-V1"


def model_lock_fingerprint(lock: dict[str, Any]) -> str:
    upstream = lock.get("upstream", {})
    identity = {
        "provider": upstream.get("provider"),
        "repo_id": upstream.get("repo_id"),
        "revision": upstream.get("revision"),
        "architecture": upstream.get("architecture"),
        "model_type": upstream.get("model_type"),
    }
    return payload_digest(identity)


def load_model_lock(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_candidate(
    output_dir: str | Path,
    *,
    hidden_size: int,
    model_lock: dict[str, Any],
    config: LatentAdapterConfig | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install neural support with: pip install -e '.[neural]'") from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = config or LatentAdapterConfig()
    adapter = LatentResidualAdapter(hidden_size, config, seed=seed)
    actual = adapter.parameter_count()
    analytical = analytical_adapter_parameter_count(hidden_size, config)
    if actual != analytical:
        raise RuntimeError(f"adapter parameter audit mismatch: actual={actual}, analytical={analytical}")

    checkpoint_path = output_dir / "latent-adapter.pt"
    torch.save(
        {
            "schema": SCHEMA,
            "hidden_size": int(hidden_size),
            "config": asdict(config),
            "seed": int(seed),
            "model_lock_fingerprint": model_lock_fingerprint(model_lock),
            "model_state": adapter.module.state_dict(),
            "authority": "SHADOW_ONLY",
        },
        checkpoint_path,
    )
    manifest = {
        "schema": SCHEMA,
        "authority": "SHADOW_ONLY",
        "hidden_size": int(hidden_size),
        "config": asdict(config),
        "seed": int(seed),
        "parameter_count": actual,
        "analytical_parameter_count": analytical,
        "model_lock_fingerprint": model_lock_fingerprint(model_lock),
        "upstream_revision": model_lock.get("upstream", {}).get("revision"),
        "adapter_state_digest": module_parameter_digest(adapter.module),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
    }
    manifest["candidate_id"] = payload_digest(manifest)
    manifest_path = output_dir / "latent-adapter-manifest.json"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return manifest


def load_candidate(
    checkpoint_path: str | Path,
    *,
    expected_model_lock_fingerprint: str,
    expected_hidden_size: int,
) -> tuple[LatentResidualAdapter, dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install neural support with: pip install -e '.[neural]'") from exc

    path = Path(checkpoint_path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported adapter candidate schema")
    if payload.get("authority") != "SHADOW_ONLY":
        raise ValueError("adapter candidate authority mismatch")
    if payload.get("model_lock_fingerprint") != expected_model_lock_fingerprint:
        raise ValueError("adapter candidate base-model mismatch")
    if int(payload.get("hidden_size", -1)) != int(expected_hidden_size):
        raise ValueError("adapter candidate hidden-size mismatch")

    config = LatentAdapterConfig(**payload["config"])
    adapter = LatentResidualAdapter(int(payload["hidden_size"]), config, seed=int(payload.get("seed", 0)))
    adapter.module.load_state_dict(payload["model_state"])
    metadata = {
        "schema": payload["schema"],
        "authority": payload["authority"],
        "hidden_size": int(payload["hidden_size"]),
        "config": payload["config"],
        "seed": int(payload.get("seed", 0)),
        "model_lock_fingerprint": payload["model_lock_fingerprint"],
        "adapter_state_digest": module_parameter_digest(adapter.module),
        "checkpoint_sha256": _sha256_file(path),
    }
    return adapter, metadata
