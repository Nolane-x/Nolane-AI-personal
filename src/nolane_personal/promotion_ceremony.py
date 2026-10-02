from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .promotion_authority_dispatch import (
    authorization_chain_sha256,
    authorization_kind,
    verify_any_promotion_authorization,
)
from .serving_coordination import verify_serving_convergence_receipt
from .state import utc_now_iso
from .store import canonical_json, payload_digest
from .transactional_registry import CheckpointRegistry


SCHEMA = "NOLANE-L36-PROMOTION-CEREMONY-V1"
AUTHORITY = "FINAL_PROMOTION_CEREMONY_EVIDENCE"


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        raise ValueError("promotion ceremony timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _validate_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a sha256 hex string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a sha256 hex string") from exc
    return value


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
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
            handle.write(canonical_json(payload) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def verify_promotion_ceremony_receipt(
    receipt: dict[str, Any],
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    if receipt.get("schema") != SCHEMA:
        raise ValueError("unsupported promotion ceremony schema")
    supplied = receipt.get("ceremony_sha256")
    body = dict(receipt)
    body.pop("ceremony_sha256", None)
    if payload_digest(body) != supplied:
        raise ValueError("promotion ceremony receipt digest mismatch")
    if receipt.get("authority") != AUTHORITY:
        raise ValueError("promotion ceremony authority mismatch")
    if receipt.get("status") not in {"COMPLETE", "BLOCKED"}:
        raise ValueError("promotion ceremony status invalid")
    if not isinstance(receipt.get("reasons"), list):
        raise ValueError("promotion ceremony reasons invalid")
    for key in (
        "authorization_sha256",
        "multicycle_chain_sha256",
        "long_horizon_retention_court_sha256",
        "candidate_checkpoint_sha256",
        "pointer_sha256",
        "serving_convergence_sha256",
        "ceremony_sha256",
    ):
        _validate_sha256(receipt.get(key), name=key)
    if not str(receipt.get("transaction_id", "")).strip():
        raise ValueError("promotion ceremony transaction_id missing")
    if int(receipt.get("pointer_generation", -1)) < 0:
        raise ValueError("promotion ceremony generation invalid")
    for key in (
        "authorization_issued_at",
        "authorization_expires_at",
        "transaction_prepared_at",
        "transaction_committed_at",
        "pointer_created_at",
        "serving_convergence_assessed_at",
        "ceremony_at",
    ):
        _parse_time(str(receipt.get(key)))
    if require_complete and receipt.get("status") != "COMPLETE":
        raise ValueError("promotion ceremony is not COMPLETE")
    return receipt


def verify_promotion_ceremony_against_registry(
    registry: CheckpointRegistry,
    receipt: dict[str, Any],
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    verify_promotion_ceremony_receipt(
        receipt,
        require_complete=require_complete,
    )
    transaction_id = str(receipt.get("transaction_id", ""))
    events = registry.transaction_events(transaction_id)
    if not events or events[0].get("state") != "PREPARED":
        raise ValueError("promotion ceremony transaction is not PREPARED")

    terminal = [
        event
        for event in events
        if event.get("state") in {"COMMITTED", "RECOVERED_COMMITTED"}
    ]
    if len(terminal) != 1:
        raise ValueError(
            "promotion ceremony transaction must have one committed terminal event"
        )
    pointer_sha = str(
        terminal[0].get("details", {}).get("pointer_sha256", "")
    )
    if pointer_sha != receipt.get("pointer_sha256"):
        raise ValueError("promotion ceremony committed pointer mismatch")
    pointer = registry.pointer_by_sha256(pointer_sha)
    if pointer.get("transaction_id") != transaction_id:
        raise ValueError("promotion ceremony pointer transaction mismatch")
    if int(pointer["generation"]) != int(receipt["pointer_generation"]):
        raise ValueError("promotion ceremony pointer generation mismatch")
    if pointer["checkpoint_sha256"] != receipt["candidate_checkpoint_sha256"]:
        raise ValueError("promotion ceremony candidate checkpoint mismatch")

    authorization = registry.promotion_authorization_for_transaction(
        transaction_id
    )
    if (
        authorization["authorization_sha256"]
        != receipt["authorization_sha256"]
    ):
        raise ValueError("promotion ceremony authorization mismatch")
    if (
        authorization_chain_sha256(authorization)
        != receipt["multicycle_chain_sha256"]
    ):
        raise ValueError("promotion ceremony continual-chain evidence mismatch")
    if (
        authorization["long_horizon_retention_court_sha256"]
        != receipt["long_horizon_retention_court_sha256"]
    ):
        raise ValueError("promotion ceremony long-horizon evidence mismatch")
    return receipt


def ceremony_path(
    registry: CheckpointRegistry,
    receipt: dict[str, Any],
) -> Path:
    verify_promotion_ceremony_receipt(receipt, require_complete=True)
    generation = int(receipt["pointer_generation"])
    return registry.root / "ceremonies" / f"{generation:012d}.json"


def persist_promotion_ceremony(
    registry: CheckpointRegistry,
    receipt: dict[str, Any],
) -> Path:
    verify_promotion_ceremony_against_registry(
        registry,
        receipt,
        require_complete=True,
    )
    path = ceremony_path(registry, receipt)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        verify_promotion_ceremony_against_registry(
            registry,
            existing,
            require_complete=True,
        )
        if existing != receipt:
            raise RuntimeError(
                "promotion ceremony generation already has different evidence"
            )
        return path
    _atomic_write_json(path, receipt)
    return path


def load_promotion_ceremony(
    registry: CheckpointRegistry,
    generation: int,
) -> dict[str, Any]:
    path = registry.root / "ceremonies" / f"{int(generation):012d}.json"
    if not path.exists():
        raise ValueError("promotion ceremony receipt not found")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    return verify_promotion_ceremony_against_registry(
        registry,
        receipt,
        require_complete=True,
    )


def audit_promotion_ceremonies(
    registry: CheckpointRegistry,
) -> dict[str, Any]:
    directory = registry.root / "ceremonies"
    receipts: list[dict[str, Any]] = []
    if directory.exists():
        for path in sorted(directory.glob("*.json")):
            receipt = json.loads(path.read_text(encoding="utf-8"))
            verify_promotion_ceremony_against_registry(
                registry,
                receipt,
                require_complete=True,
            )
            expected = f"{int(receipt['pointer_generation']):012d}.json"
            if path.name != expected:
                raise ValueError("promotion ceremony filename mismatch")
            receipts.append(receipt)
    return {
        "schema": "NOLANE-L36-PROMOTION-CEREMONY-AUDIT-V1",
        "authority": AUTHORITY,
        "status": "PASS",
        "ceremonies": len(receipts),
        "generations": [
            int(receipt["pointer_generation"])
            for receipt in receipts
        ],
        "ceremony_sha256": [
            receipt["ceremony_sha256"]
            for receipt in receipts
        ],
    }


def finalize_promotion_ceremony(
    registry: CheckpointRegistry,
    *,
    transaction_id: str,
    convergence_receipt: dict[str, Any],
    now: str | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    events = registry.transaction_events(transaction_id)
    if not events:
        raise ValueError("unknown promotion transaction")
    if events[0].get("state") != "PREPARED":
        raise ValueError("promotion transaction does not start PREPARED")

    authorization = registry.promotion_authorization_for_transaction(
        transaction_id
    )
    verify_any_promotion_authorization(
        authorization,
        require_authorized=True,
        check_expiry=False,
    )
    verify_serving_convergence_receipt(
        convergence_receipt,
        require_pass=False,
    )

    committed_states = {"COMMITTED", "RECOVERED_COMMITTED"}
    final_events = [
        event
        for event in events
        if event.get("state") in committed_states
    ]
    if len(final_events) != 1:
        raise ValueError(
            "promotion transaction must have exactly one committed terminal event"
        )
    committed_event = final_events[0]

    pointer_sha = str(
        committed_event.get("details", {}).get("pointer_sha256", "")
    )
    pointer = registry.pointer_by_sha256(pointer_sha)
    if pointer.get("transaction_id") != transaction_id:
        raise ValueError("committed pointer transaction binding mismatch")
    if pointer.get("transition") != "UPDATE":
        raise ValueError(
            "promotion ceremony requires UPDATE pointer transition"
        )
    if (
        pointer.get("checkpoint_sha256")
        != authorization.get("candidate_checkpoint_sha256")
    ):
        raise ValueError(
            "authorized candidate does not match committed pointer"
        )

    prepared_at = _parse_time(str(events[0].get("created_at")))
    pointer_at = _parse_time(str(pointer.get("created_at")))
    issued_at = _parse_time(str(authorization.get("issued_at")))
    expires_at = _parse_time(str(authorization.get("expires_at")))
    convergence_at = _parse_time(
        str(convergence_receipt.get("assessed_at"))
    )
    ceremony_at = _parse_time(now or utc_now_iso())
    committed_at = _parse_time(str(committed_event.get("created_at")))

    reasons: list[str] = []
    active = registry.active_pointer()

    if prepared_at < issued_at:
        reasons.append("transaction_predates_authorization")
    if committed_at < prepared_at:
        reasons.append("commit_predates_transaction")
    if pointer_at < prepared_at:
        reasons.append("pointer_predates_transaction")
    if committed_at < pointer_at:
        reasons.append("commit_predates_pointer_swap")
    if pointer_at > expires_at:
        reasons.append("pointer_swap_after_authorization_expiry")
    if convergence_at < pointer_at:
        reasons.append("serving_convergence_predates_pointer_swap")
    if convergence_at < committed_at:
        reasons.append("serving_convergence_predates_commit")
    if ceremony_at < convergence_at:
        reasons.append("ceremony_predates_serving_convergence")

    if active["pointer_sha256"] != pointer["pointer_sha256"]:
        reasons.append("active_pointer_moved_before_ceremony")

    if convergence_receipt.get("status") != "PASS":
        reasons.append("serving_convergence_not_pass")
    if (
        convergence_receipt.get("active_pointer_sha256")
        != pointer["pointer_sha256"]
    ):
        reasons.append("serving_convergence_pointer_mismatch")
    if (
        convergence_receipt.get("active_checkpoint_sha256")
        != pointer["checkpoint_sha256"]
    ):
        reasons.append("serving_convergence_checkpoint_mismatch")
    if int(convergence_receipt.get("active_generation", -1)) != int(
        pointer["generation"]
    ):
        reasons.append("serving_convergence_generation_mismatch")

    receipt = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "status": "COMPLETE" if not reasons else "BLOCKED",
        "reasons": sorted(set(reasons)),
        "transaction_id": transaction_id,
        "authorization_sha256": authorization["authorization_sha256"],
        "authorization_schema": authorization["schema"],
        "authorization_kind": authorization_kind(authorization),
        "multicycle_chain_sha256": authorization_chain_sha256(
            authorization
        ),
        "long_horizon_retention_court_sha256": authorization[
            "long_horizon_retention_court_sha256"
        ],
        "candidate_checkpoint_sha256": pointer["checkpoint_sha256"],
        "pointer_generation": pointer["generation"],
        "pointer_sha256": pointer["pointer_sha256"],
        "serving_convergence_sha256": convergence_receipt[
            "convergence_sha256"
        ],
        "authorization_issued_at": authorization["issued_at"],
        "authorization_expires_at": authorization["expires_at"],
        "transaction_prepared_at": events[0]["created_at"],
        "transaction_committed_at": committed_event["created_at"],
        "pointer_created_at": pointer["created_at"],
        "serving_convergence_assessed_at": convergence_receipt[
            "assessed_at"
        ],
        "ceremony_at": ceremony_at.isoformat(),
        "privacy": {
            "contains_raw_prompt_target": False,
            "contains_raw_operator_nonce": False,
            "contains_raw_process_id": False,
        },
    }
    receipt["ceremony_sha256"] = payload_digest(receipt)
    verify_promotion_ceremony_receipt(
        receipt,
        require_complete=False,
    )
    if receipt["status"] == "COMPLETE" and persist:
        persist_promotion_ceremony(registry, receipt)
    return receipt
