from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .promotion_authority import verify_promotion_authorization
from .serving_coordination import verify_serving_convergence_receipt
from .state import utc_now_iso
from .store import payload_digest
from .transactional_registry import CheckpointRegistry


SCHEMA = "NOLANE-L36-PROMOTION-CEREMONY-V1"
AUTHORITY = "FINAL_PROMOTION_CEREMONY_EVIDENCE"


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        raise ValueError("promotion ceremony timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def finalize_promotion_ceremony(
    registry: CheckpointRegistry,
    *,
    transaction_id: str,
    convergence_receipt: dict[str, Any],
    now: str | None = None,
) -> dict[str, Any]:
    events = registry.transaction_events(transaction_id)
    if not events:
        raise ValueError("unknown promotion transaction")
    if events[0].get("state") != "PREPARED":
        raise ValueError("promotion transaction does not start PREPARED")

    authorization = registry.promotion_authorization_for_transaction(
        transaction_id
    )
    verify_promotion_authorization(
        authorization,
        require_authorized=True,
        check_expiry=False,
    )
    verify_serving_convergence_receipt(
        convergence_receipt,
        require_pass=False,
    )

    committed_states = {
        "COMMITTED",
        "RECOVERED_COMMITTED",
    }
    final_events = [
        event for event in events
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
        raise ValueError("promotion ceremony requires UPDATE pointer transition")
    if (
        pointer.get("checkpoint_sha256")
        != authorization.get("candidate_checkpoint_sha256")
    ):
        raise ValueError("authorized candidate does not match committed pointer")

    prepared_at = _parse_time(str(events[0].get("created_at")))
    pointer_at = _parse_time(str(pointer.get("created_at")))
    issued_at = _parse_time(str(authorization.get("issued_at")))
    expires_at = _parse_time(str(authorization.get("expires_at")))
    convergence_at = _parse_time(
        str(convergence_receipt.get("assessed_at"))
    )
    ceremony_at = _parse_time(now or utc_now_iso())

    reasons: list[str] = []
    active = registry.active_pointer()

    if prepared_at < issued_at:
        reasons.append("transaction_predates_authorization")
    if pointer_at > expires_at:
        reasons.append("pointer_swap_after_authorization_expiry")
    if convergence_at < pointer_at:
        reasons.append("serving_convergence_predates_pointer_swap")
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
        "multicycle_chain_sha256": authorization["multicycle_chain_sha256"],
        "long_horizon_retention_court_sha256": (
            authorization["long_horizon_retention_court_sha256"]
        ),
        "candidate_checkpoint_sha256": pointer["checkpoint_sha256"],
        "pointer_generation": pointer["generation"],
        "pointer_sha256": pointer["pointer_sha256"],
        "serving_convergence_sha256": (
            convergence_receipt["convergence_sha256"]
        ),
        "authorization_issued_at": authorization["issued_at"],
        "authorization_expires_at": authorization["expires_at"],
        "transaction_prepared_at": events[0]["created_at"],
        "pointer_created_at": pointer["created_at"],
        "serving_convergence_assessed_at": (
            convergence_receipt["assessed_at"]
        ),
        "ceremony_at": ceremony_at.isoformat(),
        "privacy": {
            "contains_raw_prompt_target": False,
            "contains_raw_operator_nonce": False,
            "contains_raw_process_id": False,
        },
    }
    receipt["ceremony_sha256"] = payload_digest(receipt)
    return receipt


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
    for key in (
        "authorization_issued_at",
        "authorization_expires_at",
        "transaction_prepared_at",
        "pointer_created_at",
        "serving_convergence_assessed_at",
        "ceremony_at",
    ):
        _parse_time(str(receipt.get(key)))
    if require_complete and receipt.get("status") != "COMPLETE":
        raise ValueError("promotion ceremony is not COMPLETE")
    return receipt
