from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from .state import utc_now_iso
from .store import canonical_json, payload_digest
from .transactional_registry import CheckpointRegistry


LEASE_SCHEMA = "NOLANE-L34-SERVING-LEASE-V1"
RELOAD_PLAN_SCHEMA = "NOLANE-L34-SERVING-RELOAD-PLAN-V1"
CONVERGENCE_SCHEMA = "NOLANE-L34-SERVING-CONVERGENCE-V1"
GATE_SCHEMA = "NOLANE-L34-SERVING-GATE-V1"
AUTHORITY = "SERVING_RELOAD_COORDINATION_NO_AUTONOMOUS_PROMOTION_AUTHORITY"


def _process_hash(process_id: str) -> str:
    value = str(process_id).strip()
    if not value:
        raise ValueError("process_id is empty")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        raise ValueError("serving timestamp must be timezone-aware")
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


def _seal(payload: dict[str, Any], digest_field: str) -> dict[str, Any]:
    result = dict(payload)
    result[digest_field] = payload_digest(result)
    return result


def _verify_digest(
    payload: dict[str, Any],
    *,
    schema: str,
    digest_field: str,
    what: str,
) -> dict[str, Any]:
    if payload.get("schema") != schema:
        raise ValueError(f"unsupported {what} schema")
    supplied = payload.get(digest_field)
    body = dict(payload)
    body.pop(digest_field, None)
    if payload_digest(body) != supplied:
        raise ValueError(f"{what} digest mismatch")
    return payload


@dataclass(slots=True)
class ServingLeasePolicy:
    lease_seconds: float = 30.0
    min_live_processes: int = 1

    def validate(self) -> None:
        if self.lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if self.min_live_processes < 1:
            raise ValueError("min_live_processes must be >=1")


def verify_serving_convergence_receipt(
    receipt: dict[str, Any],
    *,
    require_pass: bool = True,
) -> dict[str, Any]:
    _verify_digest(
        receipt,
        schema=CONVERGENCE_SCHEMA,
        digest_field="convergence_sha256",
        what="serving convergence receipt",
    )
    if receipt.get("authority") != AUTHORITY:
        raise ValueError("serving convergence authority mismatch")
    if receipt.get("status") not in {"PASS", "BLOCKED"}:
        raise ValueError("serving convergence status invalid")
    if not isinstance(receipt.get("reasons"), list):
        raise ValueError("serving convergence reasons invalid")
    policy = ServingLeasePolicy(**dict(receipt.get("policy", {})))
    policy.validate()
    if int(receipt.get("active_generation", -1)) < 0:
        raise ValueError("serving convergence generation invalid")
    _validate_sha256(
        receipt.get("active_pointer_sha256"),
        name="active_pointer_sha256",
    )
    _validate_sha256(
        receipt.get("active_checkpoint_sha256"),
        name="active_checkpoint_sha256",
    )
    _validate_sha256(
        receipt.get("convergence_sha256"),
        name="convergence_sha256",
    )
    _parse_time(str(receipt.get("assessed_at")))
    if int(receipt.get("live_processes", -1)) < 0:
        raise ValueError("serving convergence live_processes invalid")
    if int(receipt.get("expired_processes", -1)) < 0:
        raise ValueError("serving convergence expired_processes invalid")
    if require_pass and receipt.get("status") != "PASS":
        raise ValueError("serving convergence is not PASS")
    return receipt


class ServingCoordinator:
    def __init__(
        self,
        registry: CheckpointRegistry,
        *,
        root: str | Path | None = None,
        policy: ServingLeasePolicy | None = None,
    ) -> None:
        self.registry = registry
        self.root = (
            Path(root)
            if root is not None
            else registry.root / "serving"
        )
        self.leases_dir = self.root / "leases"
        self.leases_dir.mkdir(parents=True, exist_ok=True)
        self.policy = policy or ServingLeasePolicy()
        self.policy.validate()

    def _lease_path(self, process_id: str) -> Path:
        return self.leases_dir / f"{_process_hash(process_id)}.json"

    def _verify_pointer_identity(
        self,
        *,
        generation: int,
        pointer_sha256: str,
        checkpoint_sha256: str,
    ) -> dict[str, Any]:
        pointer = self.registry.pointer_for_generation(generation)
        if pointer["pointer_sha256"] != pointer_sha256:
            raise ValueError("serving pointer sha does not match registry history")
        if pointer["checkpoint_sha256"] != checkpoint_sha256:
            raise ValueError("serving checkpoint sha does not match registry pointer")
        self.registry.artifact_path_for_pointer(pointer)
        return pointer

    def register_loaded(
        self,
        process_id: str,
        *,
        pointer: dict[str, Any],
        now: str | None = None,
    ) -> dict[str, Any]:
        process_hash = _process_hash(process_id)
        verified = self._verify_pointer_identity(
            generation=int(pointer["generation"]),
            pointer_sha256=str(pointer["pointer_sha256"]),
            checkpoint_sha256=str(pointer["checkpoint_sha256"]),
        )
        heartbeat = now or utc_now_iso()
        _parse_time(heartbeat)
        lease = _seal(
            {
                "schema": LEASE_SCHEMA,
                "authority": AUTHORITY,
                "process_id_sha256": process_hash,
                "loaded_generation": int(verified["generation"]),
                "loaded_pointer_sha256": verified["pointer_sha256"],
                "loaded_checkpoint_sha256": verified["checkpoint_sha256"],
                "heartbeat_at": heartbeat,
                "lease_seconds": float(self.policy.lease_seconds),
            },
            "lease_sha256",
        )
        _atomic_write_json(self._lease_path(process_id), lease)
        return lease

    def load_lease(self, process_id: str) -> dict[str, Any]:
        path = self._lease_path(process_id)
        if not path.exists():
            raise ValueError("serving process is not registered")
        lease = json.loads(path.read_text(encoding="utf-8"))
        _verify_digest(
            lease,
            schema=LEASE_SCHEMA,
            digest_field="lease_sha256",
            what="serving lease",
        )
        if lease.get("process_id_sha256") != _process_hash(process_id):
            raise ValueError("serving lease process binding mismatch")
        self._verify_pointer_identity(
            generation=int(lease["loaded_generation"]),
            pointer_sha256=str(lease["loaded_pointer_sha256"]),
            checkpoint_sha256=str(lease["loaded_checkpoint_sha256"]),
        )
        return lease

    def heartbeat(
        self,
        process_id: str,
        *,
        now: str | None = None,
    ) -> dict[str, Any]:
        lease = self.load_lease(process_id)
        pointer = self.registry.pointer_for_generation(
            int(lease["loaded_generation"])
        )
        return self.register_loaded(
            process_id,
            pointer=pointer,
            now=now,
        )

    def reload_plan(self, process_id: str) -> dict[str, Any] | None:
        lease = self.load_lease(process_id)
        active = self.registry.active_pointer()
        if lease["loaded_pointer_sha256"] == active["pointer_sha256"]:
            return None
        return _seal(
            {
                "schema": RELOAD_PLAN_SCHEMA,
                "authority": AUTHORITY,
                "process_id_sha256": _process_hash(process_id),
                "from_generation": int(lease["loaded_generation"]),
                "from_pointer_sha256": lease["loaded_pointer_sha256"],
                "from_checkpoint_sha256": lease["loaded_checkpoint_sha256"],
                "target_generation": int(active["generation"]),
                "target_pointer_sha256": active["pointer_sha256"],
                "target_checkpoint_sha256": active["checkpoint_sha256"],
                "created_at": utc_now_iso(),
            },
            "plan_sha256",
        )

    def verify_reload_plan(
        self,
        process_id: str,
        plan: dict[str, Any],
    ) -> dict[str, Any]:
        _verify_digest(
            plan,
            schema=RELOAD_PLAN_SCHEMA,
            digest_field="plan_sha256",
            what="serving reload plan",
        )
        if plan.get("process_id_sha256") != _process_hash(process_id):
            raise ValueError("reload plan process binding mismatch")
        self._verify_pointer_identity(
            generation=int(plan["target_generation"]),
            pointer_sha256=str(plan["target_pointer_sha256"]),
            checkpoint_sha256=str(plan["target_checkpoint_sha256"]),
        )
        return plan

    def ack_reload(
        self,
        process_id: str,
        *,
        plan: dict[str, Any],
        loaded_checkpoint_sha256: str,
        now: str | None = None,
    ) -> dict[str, Any]:
        plan = self.verify_reload_plan(process_id, plan)
        active = self.registry.active_pointer()
        if active["pointer_sha256"] != plan["target_pointer_sha256"]:
            raise RuntimeError("active checkpoint changed during serving reload")
        if str(loaded_checkpoint_sha256) != active["checkpoint_sha256"]:
            raise ValueError("loaded checkpoint does not match reload target")
        return self.register_loaded(
            process_id,
            pointer=active,
            now=now,
        )

    def _all_leases(self) -> list[dict[str, Any]]:
        leases: list[dict[str, Any]] = []
        for path in sorted(self.leases_dir.glob("*.json")):
            lease = json.loads(path.read_text(encoding="utf-8"))
            _verify_digest(
                lease,
                schema=LEASE_SCHEMA,
                digest_field="lease_sha256",
                what="serving lease",
            )
            self._verify_pointer_identity(
                generation=int(lease["loaded_generation"]),
                pointer_sha256=str(lease["loaded_pointer_sha256"]),
                checkpoint_sha256=str(lease["loaded_checkpoint_sha256"]),
            )
            leases.append(lease)
        return leases

    def assess_convergence(
        self,
        *,
        now: str | None = None,
    ) -> dict[str, Any]:
        active = self.registry.active_pointer()
        current = _parse_time(now or utc_now_iso())
        live: list[dict[str, Any]] = []
        expired = 0
        for lease in self._all_leases():
            heartbeat = _parse_time(lease["heartbeat_at"])
            deadline = heartbeat + timedelta(
                seconds=float(lease["lease_seconds"])
            )
            if current <= deadline:
                live.append(lease)
            else:
                expired += 1

        reasons: list[str] = []
        if len(live) < self.policy.min_live_processes:
            reasons.append("insufficient_live_serving_processes")

        loaded_pointers = {
            str(lease["loaded_pointer_sha256"])
            for lease in live
        }
        if len(loaded_pointers) > 1:
            reasons.append("split_brain_detected")
        if any(
            lease["loaded_pointer_sha256"] != active["pointer_sha256"]
            for lease in live
        ):
            reasons.append("live_processes_not_on_active_pointer")

        process_rows: list[dict[str, Any]] = []
        for index, lease in enumerate(live):
            process_rows.append(
                {
                    "process_alias": f"process_{index:03d}",
                    "loaded_generation": lease["loaded_generation"],
                    "loaded_pointer_sha256": lease["loaded_pointer_sha256"],
                    "loaded_checkpoint_sha256": lease["loaded_checkpoint_sha256"],
                    "on_active_pointer": (
                        lease["loaded_pointer_sha256"]
                        == active["pointer_sha256"]
                    ),
                }
            )

        final_active = self.registry.active_pointer()
        if final_active["pointer_sha256"] != active["pointer_sha256"]:
            reasons.append("active_pointer_changed_during_convergence")

        receipt = _seal(
            {
                "schema": CONVERGENCE_SCHEMA,
                "authority": AUTHORITY,
                "status": "PASS" if not reasons else "BLOCKED",
                "reasons": sorted(set(reasons)),
                "policy": asdict(self.policy),
                "active_generation": active["generation"],
                "active_pointer_sha256": active["pointer_sha256"],
                "active_checkpoint_sha256": active["checkpoint_sha256"],
                "assessed_at": current.isoformat(),
                "live_processes": len(live),
                "expired_processes": expired,
                "processes": process_rows,
                "privacy": {
                    "contains_raw_process_id": False,
                },
            },
            "convergence_sha256",
        )
        return receipt

    def serving_gate(
        self,
        process_id: str,
        *,
        now: str | None = None,
    ) -> dict[str, Any]:
        lease = self.load_lease(process_id)
        current = _parse_time(now or utc_now_iso())
        heartbeat = _parse_time(lease["heartbeat_at"])
        expired = current > heartbeat + timedelta(
            seconds=float(lease["lease_seconds"])
        )
        convergence = self.assess_convergence(now=now)
        active = self.registry.active_pointer()

        if expired:
            status = "LEASE_EXPIRED"
        elif lease["loaded_pointer_sha256"] != active["pointer_sha256"]:
            status = "DRAIN_RELOAD_REQUIRED"
        elif convergence["active_pointer_sha256"] != active["pointer_sha256"]:
            status = "WAITING_FOR_PEERS"
        elif convergence["status"] != "PASS":
            status = "WAITING_FOR_PEERS"
        else:
            status = "SERVE"

        return _seal(
            {
                "schema": GATE_SCHEMA,
                "authority": AUTHORITY,
                "status": status,
                "process_id_sha256": _process_hash(process_id),
                "loaded_generation": lease["loaded_generation"],
                "loaded_pointer_sha256": lease["loaded_pointer_sha256"],
                "active_generation": active["generation"],
                "active_pointer_sha256": active["pointer_sha256"],
                "convergence_sha256": convergence["convergence_sha256"],
            },
            "gate_sha256",
        )

    def verify_request_fence(
        self,
        process_id: str,
        gate: dict[str, Any],
    ) -> dict[str, Any]:
        _verify_digest(
            gate,
            schema=GATE_SCHEMA,
            digest_field="gate_sha256",
            what="serving gate",
        )
        if gate.get("process_id_sha256") != _process_hash(process_id):
            raise ValueError("serving gate process binding mismatch")
        if gate.get("status") != "SERVE":
            raise RuntimeError(
                f"serving gate blocked request: {gate.get('status')}"
            )
        active = self.registry.active_pointer()
        if gate.get("active_pointer_sha256") != active["pointer_sha256"]:
            raise RuntimeError(
                "request fence invalidated by active checkpoint change"
            )
        lease = self.load_lease(process_id)
        if lease["loaded_pointer_sha256"] != active["pointer_sha256"]:
            raise RuntimeError(
                "request fence invalidated by local checkpoint drift"
            )
        return gate


class ServingSession:
    """One process-local serving model with safe reload gating.

    loader(bundle_path, pointer) must return (model_object, checkpoint_sha256).
    The local model reference is swapped before the new lease is acknowledged.
    Until acknowledgement and cluster convergence, serving_gate() cannot return
    SERVE.
    """

    def __init__(
        self,
        coordinator: ServingCoordinator,
        *,
        process_id: str,
        loader: Callable[[Path, dict[str, Any]], tuple[Any, str]],
    ) -> None:
        self.coordinator = coordinator
        self.process_id = str(process_id)
        self.loader = loader
        self.model: Any | None = None
        self.pointer: dict[str, Any] | None = None
        self._lock = threading.RLock()

    def start(self, *, now: str | None = None) -> Any:
        with self._lock:
            pointer = self.coordinator.registry.active_pointer()
            bundle = self.coordinator.registry.artifact_path_for_pointer(pointer)
            model, checkpoint_sha = self.loader(bundle, pointer)
            if checkpoint_sha != pointer["checkpoint_sha256"]:
                raise ValueError("serving loader returned wrong checkpoint")
            self.model = model
            self.pointer = pointer
            self.coordinator.register_loaded(
                self.process_id,
                pointer=pointer,
                now=now,
            )
            return model

    def poll_reload(self, *, now: str | None = None) -> bool:
        with self._lock:
            if self.model is None or self.pointer is None:
                raise RuntimeError("serving session has not started")
            plan = self.coordinator.reload_plan(self.process_id)
            if plan is None:
                self.coordinator.heartbeat(self.process_id, now=now)
                return False
            target = self.coordinator.registry.pointer_for_generation(
                int(plan["target_generation"])
            )
            bundle = self.coordinator.registry.artifact_path_for_pointer(target)
            new_model, checkpoint_sha = self.loader(bundle, target)
            if checkpoint_sha != target["checkpoint_sha256"]:
                raise ValueError("serving loader returned wrong reload checkpoint")

            self.model = new_model
            self.pointer = target
            self.coordinator.ack_reload(
                self.process_id,
                plan=plan,
                loaded_checkpoint_sha256=checkpoint_sha,
                now=now,
            )
            return True

    def gate(self, *, now: str | None = None) -> dict[str, Any]:
        with self._lock:
            if self.model is None or self.pointer is None:
                raise RuntimeError("serving session has not started")
            return self.coordinator.serving_gate(
                self.process_id,
                now=now,
            )

    @contextmanager
    def request_model(self, *, now: str | None = None):
        """Hold one loaded generation stable for an in-flight request."""
        with self._lock:
            gate = self.gate(now=now)
            self.coordinator.verify_request_fence(
                self.process_id,
                gate,
            )
            yield self.model

    def model_for_request(self, *, now: str | None = None) -> Any:
        """Admission helper; production inference should prefer request_model()."""
        with self.request_model(now=now) as model:
            return model


def factorized_loader(
    latent_values: list[float],
    *,
    device: str = "cpu",
) -> Callable[[Path, dict[str, Any]], tuple[Any, str]]:
    def load(bundle: Path, pointer: dict[str, Any]) -> tuple[Any, str]:
        from .factorized_artifact import load_factorized_model

        model, meta = load_factorized_model(
            bundle / "factorized-nolane.pt",
            latent_values,
            device=device,
        )
        checkpoint_sha = str(meta["checkpoint_sha256"])
        if checkpoint_sha != pointer["checkpoint_sha256"]:
            raise ValueError("factorized serving checkpoint mismatch")
        return model, checkpoint_sha

    return load
