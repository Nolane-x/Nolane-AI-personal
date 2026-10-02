from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .multicycle_continual import verify_l31_run_receipt
from .promotion_authority import verify_promotion_authorization
from .state import utc_now_iso
from .store import canonical_json, payload_digest


POINTER_SCHEMA = "NOLANE-L33-ACTIVE-CHECKPOINT-POINTER-V1"
TX_EVENT_SCHEMA = "NOLANE-L33-CHECKPOINT-TRANSACTION-EVENT-V1"
ROLLBACK_SCHEMA = "NOLANE-L33-CHECKPOINT-ROLLBACK-V1"
REGISTRY_AUTHORITY = "TRANSACTIONAL_CHECKPOINT_SUBSTRATE_NO_AUTONOMOUS_PROD_AUTHORITY"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except (AttributeError, OSError):
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


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
        _fsync_dir(path.parent)
    finally:
        if tmp.exists():
            tmp.unlink()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _directory_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = item.relative_to(path).as_posix()
        rows.append(
            {
                "path": rel,
                "bytes": item.stat().st_size,
                "sha256": sha256_file(item),
            }
        )
    return rows


def _bundle_digest(path: Path) -> str:
    return payload_digest(_directory_manifest(path))


def _verify_factorized_bundle(path: Path) -> dict[str, Any]:
    checkpoint = path / "factorized-nolane.pt"
    manifest_path = path / "factorized-nolane-manifest.json"
    if not checkpoint.is_file():
        raise ValueError("factorized checkpoint missing from bundle")
    if not manifest_path.is_file():
        raise ValueError("factorized manifest missing from bundle")
    manifest = _read_json(manifest_path)
    checkpoint_sha = sha256_file(checkpoint)
    if manifest.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("factorized manifest checkpoint digest mismatch")
    if manifest.get("authority") != "FACTORIZED_CANDIDATE_UNPROMOTED":
        raise ValueError("factorized bundle authority mismatch")
    return {
        "checkpoint_sha256": checkpoint_sha,
        "manifest": manifest,
        "bundle_sha256": _bundle_digest(path),
    }


def verify_l31_candidate_bundle(path: str | Path) -> dict[str, Any]:
    bundle = Path(path)
    evidence = _verify_factorized_bundle(bundle)
    receipt_path = bundle / "l31-run-receipt.json"
    if not receipt_path.is_file():
        raise ValueError("L31 run receipt missing from candidate bundle")
    run = _read_json(receipt_path)
    verify_l31_run_receipt(run)
    if (
        run["artifact"]["checkpoint_sha256"]
        != evidence["checkpoint_sha256"]
    ):
        raise ValueError("L31 receipt candidate checkpoint mismatch")
    if run["artifact"] != evidence["manifest"]:
        raise ValueError("L31 receipt artifact manifest mismatch")
    evidence["run_receipt"] = run
    evidence["run_receipt_sha256"] = sha256_file(receipt_path)
    return evidence


def _seal_pointer(body: dict[str, Any]) -> dict[str, Any]:
    pointer = dict(body)
    pointer["pointer_sha256"] = payload_digest(pointer)
    return pointer


def _verify_pointer(pointer: dict[str, Any]) -> dict[str, Any]:
    if pointer.get("schema") != POINTER_SCHEMA:
        raise ValueError("unsupported checkpoint pointer schema")
    supplied = pointer.get("pointer_sha256")
    body = dict(pointer)
    body.pop("pointer_sha256", None)
    if payload_digest(body) != supplied:
        raise ValueError("checkpoint pointer digest mismatch")
    if pointer.get("authority") != REGISTRY_AUTHORITY:
        raise ValueError("checkpoint pointer authority mismatch")
    if int(pointer.get("generation", -1)) < 0:
        raise ValueError("checkpoint pointer generation invalid")
    checkpoint = str(pointer.get("checkpoint_sha256", ""))
    if len(checkpoint) != 64:
        raise ValueError("checkpoint pointer sha invalid")
    int(checkpoint, 16)
    return pointer


def _seal_event(body: dict[str, Any]) -> dict[str, Any]:
    event = dict(body)
    event["event_sha256"] = payload_digest(event)
    return event


def _verify_event(event: dict[str, Any]) -> dict[str, Any]:
    if event.get("schema") != TX_EVENT_SCHEMA:
        raise ValueError("unsupported checkpoint transaction event schema")
    supplied = event.get("event_sha256")
    body = dict(event)
    body.pop("event_sha256", None)
    if payload_digest(body) != supplied:
        raise ValueError("checkpoint transaction event digest mismatch")
    return event


def verify_rollback_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    if receipt.get("schema") != ROLLBACK_SCHEMA:
        raise ValueError("unsupported checkpoint rollback schema")
    supplied = receipt.get("rollback_sha256")
    body = dict(receipt)
    body.pop("rollback_sha256", None)
    if payload_digest(body) != supplied:
        raise ValueError("checkpoint rollback receipt digest mismatch")
    if receipt.get("authority") != REGISTRY_AUTHORITY:
        raise ValueError("checkpoint rollback authority mismatch")
    return receipt


class CheckpointRegistry:
    """Crash-recoverable immutable checkpoint registry.

    Candidate artifacts are copied into transaction staging first. Serving
    authority is represented only by one small self-digested pointer file,
    replaced atomically after all candidate/evidence checks pass.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.artifacts_dir = self.root / "artifacts"
        self.transactions_dir = self.root / "transactions"
        self.pointers_dir = self.root / "pointers"
        self.rollbacks_dir = self.root / "rollbacks"
        self.active_path = self.root / "active.json"
        self.lock_path = self.root / ".registry.lock"
        for directory in (
            self.artifacts_dir,
            self.transactions_dir,
            self.pointers_dir,
            self.rollbacks_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def lock(self, *, break_stale_lock: bool = False) -> Iterator[None]:
        if break_stale_lock and self.lock_path.exists():
            self.lock_path.unlink()
        try:
            fd = os.open(
                self.lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
        except FileExistsError as exc:
            raise RuntimeError(
                "checkpoint registry is locked; use explicit recovery to break a stale lock"
            ) from exc
        try:
            payload = {
                "pid": os.getpid(),
                "created_at": utc_now_iso(),
            }
            os.write(fd, (canonical_json(payload) + "\n").encode("utf-8"))
            os.fsync(fd)
            os.close(fd)
            _fsync_dir(self.root)
            yield
        finally:
            try:
                os.close(fd)
            except OSError:
                pass
            if self.lock_path.exists():
                self.lock_path.unlink()
                _fsync_dir(self.root)

    def active_pointer(self) -> dict[str, Any]:
        if not self.active_path.exists():
            raise RuntimeError("checkpoint registry is not initialized")
        return _verify_pointer(_read_json(self.active_path))

    def _pointer_snapshot_path(self, generation: int) -> Path:
        return self.pointers_dir / f"{int(generation):012d}.json"

    def pointer_for_generation(self, generation: int) -> dict[str, Any]:
        path = self._pointer_snapshot_path(int(generation))
        if not path.exists():
            raise ValueError("unknown checkpoint pointer generation")
        return _verify_pointer(_read_json(path))

    def artifact_path_for_pointer(
        self,
        pointer: dict[str, Any],
    ) -> Path:
        verified = _verify_pointer(pointer)
        path = self.root / str(verified["artifact_relpath"])
        evidence = _verify_factorized_bundle(path)
        if evidence["checkpoint_sha256"] != verified["checkpoint_sha256"]:
            raise ValueError("checkpoint pointer artifact digest mismatch")
        return path

    def pointer_by_sha256(self, pointer_sha256: str) -> dict[str, Any]:
        value = str(pointer_sha256)
        if len(value) != 64:
            raise ValueError("pointer sha must be a sha256 hex string")
        try:
            int(value, 16)
        except ValueError as exc:
            raise ValueError("pointer sha must be a sha256 hex string") from exc
        for path in sorted(self.pointers_dir.glob("*.json")):
            pointer = _verify_pointer(_read_json(path))
            if pointer["pointer_sha256"] == value:
                return pointer
        raise ValueError("unknown checkpoint pointer sha")

    def promotion_authorization_for_transaction(
        self,
        transaction_id: str,
    ) -> dict[str, Any]:
        events = self.transaction_events(transaction_id)
        if not events or events[0].get("state") != "PREPARED":
            raise ValueError("transaction has no PREPARED authorization binding")
        prepared = events[0].get("details", {})
        auth_sha = prepared.get("promotion_authorization_sha256")
        if auth_sha is None:
            raise ValueError("transaction is not promotion-authorized")
        auth_path = (
            self._tx_dir(transaction_id)
            / "promotion-authorization.json"
        )
        if not auth_path.exists():
            raise ValueError("promotion authorization file missing")
        authorization = verify_promotion_authorization(
            _read_json(auth_path),
            require_authorized=True,
            check_expiry=False,
        )
        if authorization["authorization_sha256"] != auth_sha:
            raise ValueError("promotion authorization transaction mismatch")
        if (
            authorization["active_parent_checkpoint_sha256"]
            != prepared.get("parent_checkpoint_sha256")
        ):
            raise ValueError("promotion authorization parent binding mismatch")
        if (
            authorization["candidate_checkpoint_sha256"]
            != prepared.get("candidate_checkpoint_sha256")
        ):
            raise ValueError("promotion authorization candidate binding mismatch")
        return authorization

    def _write_pointer_snapshot(self, pointer: dict[str, Any]) -> None:
        path = self._pointer_snapshot_path(int(pointer["generation"]))
        if path.exists():
            existing = _verify_pointer(_read_json(path))
            if existing != pointer:
                raise RuntimeError("pointer generation already exists with different evidence")
            return
        _atomic_write_json(path, pointer)

    def _install_bundle(self, bundle: Path, checkpoint_sha256: str) -> Path:
        destination = self.artifacts_dir / checkpoint_sha256
        if destination.exists():
            existing = _verify_factorized_bundle(destination)
            incoming = _verify_factorized_bundle(bundle)
            if existing["bundle_sha256"] != incoming["bundle_sha256"]:
                raise RuntimeError(
                    "checkpoint SHA collision with different bundle contents"
                )
            return destination
        temp = self.artifacts_dir / f".{checkpoint_sha256}.installing"
        if temp.exists():
            shutil.rmtree(temp)
        shutil.copytree(bundle, temp)
        _fsync_dir(self.artifacts_dir)
        os.replace(temp, destination)
        _fsync_dir(self.artifacts_dir)
        return destination

    def initialize(self, bundle_path: str | Path) -> dict[str, Any]:
        with self.lock():
            if self.active_path.exists():
                raise RuntimeError("checkpoint registry already initialized")
            bundle = Path(bundle_path)
            evidence = _verify_factorized_bundle(bundle)
            self._install_bundle(bundle, evidence["checkpoint_sha256"])
            pointer = _seal_pointer(
                {
                    "schema": POINTER_SCHEMA,
                    "authority": REGISTRY_AUTHORITY,
                    "generation": 0,
                    "checkpoint_sha256": evidence["checkpoint_sha256"],
                    "artifact_relpath": (
                        f"artifacts/{evidence['checkpoint_sha256']}"
                    ),
                    "previous_pointer_sha256": None,
                    "transaction_id": None,
                    "transition": "INITIALIZE",
                    "rollback_target_generation": None,
                    "created_at": utc_now_iso(),
                }
            )
            self._write_pointer_snapshot(pointer)
            _atomic_write_json(self.active_path, pointer)
            return pointer

    def _tx_dir(self, transaction_id: str) -> Path:
        return self.transactions_dir / transaction_id

    def _event_files(self, transaction_id: str) -> list[Path]:
        event_dir = self._tx_dir(transaction_id) / "events"
        if not event_dir.exists():
            return []
        return sorted(event_dir.glob("*.json"))

    def transaction_events(self, transaction_id: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        previous: str | None = None
        for path in self._event_files(transaction_id):
            event = _verify_event(_read_json(path))
            if event.get("previous_event_sha256") != previous:
                raise ValueError("checkpoint transaction event-chain mismatch")
            events.append(event)
            previous = event["event_sha256"]
        return events

    def _append_event(
        self,
        transaction_id: str,
        *,
        state: str,
        details: dict[str, Any],
    ) -> dict[str, Any]:
        tx_dir = self._tx_dir(transaction_id)
        event_dir = tx_dir / "events"
        event_dir.mkdir(parents=True, exist_ok=True)
        events = self.transaction_events(transaction_id)
        sequence = len(events)
        event = _seal_event(
            {
                "schema": TX_EVENT_SCHEMA,
                "authority": REGISTRY_AUTHORITY,
                "transaction_id": transaction_id,
                "sequence": sequence,
                "state": state,
                "previous_event_sha256": (
                    events[-1]["event_sha256"] if events else None
                ),
                "details": details,
                "created_at": utc_now_iso(),
            }
        )
        path = event_dir / f"{sequence:06d}-{state.lower()}.json"
        if path.exists():
            raise RuntimeError("transaction event sequence already exists")
        _atomic_write_json(path, event)
        return event

    def _authorization_seen(self, authorization_sha256: str) -> bool:
        for tx_dir in sorted(
            p for p in self.transactions_dir.iterdir() if p.is_dir()
        ):
            events = self.transaction_events(tx_dir.name)
            if not events:
                continue
            if (
                events[0].get("details", {}).get(
                    "promotion_authorization_sha256"
                )
                == authorization_sha256
            ):
                return True
        return False

    def begin_authorized_update(
        self,
        candidate_bundle: str | Path,
        authorization: dict[str, Any],
        *,
        now: str | None = None,
    ) -> str:
        with self.lock():
            verified_auth = verify_promotion_authorization(
                authorization,
                now=now,
            )
            auth_sha = str(verified_auth["authorization_sha256"])
            if self._authorization_seen(auth_sha):
                raise RuntimeError("promotion authorization has already been used")

            parent = self.active_pointer()
            if (
                verified_auth["active_parent_checkpoint_sha256"]
                != parent["checkpoint_sha256"]
            ):
                raise ValueError(
                    "promotion authorization parent is not the active checkpoint"
                )

            source = Path(candidate_bundle)
            evidence = verify_l31_candidate_bundle(source)
            if (
                verified_auth["candidate_checkpoint_sha256"]
                != evidence["checkpoint_sha256"]
            ):
                raise ValueError(
                    "promotion authorization candidate does not match bundle"
                )

            transaction_id = payload_digest(
                {
                    "parent_pointer_sha256": parent["pointer_sha256"],
                    "candidate_checkpoint_sha256": evidence["checkpoint_sha256"],
                    "candidate_bundle_sha256": evidence["bundle_sha256"],
                    "l31_run_receipt_sha256": evidence["run_receipt_sha256"],
                    "promotion_authorization_sha256": auth_sha,
                }
            )
            tx_dir = self._tx_dir(transaction_id)
            if tx_dir.exists():
                raise RuntimeError("identical checkpoint transaction already exists")
            staging = tx_dir / "staging"
            staging.parent.mkdir(parents=True, exist_ok=False)
            shutil.copytree(source, staging)
            staged = verify_l31_candidate_bundle(staging)
            if staged["bundle_sha256"] != evidence["bundle_sha256"]:
                raise RuntimeError("candidate bundle changed while staging")
            _atomic_write_json(
                tx_dir / "promotion-authorization.json",
                verified_auth,
            )
            self._append_event(
                transaction_id,
                state="PREPARED",
                details={
                    "parent_pointer_sha256": parent["pointer_sha256"],
                    "parent_generation": parent["generation"],
                    "parent_checkpoint_sha256": parent["checkpoint_sha256"],
                    "candidate_checkpoint_sha256": evidence["checkpoint_sha256"],
                    "candidate_bundle_sha256": evidence["bundle_sha256"],
                    "l31_run_receipt_sha256": evidence["run_receipt_sha256"],
                    "promotion_authorization_sha256": auth_sha,
                },
            )
            return transaction_id

    def begin_l31_update(self, candidate_bundle: str | Path) -> str:
        with self.lock():
            parent = self.active_pointer()
            source = Path(candidate_bundle)
            evidence = verify_l31_candidate_bundle(source)
            if (
                evidence["run_receipt"]["parent_factorized_checkpoint_sha256"]
                != parent["checkpoint_sha256"]
            ):
                raise ValueError(
                    "L31 candidate parent does not match active checkpoint"
                )
            transaction_id = payload_digest(
                {
                    "parent_pointer_sha256": parent["pointer_sha256"],
                    "candidate_checkpoint_sha256": evidence["checkpoint_sha256"],
                    "candidate_bundle_sha256": evidence["bundle_sha256"],
                    "l31_run_receipt_sha256": evidence["run_receipt_sha256"],
                }
            )
            tx_dir = self._tx_dir(transaction_id)
            if tx_dir.exists():
                raise RuntimeError("identical checkpoint transaction already exists")
            staging = tx_dir / "staging"
            staging.parent.mkdir(parents=True, exist_ok=False)
            shutil.copytree(source, staging)
            staged = verify_l31_candidate_bundle(staging)
            if staged["bundle_sha256"] != evidence["bundle_sha256"]:
                raise RuntimeError("candidate bundle changed while staging")
            self._append_event(
                transaction_id,
                state="PREPARED",
                details={
                    "parent_pointer_sha256": parent["pointer_sha256"],
                    "parent_generation": parent["generation"],
                    "parent_checkpoint_sha256": parent["checkpoint_sha256"],
                    "candidate_checkpoint_sha256": evidence["checkpoint_sha256"],
                    "candidate_bundle_sha256": evidence["bundle_sha256"],
                    "l31_run_receipt_sha256": evidence["run_receipt_sha256"],
                },
            )
            return transaction_id

    def verify_update(
        self,
        transaction_id: str,
        *,
        now: str | None = None,
    ) -> dict[str, Any]:
        with self.lock():
            events = self.transaction_events(transaction_id)
            if not events or events[0]["state"] != "PREPARED":
                raise RuntimeError("transaction is not prepared")
            if any(
                event["state"] in {"COMMITTED", "ABORTED", "RECOVERED_COMMITTED", "RECOVERED_ABORTED"}
                for event in events
            ):
                raise RuntimeError("transaction is already final")
            prepared = events[0]["details"]
            active = self.active_pointer()
            if active["pointer_sha256"] != prepared["parent_pointer_sha256"]:
                raise RuntimeError("active pointer changed before transaction verification")

            auth_sha = prepared.get("promotion_authorization_sha256")
            if auth_sha is not None:
                auth_path = (
                    self._tx_dir(transaction_id)
                    / "promotion-authorization.json"
                )
                if not auth_path.exists():
                    raise RuntimeError("promotion authorization file missing")
                authorization = verify_promotion_authorization(
                    _read_json(auth_path),
                    now=now,
                )
                if authorization["authorization_sha256"] != auth_sha:
                    raise RuntimeError("promotion authorization digest drift")
                if (
                    authorization["active_parent_checkpoint_sha256"]
                    != active["checkpoint_sha256"]
                ):
                    raise RuntimeError("promotion authorization parent drift")
                if (
                    authorization["candidate_checkpoint_sha256"]
                    != prepared["candidate_checkpoint_sha256"]
                ):
                    raise RuntimeError("promotion authorization candidate drift")

            staging = self._tx_dir(transaction_id) / "staging"
            evidence = verify_l31_candidate_bundle(staging)
            if evidence["checkpoint_sha256"] != prepared["candidate_checkpoint_sha256"]:
                raise RuntimeError("staged candidate checkpoint drift")
            if evidence["bundle_sha256"] != prepared["candidate_bundle_sha256"]:
                raise RuntimeError("staged candidate bundle drift")
            if evidence["run_receipt_sha256"] != prepared["l31_run_receipt_sha256"]:
                raise RuntimeError("staged L31 receipt drift")
            return self._append_event(
                transaction_id,
                state="VERIFIED",
                details={
                    "candidate_checkpoint_sha256": evidence["checkpoint_sha256"],
                    "candidate_bundle_sha256": evidence["bundle_sha256"],
                },
            )

    def commit_update(
        self,
        transaction_id: str,
        *,
        now: str | None = None,
    ) -> dict[str, Any]:
        with self.lock():
            events = self.transaction_events(transaction_id)
            if not events:
                raise RuntimeError("unknown transaction")
            if events[-1]["state"] != "VERIFIED":
                raise RuntimeError("transaction must be VERIFIED before commit")
            prepared = events[0]["details"]
            active = self.active_pointer()
            if active["pointer_sha256"] != prepared["parent_pointer_sha256"]:
                raise RuntimeError("active pointer changed before commit")

            auth_sha = prepared.get("promotion_authorization_sha256")
            if auth_sha is not None:
                auth_path = (
                    self._tx_dir(transaction_id)
                    / "promotion-authorization.json"
                )
                authorization = verify_promotion_authorization(
                    _read_json(auth_path),
                    now=now,
                )
                if authorization["authorization_sha256"] != auth_sha:
                    raise RuntimeError("promotion authorization digest drift")
                if (
                    authorization["active_parent_checkpoint_sha256"]
                    != active["checkpoint_sha256"]
                ):
                    raise RuntimeError("promotion authorization parent drift")
                if (
                    authorization["candidate_checkpoint_sha256"]
                    != prepared["candidate_checkpoint_sha256"]
                ):
                    raise RuntimeError("promotion authorization candidate drift")

            staging = self._tx_dir(transaction_id) / "staging"
            evidence = verify_l31_candidate_bundle(staging)
            if evidence["bundle_sha256"] != prepared["candidate_bundle_sha256"]:
                raise RuntimeError("candidate bundle drift before commit")
            self._install_bundle(staging, evidence["checkpoint_sha256"])

            pointer = _seal_pointer(
                {
                    "schema": POINTER_SCHEMA,
                    "authority": REGISTRY_AUTHORITY,
                    "generation": int(active["generation"]) + 1,
                    "checkpoint_sha256": evidence["checkpoint_sha256"],
                    "artifact_relpath": (
                        f"artifacts/{evidence['checkpoint_sha256']}"
                    ),
                    "previous_pointer_sha256": active["pointer_sha256"],
                    "transaction_id": transaction_id,
                    "transition": "UPDATE",
                    "rollback_target_generation": None,
                    "created_at": utc_now_iso(),
                }
            )
            self._write_pointer_snapshot(pointer)
            _atomic_write_json(self.active_path, pointer)
            self._append_event(
                transaction_id,
                state="POINTER_SWAPPED",
                details={
                    "pointer_sha256": pointer["pointer_sha256"],
                    "generation": pointer["generation"],
                },
            )
            self._append_event(
                transaction_id,
                state="COMMITTED",
                details={
                    "pointer_sha256": pointer["pointer_sha256"],
                    "candidate_checkpoint_sha256": evidence["checkpoint_sha256"],
                },
            )
            return pointer

    def abort_update(self, transaction_id: str, *, reason: str) -> dict[str, Any]:
        with self.lock():
            events = self.transaction_events(transaction_id)
            if not events:
                raise RuntimeError("unknown transaction")
            if any(
                event["state"] in {"COMMITTED", "RECOVERED_COMMITTED"}
                for event in events
            ):
                raise RuntimeError("cannot abort a committed transaction")
            return self._append_event(
                transaction_id,
                state="ABORTED",
                details={"reason": str(reason)},
            )

    def recover(
        self,
        *,
        break_stale_lock: bool = False,
    ) -> list[dict[str, Any]]:
        recovered: list[dict[str, Any]] = []
        with self.lock(break_stale_lock=break_stale_lock):
            active = self.active_pointer()
            for tx_dir in sorted(
                p for p in self.transactions_dir.iterdir() if p.is_dir()
            ):
                transaction_id = tx_dir.name
                events = self.transaction_events(transaction_id)
                if not events:
                    continue
                states = {event["state"] for event in events}
                if states & {
                    "COMMITTED",
                    "ABORTED",
                    "RECOVERED_COMMITTED",
                    "RECOVERED_ABORTED",
                    "RECOVERY_CONFLICT",
                }:
                    continue
                prepared = events[0]["details"]
                candidate_sha = prepared["candidate_checkpoint_sha256"]
                parent_pointer = prepared["parent_pointer_sha256"]
                if (
                    active.get("transaction_id") == transaction_id
                    and active.get("checkpoint_sha256") == candidate_sha
                ):
                    recovered.append(
                        self._append_event(
                            transaction_id,
                            state="RECOVERED_COMMITTED",
                            details={
                                "pointer_sha256": active["pointer_sha256"],
                                "reason": "active_pointer_already_swapped",
                            },
                        )
                    )
                elif active.get("pointer_sha256") == parent_pointer:
                    staging = tx_dir / "staging"
                    if staging.exists():
                        shutil.rmtree(staging)
                    recovered.append(
                        self._append_event(
                            transaction_id,
                            state="RECOVERED_ABORTED",
                            details={
                                "reason": "active_pointer_never_swapped",
                            },
                        )
                    )
                else:
                    recovered.append(
                        self._append_event(
                            transaction_id,
                            state="RECOVERY_CONFLICT",
                            details={
                                "active_pointer_sha256": active["pointer_sha256"],
                                "expected_parent_pointer_sha256": parent_pointer,
                                "candidate_checkpoint_sha256": candidate_sha,
                            },
                        )
                    )
            return recovered

    def rollback_to_generation(self, generation: int) -> dict[str, Any]:
        with self.lock():
            active = self.active_pointer()
            target_path = self._pointer_snapshot_path(int(generation))
            if not target_path.exists():
                raise ValueError("unknown rollback target generation")
            target = _verify_pointer(_read_json(target_path))
            if int(target["generation"]) >= int(active["generation"]):
                raise ValueError("rollback target must be an earlier generation")
            artifact = self.root / str(target["artifact_relpath"])
            evidence = _verify_factorized_bundle(artifact)
            if evidence["checkpoint_sha256"] != target["checkpoint_sha256"]:
                raise RuntimeError("rollback target artifact digest mismatch")

            rollback_id = payload_digest(
                {
                    "active_pointer_sha256": active["pointer_sha256"],
                    "target_pointer_sha256": target["pointer_sha256"],
                    "next_generation": int(active["generation"]) + 1,
                }
            )
            pointer = _seal_pointer(
                {
                    "schema": POINTER_SCHEMA,
                    "authority": REGISTRY_AUTHORITY,
                    "generation": int(active["generation"]) + 1,
                    "checkpoint_sha256": target["checkpoint_sha256"],
                    "artifact_relpath": target["artifact_relpath"],
                    "previous_pointer_sha256": active["pointer_sha256"],
                    "transaction_id": rollback_id,
                    "transition": "ROLLBACK",
                    "rollback_target_generation": int(target["generation"]),
                    "created_at": utc_now_iso(),
                }
            )
            self._write_pointer_snapshot(pointer)
            _atomic_write_json(self.active_path, pointer)

            receipt = {
                "schema": ROLLBACK_SCHEMA,
                "authority": REGISTRY_AUTHORITY,
                "rollback_id": rollback_id,
                "from_generation": active["generation"],
                "from_checkpoint_sha256": active["checkpoint_sha256"],
                "target_generation": target["generation"],
                "target_checkpoint_sha256": target["checkpoint_sha256"],
                "new_generation": pointer["generation"],
                "new_pointer_sha256": pointer["pointer_sha256"],
                "created_at": utc_now_iso(),
            }
            receipt["rollback_sha256"] = payload_digest(receipt)
            path = self.rollbacks_dir / f"{int(pointer['generation']):012d}.json"
            _atomic_write_json(path, receipt)
            return receipt

    def verify_registry(self) -> dict[str, Any]:
        active = self.active_pointer()
        snapshots = sorted(self.pointers_dir.glob("*.json"))
        expected_previous: str | None = None
        expected_generation = 0
        for path in snapshots:
            pointer = _verify_pointer(_read_json(path))
            if int(pointer["generation"]) != expected_generation:
                raise ValueError("checkpoint pointer generation gap")
            if pointer["previous_pointer_sha256"] != expected_previous:
                raise ValueError("checkpoint pointer ancestry mismatch")
            artifact = self.root / str(pointer["artifact_relpath"])
            evidence = _verify_factorized_bundle(artifact)
            if evidence["checkpoint_sha256"] != pointer["checkpoint_sha256"]:
                raise ValueError("checkpoint pointer artifact mismatch")
            expected_previous = pointer["pointer_sha256"]
            expected_generation += 1
        if not snapshots:
            raise ValueError("checkpoint pointer history is empty")
        if active != _verify_pointer(_read_json(snapshots[-1])):
            raise ValueError("active pointer is not latest history generation")

        authorization_count = 0
        for tx_dir in sorted(
            p for p in self.transactions_dir.iterdir() if p.is_dir()
        ):
            events = self.transaction_events(tx_dir.name)
            if not events:
                continue
            auth_sha = events[0].get("details", {}).get(
                "promotion_authorization_sha256"
            )
            if auth_sha is None:
                continue
            auth_path = tx_dir / "promotion-authorization.json"
            if not auth_path.exists():
                raise ValueError("authorized transaction missing authorization")
            authorization = verify_promotion_authorization(
                _read_json(auth_path),
                check_expiry=False,
            )
            if authorization["authorization_sha256"] != auth_sha:
                raise ValueError("authorized transaction authorization mismatch")
            authorization_count += 1

        rollback_count = 0
        for path in sorted(self.rollbacks_dir.glob("*.json")):
            receipt = verify_rollback_receipt(_read_json(path))
            pointer_path = self._pointer_snapshot_path(
                int(receipt["new_generation"])
            )
            if not pointer_path.exists():
                raise ValueError("rollback receipt points to missing generation")
            pointer = _verify_pointer(_read_json(pointer_path))
            if pointer.get("transition") != "ROLLBACK":
                raise ValueError("rollback receipt generation is not rollback")
            if pointer.get("pointer_sha256") != receipt.get("new_pointer_sha256"):
                raise ValueError("rollback receipt pointer mismatch")
            if (
                int(pointer.get("rollback_target_generation"))
                != int(receipt.get("target_generation"))
            ):
                raise ValueError("rollback target generation mismatch")
            rollback_count += 1
        return {
            "schema": "NOLANE-L33-CHECKPOINT-REGISTRY-AUDIT-V1",
            "authority": REGISTRY_AUTHORITY,
            "status": "PASS",
            "generations": len(snapshots),
            "active_generation": active["generation"],
            "active_checkpoint_sha256": active["checkpoint_sha256"],
            "active_pointer_sha256": active["pointer_sha256"],
            "rollback_receipts": rollback_count,
            "promotion_authorizations": authorization_count,
        }
