from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any

from .interactive_review import LocalReviewSession
from .local_evidence_workbench import (
    finalize_workbench,
    paths_for,
    workbench_status,
)
from .product_evidence_bridge import (
    ProductEvidenceExportPolicy,
    prepare_product_evidence_window,
    verify_product_evidence_window,
)
from .store import canonical_json, payload_digest


REGISTRY_SCHEMA = "NOLANE-L45-INAPP-EVIDENCE-REGISTRY-V1"
AUTHORITY = "LOCAL_EXPLICIT_REVIEW_ONLY_NO_AUTO_TRAINING_AUTHORITY"


class ProductLearningWorkspace:
    """Local in-app human-review workspace over real product conversations.

    This class can prepare evidence windows and record explicit human review
    decisions. It cannot train or promote a model.
    """

    def __init__(
        self,
        *,
        data_dir: str | Path,
        policy: ProductEvidenceExportPolicy | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.db_path = self.data_dir / "living.db"
        self.profile_path = self.data_dir / "personalization.json"
        self.root = self.data_dir / "learning-evidence"
        self.windows_dir = self.root / "windows"
        self.registry_path = self.root / "learning-registry.json"
        self.policy = policy or ProductEvidenceExportPolicy()
        self.policy.validate()
        self._lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        self.windows_dir.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self._write_registry(self._empty_registry())

    @staticmethod
    def _empty_registry() -> dict[str, Any]:
        payload = {
            "schema": REGISTRY_SCHEMA,
            "authority": AUTHORITY,
            "next_window_index": 1,
            "last_exported_rowid": 0,
            "windows": [],
            "privacy": {
                "contains_raw_prompt_target": False,
                "contains_candidate_ids": False,
                "auto_approval": False,
                "auto_training": False,
                "cloud_upload": False,
            },
        }
        payload["registry_sha256"] = payload_digest(payload)
        return payload

    def _write_registry(self, payload: dict[str, Any]) -> None:
        body = dict(payload)
        body.pop("registry_sha256", None)
        body["registry_sha256"] = payload_digest(body)
        self.root.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self.root,
            prefix="learning-registry.",
            suffix=".tmp",
            delete=False,
        )
        temp = Path(handle.name)
        try:
            with handle:
                handle.write(canonical_json(body) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, self.registry_path)
        finally:
            if temp.exists():
                temp.unlink()

    def _load_registry(self) -> dict[str, Any]:
        payload = json.loads(
            self.registry_path.read_text(encoding="utf-8")
        )
        if payload.get("schema") != REGISTRY_SCHEMA:
            raise ValueError("unsupported in-app evidence registry schema")
        supplied = payload.get("registry_sha256")
        body = dict(payload)
        body.pop("registry_sha256", None)
        if payload_digest(body) != supplied:
            raise ValueError("in-app evidence registry digest mismatch")
        if payload.get("authority") != AUTHORITY:
            raise ValueError("in-app evidence registry authority mismatch")
        if not isinstance(payload.get("windows"), list):
            raise ValueError("in-app evidence registry windows invalid")
        return payload

    def _window_root(self, window_id: str) -> Path:
        rendered = str(window_id).strip()
        if not rendered.startswith("window-"):
            raise ValueError("invalid product evidence window id")
        if "/" in rendered or "\\" in rendered or ".." in rendered:
            raise ValueError("invalid product evidence window id")
        return self.windows_dir / rendered

    def verify_registry(self) -> dict[str, Any]:
        with self._lock:
            registry = self._load_registry()
            seen: set[str] = set()
            previous_high_water = 0
            for row in registry["windows"]:
                window_id = str(row.get("window_id", ""))
                if not window_id or window_id in seen:
                    raise ValueError("duplicate/invalid evidence window id")
                seen.add(window_id)
                window = verify_product_evidence_window(
                    self._window_root(window_id)
                )
                if (
                    str(window["manifest_sha256"])
                    != str(row.get("window_manifest_sha256"))
                ):
                    raise ValueError(
                        "in-app evidence window manifest lineage mismatch"
                    )
                source = window["source_event_range"]
                high_water = int(source["through_rowid_inclusive"])
                after = int(source["after_rowid_exclusive"])
                if after != previous_high_water:
                    raise ValueError(
                        "in-app evidence windows are not a contiguous export chain"
                    )
                if high_water <= after:
                    raise ValueError(
                        "in-app evidence window high-water mark invalid"
                    )
                if int(row.get("through_rowid_inclusive", -1)) != high_water:
                    raise ValueError(
                        "in-app evidence registry high-water mismatch"
                    )
                previous_high_water = high_water

            # Recover a crash that happened after the fully verified window
            # directory was atomically installed but before the registry
            # pointer was written. Only the exact next contiguous window may be
            # adopted; gaps or unrelated directories remain fail-closed.
            changed = False
            expected_next = len(registry["windows"]) + 1
            while True:
                window_id = f"window-{expected_next:04d}"
                root = self._window_root(window_id)
                if not root.is_dir():
                    break
                window = verify_product_evidence_window(root)
                source = window["source_event_range"]
                after = int(source["after_rowid_exclusive"])
                high_water = int(source["through_rowid_inclusive"])
                if after != previous_high_water or high_water <= after:
                    raise ValueError(
                        "orphan evidence window is not contiguous with registry"
                    )
                registry["windows"].append(
                    {
                        "window_id": window_id,
                        "window_manifest_sha256": window["manifest_sha256"],
                        "through_rowid_inclusive": high_water,
                    }
                )
                seen.add(window_id)
                previous_high_water = high_water
                expected_next += 1
                changed = True

            unregistered = sorted(
                path.name
                for path in self.windows_dir.iterdir()
                if (
                    path.is_dir()
                    and path.name.startswith("window-")
                    and path.name not in seen
                )
            )
            if unregistered:
                raise ValueError(
                    "unregistered evidence window gap/conflict: "
                    + ",".join(unregistered)
                )

            expected_next = len(registry["windows"]) + 1
            if changed:
                registry["last_exported_rowid"] = previous_high_water
                registry["next_window_index"] = expected_next
                self._write_registry(registry)
                registry = self._load_registry()

            if int(registry["last_exported_rowid"]) != previous_high_water:
                raise ValueError(
                    "in-app evidence registry final cursor mismatch"
                )
            if int(registry["next_window_index"]) != expected_next:
                raise ValueError(
                    "in-app evidence registry next-window index mismatch"
                )
            return registry

    def create_window(self) -> dict[str, Any]:
        with self._lock:
            registry = self.verify_registry()

            # A retry/double-click while the newest window is still under
            # review must return that same window instead of consuming the
            # next slice of product history.
            if registry["windows"]:
                latest_id = str(registry["windows"][-1]["window_id"])
                latest_root = self._window_root(latest_id)
                latest_status = workbench_status(
                    latest_root / "workbench"
                )
                if not bool(latest_status["intake_ready"]):
                    return self.window_status(latest_id)

            index = int(registry["next_window_index"])
            window_id = f"window-{index:04d}"
            final_root = self._window_root(window_id)
            if final_root.exists():
                raise FileExistsError(
                    f"evidence window already exists: {final_root}"
                )

            temp_root = Path(
                tempfile.mkdtemp(
                    dir=self.windows_dir,
                    prefix=f".{window_id}.",
                )
            )
            try:
                window = prepare_product_evidence_window(
                    self.db_path,
                    temp_root,
                    after_rowid=int(registry["last_exported_rowid"]),
                    profile_path=self.profile_path,
                    policy=self.policy,
                )
                high_water = int(
                    window["source_event_range"]["through_rowid_inclusive"]
                )
                if high_water <= int(registry["last_exported_rowid"]):
                    raise ValueError(
                        "new evidence window did not advance product event cursor"
                    )
                os.replace(temp_root, final_root)
            except Exception:
                if temp_root.exists():
                    shutil.rmtree(temp_root, ignore_errors=True)
                raise

            row = {
                "window_id": window_id,
                "window_manifest_sha256": window["manifest_sha256"],
                "through_rowid_inclusive": high_water,
            }
            registry["windows"].append(row)
            registry["last_exported_rowid"] = high_water
            registry["next_window_index"] = index + 1
            self._write_registry(registry)
            return self.window_status(window_id)

    def window_status(self, window_id: str) -> dict[str, Any]:
        with self._lock:
            registry = self.verify_registry()
            known = {
                str(row["window_id"]): row
                for row in registry["windows"]
            }
            if window_id not in known:
                raise ValueError("unknown product evidence window")
            root = self._window_root(window_id)
            window = verify_product_evidence_window(root)
            status = workbench_status(root / "workbench")
            return {
                "window_id": window_id,
                "phase": status["phase"],
                "review_progress": status["review_progress"],
                "intake_ready": bool(status["intake_ready"]),
                "approved_manifest_sha256": status[
                    "approved_manifest_sha256"
                ],
                "quality_status": status["quality_status"],
                "quality_court_sha256": status["quality_court_sha256"],
                "through_rowid_inclusive": int(
                    window["source_event_range"]["through_rowid_inclusive"]
                ),
                "authority": AUTHORITY,
            }

    def list_windows(self) -> list[dict[str, Any]]:
        with self._lock:
            registry = self.verify_registry()
            return [
                self.window_status(str(row["window_id"]))
                for row in registry["windows"]
            ]

    def next_candidate(self, window_id: str) -> dict[str, Any] | None:
        with self._lock:
            self.window_status(window_id)
            root = self._window_root(window_id)
            paths = paths_for(root / "workbench")
            session = LocalReviewSession(
                paths.queue_manifest,
                paths.decisions,
                paths.review_progress,
            )
            pending = session.pending_candidates()
            if not pending:
                workbench_status(root / "workbench")
                return None
            candidate = pending[0]
            progress = session.progress()
            return {
                "window_id": window_id,
                "candidate_id": str(candidate["candidate_id"]),
                "prompt": str(candidate["prompt"]),
                "target": str(candidate["target"]),
                "language": candidate.get("language"),
                "progress": progress.to_dict(),
                "privacy": {
                    "raw_text_returned_to_local_authenticated_ui": True,
                    "network_model_call": False,
                },
            }

    def record_decision(
        self,
        window_id: str,
        candidate_id: str,
        *,
        decision: str,
        language: str | None = None,
        weight: float = 1.0,
        corrected_target: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self.window_status(window_id)
            command = str(decision).strip().lower()
            if command not in {"approve", "reject", "sensitive"}:
                raise ValueError(
                    "decision must be approve, reject, or sensitive"
                )
            root = self._window_root(window_id)
            paths = paths_for(root / "workbench")
            session = LocalReviewSession(
                paths.queue_manifest,
                paths.decisions,
                paths.review_progress,
            )
            session.record_decision(
                candidate_id,
                approved=command == "approve",
                sensitive=command == "sensitive",
                language=language,
                weight=float(weight),
                corrected_target=corrected_target,
            )
            workbench_status(root / "workbench")
            return {
                "window": self.window_status(window_id),
                "next_candidate": self.next_candidate(window_id),
            }

    def finalize_window(
        self,
        window_id: str,
        *,
        allow_undecided: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            status = self.window_status(window_id)
            if status["intake_ready"]:
                return status
            root = self._window_root(window_id)
            finalize_workbench(
                root / "workbench",
                allow_undecided=allow_undecided,
            )
            return self.window_status(window_id)
