from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Callable

from .cortex import Cortex, NullCortex
from .engine import LivingEngine
from .product_cortex import FactorizedProductCortex
from .product_profile import ProductProfile, ProductProfileStore
from .product_learning_workspace import ProductLearningWorkspace
from .promotion_ceremony import verify_promotion_ceremony_receipt
from .store import LivingStore


CortexFactory = Callable[[str, Callable[[], ProductProfile]], Cortex]


class ProductRuntime:
    """Thread-safe local product facade over the persistent LivingEngine."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        checkpoint: str | Path | None = None,
        tokenizer_path: str | Path | None = None,
        device: str = "auto",
        cortex_factory: CortexFactory | None = None,
        enable_rest: bool = True,
        release_ceremony: str | Path | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.profile_store = ProductProfileStore(
            self.data_dir / "personalization.json"
        )
        self.profile = self.profile_store.load()
        self._rest_allowed = bool(enable_rest)
        # ProductHTTPServer dispatches requests on worker threads.
        # ProductRuntime's RLock serializes every store/engine operation, so
        # this one product-owned connection may safely cross those threads.
        self.store = LivingStore(
            self.data_dir / "living.db",
            allow_cross_thread=True,
        )
        self.engine = LivingEngine(
            self.store,
            cortex=NullCortex(),
            enable_rest=self._rest_allowed and self.profile.memory_enabled,
            memory_enabled=self.profile.memory_enabled,
        )
        self.checkpoint = None if checkpoint is None else Path(checkpoint)
        self.tokenizer_path = (
            None if tokenizer_path is None else Path(tokenizer_path)
        )
        self.device = str(device)
        self.release_ceremony = (
            None if release_ceremony is None else Path(release_ceremony)
        )
        if self.release_ceremony is not None:
            if self.checkpoint is None:
                raise ValueError(
                    "release ceremony requires a configured checkpoint"
                )
            checkpoint_file = self.checkpoint
            if checkpoint_file.is_dir():
                checkpoint_file = checkpoint_file / "factorized-nolane.pt"
            if not checkpoint_file.is_file():
                raise FileNotFoundError(
                    f"release checkpoint not found: {checkpoint_file}"
                )
            if not self.release_ceremony.is_file():
                raise FileNotFoundError(
                    f"release ceremony not found: {self.release_ceremony}"
                )
            ceremony_payload = json.loads(
                self.release_ceremony.read_text(encoding="utf-8")
            )
            verify_promotion_ceremony_receipt(
                ceremony_payload,
                require_complete=True,
            )
            actual_sha = hashlib.sha256(
                checkpoint_file.read_bytes()
            ).hexdigest()
            if (
                ceremony_payload["candidate_checkpoint_sha256"]
                != actual_sha
            ):
                raise ValueError(
                    "release checkpoint does not match COMPLETE ceremony"
                )
        self.learning = ProductLearningWorkspace(data_dir=self.data_dir)
        self._factory = cortex_factory
        self._cortex: Cortex | None = None
        self._phase = "off"
        self._error: str | None = None
        self._model_checkpoint_sha256: str | None = None
        self._lock = threading.RLock()

    def current_profile(self) -> ProductProfile:
        return self.profile

    def _default_factory(
        self,
        identity_id: str,
        profile_getter: Callable[[], ProductProfile],
    ) -> Cortex:
        if self.checkpoint is None:
            raise FileNotFoundError(
                "release model checkpoint is not configured"
            )
        if self.tokenizer_path is None:
            raise FileNotFoundError(
                "release tokenizer assets are not configured"
            )
        return FactorizedProductCortex(
            checkpoint=self.checkpoint,
            tokenizer_path=self.tokenizer_path,
            latent_path=self.data_dir / "latent.json",
            identity_id=identity_id,
            profile_getter=profile_getter,
            device=self.device,
        )

    def power_on(self) -> dict[str, Any]:
        with self._lock:
            if self._phase == "on" and self._cortex is not None:
                return self.status()
            self._phase = "starting"
            self._error = None
            try:
                factory = self._factory or self._default_factory
                cortex = factory(
                    self.engine.state.identity_id,
                    self.current_profile,
                )
                self._cortex = cortex
                self.engine.cortex = cortex
                self.engine.memory_enabled = self.profile.memory_enabled
                self._model_checkpoint_sha256 = getattr(
                    cortex,
                    "checkpoint_sha256",
                    None,
                )
                self._phase = "on"
            except Exception as exc:
                self._cortex = None
                self.engine.cortex = NullCortex()
                self._phase = "error"
                self._error = f"{type(exc).__name__}: {exc}"
            return self.status()

    def power_off(self) -> dict[str, Any]:
        with self._lock:
            cortex = self._cortex
            self._cortex = None
            self.engine.cortex = NullCortex()
            self._phase = "off"
            self._error = None
            self._model_checkpoint_sha256 = None
            close = getattr(cortex, "close", None)
            if callable(close):
                close()
            return self.status()

    def set_power(self, enabled: bool) -> dict[str, Any]:
        return self.power_on() if bool(enabled) else self.power_off()

    def status(self) -> dict[str, Any]:
        with self._lock:
            state = self.engine.state
            return {
                "schema": "NOLANE-PRODUCT-RUNTIME-STATUS-V1",
                "phase": self._phase,
                "powered": self._phase == "on",
                "error": self._error,
                "identity_id": state.identity_id,
                "state_version": state.version,
                "interactions": state.relationship.interaction_count,
                "open_threads": sum(
                    1 for thread in state.open_threads if thread.unresolved
                ),
                "memory_enabled": bool(self.profile.memory_enabled),
                "initiative": self.profile.initiative,
                "model_checkpoint_sha256": self._model_checkpoint_sha256,
                "device": self.device,
            }

    def history(self, limit: int = 200) -> list[dict[str, object]]:
        with self._lock:
            return self.store.conversation_messages(limit=limit)

    def send_message(self, text: str) -> dict[str, Any]:
        clean = str(text).strip()
        if not clean:
            raise ValueError("message is empty")
        if len(clean) > 12000:
            raise ValueError("message is too long")
        with self._lock:
            if self._phase != "on" or self._cortex is None:
                raise RuntimeError("Nolane AI is not running")
            result = self.engine.handle_user_message(clean)
            return {
                "reply": result.speech,
                "state_version": result.state.version,
                "observer_error": result.observer_error,
            }

    def tick(self) -> dict[str, Any]:
        with self._lock:
            if self._phase != "on" or self._cortex is None:
                return {"speech": "", "skipped": "ai_off"}
            if self.profile.initiative == "off":
                return {"speech": "", "skipped": "initiative_off"}
            result = self.engine.tick()
            return {
                "speech": result.speech,
                "state_version": result.state.version,
                "rest_error": result.rest_error,
            }

    def get_profile(self) -> dict[str, Any]:
        with self._lock:
            return self.profile.to_dict()

    def update_profile(self, patch: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.profile = self.profile_store.update(dict(patch))
            self.engine.memory_enabled = self.profile.memory_enabled
            self.engine.enable_rest = (
                self._rest_allowed and self.profile.memory_enabled
            )
            return self.profile.to_dict()

    def learning_windows(self) -> list[dict[str, Any]]:
        with self._lock:
            return self.learning.list_windows()

    def create_learning_window(self) -> dict[str, Any]:
        with self._lock:
            return self.learning.create_window()

    def next_learning_candidate(
        self,
        window_id: str,
    ) -> dict[str, Any] | None:
        with self._lock:
            return self.learning.next_candidate(window_id)

    def record_learning_decision(
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
            return self.learning.record_decision(
                window_id,
                candidate_id,
                decision=decision,
                language=language,
                weight=weight,
                corrected_target=corrected_target,
            )

    def finalize_learning_window(
        self,
        window_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            return self.learning.finalize_window(window_id)

    def close(self) -> None:
        with self._lock:
            cortex = self._cortex
            self._cortex = None
            close = getattr(cortex, "close", None)
            if callable(close):
                close()
            self.store.close()
