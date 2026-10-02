from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable

from .cortex import Cortex, NullCortex
from .engine import LivingEngine
from .product_cortex import FactorizedProductCortex
from .product_profile import ProductProfile, ProductProfileStore
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
    ) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.profile_store = ProductProfileStore(
            self.data_dir / "personalization.json"
        )
        self.profile = self.profile_store.load()
        self.store = LivingStore(self.data_dir / "living.db")
        self.engine = LivingEngine(
            self.store,
            cortex=NullCortex(),
            enable_rest=enable_rest,
            memory_enabled=self.profile.memory_enabled,
        )
        self.checkpoint = None if checkpoint is None else Path(checkpoint)
        self.tokenizer_path = (
            None if tokenizer_path is None else Path(tokenizer_path)
        )
        self.device = str(device)
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
            return self.profile.to_dict()

    def close(self) -> None:
        with self._lock:
            cortex = self._cortex
            self._cortex = None
            close = getattr(cortex, "close", None)
            if callable(close):
                close()
            self.store.close()
