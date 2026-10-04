from __future__ import annotations

import hashlib
import secrets
from pathlib import Path
from typing import Callable

from .cortex import CortexReply, CortexRequest
from .factorized_artifact import load_factorized_model
from .latent import LatentBindingError, LatentStore
from .product_profile import ProductProfile
from .product_prompt_payload import (
    build_product_messages,
    product_profile_summary,
    product_state_summary,
)


_LENGTH_TOKENS = {
    "compact": 128,
    "balanced": 256,
    "expansive": 512,
}

class FactorizedProductCortex:
    """Standalone Nolane cortex adapter for the product runtime.

    The model object is Nolane's factorized boundary + recurrent cortex.
    Qwen is used only for the tokenizer/chat-template boundary.
    """

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        tokenizer_path: str | Path,
        latent_path: str | Path,
        identity_id: str,
        profile_getter: Callable[[], ProductProfile],
        device: str = "auto",
        sampling_seed_getter: Callable[[], int] | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Product inference requires the local neural/tokenizer runtime"
            ) from exc

        self.torch = torch
        checkpoint = Path(checkpoint)
        if checkpoint.is_dir():
            checkpoint = checkpoint / "factorized-nolane.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Nolane factorized checkpoint not found: {checkpoint}"
            )
        tokenizer_path = Path(tokenizer_path)
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Nolane tokenizer assets not found: {tokenizer_path}"
            )

        resolved_device = device
        if resolved_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = resolved_device
        self.checkpoint = checkpoint
        self.profile_getter = profile_getter
        self.sampling_seed_getter = (
            sampling_seed_getter
            if sampling_seed_getter is not None
            else lambda: secrets.randbits(64)
        )

        try:
            payload = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            payload = torch.load(checkpoint, map_location="cpu")
        cortex_config = dict(payload.get("cortex_config", {}))
        latent_dim = int(cortex_config.get("latent_dim", 0))
        if latent_dim <= 0:
            raise ValueError("factorized checkpoint is missing cortex latent_dim")

        checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        latent_store = LatentStore(latent_path)
        latent = None
        try:
            latent = latent_store.load_bound(
                identity_id=identity_id,
                checkpoint_sha256=checkpoint_sha,
                latent_dim=latent_dim,
                protocol_sha256=None,
            )
        except LatentBindingError:
            # Never silently carry neural latent values across a checkpoint
            # identity change. State/memory remain persistent; the latent starts
            # clean until a future explicit migration court exists.
            old_path = Path(latent_path)
            if old_path.exists():
                archive = old_path.with_name(
                    f"{old_path.stem}.{checkpoint_sha[:12]}.previous"
                    f"{old_path.suffix}"
                )
                if not archive.exists():
                    old_path.replace(archive)
        if latent is None:
            latent = latent_store.initialize(
                identity_id=identity_id,
                checkpoint_sha256=checkpoint_sha,
                latent_dim=latent_dim,
                protocol_sha256=None,
            )

        self.model, self.meta = load_factorized_model(
            checkpoint,
            latent.values,
            device=self.device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            local_files_only=True,
        )
        self.checkpoint_sha256 = str(self.meta["checkpoint_sha256"])

    @staticmethod
    def _state_summary(request: CortexRequest) -> str:
        return product_state_summary(request)

    @staticmethod
    def _profile_summary(profile: ProductProfile) -> str:
        return product_profile_summary(profile)

    def generate(self, request: CortexRequest) -> CortexReply:
        profile = self.profile_getter()
        messages = build_product_messages(profile, request)
        try:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = self.tokenizer(
            rendered,
            return_tensors="pt",
        )
        ids = inputs["input_ids"].to(self.device)
        max_new_tokens = _LENGTH_TOKENS[profile.response_length]
        with self.torch.inference_mode():
            generated = self.model.generate(
                input_ids=ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.55,
                top_p=0.9,
                sampling_seed=int(self.sampling_seed_getter()),
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = generated[0, ids.shape[1] :]
        text = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()
        return CortexReply(text, intent=request.intent)

    def self_test(self) -> dict[str, object]:
        """Run one real local token through the loaded neural runtime.

        This is intentionally tiny: it proves tokenizer -> device -> model
        generation works before ProductRuntime reports AI ON.
        """
        inputs = self.tokenizer(
            "Nolane readiness check.",
            return_tensors="pt",
        )
        ids = inputs["input_ids"].to(self.device)
        if ids.ndim != 2 or ids.shape[1] < 1:
            raise RuntimeError("product tokenizer smoke test produced no input ids")
        with self.torch.inference_mode():
            generated = self.model.generate(
                input_ids=ids,
                max_new_tokens=1,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        if generated.ndim != 2 or generated.shape[0] != 1:
            raise RuntimeError("product model smoke test returned invalid shape")
        produced = int(generated.shape[1] - ids.shape[1])
        if produced < 1:
            raise RuntimeError("product model smoke test produced no new token")
        return {
            "status": "PASS",
            "generated_tokens": produced,
            "checkpoint_sha256": self.checkpoint_sha256,
            "device": self.device,
        }

    def close(self) -> None:
        self.model = None
        if self.device == "cuda" and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
