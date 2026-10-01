from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .deep_recurrent_cortex import DeepRecurrentStateSpaceCortex
from .surgery import parameter_guard_snapshot


@dataclass(slots=True)
class NativeBoundaryConfig:
    carry_recurrent_state: bool = False

    def validate(self) -> None:
        return None


@dataclass(slots=True)
class NativeBoundaryOutput:
    logits: Any
    loss: Any | None = None
    state: Any | None = None
    trace: dict[str, Any] | None = None


class NativeNolaneBoundaryModel:
    """Qwen decoder-free model.

    Reuses only Qwen token embeddings, final norm and LM head. Every Qwen
    Transformer decoder block is absent from forward and generation.
    Autoregressive continuity is owned by the Nolane recurrent cortex state.
    """

    def __init__(
        self,
        qwen_model,
        cortex: DeepRecurrentStateSpaceCortex,
        latent,
        *,
        config: NativeBoundaryConfig | None = None,
    ) -> None:
        self.qwen_model = qwen_model
        self.cortex = cortex
        self.latent = latent
        self.config = config or NativeBoundaryConfig()
        self.config.validate()
        self.state = None

        for parameter in self.qwen_model.parameters():
            parameter.requires_grad_(False)

        if not hasattr(self.qwen_model, "model"):
            raise ValueError("unsupported Qwen model: missing decoder backbone")
        backbone = self.qwen_model.model
        if not hasattr(backbone, "embed_tokens") or not hasattr(backbone, "norm"):
            raise ValueError("unsupported Qwen model: missing embedding/final norm")
        if not hasattr(self.qwen_model, "lm_head"):
            raise ValueError("unsupported Qwen model: missing LM head")

    @property
    def embed_tokens(self):
        return self.qwen_model.model.embed_tokens

    @property
    def final_norm(self):
        return self.qwen_model.model.norm

    @property
    def lm_head(self):
        return self.qwen_model.lm_head

    def set_latent(self, latent) -> None:
        self.latent = latent

    def reset_state(self) -> None:
        self.state = None

    def trainable_parameters(self):
        return [p for p in self.cortex.module.parameters() if p.requires_grad]

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def frozen_qwen_parameter_count(self) -> int:
        return sum(p.numel() for p in self.qwen_model.parameters())

    def assert_gradient_boundary(self) -> None:
        if any(p.requires_grad for p in self.qwen_model.parameters()):
            raise RuntimeError("Qwen gradient boundary violated")
        if not self.trainable_parameters():
            raise RuntimeError("native cortex exposes no trainable parameters")

    def base_guard(self):
        return parameter_guard_snapshot(self.qwen_model)

    def _loss(self, logits, labels):
        torch = self.cortex.torch
        if labels is None:
            return None
        if logits.shape[1] < 2:
            return logits.sum() * 0.0
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous().to(logits.device)
        return torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    def forward(
        self,
        *,
        input_ids,
        labels=None,
        state=None,
        update_persistent_state: bool = False,
    ) -> NativeBoundaryOutput:
        self.assert_gradient_boundary()
        hidden = self.embed_tokens(input_ids)
        initial = self.state if state is None and self.config.carry_recurrent_state else state
        updated, next_state, trace = self.cortex.scan(
            hidden,
            self.latent,
            state=initial,
        )
        normalized = self.final_norm(updated)
        logits = self.lm_head(normalized).float()
        loss = self._loss(logits, labels)
        if update_persistent_state and self.config.carry_recurrent_state:
            self.state = next_state.detach().clone()
        return NativeBoundaryOutput(
            logits=logits,
            loss=loss,
            state=next_state,
            trace=trace,
        )

    def _sample_next(
        self,
        logits,
        *,
        do_sample: bool,
        temperature: float,
        top_p: float,
    ):
        torch = self.cortex.torch
        if not do_sample:
            return torch.argmax(logits, dim=-1, keepdim=True)
        temp = max(float(temperature), 1e-5)
        probs = torch.softmax(logits / temp, dim=-1)
        if 0.0 < float(top_p) < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            mask = cumulative - sorted_probs > float(top_p)
            sorted_probs = sorted_probs.masked_fill(mask, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            sampled = torch.multinomial(sorted_probs, num_samples=1)
            return sorted_indices.gather(-1, sampled)
        return torch.multinomial(probs, num_samples=1)

    def generate(
        self,
        *,
        input_ids,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 0.8,
        top_p: float = 0.9,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
    ):
        torch = self.cortex.torch
        del pad_token_id
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [batch, sequence]")
        if input_ids.shape[0] != 1:
            raise ValueError("native generation currently supports batch size 1")
        if input_ids.shape[1] < 1:
            raise ValueError("native generation requires a non-empty prompt")

        self.qwen_model.eval()
        self.cortex.eval()
        generated = input_ids.clone()
        state = None
        with torch.no_grad():
            first = self.forward(input_ids=generated, state=None)
            state = first.state
            next_logits = first.logits[:, -1, :]
            for _ in range(int(max_new_tokens)):
                next_token = self._sample_next(
                    next_logits,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                )
                generated = torch.cat([generated, next_token], dim=1)
                if eos_token_id is not None and bool((next_token == int(eos_token_id)).all()):
                    break
                step = self.forward(input_ids=next_token, state=state)
                state = step.state
                next_logits = step.logits[:, -1, :]

        if self.config.carry_recurrent_state:
            self.state = None if state is None else state.detach().clone()
        return generated

    def prompt_scan_equivalent(self, input_ids, *, atol: float = 1e-5, rtol: float = 1e-5) -> bool:
        torch = self.cortex.torch
        self.qwen_model.eval()
        self.cortex.eval()
        with torch.no_grad():
            full = self.forward(input_ids=input_ids, state=None)
            pieces = []
            state = None
            for index in range(input_ids.shape[1]):
                step = self.forward(
                    input_ids=input_ids[:, index:index+1],
                    state=state,
                )
                state = step.state
                pieces.append(step.logits)
            incremental = torch.cat(pieces, dim=1)
        return bool(
            torch.allclose(full.logits, incremental, atol=atol, rtol=rtol)
            and torch.allclose(full.state, state, atol=atol, rtol=rtol)
        )
