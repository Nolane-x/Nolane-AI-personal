from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .deep_recurrent_cortex import (
    DeepRecurrentCortexConfig,
    DeepRecurrentStateSpaceCortex,
)


@dataclass(slots=True)
class StandaloneNolaneConfig:
    vocab_size: int
    hidden_size: int
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    padding_idx: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    boundary_dtype: str = "float32"

    def validate(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")
        if self.boundary_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("unsupported boundary_dtype")


@dataclass(slots=True)
class StandaloneOutput:
    logits: Any
    loss: Any | None = None
    state: Any | None = None
    trace: dict[str, Any] | None = None


def _torch_dtype(torch, name: str):
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"unsupported boundary dtype: {name}")
    return mapping[name]


class StandaloneBoundaryModule:
    """Owned input/output boundary with no Qwen model object dependency."""

    def __init__(
        self,
        config: StandaloneNolaneConfig,
        *,
        dtype=None,
        device=None,
    ) -> None:
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise RuntimeError("Install neural support with: pip install -e '.[neural]'") from exc

        config.validate()
        if dtype is None:
            dtype = _torch_dtype(torch, config.boundary_dtype)
        self.torch = torch
        self.nn = nn
        self.config = config

        class RMSNorm(nn.Module):
            def __init__(self, hidden_size: int, eps: float) -> None:
                super().__init__()
                factory = {}
                if dtype is not None:
                    factory["dtype"] = dtype
                if device is not None:
                    factory["device"] = device
                self.weight = nn.Parameter(torch.ones(hidden_size, **factory))
                self.variance_epsilon = float(eps)

            def forward(self, hidden_states):
                input_dtype = hidden_states.dtype
                hidden_states = hidden_states.to(torch.float32)
                variance = hidden_states.pow(2).mean(-1, keepdim=True)
                hidden_states = hidden_states * torch.rsqrt(
                    variance + self.variance_epsilon
                )
                return self.weight * hidden_states.to(input_dtype)

        factory = {}
        if dtype is not None:
            factory["dtype"] = dtype
        if device is not None:
            factory["device"] = device
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.padding_idx,
            **factory,
        )
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = None
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                **factory,
            )

        class Module(nn.Module):
            def __init__(self, outer) -> None:
                super().__init__()
                self.embed_tokens = outer.embed_tokens
                self.final_norm = outer.final_norm
                if outer.lm_head is not None:
                    self.lm_head = outer.lm_head

        self.module = Module(self)

    def to(self, device: str):
        self.module.to(device)
        return self

    def eval(self):
        self.module.eval()
        return self

    def train(self):
        self.module.train()
        return self

    def logits(self, hidden_states):
        torch = self.torch
        normalized = self.final_norm(hidden_states)
        if self.config.tie_word_embeddings:
            return torch.nn.functional.linear(
                normalized,
                self.embed_tokens.weight,
            ).float()
        return self.lm_head(normalized).float()

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.module.parameters())

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        return self.module.load_state_dict(state_dict)


class StandaloneNolaneLM:
    """Standalone Nolane language model.

    Runtime dependencies are this owned boundary, the Nolane recurrent cortex,
    a latent vector and PyTorch. No Qwen model object is held or called.
    """

    def __init__(
        self,
        boundary: StandaloneBoundaryModule,
        cortex: DeepRecurrentStateSpaceCortex,
        latent,
        *,
        carry_recurrent_state: bool = False,
    ) -> None:
        if boundary.config.hidden_size != cortex.hidden_size:
            raise ValueError("boundary/cortex hidden-size mismatch")
        self.boundary = boundary
        self.cortex = cortex
        self.latent = latent
        self.carry_recurrent_state = bool(carry_recurrent_state)
        self.state = None

    @property
    def config(self):
        return self.boundary.config

    def set_latent(self, latent) -> None:
        self.latent = latent

    def reset_state(self) -> None:
        self.state = None

    def to(self, device: str):
        self.boundary.to(device)
        self.cortex.to(device)
        return self

    def eval(self):
        self.boundary.eval()
        self.cortex.eval()
        return self

    def cortex_parameter_count(self) -> int:
        return self.cortex.parameter_count()

    def boundary_parameter_count(self) -> int:
        return self.boundary.parameter_count()

    def total_parameter_count(self) -> int:
        return self.cortex_parameter_count() + self.boundary_parameter_count()

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
    ) -> StandaloneOutput:
        hidden = self.boundary.embed_tokens(input_ids)
        initial = (
            self.state
            if state is None and self.carry_recurrent_state
            else state
        )
        updated, next_state, trace = self.cortex.scan(
            hidden,
            self.latent,
            state=initial,
        )
        logits = self.boundary.logits(updated)
        loss = self._loss(logits, labels)
        if update_persistent_state and self.carry_recurrent_state:
            self.state = next_state.detach().clone()
        return StandaloneOutput(
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
        temperature = max(float(temperature), 1e-5)
        probs = torch.softmax(logits / temperature, dim=-1)
        if 0.0 < float(top_p) < 1.0:
            sorted_probs, sorted_indices = torch.sort(
                probs,
                descending=True,
                dim=-1,
            )
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            mask = cumulative - sorted_probs > float(top_p)
            sorted_probs = sorted_probs.masked_fill(mask, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-12)
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
            raise ValueError("standalone generation currently supports batch size 1")
        if input_ids.shape[1] < 1:
            raise ValueError("standalone generation requires a non-empty prompt")

        self.eval()
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
                if eos_token_id is not None and bool(
                    (next_token == int(eos_token_id)).all()
                ):
                    break
                step = self.forward(input_ids=next_token, state=state)
                state = step.state
                next_logits = step.logits[:, -1, :]

        if self.carry_recurrent_state:
            self.state = None if state is None else state.detach().clone()
        return generated

    def prompt_scan_equivalent(
        self,
        input_ids,
        *,
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ) -> bool:
        torch = self.cortex.torch
        self.eval()
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


def standalone_config_from_qwen(qwen_model) -> StandaloneNolaneConfig:
    embed = qwen_model.model.embed_tokens
    norm = qwen_model.model.norm
    lm_head = qwen_model.lm_head
    tied = bool(embed.weight.data_ptr() == lm_head.weight.data_ptr())
    eps = getattr(norm, "variance_epsilon", None)
    if eps is None:
        eps = getattr(norm, "eps", None)
    if eps is None:
        eps = getattr(qwen_model.config, "rms_norm_eps", 1e-6)
    return StandaloneNolaneConfig(
        vocab_size=int(embed.num_embeddings),
        hidden_size=int(embed.embedding_dim),
        rms_norm_eps=float(eps),
        tie_word_embeddings=tied,
        padding_idx=embed.padding_idx,
        bos_token_id=getattr(qwen_model.config, "bos_token_id", None),
        eos_token_id=getattr(qwen_model.config, "eos_token_id", None),
        pad_token_id=getattr(qwen_model.config, "pad_token_id", None),
        boundary_dtype=str(embed.weight.dtype).replace("torch.", ""),
    )


def clone_boundary_from_qwen(qwen_model) -> StandaloneBoundaryModule:
    torch = __import__("torch")
    config = standalone_config_from_qwen(qwen_model)
    source_dtype = qwen_model.model.embed_tokens.weight.dtype
    source_device = qwen_model.model.embed_tokens.weight.device
    boundary = StandaloneBoundaryModule(
        config,
        dtype=source_dtype,
        device=source_device,
    )
    with torch.no_grad():
        boundary.embed_tokens.weight.copy_(
            qwen_model.model.embed_tokens.weight.detach()
        )
        boundary.final_norm.weight.copy_(
            qwen_model.model.norm.weight.detach()
        )
        if not config.tie_word_embeddings:
            boundary.lm_head.weight.copy_(
                qwen_model.lm_head.weight.detach()
            )
    return boundary
