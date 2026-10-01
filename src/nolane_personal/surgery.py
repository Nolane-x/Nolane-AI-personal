from __future__ import annotations

import contextlib
import hashlib
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .store import payload_digest


@dataclass(slots=True)
class LatentAdapterConfig:
    latent_dim: int = 32
    bottleneck_dim: int = 16
    max_abs_gate: float = 0.10

    def validate(self) -> None:
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be positive")
        if not 0.0 < self.max_abs_gate <= 0.25:
            raise ValueError("max_abs_gate must be in (0, 0.25]")


def analytical_adapter_parameter_count(hidden_size: int, config: LatentAdapterConfig | None = None) -> int:
    c = config or LatentAdapterConfig()
    c.validate()
    hidden_size = int(hidden_size)
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    layer_norm = 2 * c.latent_dim
    first = c.latent_dim * c.bottleneck_dim + c.bottleneck_dim
    second = c.bottleneck_dim * hidden_size + hidden_size
    gate = 1
    return layer_norm + first + second + gate


class LatentResidualAdapter:
    """Tiny residual bridge from the 32D Living latent into LM hidden space.

    The adapter is an isolated candidate artifact. It does not patch or replace
    any base-model parameter. Counterfactual hooks may apply its residual only
    during a shadow forward.
    """

    def __init__(
        self,
        hidden_size: int,
        config: LatentAdapterConfig | None = None,
        *,
        seed: int = 0,
    ) -> None:
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise RuntimeError("Install neural support with: pip install -e '.[neural]'") from exc

        self.torch = torch
        self.nn = nn
        self.config = config or LatentAdapterConfig()
        self.config.validate()
        self.hidden_size = int(hidden_size)
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.module = nn.ModuleDict(
                {
                    "norm": nn.LayerNorm(self.config.latent_dim),
                    "down": nn.Linear(self.config.latent_dim, self.config.bottleneck_dim),
                    "up": nn.Linear(self.config.bottleneck_dim, self.hidden_size),
                }
            )
            self.raw_gate = nn.Parameter(torch.zeros(()))
            self.module.register_parameter("raw_gate", self.raw_gate)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.module.parameters())

    def effective_gate(self, override: float | None = None):
        torch = self.torch
        if override is not None:
            value = max(-self.config.max_abs_gate, min(self.config.max_abs_gate, float(override)))
            return torch.tensor(value, dtype=self.raw_gate.dtype, device=self.raw_gate.device)
        return self.config.max_abs_gate * torch.tanh(self.raw_gate)

    def residual(self, latent, *, dtype, device, gate_override: float | None = None):
        torch = self.torch
        if not torch.is_tensor(latent):
            latent = torch.tensor(latent, dtype=torch.float32)
        if latent.ndim == 1:
            latent = latent.unsqueeze(0)
        if latent.ndim != 2 or latent.shape[-1] != self.config.latent_dim:
            raise ValueError("latent shape mismatch")
        latent = latent.to(device=device, dtype=self.module["down"].weight.dtype)
        x = self.module["norm"](latent)
        x = torch.nn.functional.silu(self.module["down"](x))
        x = self.module["up"](x)
        x = x * self.effective_gate(gate_override)
        return x.to(dtype=dtype)

    def to(self, device: str):
        self.module.to(device)
        return self

    def eval(self):
        self.module.eval()
        return self


def tensor_state_digest(state_dict: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def module_parameter_digest(module) -> str:
    return tensor_state_digest(module.state_dict())


def _nested_attr(obj: Any, path: str) -> Any | None:
    current = obj
    for part in path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def resolve_transformer_layers(model) -> list[Any]:
    """Resolve common Hugging Face decoder-layer containers without Qwen-only imports."""
    for path in ("model.layers", "model.model.layers", "transformer.h", "layers"):
        layers = _nested_attr(model, path)
        if layers is not None:
            try:
                result = list(layers)
            except TypeError:
                continue
            if result:
                return result
    raise ValueError("unable to resolve transformer layer container")


def select_layer_indices(total_layers: int, requested: Iterable[int] | None = None) -> list[int]:
    total = int(total_layers)
    if total <= 0:
        raise ValueError("total_layers must be positive")
    if requested is not None:
        result = sorted(set(int(i) for i in requested))
        if not result:
            raise ValueError("at least one layer index is required")
        if result[0] < 0 or result[-1] >= total:
            raise ValueError("layer index out of range")
        return result
    raw = [max(0, total // 4), max(0, total // 2), max(0, (3 * total) // 4)]
    return sorted(set(min(total - 1, i) for i in raw))


def _inject_hidden(output, residual, *, token_scope: str):
    torch = residual.new_zeros(())  # keeps torch/device dependency local
    del torch
    if isinstance(output, tuple):
        hidden = output[0]
        replaced = _inject_tensor(hidden, residual, token_scope=token_scope)
        return (replaced,) + output[1:]
    if isinstance(output, list):
        hidden = output[0]
        replaced = _inject_tensor(hidden, residual, token_scope=token_scope)
        return [replaced] + output[1:]
    return _inject_tensor(output, residual, token_scope=token_scope)


def _inject_tensor(hidden, residual, *, token_scope: str):
    if not hasattr(hidden, "ndim") or hidden.ndim != 3:
        raise ValueError("expected decoder hidden state shaped [batch, sequence, hidden]")
    if hidden.shape[-1] != residual.shape[-1]:
        raise ValueError("adapter hidden size mismatch")
    if residual.shape[0] == 1 and hidden.shape[0] > 1:
        residual = residual.expand(hidden.shape[0], -1)
    if residual.shape[0] != hidden.shape[0]:
        raise ValueError("latent batch does not match model batch")
    delta = residual.unsqueeze(1)
    if token_scope == "all":
        return hidden + delta
    if token_scope != "last":
        raise ValueError("token_scope must be 'last' or 'all'")
    updated = hidden.clone()
    updated[:, -1:, :] = updated[:, -1:, :] + delta
    return updated


class LatentHookSession(contextlib.AbstractContextManager):
    def __init__(
        self,
        model,
        adapter: LatentResidualAdapter,
        latent,
        *,
        layer_indices: Iterable[int] | None = None,
        gate_override: float | None = None,
        token_scope: str = "last",
    ) -> None:
        self.model = model
        self.adapter = adapter
        self.latent = latent
        self.layers = resolve_transformer_layers(model)
        self.layer_indices = select_layer_indices(len(self.layers), layer_indices)
        self.gate_override = gate_override
        self.token_scope = token_scope
        self.handles: list[Any] = []

    def __enter__(self):
        for index in self.layer_indices:
            layer = self.layers[index]

            def hook(_module, _inputs, output, *, _index=index):
                del _index
                hidden = output[0] if isinstance(output, (tuple, list)) else output
                residual = self.adapter.residual(
                    self.latent,
                    dtype=hidden.dtype,
                    device=hidden.device,
                    gate_override=self.gate_override,
                )
                return _inject_hidden(output, residual, token_scope=self.token_scope)

            self.handles.append(layer.register_forward_hook(hook))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        return False


@dataclass(slots=True)
class CounterfactualReceipt:
    schema: str
    authority: str
    candidate_id: str
    base_model_fingerprint: str
    latent_digest: str
    adapter_digest: str
    adapter_parameters: int
    layer_indices: list[int]
    token_scope: str
    gate: float
    baseline_latency_ms: float
    counterfactual_latency_ms: float
    overhead_ratio: float
    kl_baseline_to_counterfactual: float
    mean_abs_logit_shift: float
    max_abs_logit_shift: float
    cosine_similarity: float
    baseline_top1: int
    counterfactual_top1: int
    top1_changed: bool
    base_model_unchanged: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CounterfactualSurgeryProbe:
    """Paired baseline/counterfactual court. Baseline output is always returned."""

    def __init__(
        self,
        model,
        adapter: LatentResidualAdapter,
        *,
        base_model_fingerprint: str,
        latent_digest: str,
    ) -> None:
        self.model = model
        self.adapter = adapter
        self.base_model_fingerprint = str(base_model_fingerprint)
        self.latent_digest = str(latent_digest)

    def run(
        self,
        model_inputs: dict[str, Any],
        latent,
        *,
        layer_indices: Iterable[int] | None = None,
        gate: float = 0.02,
        token_scope: str = "last",
    ) -> tuple[Any, CounterfactualReceipt]:
        torch = self.adapter.torch
        self.model.eval()
        self.adapter.eval()
        before_digest = module_parameter_digest(self.model)
        adapter_digest = module_parameter_digest(self.adapter.module)
        layers = resolve_transformer_layers(self.model)
        selected = select_layer_indices(len(layers), layer_indices)
        bounded_gate = max(-self.adapter.config.max_abs_gate, min(self.adapter.config.max_abs_gate, float(gate)))

        with torch.inference_mode():
            start = time.perf_counter()
            baseline = self.model(**model_inputs)
            baseline_ms = (time.perf_counter() - start) * 1000.0

            start = time.perf_counter()
            with LatentHookSession(
                self.model,
                self.adapter,
                latent,
                layer_indices=selected,
                gate_override=bounded_gate,
                token_scope=token_scope,
            ):
                counterfactual = self.model(**model_inputs)
            counterfactual_ms = (time.perf_counter() - start) * 1000.0

        base_logits = baseline.logits[:, -1, :].float()
        cf_logits = counterfactual.logits[:, -1, :].float()
        log_p = torch.log_softmax(base_logits, dim=-1)
        log_q = torch.log_softmax(cf_logits, dim=-1)
        p = torch.softmax(base_logits, dim=-1)
        kl = torch.sum(p * (log_p - log_q), dim=-1).mean()
        shift = cf_logits - base_logits
        cosine = torch.nn.functional.cosine_similarity(base_logits, cf_logits, dim=-1).mean()
        base_top1 = int(torch.argmax(base_logits[0]).item())
        cf_top1 = int(torch.argmax(cf_logits[0]).item())
        after_digest = module_parameter_digest(self.model)

        candidate_payload = {
            "base_model_fingerprint": self.base_model_fingerprint,
            "latent_digest": self.latent_digest,
            "adapter_digest": adapter_digest,
            "layer_indices": selected,
            "token_scope": token_scope,
            "gate": bounded_gate,
        }
        receipt = CounterfactualReceipt(
            schema="NOLANE-L5-COUNTERFACTUAL-SURGERY-V1",
            authority="COUNTERFACTUAL_ONLY_BASELINE_OUTPUT",
            candidate_id=payload_digest(candidate_payload),
            base_model_fingerprint=self.base_model_fingerprint,
            latent_digest=self.latent_digest,
            adapter_digest=adapter_digest,
            adapter_parameters=self.adapter.parameter_count(),
            layer_indices=selected,
            token_scope=token_scope,
            gate=bounded_gate,
            baseline_latency_ms=baseline_ms,
            counterfactual_latency_ms=counterfactual_ms,
            overhead_ratio=counterfactual_ms / max(baseline_ms, 1e-9),
            kl_baseline_to_counterfactual=float(kl.item()),
            mean_abs_logit_shift=float(torch.mean(torch.abs(shift)).item()),
            max_abs_logit_shift=float(torch.max(torch.abs(shift)).item()),
            cosine_similarity=float(cosine.item()),
            baseline_top1=base_top1,
            counterfactual_top1=cf_top1,
            top1_changed=base_top1 != cf_top1,
            base_model_unchanged=before_digest == after_digest,
        )
        return baseline, receipt
