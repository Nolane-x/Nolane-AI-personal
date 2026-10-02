from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class QuantizedBoundaryConfig:
    vocab_size: int
    hidden_size: int
    rank: int
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    padding_idx: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    quantization: str = "int8_symmetric_rowwise"
    dequant_dtype: str = "float32"
    logit_chunk_size: int = 4096

    def validate(self) -> None:
        if self.vocab_size <= 0 or self.hidden_size <= 0 or self.rank <= 0:
            raise ValueError("vocab_size, hidden_size and rank must be positive")
        if self.rank > min(self.vocab_size, self.hidden_size):
            raise ValueError("rank must be <= min(vocab_size, hidden_size)")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")
        if self.quantization != "int8_symmetric_rowwise":
            raise ValueError("unsupported quantization")
        if self.dequant_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("unsupported dequant_dtype")
        if self.logit_chunk_size < 1:
            raise ValueError("logit_chunk_size must be positive")


def analytical_factorized_storage_bytes(config: QuantizedBoundaryConfig, *, source_element_size: int = 4) -> int:
    config.validate()
    copies = 1 if config.tie_word_embeddings else 2
    factor_elements = copies * (
        config.vocab_size * config.rank + config.rank * config.hidden_size
    )
    return factor_elements * int(source_element_size) + config.hidden_size * int(source_element_size)


def analytical_quantized_storage_bytes(config: QuantizedBoundaryConfig) -> int:
    config.validate()
    copies = 1 if config.tie_word_embeddings else 2
    factor_bytes = copies * (
        config.vocab_size * config.rank
        + config.vocab_size * 4
        + config.rank * config.hidden_size
        + config.rank * 4
    )
    norm_bytes = config.hidden_size * 4
    return factor_bytes + norm_bytes


def _dtype(torch, name: str):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def quantize_rows(tensor):
    torch = __import__("torch")
    matrix = tensor.detach().float().cpu()
    scales = matrix.abs().amax(dim=1).clamp_min(1e-12) / 127.0
    quantized = torch.round(matrix / scales.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
    return quantized, scales.to(torch.float32)


def dequantize_rows(quantized, scales, *, dtype=None):
    torch = __import__("torch")
    dtype = dtype or torch.float32
    return quantized.to(dtype) * scales.to(dtype).unsqueeze(1)


def _relative_error(reference, reconstructed) -> float:
    denominator = reference.detach().float().norm().clamp_min(1e-12)
    return float(
        ((reference.detach().float() - reconstructed.detach().float()).norm() / denominator)
        .cpu()
    )


class QuantizedBoundaryModule:
    """Inference-only int8 low-rank language boundary.

    Quantized factors remain int8 in the persistent state. Only selected token
    rows and one vocabulary chunk are dequantized at a time.
    """

    def __init__(
        self,
        config: QuantizedBoundaryConfig,
        *,
        input_codes_q,
        input_code_scales,
        input_basis_q,
        input_basis_scales,
        final_norm_weight,
        output_codes_q=None,
        output_code_scales=None,
        output_basis_q=None,
        output_basis_scales=None,
        device=None,
    ) -> None:
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise RuntimeError("Install neural support with: pip install -e '.[neural]'") from exc

        config.validate()
        self.torch = torch
        self.nn = nn
        self.config = config
        dequant_dtype = _dtype(torch, config.dequant_dtype)

        class QuantizedEmbedding(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("codes_q", input_codes_q.detach().to(torch.int8))
                self.register_buffer("code_scales", input_code_scales.detach().to(torch.float32))
                self.register_buffer("basis_q", input_basis_q.detach().to(torch.int8))
                self.register_buffer("basis_scales", input_basis_scales.detach().to(torch.float32))

            def basis(self):
                return dequantize_rows(
                    self.basis_q,
                    self.basis_scales,
                    dtype=dequant_dtype,
                )

            def forward(self, input_ids):
                codes = self.codes_q[input_ids].to(dequant_dtype)
                scales = self.code_scales[input_ids].to(dequant_dtype).unsqueeze(-1)
                return (codes * scales) @ self.basis()

        class RMSNorm(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(final_norm_weight.detach().to(dequant_dtype).clone())
                self.variance_epsilon = float(config.rms_norm_eps)

            def forward(self, hidden_states):
                input_dtype = hidden_states.dtype
                states = hidden_states.float()
                states = states * torch.rsqrt(
                    states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon
                )
                return self.weight * states.to(input_dtype)

        class QuantizedOutput(nn.Module):
            def __init__(self):
                super().__init__()
                if output_codes_q is None or output_code_scales is None:
                    raise ValueError("untied boundary requires output code factors")
                if output_basis_q is None or output_basis_scales is None:
                    raise ValueError("untied boundary requires output basis factors")
                self.register_buffer("codes_q", output_codes_q.detach().to(torch.int8))
                self.register_buffer("code_scales", output_code_scales.detach().to(torch.float32))
                self.register_buffer("basis_q", output_basis_q.detach().to(torch.int8))
                self.register_buffer("basis_scales", output_basis_scales.detach().to(torch.float32))

            def basis(self):
                return dequantize_rows(
                    self.basis_q,
                    self.basis_scales,
                    dtype=dequant_dtype,
                )

        self.embed_tokens = QuantizedEmbedding()
        self.final_norm = RMSNorm()
        self.output = None if config.tie_word_embeddings else QuantizedOutput()

        class Module(nn.Module):
            def __init__(self, outer):
                super().__init__()
                self.embed_tokens = outer.embed_tokens
                self.final_norm = outer.final_norm
                if outer.output is not None:
                    self.output = outer.output

        self.module = Module(self)
        if device is not None:
            self.module.to(device)

    def to(self, device: str):
        self.module.to(device)
        return self

    def eval(self):
        self.module.eval()
        return self

    def train(self):
        raise RuntimeError("quantized boundary is inference-only; train factorized weights before quantization")

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.module.parameters())

    def tensor_count(self) -> int:
        return sum(tensor.numel() for tensor in self.module.state_dict().values())

    def storage_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in self.module.state_dict().values()
        )

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state):
        return self.module.load_state_dict(state)

    def reconstructed_input_weight(self):
        codes = dequantize_rows(
            self.embed_tokens.codes_q,
            self.embed_tokens.code_scales,
            dtype=self.torch.float32,
        )
        basis = dequantize_rows(
            self.embed_tokens.basis_q,
            self.embed_tokens.basis_scales,
            dtype=self.torch.float32,
        )
        return codes @ basis

    def reconstructed_output_weight(self):
        if self.config.tie_word_embeddings:
            return self.reconstructed_input_weight()
        codes = dequantize_rows(
            self.output.codes_q,
            self.output.code_scales,
            dtype=self.torch.float32,
        )
        basis = dequantize_rows(
            self.output.basis_q,
            self.output.basis_scales,
            dtype=self.torch.float32,
        )
        return codes @ basis

    def _chunked_project(self, rank_hidden, codes_q, scales):
        torch = self.torch
        dtype = rank_hidden.dtype
        chunks = []
        size = int(self.config.logit_chunk_size)
        for start in range(0, self.config.vocab_size, size):
            end = min(start + size, self.config.vocab_size)
            codes = codes_q[start:end].to(dtype)
            row_scales = scales[start:end].to(dtype).unsqueeze(1)
            chunks.append(rank_hidden @ (codes * row_scales).t())
        return torch.cat(chunks, dim=-1)

    def logits(self, hidden_states):
        normalized = self.final_norm(hidden_states)
        if self.config.tie_word_embeddings:
            basis = self.embed_tokens.basis()
            rank_hidden = normalized @ basis.t()
            logits = self._chunked_project(
                rank_hidden,
                self.embed_tokens.codes_q,
                self.embed_tokens.code_scales,
            )
            return logits.float()

        basis = self.output.basis()
        rank_hidden = normalized @ basis.t()
        logits = self._chunked_project(
            rank_hidden,
            self.output.codes_q,
            self.output.code_scales,
        )
        return logits.float()


def quantize_factorized_boundary(source, *, logit_chunk_size: int = 4096):
    torch = __import__("torch")
    source_config = source.config
    config = QuantizedBoundaryConfig(
        vocab_size=int(source_config.vocab_size),
        hidden_size=int(source_config.hidden_size),
        rank=int(source_config.rank),
        rms_norm_eps=float(source_config.rms_norm_eps),
        tie_word_embeddings=bool(source_config.tie_word_embeddings),
        padding_idx=source_config.padding_idx,
        bos_token_id=source_config.bos_token_id,
        eos_token_id=source_config.eos_token_id,
        pad_token_id=source_config.pad_token_id,
        dequant_dtype="float32",
        logit_chunk_size=int(logit_chunk_size),
    )

    input_codes_q, input_code_scales = quantize_rows(source.embed_tokens.codes.weight)
    input_basis_q, input_basis_scales = quantize_rows(source.embed_tokens.basis)

    kwargs = {}
    code_error = _relative_error(
        source.embed_tokens.codes.weight,
        dequantize_rows(input_codes_q, input_code_scales),
    )
    basis_error = _relative_error(
        source.embed_tokens.basis,
        dequantize_rows(input_basis_q, input_basis_scales),
    )
    output_code_error = code_error
    output_basis_error = basis_error

    if not config.tie_word_embeddings:
        output_codes_q, output_code_scales = quantize_rows(source.output_codes)
        output_basis_q, output_basis_scales = quantize_rows(source.output_basis)
        kwargs = {
            "output_codes_q": output_codes_q,
            "output_code_scales": output_code_scales,
            "output_basis_q": output_basis_q,
            "output_basis_scales": output_basis_scales,
        }
        output_code_error = _relative_error(
            source.output_codes,
            dequantize_rows(output_codes_q, output_code_scales),
        )
        output_basis_error = _relative_error(
            source.output_basis,
            dequantize_rows(output_basis_q, output_basis_scales),
        )

    boundary = QuantizedBoundaryModule(
        config,
        input_codes_q=input_codes_q,
        input_code_scales=input_code_scales,
        input_basis_q=input_basis_q,
        input_basis_scales=input_basis_scales,
        final_norm_weight=source.final_norm.weight.detach().cpu(),
        **kwargs,
    )
    float_storage_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in source.module.state_dict().values()
    )
    receipt = {
        "schema": "NOLANE-L19-INT8-FACTORIZED-BOUNDARY-V1",
        "quantization": config.quantization,
        "rank": config.rank,
        "tie_word_embeddings": config.tie_word_embeddings,
        "input_code_relative_error": code_error,
        "input_basis_relative_error": basis_error,
        "output_code_relative_error": output_code_error,
        "output_basis_relative_error": output_basis_error,
        "source_float_storage_bytes": int(float_storage_bytes),
        "quantized_storage_bytes": int(boundary.storage_bytes()),
        "storage_ratio": float(boundary.storage_bytes() / max(1, float_storage_bytes)),
        "config": asdict(config),
    }
    return boundary, receipt
