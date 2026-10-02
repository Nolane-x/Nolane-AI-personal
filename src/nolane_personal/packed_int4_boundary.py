from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class PackedInt4BoundaryConfig:
    vocab_size: int
    hidden_size: int
    rank: int
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    padding_idx: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    quantization: str = "int4_symmetric_rowwise_packed"
    dequant_dtype: str = "float32"
    logit_chunk_size: int = 4096

    def validate(self) -> None:
        if self.vocab_size <= 0 or self.hidden_size <= 0 or self.rank <= 0:
            raise ValueError("vocab_size, hidden_size and rank must be positive")
        if self.rank > min(self.vocab_size, self.hidden_size):
            raise ValueError("rank must be <= min(vocab_size, hidden_size)")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")
        if self.quantization != "int4_symmetric_rowwise_packed":
            raise ValueError("unsupported quantization")
        if self.dequant_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("unsupported dequant_dtype")
        if self.logit_chunk_size < 1:
            raise ValueError("logit_chunk_size must be positive")


def _dtype(torch, name: str):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _packed_columns(columns: int) -> int:
    return (int(columns) + 1) // 2


def quantize_pack_rows(tensor):
    """Symmetric row-wise INT4 with two signed values packed into one byte.

    Logical quantized values are in [-7, 7]. They are encoded as [1, 15]
    using q + 8, leaving nibble 8 as exact logical zero.
    """
    torch = __import__("torch")
    matrix = tensor.detach().float().cpu()
    if matrix.ndim != 2:
        raise ValueError("packed INT4 quantization expects a 2D tensor")
    rows, columns = matrix.shape
    scales = matrix.abs().amax(dim=1).clamp_min(1e-12) / 7.0
    q = torch.round(matrix / scales.unsqueeze(1)).clamp(-7, 7).to(torch.int16)
    encoded = (q + 8).to(torch.uint8)
    if columns % 2:
        zero_pad = torch.full((rows, 1), 8, dtype=torch.uint8)
        encoded = torch.cat([encoded, zero_pad], dim=1)
    low = encoded[:, 0::2]
    high = encoded[:, 1::2]
    packed = low | (high << 4)
    return packed.contiguous(), scales.to(torch.float32), int(columns)


def unpack_rows(packed, columns: int):
    torch = __import__("torch")
    if packed.ndim != 2:
        raise ValueError("packed INT4 tensor must be 2D")
    columns = int(columns)
    if columns < 1:
        raise ValueError("columns must be positive")
    low = (packed & 0x0F).to(torch.int16) - 8
    high = ((packed >> 4) & 0x0F).to(torch.int16) - 8
    rows = packed.shape[0]
    values = torch.empty(
        (rows, packed.shape[1] * 2),
        dtype=torch.int16,
        device=packed.device,
    )
    values[:, 0::2] = low
    values[:, 1::2] = high
    return values[:, :columns]


def dequantize_packed_rows(packed, scales, columns: int, *, dtype=None):
    torch = __import__("torch")
    dtype = dtype or torch.float32
    logical = unpack_rows(packed, columns).to(dtype)
    return logical * scales.to(dtype).unsqueeze(1)


def analytical_int8_storage_bytes(config: PackedInt4BoundaryConfig) -> int:
    config.validate()
    copies = 1 if config.tie_word_embeddings else 2
    factors = copies * (
        config.vocab_size * config.rank
        + config.vocab_size * 4
        + config.rank * config.hidden_size
        + config.rank * 4
    )
    return factors + config.hidden_size * 4


def analytical_packed_int4_storage_bytes(config: PackedInt4BoundaryConfig) -> int:
    config.validate()
    copies = 1 if config.tie_word_embeddings else 2
    factors = copies * (
        config.vocab_size * _packed_columns(config.rank)
        + config.vocab_size * 4
        + config.rank * _packed_columns(config.hidden_size)
        + config.rank * 4
    )
    return factors + config.hidden_size * 4


def _relative_error(reference, reconstructed) -> float:
    denominator = reference.detach().float().norm().clamp_min(1e-12)
    return float(
        ((reference.detach().float() - reconstructed.detach().float()).norm() / denominator)
        .cpu()
    )


class PackedInt4BoundaryModule:
    """Inference-only packed INT4 low-rank language boundary."""

    def __init__(
        self,
        config: PackedInt4BoundaryConfig,
        *,
        input_codes_packed,
        input_code_scales,
        input_basis_packed,
        input_basis_scales,
        final_norm_weight,
        output_codes_packed=None,
        output_code_scales=None,
        output_basis_packed=None,
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

        class PackedEmbedding(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("codes_packed", input_codes_packed.detach().to(torch.uint8))
                self.register_buffer("code_scales", input_code_scales.detach().to(torch.float32))
                self.register_buffer("basis_packed", input_basis_packed.detach().to(torch.uint8))
                self.register_buffer("basis_scales", input_basis_scales.detach().to(torch.float32))

            def basis(self):
                return dequantize_packed_rows(
                    self.basis_packed,
                    self.basis_scales,
                    config.hidden_size,
                    dtype=dequant_dtype,
                )

            def forward(self, input_ids):
                packed = self.codes_packed[input_ids]
                scales = self.code_scales[input_ids]
                logical = unpack_rows(
                    packed.reshape(-1, packed.shape[-1]),
                    config.rank,
                ).reshape(*input_ids.shape, config.rank).to(dequant_dtype)
                codes = logical * scales.to(dequant_dtype).unsqueeze(-1)
                return codes @ self.basis()

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

        class PackedOutput(nn.Module):
            def __init__(self):
                super().__init__()
                if output_codes_packed is None or output_code_scales is None:
                    raise ValueError("untied boundary requires packed output code factors")
                if output_basis_packed is None or output_basis_scales is None:
                    raise ValueError("untied boundary requires packed output basis factors")
                self.register_buffer("codes_packed", output_codes_packed.detach().to(torch.uint8))
                self.register_buffer("code_scales", output_code_scales.detach().to(torch.float32))
                self.register_buffer("basis_packed", output_basis_packed.detach().to(torch.uint8))
                self.register_buffer("basis_scales", output_basis_scales.detach().to(torch.float32))

            def basis(self):
                return dequantize_packed_rows(
                    self.basis_packed,
                    self.basis_scales,
                    config.hidden_size,
                    dtype=dequant_dtype,
                )

        self.embed_tokens = PackedEmbedding()
        self.final_norm = RMSNorm()
        self.output = None if config.tie_word_embeddings else PackedOutput()

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
        raise RuntimeError("packed INT4 boundary is inference-only; train float factors before packing")

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.module.parameters())

    def storage_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in self.module.state_dict().values()
        )

    def tensor_count(self) -> int:
        return sum(tensor.numel() for tensor in self.module.state_dict().values())

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state):
        return self.module.load_state_dict(state)

    def reconstructed_input_weight(self):
        codes = dequantize_packed_rows(
            self.embed_tokens.codes_packed,
            self.embed_tokens.code_scales,
            self.config.rank,
            dtype=self.torch.float32,
        )
        basis = self.embed_tokens.basis().float()
        return codes @ basis

    def reconstructed_output_weight(self):
        if self.config.tie_word_embeddings:
            return self.reconstructed_input_weight()
        codes = dequantize_packed_rows(
            self.output.codes_packed,
            self.output.code_scales,
            self.config.rank,
            dtype=self.torch.float32,
        )
        basis = self.output.basis().float()
        return codes @ basis

    def _chunked_project(self, rank_hidden, packed_codes, scales):
        chunks = []
        size = int(self.config.logit_chunk_size)
        dtype = rank_hidden.dtype
        for start in range(0, self.config.vocab_size, size):
            end = min(start + size, self.config.vocab_size)
            codes = dequantize_packed_rows(
                packed_codes[start:end],
                scales[start:end],
                self.config.rank,
                dtype=dtype,
            )
            chunks.append(rank_hidden @ codes.t())
        return self.torch.cat(chunks, dim=-1)

    def logits(self, hidden_states):
        normalized = self.final_norm(hidden_states)
        if self.config.tie_word_embeddings:
            basis = self.embed_tokens.basis()
            rank_hidden = normalized @ basis.t()
            return self._chunked_project(
                rank_hidden,
                self.embed_tokens.codes_packed,
                self.embed_tokens.code_scales,
            ).float()
        basis = self.output.basis()
        rank_hidden = normalized @ basis.t()
        return self._chunked_project(
            rank_hidden,
            self.output.codes_packed,
            self.output.code_scales,
        ).float()


def pack_factorized_boundary(source, *, logit_chunk_size: int = 4096):
    source_config = source.config
    config = PackedInt4BoundaryConfig(
        vocab_size=int(source_config.vocab_size),
        hidden_size=int(source_config.hidden_size),
        rank=int(source_config.rank),
        rms_norm_eps=float(source_config.rms_norm_eps),
        tie_word_embeddings=bool(source_config.tie_word_embeddings),
        padding_idx=source_config.padding_idx,
        bos_token_id=source_config.bos_token_id,
        eos_token_id=source_config.eos_token_id,
        pad_token_id=source_config.pad_token_id,
        logit_chunk_size=int(logit_chunk_size),
    )

    in_codes_packed, in_code_scales, _ = quantize_pack_rows(source.embed_tokens.codes.weight)
    in_basis_packed, in_basis_scales, _ = quantize_pack_rows(source.embed_tokens.basis)

    in_codes_recon = dequantize_packed_rows(
        in_codes_packed, in_code_scales, config.rank
    )
    in_basis_recon = dequantize_packed_rows(
        in_basis_packed, in_basis_scales, config.hidden_size
    )
    input_code_error = _relative_error(source.embed_tokens.codes.weight, in_codes_recon)
    input_basis_error = _relative_error(source.embed_tokens.basis, in_basis_recon)

    kwargs = {}
    output_code_error = input_code_error
    output_basis_error = input_basis_error
    if not config.tie_word_embeddings:
        out_codes_packed, out_code_scales, _ = quantize_pack_rows(source.output_codes)
        out_basis_packed, out_basis_scales, _ = quantize_pack_rows(source.output_basis)
        kwargs = {
            "output_codes_packed": out_codes_packed,
            "output_code_scales": out_code_scales,
            "output_basis_packed": out_basis_packed,
            "output_basis_scales": out_basis_scales,
        }
        output_code_error = _relative_error(
            source.output_codes,
            dequantize_packed_rows(out_codes_packed, out_code_scales, config.rank),
        )
        output_basis_error = _relative_error(
            source.output_basis,
            dequantize_packed_rows(out_basis_packed, out_basis_scales, config.hidden_size),
        )

    boundary = PackedInt4BoundaryModule(
        config,
        input_codes_packed=in_codes_packed,
        input_code_scales=in_code_scales,
        input_basis_packed=in_basis_packed,
        input_basis_scales=in_basis_scales,
        final_norm_weight=source.final_norm.weight.detach().cpu(),
        **kwargs,
    )

    source_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in source.module.state_dict().values()
    )
    int8_bytes = analytical_int8_storage_bytes(config)
    receipt = {
        "schema": "NOLANE-L20-PACKED-INT4-BOUNDARY-V1",
        "quantization": config.quantization,
        "rank": config.rank,
        "tie_word_embeddings": config.tie_word_embeddings,
        "input_code_relative_error": input_code_error,
        "input_basis_relative_error": input_basis_error,
        "output_code_relative_error": output_code_error,
        "output_basis_relative_error": output_basis_error,
        "source_float_storage_bytes": int(source_bytes),
        "analytical_int8_storage_bytes": int(int8_bytes),
        "packed_int4_storage_bytes": int(boundary.storage_bytes()),
        "storage_ratio_vs_float": float(boundary.storage_bytes() / max(1, source_bytes)),
        "storage_ratio_vs_int8": float(boundary.storage_bytes() / max(1, int8_bytes)),
        "config": asdict(config),
    }
    return boundary, receipt
