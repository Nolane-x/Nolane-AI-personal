from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class RealQwen06Evidence:
    repo_id: str
    requested_revision: str
    resolved_revision: str
    model_type: str
    parameter_count: int
    expected_parameter_count: int
    vocab_size: int
    hidden_size: int
    decoder_layers: int
    tie_word_embeddings: bool
    forward_sequence_tokens: int
    forward_logits_finite: bool
    forward_top_token_id: int
    sampled_vocab_rows: int
    sampled_rank: int
    sample_input_factorization_error: float
    sample_output_factorization_error: float
    sample_input_quantization_error: float
    sample_output_quantization_error: float
    sample_logit_relative_error: float
    full_dense_boundary_parameters: int
    full_factorized_boundary_parameters: int
    full_factorized_parameter_ratio: float
    full_factorized_fp32_bytes: int
    full_factorized_bf16_bytes: int
    full_quantized_bytes: int
    full_quantized_ratio_vs_fp32: float
    full_quantized_ratio_vs_bf16: float
    source_boundary_storage_bytes: int
    runtime_requires_qwen_for_court: bool


@dataclass(slots=True)
class RealQwen06Thresholds:
    expected_model_type: str = "qwen3"
    max_sample_quantization_error: float = 0.03
    max_sample_logit_relative_error: float = 0.06
    max_quantized_ratio_vs_fp32: float = 0.30
    max_quantized_ratio_vs_bf16: float = 0.60
    min_decoder_layers: int = 1
    min_sampled_vocab_rows: int = 256


def decide_real_qwen06_court(evidence, thresholds=None):
    t=thresholds or RealQwen06Thresholds()
    reasons=[]
    if evidence.requested_revision!=evidence.resolved_revision:
        reasons.append("revision_mismatch")
    if evidence.model_type!=t.expected_model_type:
        reasons.append("model_type_mismatch")
    if evidence.parameter_count!=evidence.expected_parameter_count:
        reasons.append("parameter_count_mismatch")
    if evidence.decoder_layers<t.min_decoder_layers:
        reasons.append("decoder_layer_count_invalid")
    if not evidence.forward_logits_finite:
        reasons.append("real_forward_nonfinite")
    if evidence.sampled_vocab_rows<t.min_sampled_vocab_rows:
        reasons.append("insufficient_real_weight_rows")
    for name,value in (
        ("sample_input_quantization_error",evidence.sample_input_quantization_error),
        ("sample_output_quantization_error",evidence.sample_output_quantization_error),
    ):
        if value>t.max_sample_quantization_error:
            reasons.append(name+"_failed")
    if evidence.sample_logit_relative_error>t.max_sample_logit_relative_error:
        reasons.append("sample_logit_relative_error_failed")
    if evidence.full_quantized_ratio_vs_fp32>=t.max_quantized_ratio_vs_fp32:
        reasons.append("fp32_storage_ratio_failed")
    if evidence.full_quantized_ratio_vs_bf16>=t.max_quantized_ratio_vs_bf16:
        reasons.append("bf16_storage_ratio_failed")
    if not evidence.runtime_requires_qwen_for_court:
        reasons.append("court_did_not_load_real_qwen")
    return {
        "status":"REAL_QWEN06_WEIGHT_COURT_PASS" if not reasons else "REAL_QWEN06_WEIGHT_COURT_BLOCKED",
        "reasons":reasons,
        "evidence":asdict(evidence),
        "thresholds":asdict(t),
    }
