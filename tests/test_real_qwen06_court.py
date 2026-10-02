from nolane_personal.real_qwen06_court import RealQwen06Evidence,decide_real_qwen06_court


def evidence(**updates):
    values=dict(
        repo_id="Qwen/Qwen3-0.6B",
        requested_revision="abc",
        resolved_revision="abc",
        model_type="qwen3",
        parameter_count=751632384,
        expected_parameter_count=751632384,
        vocab_size=151936,
        hidden_size=1024,
        decoder_layers=28,
        tie_word_embeddings=True,
        forward_sequence_tokens=5,
        forward_logits_finite=True,
        forward_top_token_id=42,
        sampled_vocab_rows=512,
        sampled_rank=128,
        sample_input_factorization_error=.5,
        sample_output_factorization_error=.5,
        sample_input_quantization_error=.01,
        sample_output_quantization_error=.01,
        sample_logit_relative_error=.02,
        full_dense_boundary_parameters=155000000,
        full_factorized_boundary_parameters=19500000,
        full_factorized_parameter_ratio=.126,
        full_factorized_fp32_bytes=78000000,
        full_factorized_bf16_bytes=39000000,
        full_quantized_bytes=21000000,
        full_quantized_ratio_vs_fp32=.269,
        full_quantized_ratio_vs_bf16=.538,
        source_boundary_storage_bytes=310000000,
        runtime_requires_qwen_for_court=True,
    )
    values.update(updates)
    return RealQwen06Evidence(**values)


def test_real_qwen06_court_passes_only_matching_real_weight_evidence():
    assert decide_real_qwen06_court(evidence())["status"]=="REAL_QWEN06_WEIGHT_COURT_PASS"
    assert "revision_mismatch" in decide_real_qwen06_court(evidence(resolved_revision="wrong"))["reasons"]
    assert "parameter_count_mismatch" in decide_real_qwen06_court(evidence(parameter_count=1))["reasons"]
    assert "real_forward_nonfinite" in decide_real_qwen06_court(evidence(forward_logits_finite=False))["reasons"]
    assert "sample_logit_relative_error_failed" in decide_real_qwen06_court(evidence(sample_logit_relative_error=.2))["reasons"]
    assert "court_did_not_load_real_qwen" in decide_real_qwen06_court(evidence(runtime_requires_qwen_for_court=False))["reasons"]
