from nolane_personal.quantized_evaluation import QuantizedQualityEvidence, QuantizedResourceEvidence, decide_quantized_quality, decide_quantized_resources
from nolane_personal.quantized_promotion import decide_quantized_promotion


def q(**kw):
    d=dict(
        test_examples=20,anchor_examples=8,rank=96,
        source_factorized_nll=1.70,quantized_nll=1.71,nll_regression_vs_factorized=.01,
        anchor_source_nll=1.50,anchor_quantized_nll=1.52,anchor_nll_regression=.02,
        greedy_token_agreement=.99,prompt_scan_equivalence_passed=True,
        cortex_digest_equal_to_source=True,source_rank_equal=True,
        runtime_requires_qwen_model=False,runtime_requires_transformers=False,
    )
    d.update(kw); return QuantizedQualityEvidence(**d)


def r(**kw):
    d=dict(
        prompts=8,source_median_ms=50.,quantized_median_ms=65.,latency_ratio_vs_source=1.3,
        source_checkpoint_bytes=100_000_000,quantized_checkpoint_bytes=28_000_000,checkpoint_ratio=.28,
        source_boundary_storage_bytes=90_000_000,quantized_boundary_storage_bytes=25_000_000,
        boundary_storage_ratio=.278,tokens_per_second=20.,
    )
    d.update(kw); return QuantizedResourceEvidence(**d)


def test_quantized_quality_court():
    assert decide_quantized_quality(q())["status"]=="QUANTIZED_FACTOR_QUALITY_PASS"
    assert "factorized_noninferiority_failed" in decide_quantized_quality(q(nll_regression_vs_factorized=.08))["reasons"]
    assert "cortex_digest_drift" in decide_quantized_quality(q(cortex_digest_equal_to_source=False))["reasons"]
    assert "rank_mismatch" in decide_quantized_quality(q(source_rank_equal=False))["reasons"]


def test_quantized_resource_court():
    assert decide_quantized_resources(r())["status"]=="QUANTIZED_FACTOR_RESOURCE_PASS"
    assert "checkpoint_compression_failed" in decide_quantized_resources(r(checkpoint_ratio=.8))["reasons"]
    assert "latency_regression_failed" in decide_quantized_resources(r(latency_ratio_vs_source=2.0))["reasons"]


def test_quantized_promotion_binds_all_lineage():
    p=decide_quantized_promotion(
        quality_status="QUANTIZED_FACTOR_QUALITY_PASS",
        quality_checkpoint_sha256="q",
        quality_source_factorized_checkpoint_sha256="f",
        quality_source_l16_checkpoint_sha256="l16",
        resource_status="QUANTIZED_FACTOR_RESOURCE_PASS",
        resource_checkpoint_sha256="q",
        resource_source_factorized_checkpoint_sha256="f",
        resource_source_l16_checkpoint_sha256="l16",
    )
    assert p.status=="QUANTIZED_FACTOR_PROMOTION_PASS"
    b=decide_quantized_promotion(
        quality_status="QUANTIZED_FACTOR_QUALITY_PASS",
        quality_checkpoint_sha256="q1",
        quality_source_factorized_checkpoint_sha256="f1",
        quality_source_l16_checkpoint_sha256="l1",
        resource_status="QUANTIZED_FACTOR_RESOURCE_PASS",
        resource_checkpoint_sha256="q2",
        resource_source_factorized_checkpoint_sha256="f2",
        resource_source_l16_checkpoint_sha256="l2",
    )
    assert "quantized_checkpoint_identity_mismatch" in b.reasons
    assert "source_factorized_checkpoint_mismatch" in b.reasons
    assert "source_l16_checkpoint_mismatch" in b.reasons
