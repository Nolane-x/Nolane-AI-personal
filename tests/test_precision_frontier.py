from nolane_personal.precision_frontier import (
    PackedInt4QualityEvidence,
    PackedInt4ResourceEvidence,
    decide_packed_int4_quality,
    decide_packed_int4_resources,
    decide_precision_frontier,
)


def q(**kw):
    d=dict(
        test_examples=20,anchor_examples=8,rank=96,
        factorized_nll=1.70,int8_nll=1.71,int4_nll=1.73,
        int4_regression_vs_factorized=.03,int4_regression_vs_int8=.02,
        anchor_factorized_nll=1.50,anchor_int8_nll=1.51,anchor_int4_nll=1.54,
        anchor_int4_regression_vs_factorized=.04,anchor_int4_regression_vs_int8=.03,
        greedy_agreement_int4_vs_factorized=.96,greedy_agreement_int4_vs_int8=.98,
        prompt_scan_equivalence_passed=True,cortex_digest_equal=True,rank_equal=True,
        runtime_requires_qwen_model=False,runtime_requires_transformers=False,
    )
    d.update(kw); return PackedInt4QualityEvidence(**d)


def r(**kw):
    d=dict(
        prompts=8,int8_median_ms=60.,int4_median_ms=75.,latency_ratio_vs_int8=1.25,
        int8_checkpoint_bytes=30_000_000,int4_checkpoint_bytes=17_000_000,
        checkpoint_ratio_vs_int8=.567,int8_boundary_storage_bytes=25_000_000,
        int4_boundary_storage_bytes=13_000_000,boundary_storage_ratio_vs_int8=.52,
        int4_tokens_per_second=15.,
    )
    d.update(kw); return PackedInt4ResourceEvidence(**d)


def test_int4_quality_and_resource_courts():
    assert decide_packed_int4_quality(q())["status"]=="PACKED_INT4_QUALITY_PASS"
    assert "int8_noninferiority_failed" in decide_packed_int4_quality(q(int4_regression_vs_int8=.08))["reasons"]
    assert decide_packed_int4_resources(r())["status"]=="PACKED_INT4_RESOURCE_PASS"
    assert "latency_regression_failed" in decide_packed_int4_resources(r(latency_ratio_vs_int8=2.0))["reasons"]


def test_precision_frontier_selects_int4_only_when_both_courts_pass():
    d=decide_precision_frontier(
        int8_quality_status="QUANTIZED_FACTOR_QUALITY_PASS",
        int8_resource_status="QUANTIZED_FACTOR_RESOURCE_PASS",
        int8_checkpoint_sha256="i8",
        int8_source_factorized_checkpoint_sha256="f",
        int4_quality_status="PACKED_INT4_QUALITY_PASS",
        int4_resource_status="PACKED_INT4_RESOURCE_PASS",
        int4_checkpoint_sha256="i4",
        int4_source_factorized_checkpoint_sha256="f",
    )
    assert d.status=="PRECISION_FRONTIER_SELECT_INT4"
    assert d.selected_precision=="int4"
    assert d.selected_checkpoint_sha256=="i4"


def test_precision_frontier_keeps_int8_when_int4_fails():
    d=decide_precision_frontier(
        int8_quality_status="QUANTIZED_FACTOR_QUALITY_PASS",
        int8_resource_status="QUANTIZED_FACTOR_RESOURCE_PASS",
        int8_checkpoint_sha256="i8",
        int8_source_factorized_checkpoint_sha256="f",
        int4_quality_status="PACKED_INT4_QUALITY_BLOCKED",
        int4_resource_status="PACKED_INT4_RESOURCE_PASS",
        int4_checkpoint_sha256="i4",
        int4_source_factorized_checkpoint_sha256="f",
    )
    assert d.status=="PRECISION_FRONTIER_KEEP_INT8"
    assert d.selected_precision=="int8"
    assert "int4_quality_not_passed" in d.reasons


def test_precision_frontier_blocks_without_valid_int8_fallback():
    d=decide_precision_frontier(
        int8_quality_status="QUANTIZED_FACTOR_QUALITY_BLOCKED",
        int8_resource_status="QUANTIZED_FACTOR_RESOURCE_PASS",
        int8_checkpoint_sha256="i8",
        int8_source_factorized_checkpoint_sha256="f",
        int4_quality_status="PACKED_INT4_QUALITY_PASS",
        int4_resource_status="PACKED_INT4_RESOURCE_PASS",
        int4_checkpoint_sha256="i4",
        int4_source_factorized_checkpoint_sha256="f",
    )
    assert d.status=="PRECISION_FRONTIER_BLOCKED"
    assert d.selected_precision is None


def test_precision_frontier_keeps_int8_on_mismatched_int4_evidence_lineage():
    d=decide_precision_frontier(
        int8_quality_status="QUANTIZED_FACTOR_QUALITY_PASS",
        int8_resource_status="QUANTIZED_FACTOR_RESOURCE_PASS",
        int8_checkpoint_sha256="i8",
        int8_source_factorized_checkpoint_sha256="f",
        int4_quality_status="PACKED_INT4_QUALITY_PASS",
        int4_resource_status="PACKED_INT4_RESOURCE_PASS",
        int4_checkpoint_sha256="i4",
        int4_source_factorized_checkpoint_sha256="f",
        int4_quality_reference_int8_checkpoint_sha256="other-i8",
        int4_resource_reference_int8_checkpoint_sha256="i8",
        int4_resource_checkpoint_sha256="i4",
    )
    assert d.status=="PRECISION_FRONTIER_KEEP_INT8"
    assert d.selected_checkpoint_sha256=="i8"
    assert "int4_source_lineage_mismatch" in d.reasons
