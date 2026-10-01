from nolane_personal.replacement_resources import (
    ReplacementResourceEvidence,
    decide_replacement_resources,
)


def _evidence(**overrides):
    values = dict(
        prompts=8,
        total_decoder_layers=28,
        skipped_layers=2,
        baseline_median_ms=100.0,
        replacement_median_ms=88.0,
        latency_ratio_vs_base=0.88,
        artifact_bytes=500_000,
        replacement_parameters=84_289,
    )
    values.update(overrides)
    return ReplacementResourceEvidence(**values)


def test_resource_court_requires_actual_speed_gain_and_real_skipping():
    assert decide_replacement_resources(_evidence())["status"] == "BLOCK_REPLACEMENT_RESOURCE_PASS"

    bad = decide_replacement_resources(_evidence(latency_ratio_vs_base=1.02))
    assert "latency_gain_gate_failed" in bad["reasons"]

    bad = decide_replacement_resources(_evidence(skipped_layers=0))
    assert "no_decoder_blocks_skipped" in bad["reasons"]

    bad = decide_replacement_resources(_evidence(skipped_layers=28))
    assert "cannot_replace_all_decoder_layers" in bad["reasons"]
