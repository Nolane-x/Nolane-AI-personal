from __future__ import annotations

import json
from pathlib import Path

import pytest

from nolane_personal.cortex import CortexReply
from nolane_personal.learning_campaign import (
    CAMPAIGN_SPEC_SCHEMA,
    build_real_learning_campaign,
    verify_real_learning_campaign,
)
from nolane_personal.local_evidence_workbench import paths_for
from nolane_personal.personal_dataset import load_jsonl
from nolane_personal.personal_protocol import load_protocol
from nolane_personal.product_runtime import ProductRuntime


class CampaignCortex:
    checkpoint_sha256 = "a" * 64

    def generate(self, request):
        return CortexReply(
            "Runtime answer for " + str(request.user_text or request.intent),
            intent=request.intent,
        )

    def close(self):
        return None


def factory(_identity_id, _profile_getter):
    return CampaignCortex()


def add_window_turns(runtime: ProductRuntime, seed: str, *, count: int = 12):
    if runtime.status()["phase"] != "on":
        runtime.power_on()
    for i in range(count):
        runtime.send_message(
            f"{seed} request {i}: explain the unique concept {seed}-{i} "
            f"using one concrete example and avoid unrelated material."
        )


def approve_and_finalize(runtime: ProductRuntime, window_id: str):
    while True:
        candidate = runtime.next_learning_candidate(window_id)
        if candidate is None:
            break
        runtime.record_learning_decision(
            window_id,
            candidate["candidate_id"],
            decision="approve",
            language="en",
            corrected_target=(
                f"Approved {window_id} target for {candidate['candidate_id']}: "
                "give one specific, bounded answer."
            ),
        )
    result = runtime.finalize_learning_window(window_id)
    assert result["quality_status"] == "PASS"
    return result


def make_registry_windows(tmp_path: Path):
    runtime = ProductRuntime(tmp_path, cortex_factory=factory)
    roots = []
    try:
        for i in range(7):
            add_window_turns(runtime, f"topic-{i}")
            row = runtime.create_learning_window()
            approve_and_finalize(runtime, row["window_id"])
            roots.append(
                tmp_path
                / "learning-evidence"
                / "windows"
                / row["window_id"]
            )
    finally:
        runtime.close()
    return roots


def make_model_inputs(tmp_path: Path):
    checkpoint = tmp_path / "factorized-nolane.pt"
    checkpoint.write_bytes(b"checkpoint")
    latent = tmp_path / "latent.json"
    latent.write_text('{"values":[0.0]}', encoding="utf-8")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tokenizer / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return checkpoint, latent, tokenizer


def test_campaign_builds_five_cycles_without_rehearsing_prior_test_splits(tmp_path):
    windows = make_registry_windows(tmp_path)
    checkpoint, latent, tokenizer = make_model_inputs(tmp_path)

    spec = {
        "schema": CAMPAIGN_SPEC_SCHEMA,
        "initial_factorized": str(checkpoint),
        "latent": str(latent),
        "tokenizer": str(tokenizer),
        "device": "cpu",
        "fixed_window": str(windows[0]),
        "baseline_retention_window": str(windows[1]),
        "adaptation_windows": [str(path) for path in windows[2:]],
        "isolation_policy": {
            "near_duplicate_token_jaccard": 1.0,
            "near_duplicate_sequence_ratio": 1.0,
            "min_tokens_for_near_duplicate": 4,
        },
    }
    spec_path = tmp_path / "campaign-spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    output = tmp_path / "campaign"

    receipt = build_real_learning_campaign(spec_path, output)
    assert receipt["status"] == "READY_FOR_EXPLICIT_L43_EXECUTION"
    assert len(receipt["adaptations"]) == 5
    assert len(receipt["retention_cycles"]) == 5
    verify_real_learning_campaign(output)

    adaptation1 = windows[2] / "workbench"
    wb = paths_for(adaptation1)
    approved = json.loads(
        wb.approved_manifest.read_text(encoding="utf-8")
    )
    dataset_path = (
        wb.approved_manifest.parent / approved["dataset_filename"]
    )
    protocol_path = (
        wb.approved_manifest.parent / approved["protocol_filename"]
    )
    examples = load_jsonl(dataset_path)
    protocol = load_protocol(protocol_path)

    test_indices = [
        int(row["index"])
        for row in protocol["splits"]["test"]
    ]
    train_dev_indices = [
        int(row["index"])
        for split in ("train", "dev")
        for row in protocol["splits"][split]
    ]
    assert test_indices
    assert train_dev_indices

    cycle2_manifest = json.loads(
        (
            output
            / "retention"
            / "cycle-002"
            / "approved-evidence-manifest.json"
        ).read_text(encoding="utf-8")
    )
    cycle2_dataset = (
        output
        / "retention"
        / "cycle-002"
        / cycle2_manifest["dataset_filename"]
    )
    cycle2_rows = [
        json.loads(line)
        for line in cycle2_dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cycle2_pairs = {
        (row["prompt"], row["target"])
        for row in cycle2_rows
    }

    for index in test_indices:
        example = examples[index]
        assert (example.prompt, example.target) not in cycle2_pairs
    for index in train_dev_indices:
        example = examples[index]
        assert (example.prompt, example.target) in cycle2_pairs

    retention_row = receipt["retention_cycles"][1]
    assert retention_row["kind"] == "COMPOSED_PRIOR_TRAIN_DEV_ONLY"
    assert retention_row["original_test_examples_excluded"] > 0


def test_campaign_rejects_non_chronological_window_roles(tmp_path):
    windows = make_registry_windows(tmp_path)
    checkpoint, latent, tokenizer = make_model_inputs(tmp_path)
    spec = {
        "schema": CAMPAIGN_SPEC_SCHEMA,
        "initial_factorized": str(checkpoint),
        "latent": str(latent),
        "tokenizer": str(tokenizer),
        "fixed_window": str(windows[1]),
        "baseline_retention_window": str(windows[0]),
        "adaptation_windows": [str(path) for path in windows[2:]],
    }
    spec_path = tmp_path / "bad-order.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    with pytest.raises(ValueError, match="must be chronological"):
        build_real_learning_campaign(
            spec_path,
            tmp_path / "bad-order-campaign",
        )


def test_campaign_rejects_window_role_reuse(tmp_path):
    windows = make_registry_windows(tmp_path)
    checkpoint, latent, tokenizer = make_model_inputs(tmp_path)
    spec = {
        "schema": CAMPAIGN_SPEC_SCHEMA,
        "initial_factorized": str(checkpoint),
        "latent": str(latent),
        "tokenizer": str(tokenizer),
        "fixed_window": str(windows[0]),
        "baseline_retention_window": str(windows[1]),
        "adaptation_windows": [
            str(windows[2]),
            str(windows[2]),
            str(windows[4]),
            str(windows[5]),
            str(windows[6]),
        ],
    }
    spec_path = tmp_path / "reused.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    with pytest.raises(ValueError, match="window role reused"):
        build_real_learning_campaign(
            spec_path,
            tmp_path / "reused-campaign",
        )
