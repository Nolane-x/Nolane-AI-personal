import hashlib
import json
from pathlib import Path

from nolane_personal.latent import LatentStore
from nolane_personal.personal_dataset import load_jsonl
from nolane_personal.personal_protocol import (
    PersonalizationSplitPolicy,
    build_personalization_protocol,
)
from nolane_personal.real_candidate_readiness import assess_real_candidate_readiness


def write_dataset(path: Path, count: int):
    rows=[]
    for i in range(count):
        rows.append({
            "prompt":f"prompt {i}",
            "target":f"target {i}",
            "language":"vi" if i%2==0 else "en",
        })
    path.write_text(
        "".join(json.dumps(row,ensure_ascii=False)+"\n" for row in rows),
        encoding="utf-8",
    )


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_protocol(dataset: Path, protocol: Path):
    examples=load_jsonl(dataset)
    groups=[
        hashlib.sha256(f"readiness-source-group-{i}".encode("utf-8")).hexdigest()
        for i in range(len(examples))
    ]
    payload=build_personalization_protocol(
        examples,
        dataset_sha256=sha256(dataset),
        policy=PersonalizationSplitPolicy(),
        source_group_sha256=groups,
    )
    protocol.write_text(json.dumps(payload,ensure_ascii=False),encoding="utf-8")
    return payload


def fixture(tmp_path: Path, count: int = 7):
    dataset=tmp_path/"personalization.jsonl"
    protocol=tmp_path/"protocol.json"
    anchor=tmp_path/"anchor.jsonl"
    latent=tmp_path/"latent.json"
    lock=tmp_path/"model.lock.json"
    model=tmp_path/"model"
    l14=tmp_path/"l14.pt"

    write_dataset(dataset,count)
    frozen=write_protocol(dataset,protocol)
    write_dataset(anchor,4)

    store=LatentStore(latent)
    store.initialize(
        identity_id="test-ai",
        checkpoint_sha256="living-core",
        latent_dim=32,
        protocol_sha256=None,
    )

    revision="abc123"
    lock.write_text(json.dumps({
        "upstream":{
            "repo_id":"Qwen/Qwen3-0.6B",
            "revision":revision,
        }
    }),encoding="utf-8")
    model.mkdir()
    (model/"config.json").write_text("{}",encoding="utf-8")
    (model/"model.safetensors").write_bytes(b"weights")
    (model/".nolane-model-revision").write_text(revision,encoding="utf-8")
    l14.write_bytes(b"candidate")

    return {
        "dataset":dataset,
        "protocol":protocol,
        "anchor":anchor,
        "latent":latent,
        "model_lock":lock,
        "model_dir":model,
        "l14_anchor":l14,
        "protocol_payload":frozen,
    }


def assess(paths):
    return assess_real_candidate_readiness(
        dataset=paths["dataset"],
        protocol=paths["protocol"],
        anchor=paths["anchor"],
        latent=paths["latent"],
        model_lock=paths["model_lock"],
        model_dir=paths["model_dir"],
        l14_anchor=paths["l14_anchor"],
    )


def test_real_candidate_readiness_passes_only_with_auditable_inputs(tmp_path):
    paths=fixture(tmp_path,7)
    decision=assess(paths)
    assert decision.status=="REAL_CANDIDATE_INPUTS_READY"
    assert decision.reasons==[]
    assert decision.evidence["dataset_examples"]==7
    assert decision.evidence["train_examples"]>=1
    assert decision.evidence["dev_examples"]>=1
    assert decision.evidence["test_examples"]>=2
    assert decision.evidence["anchor_examples"]==4
    assert decision.evidence["model_revision_matches"] is True
    assert decision.evidence["latent_valid"] is True
    assert decision.evidence["dataset_sha256"]==sha256(paths["dataset"])
    assert decision.evidence["protocol_sha256"]==paths["protocol_payload"]["protocol_sha256"]
    assert decision.evidence["evidence_quality_status"]=="PASS"
    assert decision.evidence["evidence_quality_court_sha256"]


def test_readiness_blocks_protocol_with_only_one_test_example(tmp_path):
    paths=fixture(tmp_path,6)
    decision=assess(paths)
    assert decision.status=="REAL_CANDIDATE_INPUTS_BLOCKED"
    assert "insufficient_test_examples" in decision.reasons


def test_readiness_blocks_revision_drift(tmp_path):
    paths=fixture(tmp_path,7)
    (paths["model_dir"]/".nolane-model-revision").write_text("wrong",encoding="utf-8")
    decision=assess(paths)
    assert "pinned_model_revision_mismatch" in decision.reasons


def test_readiness_blocks_missing_l14_and_latent(tmp_path):
    paths=fixture(tmp_path,7)
    paths["l14_anchor"].unlink()
    paths["latent"].unlink()
    decision=assess(paths)
    assert "l14_anchor_candidate_missing" in decision.reasons
    assert "persistent_latent_missing" in decision.reasons


def test_readiness_blocks_dataset_protocol_drift(tmp_path):
    paths=fixture(tmp_path,7)
    with paths["dataset"].open("a",encoding="utf-8") as fh:
        fh.write(json.dumps({"prompt":"new","target":"new"})+"\n")
    decision=assess(paths)
    assert decision.status=="REAL_CANDIDATE_INPUTS_BLOCKED"
    assert any(reason.startswith("personalization_protocol_invalid:") for reason in decision.reasons)


def test_readiness_blocks_protocol_without_source_group_lineage(tmp_path):
    paths=fixture(tmp_path,7)
    examples=load_jsonl(paths["dataset"])
    protocol=build_personalization_protocol(
        examples,
        dataset_sha256=sha256(paths["dataset"]),
        policy=PersonalizationSplitPolicy(),
    )
    paths["protocol"].write_text(
        json.dumps(protocol,ensure_ascii=False),
        encoding="utf-8",
    )
    decision=assess(paths)
    assert decision.status=="REAL_CANDIDATE_INPUTS_BLOCKED"
    assert "evidence_quality:incomplete_source_group_lineage" in decision.reasons
    assert "evidence_quality:non_grouped_split_strategy" in decision.reasons
