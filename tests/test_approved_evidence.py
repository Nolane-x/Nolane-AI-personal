import json
from pathlib import Path

import pytest

from nolane_personal.approved_evidence import (
    ApprovedEvidencePolicy,
    build_approved_evidence_pack,
    verify_approved_evidence_pack,
)
from nolane_personal.latent import LatentStore
from nolane_personal.real_candidate_readiness import assess_real_candidate_readiness


def write_source(path: Path, rows):
    with path.open("w",encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row,ensure_ascii=False)+"\n")


def approved_rows(count=7):
    rows=[]
    for i in range(count):
        rows.append({
            "prompt":f"PRIVATE-PROMPT-{i}",
            "target":f"PRIVATE-TARGET-{i}",
            "language":"vi" if i%2==0 else "en",
            "approved":True,
            "source_id":f"raw-conversation-id-{i}",
        })
    return rows


def test_builder_filters_unapproved_sensitive_and_keeps_manifest_private(tmp_path):
    source=tmp_path/"source.jsonl"
    rows=approved_rows(7)+[
        {
            "prompt":"UNAPPROVED-PRIVATE-PROMPT",
            "target":"UNAPPROVED-PRIVATE-TARGET",
            "language":"vi",
            "approved":False,
            "source_id":"raw-unapproved-id",
        },
        {
            "prompt":"SENSITIVE-PRIVATE-PROMPT",
            "target":"SENSITIVE-PRIVATE-TARGET",
            "language":"en",
            "approved":True,
            "sensitive":True,
            "source_id":"raw-sensitive-id",
        },
    ]
    write_source(source,rows)
    result=build_approved_evidence_pack(source,tmp_path/"pack")
    manifest=verify_approved_evidence_pack(result.manifest_path)

    assert manifest["authority"]=="USER_APPROVED_LOCAL_EVIDENCE_UNPROMOTED"
    assert manifest["stats"]["source_rows"]==9
    assert manifest["stats"]["output_examples"]==7
    assert manifest["stats"]["excluded_unapproved"]==1
    assert manifest["stats"]["excluded_sensitive"]==1
    assert manifest["stats"]["train_examples"]==4
    assert manifest["stats"]["dev_examples"]==1
    assert manifest["stats"]["test_examples"]==2
    assert manifest["stats"]["language_counts"]=={"en":3,"vi":4}

    rendered=result.manifest_path.read_text(encoding="utf-8")
    for forbidden in (
        "PRIVATE-PROMPT",
        "PRIVATE-TARGET",
        "raw-conversation-id",
        "raw-unapproved-id",
        "raw-sensitive-id",
    ):
        assert forbidden not in rendered

    dataset=result.dataset_path.read_text(encoding="utf-8")
    assert "UNAPPROVED-PRIVATE-PROMPT" not in dataset
    assert "SENSITIVE-PRIVATE-PROMPT" not in dataset
    assert "PRIVATE-PROMPT-0" in dataset


def test_builder_output_is_l21_readiness_compatible(tmp_path):
    source=tmp_path/"source.jsonl"
    write_source(source,approved_rows(8))
    result=build_approved_evidence_pack(source,tmp_path/"pack")

    anchor=tmp_path/"anchor.jsonl"
    write_source(anchor,[
        {"prompt":f"anchor-{i}","target":f"answer-{i}","language":"vi","approved":True}
        for i in range(4)
    ])
    # Readiness anchor loader only needs personalization fields; remove approval-only metadata.
    anchor.write_text("".join(
        json.dumps({"prompt":f"anchor-{i}","target":f"answer-{i}","language":"vi"})+"\n"
        for i in range(4)
    ),encoding="utf-8")

    latent=tmp_path/"latent.json"
    LatentStore(latent).initialize(
        identity_id="approved-pack-test",
        checkpoint_sha256="core",
        latent_dim=32,
        protocol_sha256=None,
    )

    revision="pinned-revision"
    model=tmp_path/"model"
    model.mkdir()
    (model/"config.json").write_text("{}",encoding="utf-8")
    (model/"model.safetensors").write_bytes(b"x")
    (model/".nolane-model-revision").write_text(revision,encoding="utf-8")
    lock=tmp_path/"model.lock.json"
    lock.write_text(json.dumps({"upstream":{"revision":revision}}),encoding="utf-8")
    l14=tmp_path/"l14.pt"
    l14.write_bytes(b"candidate")

    decision=assess_real_candidate_readiness(
        dataset=result.dataset_path,
        protocol=result.protocol_path,
        anchor=anchor,
        latent=latent,
        model_lock=lock,
        model_dir=model,
        l14_anchor=l14,
    )
    assert decision.status=="REAL_CANDIDATE_INPUTS_READY"
    assert decision.evidence["train_examples"]>=1
    assert decision.evidence["dev_examples"]>=1
    assert decision.evidence["test_examples"]>=2


def test_duplicate_approved_pair_fails_closed(tmp_path):
    source=tmp_path/"source.jsonl"
    rows=approved_rows(7)
    rows.append(dict(rows[0]))
    rows[-1]["source_id"]="different-source-id"
    write_source(source,rows)
    with pytest.raises(ValueError,match="duplicate approved prompt/target pair"):
        build_approved_evidence_pack(source,tmp_path/"pack")


def test_disallowed_language_fails_closed(tmp_path):
    source=tmp_path/"source.jsonl"
    rows=approved_rows(7)
    rows[2]["language"]="fr"
    write_source(source,rows)
    with pytest.raises(ValueError,match="not allowed"):
        build_approved_evidence_pack(source,tmp_path/"pack")


def test_insufficient_approved_rows_fails_closed(tmp_path):
    source=tmp_path/"source.jsonl"
    write_source(source,approved_rows(6))
    with pytest.raises(ValueError,match="need at least 7"):
        build_approved_evidence_pack(source,tmp_path/"pack")


def test_manifest_verification_detects_dataset_tamper(tmp_path):
    source=tmp_path/"source.jsonl"
    write_source(source,approved_rows(7))
    result=build_approved_evidence_pack(source,tmp_path/"pack")
    result.dataset_path.write_text(
        result.dataset_path.read_text(encoding="utf-8")
        + json.dumps({"prompt":"tamper","target":"tamper","language":"vi"})+"\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError,match="dataset digest mismatch"):
        verify_approved_evidence_pack(result.manifest_path)


def test_non_empty_output_directory_is_never_overwritten(tmp_path):
    source=tmp_path/"source.jsonl"
    write_source(source,approved_rows(7))
    output=tmp_path/"pack"
    output.mkdir()
    (output/"existing.txt").write_text("do not overwrite",encoding="utf-8")
    with pytest.raises(FileExistsError,match="refusing to overwrite"):
        build_approved_evidence_pack(source,output)


def test_policy_requires_l21_ready_minimum():
    with pytest.raises(ValueError,match="min_approved_examples must be >=7"):
        ApprovedEvidencePolicy(min_approved_examples=6).validate()


def test_evidence_pack_drives_l21_check_only_without_private_receipt_leak(tmp_path):
    import subprocess
    import sys

    root=Path(__file__).resolve().parents[1]
    source=tmp_path/"source.jsonl"
    write_source(source,approved_rows(8))
    result=build_approved_evidence_pack(source,tmp_path/"pack")

    anchor=tmp_path/"anchor.jsonl"
    anchor.write_text("".join(
        json.dumps({
            "prompt":f"anchor-{i}",
            "target":f"answer-{i}",
            "language":"vi" if i%2==0 else "en",
        },ensure_ascii=False)+"\n"
        for i in range(4)
    ),encoding="utf-8")

    latent=tmp_path/"latent.json"
    LatentStore(latent).initialize(
        identity_id="approved-pack-pipeline-test",
        checkpoint_sha256="core",
        latent_dim=32,
        protocol_sha256=None,
    )

    revision="pinned-revision"
    model=tmp_path/"model"
    model.mkdir()
    (model/"config.json").write_text("{}",encoding="utf-8")
    (model/"model.safetensors").write_bytes(b"x")
    (model/".nolane-model-revision").write_text(revision,encoding="utf-8")
    lock=tmp_path/"model.lock.json"
    lock.write_text(json.dumps({"upstream":{"revision":revision}}),encoding="utf-8")
    l14=tmp_path/"l14.pt"
    l14.write_bytes(b"candidate")
    workspace=tmp_path/"workspace"

    completed=subprocess.run(
        [
            sys.executable,
            str(root/"scripts/run_real_candidate_pipeline.py"),
            "--evidence-pack",str(result.manifest_path),
            "--anchor",str(anchor),
            "--latent",str(latent),
            "--model-lock",str(lock),
            "--model",str(model),
            "--l14-anchor",str(l14),
            "--workspace",str(workspace),
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert completed.returncode==0,completed.stderr
    receipt_path=workspace/"pipeline-receipt.json"
    receipt=json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"]=="REAL_CANDIDATE_READY_NOT_EXECUTED"
    assert receipt["approved_evidence_manifest_sha256"]==result.manifest["manifest_sha256"]
    assert receipt["approved_evidence_authority"]=="USER_APPROVED_LOCAL_EVIDENCE_UNPROMOTED"
    assert receipt["dataset_sha256"]==result.manifest["dataset_sha256"]
    assert receipt["protocol_sha256"]==result.manifest["protocol_sha256"]
    rendered=receipt_path.read_text(encoding="utf-8")
    assert "PRIVATE-PROMPT" not in rendered
    assert "PRIVATE-TARGET" not in rendered
    assert "raw-conversation-id" not in rendered


def test_readiness_cli_accepts_verified_evidence_pack(tmp_path):
    import subprocess
    import sys

    root=Path(__file__).resolve().parents[1]
    source=tmp_path/"source.jsonl"
    write_source(source,approved_rows(7))
    result=build_approved_evidence_pack(source,tmp_path/"pack")

    anchor=tmp_path/"anchor.jsonl"
    anchor.write_text("".join(
        json.dumps({"prompt":f"a{i}","target":f"b{i}","language":"vi"})+"\n"
        for i in range(4)
    ),encoding="utf-8")
    latent=tmp_path/"latent.json"
    LatentStore(latent).initialize(
        identity_id="readiness-pack-test",
        checkpoint_sha256="core",
        latent_dim=32,
        protocol_sha256=None,
    )
    revision="pinned"
    model=tmp_path/"model"
    model.mkdir()
    (model/"config.json").write_text("{}",encoding="utf-8")
    (model/"model.safetensors").write_bytes(b"x")
    (model/".nolane-model-revision").write_text(revision,encoding="utf-8")
    lock=tmp_path/"lock.json"
    lock.write_text(json.dumps({"upstream":{"revision":revision}}),encoding="utf-8")
    l14=tmp_path/"l14.pt"
    l14.write_bytes(b"x")

    completed=subprocess.run(
        [
            sys.executable,
            str(root/"scripts/assess_real_candidate_readiness.py"),
            "--evidence-pack",str(result.manifest_path),
            "--anchor",str(anchor),
            "--latent",str(latent),
            "--model-lock",str(lock),
            "--model",str(model),
            "--l14-anchor",str(l14),
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert completed.returncode==0,completed.stderr
    payload=json.loads(completed.stdout)
    assert payload["decision"]["status"]=="REAL_CANDIDATE_INPUTS_READY"
    assert payload["approved_evidence_manifest_sha256"]==result.manifest["manifest_sha256"]
