import json
import subprocess
import sys
from pathlib import Path

import pytest

from nolane_personal.latent import LatentStore
from nolane_personal.local_evidence_intake import (
    finalize_local_evidence_intake,
    verify_local_evidence_intake,
)
from nolane_personal.review_queue import build_review_queue, verify_review_queue


ROOT=Path(__file__).resolve().parents[1]


def write_export(path: Path, count: int=9):
    conversations=[]
    for i in range(count):
        conversations.append({
            "conversation_id":f"PRIVATE-CONVERSATION-{i}",
            "language":"vi" if i%2==0 else "en",
            "messages":[
                {"role":"user","content":f"L25-PRIVATE-PROMPT-{i}"},
                {"role":"assistant","content":f"L25-PRIVATE-TARGET-{i}"},
            ],
        })
    path.write_text(json.dumps(conversations,ensure_ascii=False),encoding="utf-8")


def prepare_review(tmp_path: Path, *, approved_count: int=7):
    source=tmp_path/"conversations.json"
    write_export(source,9)
    queue=build_review_queue(source,tmp_path/"queue")
    _manifest,candidates=verify_review_queue(queue["manifest_path"])

    decisions=[]
    for index,candidate in enumerate(candidates):
        if index<approved_count:
            decisions.append({
                "candidate_id":candidate["candidate_id"],
                "approved":True,
                "sensitive":False,
                "language":candidate["language"],
            })
        elif index==approved_count:
            decisions.append({
                "candidate_id":candidate["candidate_id"],
                "approved":True,
                "sensitive":True,
                "language":candidate["language"],
            })
        else:
            decisions.append({
                "candidate_id":candidate["candidate_id"],
                "approved":False,
                "sensitive":False,
                "language":candidate["language"],
            })
    decisions_path=tmp_path/"decisions.jsonl"
    decisions_path.write_text(
        "".join(json.dumps(row)+"\n" for row in decisions),
        encoding="utf-8",
    )
    return queue,decisions_path


def test_local_intake_builds_hash_bound_private_lineage(tmp_path):
    queue,decisions=prepare_review(tmp_path)
    result=finalize_local_evidence_intake(
        queue["manifest_path"],
        decisions,
        tmp_path/"intake",
    )
    verified=verify_local_evidence_intake(
        result.manifest_path,
        queue_manifest_path=queue["manifest_path"],
        decisions_path=decisions,
    )

    assert verified["authority"]=="EXPLICIT_REVIEWED_LOCAL_EVIDENCE_UNPROMOTED"
    assert verified["stats"]["queue_candidates"]==9
    assert verified["stats"]["approved"]==7
    assert verified["stats"]["sensitive"]==1
    assert verified["stats"]["rejected"]==2
    assert verified["stats"]["undecided"]==0
    assert verified["stats"]["output_examples"]==7
    assert verified["stats"]["train_examples"]==4
    assert verified["stats"]["dev_examples"]==1
    assert verified["stats"]["test_examples"]==2

    rendered=result.manifest_path.read_text(encoding="utf-8")
    for forbidden in (
        "L25-PRIVATE-PROMPT",
        "L25-PRIVATE-TARGET",
        "PRIVATE-CONVERSATION",
    ):
        assert forbidden not in rendered

    approved_manifest=json.loads(
        result.approved_manifest_path.read_text(encoding="utf-8")
    )
    assert verified["approved_manifest_sha256"]==approved_manifest["manifest_sha256"]
    assert verified["dataset_sha256"]==approved_manifest["dataset_sha256"]
    assert verified["protocol_sha256"]==approved_manifest["protocol_sha256"]


def test_local_intake_fails_if_decisions_change_after_freeze(tmp_path):
    queue,decisions=prepare_review(tmp_path)
    result=finalize_local_evidence_intake(
        queue["manifest_path"],decisions,tmp_path/"intake"
    )
    decisions.write_text(
        decisions.read_text(encoding="utf-8")
        + json.dumps({
            "candidate_id":"tamper",
            "approved":False,
            "sensitive":False,
        })+"\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError,match="decisions digest mismatch"):
        verify_local_evidence_intake(
            result.manifest_path,
            queue_manifest_path=queue["manifest_path"],
            decisions_path=decisions,
        )


def test_local_intake_fails_if_reviewed_source_is_tampered(tmp_path):
    queue,decisions=prepare_review(tmp_path)
    result=finalize_local_evidence_intake(
        queue["manifest_path"],decisions,tmp_path/"intake"
    )
    reviewed_source=result.reviewed_manifest_path.parent/"reviewed-evidence-source.jsonl"
    reviewed_source.write_text(
        reviewed_source.read_text(encoding="utf-8")+"{}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError,match="reviewed evidence source digest mismatch"):
        verify_local_evidence_intake(
            result.manifest_path,
            queue_manifest_path=queue["manifest_path"],
            decisions_path=decisions,
        )


def test_local_intake_requires_l21_ready_approved_minimum(tmp_path):
    queue,decisions=prepare_review(tmp_path,approved_count=6)
    with pytest.raises(ValueError,match="need at least 7"):
        finalize_local_evidence_intake(
            queue["manifest_path"],decisions,tmp_path/"intake"
        )


def test_local_intake_never_overwrites_existing_workspace(tmp_path):
    queue,decisions=prepare_review(tmp_path)
    output=tmp_path/"intake"
    output.mkdir()
    (output/"keep.txt").write_text("keep",encoding="utf-8")
    with pytest.raises(FileExistsError,match="refusing to overwrite"):
        finalize_local_evidence_intake(
            queue["manifest_path"],decisions,output
        )


def test_local_intake_approved_pack_drives_l21_check_only(tmp_path):
    queue,decisions=prepare_review(tmp_path)
    result=finalize_local_evidence_intake(
        queue["manifest_path"],decisions,tmp_path/"intake"
    )

    anchor=tmp_path/"anchor.jsonl"
    anchor.write_text("".join(
        json.dumps({
            "prompt":f"anchor-{i}",
            "target":f"answer-{i}",
            "language":"vi" if i%2==0 else "en",
        })+"\n"
        for i in range(4)
    ),encoding="utf-8")

    latent=tmp_path/"latent.json"
    LatentStore(latent).initialize(
        identity_id="l25-integration",
        checkpoint_sha256="core",
        latent_dim=32,
        protocol_sha256=None,
    )
    revision="pinned-l25"
    model=tmp_path/"model"
    model.mkdir()
    (model/"config.json").write_text("{}",encoding="utf-8")
    (model/"model.safetensors").write_bytes(b"x")
    (model/".nolane-model-revision").write_text(revision,encoding="utf-8")
    lock=tmp_path/"model.lock.json"
    lock.write_text(json.dumps({"upstream":{"revision":revision}}),encoding="utf-8")
    l14=tmp_path/"l14.pt"
    l14.write_bytes(b"x")
    workspace=tmp_path/"l21"

    completed=subprocess.run(
        [
            sys.executable,
            str(ROOT/"scripts/run_real_candidate_pipeline.py"),
            "--evidence-pack",str(result.approved_manifest_path),
            "--anchor",str(anchor),
            "--latent",str(latent),
            "--model-lock",str(lock),
            "--model",str(model),
            "--l14-anchor",str(l14),
            "--workspace",str(workspace),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode==0,completed.stderr
    receipt=json.loads(
        (workspace/"pipeline-receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["status"]=="REAL_CANDIDATE_READY_NOT_EXECUTED"
    assert receipt["approved_evidence_manifest_sha256"]==result.manifest["approved_manifest_sha256"]
    rendered=json.dumps(receipt,sort_keys=True)
    assert "L25-PRIVATE-PROMPT" not in rendered
    assert "L25-PRIVATE-TARGET" not in rendered
