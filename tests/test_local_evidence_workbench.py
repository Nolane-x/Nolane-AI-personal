import json
from pathlib import Path

import pytest

from nolane_personal.interactive_review import LocalReviewSession
from nolane_personal.latent import LatentStore
from nolane_personal.local_evidence_workbench import (
    assess_workbench_readiness,
    finalize_workbench,
    initialize_workbench,
    paths_for,
    verify_workbench_manifest,
    workbench_status,
)


def write_export(path: Path, count: int=9):
    rows=[]
    for i in range(count):
        rows.append({
            "conversation_id":f"L27-PRIVATE-CONVERSATION-{i}",
            "language":"vi" if i%2==0 else "en",
            "messages":[
                {"role":"user","content":f"L27-PRIVATE-PROMPT-{i}"},
                {"role":"assistant","content":f"L27-PRIVATE-TARGET-{i}"},
            ],
        })
    path.write_text(json.dumps(rows,ensure_ascii=False),encoding="utf-8")


def init(tmp_path: Path):
    source=tmp_path/"export.json"
    write_export(source)
    root=tmp_path/"workbench"
    status=initialize_workbench(source,root)
    return root,status


def review(root: Path, decisions):
    paths=paths_for(root)
    session=LocalReviewSession(
        paths.queue_manifest,
        paths.decisions,
        paths.review_progress,
    )
    pending=session.pending_candidates()
    for index,action in enumerate(decisions):
        candidate=pending[index]
        approved,sensitive=action
        session.record_decision(
            candidate["candidate_id"],
            approved=approved,
            sensitive=sensitive,
        )
    return session


def test_workbench_initializes_private_queue_ready_state(tmp_path):
    root,status=init(tmp_path)
    assert status["phase"]=="QUEUE_READY"
    assert status["review_progress"]["total"]==9
    assert status["review_progress"]["decided"]==0
    assert status["review_progress"]["approved_non_sensitive"]==0
    assert status["intake_ready"] is False

    paths=paths_for(root)
    rendered=paths.workbench_manifest.read_text(encoding="utf-8")
    for forbidden in (
        "L27-PRIVATE-PROMPT",
        "L27-PRIVATE-TARGET",
        "L27-PRIVATE-CONVERSATION",
    ):
        assert forbidden not in rendered
    verified=verify_workbench_manifest(root)
    assert verified["manifest_sha256"]==status["manifest_sha256"]


def test_workbench_phase_transitions_and_finalize(tmp_path):
    root,_=init(tmp_path)
    session=review(root,[(True,False),(False,False)])
    status=workbench_status(root)
    assert status["phase"]=="REVIEW_IN_PROGRESS"
    assert status["review_progress"]["decided"]==2
    assert status["review_progress"]["remaining"]==7

    remaining=session.pending_candidates()
    # Need seven eligible total: first already approved, approve next six,
    # then reject the final candidate.
    for candidate in remaining[:6]:
        session.record_decision(
            candidate["candidate_id"],
            approved=True,
            sensitive=False,
        )
    for candidate in session.pending_candidates():
        session.record_decision(
            candidate["candidate_id"],
            approved=False,
            sensitive=False,
        )

    status=workbench_status(root)
    assert status["phase"]=="REVIEW_COMPLETE"
    assert status["review_progress"]["approved_non_sensitive"]==7
    assert status["review_progress"]["remaining"]==0

    finalized=finalize_workbench(root)
    assert finalized["phase"]=="INTAKE_READY"
    assert finalized["intake_ready"] is True
    assert finalized["approved_manifest_sha256"]
    assert finalized["dataset_sha256"]
    assert finalized["protocol_sha256"]
    assert verify_workbench_manifest(root)["phase"]=="INTAKE_READY"


def test_workbench_refuses_implicit_finalize_with_undecided(tmp_path):
    root,_=init(tmp_path)
    session=review(root,[(True,False)]*7)
    assert session.progress().remaining==2
    with pytest.raises(ValueError,match="remain undecided"):
        finalize_workbench(root)

    finalized=finalize_workbench(root,allow_undecided=True)
    assert finalized["phase"]=="INTAKE_READY"
    assert finalized["review_progress"]["remaining"]==2
    assert finalized["review_progress"]["approved_non_sensitive"]==7


def test_workbench_refuses_finalize_without_seven_approved(tmp_path):
    root,_=init(tmp_path)
    session=review(root,[(True,False)]*6)
    for candidate in session.pending_candidates():
        session.record_decision(
            candidate["candidate_id"],
            approved=False,
            sensitive=False,
        )
    with pytest.raises(ValueError,match="need at least 7"):
        finalize_workbench(root)


def test_workbench_manifest_tamper_is_detected(tmp_path):
    root,_=init(tmp_path)
    paths=paths_for(root)
    payload=json.loads(paths.workbench_manifest.read_text(encoding="utf-8"))
    payload["phase"]="INTAKE_READY"
    paths.workbench_manifest.write_text(json.dumps(payload),encoding="utf-8")
    with pytest.raises(ValueError,match="manifest digest mismatch"):
        verify_workbench_manifest(root)


def test_workbench_detects_decisions_tamper_after_intake(tmp_path):
    root,_=init(tmp_path)
    session=review(root,[(True,False)]*7)
    for candidate in session.pending_candidates():
        session.record_decision(
            candidate["candidate_id"],
            approved=False,
            sensitive=False,
        )
    finalize_workbench(root)

    paths=paths_for(root)
    paths.decisions.write_text(
        paths.decisions.read_text(encoding="utf-8")
        + json.dumps({
            "candidate_id":"tampered",
            "approved":False,
            "sensitive":False,
            "language":None,
            "weight":1.0,
        })+"\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        workbench_status(root)


def test_workbench_readiness_uses_exact_finalized_pack(tmp_path):
    root,_=init(tmp_path)
    session=review(root,[(True,False)]*7)
    for candidate in session.pending_candidates():
        session.record_decision(
            candidate["candidate_id"],
            approved=False,
            sensitive=False,
        )
    finalized=finalize_workbench(root)

    anchor=tmp_path/"anchor.jsonl"
    anchor.write_text("".join(
        json.dumps({
            "prompt":f"anchor-{i}",
            "target":f"target-{i}",
            "language":"vi" if i%2==0 else "en",
        })+"\n"
        for i in range(4)
    ),encoding="utf-8")
    latent=tmp_path/"latent.json"
    LatentStore(latent).initialize(
        identity_id="l27",
        checkpoint_sha256="core",
        latent_dim=32,
        protocol_sha256=None,
    )
    model=tmp_path/"model"
    model.mkdir()
    revision="l27-revision"
    (model/"config.json").write_text("{}",encoding="utf-8")
    (model/"model.safetensors").write_bytes(b"x")
    (model/".nolane-model-revision").write_text(revision,encoding="utf-8")
    lock=tmp_path/"model.lock.json"
    lock.write_text(json.dumps({"upstream":{"revision":revision}}),encoding="utf-8")
    l14=tmp_path/"l14.pt"
    l14.write_bytes(b"x")

    result=assess_workbench_readiness(
        root,
        anchor=anchor,
        latent=latent,
        model_lock=lock,
        model_dir=model,
        l14_anchor=l14,
    )
    assert result["readiness"]["status"]=="REAL_CANDIDATE_INPUTS_READY"
    assert (
        result["approved_evidence_manifest_sha256"]
        == finalized["approved_manifest_sha256"]
    )
    rendered=json.dumps(result,sort_keys=True)
    assert "L27-PRIVATE-PROMPT" not in rendered
    assert "L27-PRIVATE-TARGET" not in rendered
