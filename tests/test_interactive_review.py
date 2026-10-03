import json
from pathlib import Path

import pytest

from nolane_personal.interactive_review import LocalReviewSession
from nolane_personal.review_queue import (
    apply_review_decisions,
    build_review_queue,
    verify_review_queue,
)


def write_export(path: Path, *, languages=True):
    conversations=[]
    for index in range(3):
        row={
            "conversation_id":f"L26-PRIVATE-CONVERSATION-{index}",
            "messages":[
                {"role":"user","content":f"L26-PRIVATE-PROMPT-{index}"},
                {"role":"assistant","content":f"L26-PRIVATE-TARGET-{index}"},
            ],
        }
        if languages:
            row["language"]="vi" if index%2==0 else "en"
        conversations.append(row)
    path.write_text(json.dumps(conversations,ensure_ascii=False),encoding="utf-8")


def session(tmp_path: Path, *, languages=True):
    source=tmp_path/"export.json"
    write_export(source,languages=languages)
    queued=build_review_queue(source,tmp_path/"queue")
    decisions=tmp_path/"decisions.jsonl"
    progress=tmp_path/"progress.json"
    review=LocalReviewSession(
        queued["manifest_path"],
        decisions,
        progress,
    )
    return review,queued,decisions,progress


def load_jsonl(path: Path):
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_opening_reviewer_never_auto_approves_and_manifest_is_private(tmp_path):
    review,queued,decisions,progress=session(tmp_path)
    assert not decisions.exists()
    assert len(review.pending_candidates())==3
    state=review.progress()
    assert state.total==3
    assert state.decided==0
    assert state.approved_non_sensitive==0
    assert state.remaining==3

    manifest=review.verify_progress_manifest()
    assert manifest["authority"]=="HUMAN_REVIEW_SESSION_NO_AUTO_APPROVAL"
    assert manifest["decisions_sha256"] is None
    rendered=progress.read_text(encoding="utf-8")
    for forbidden in (
        "L26-PRIVATE-PROMPT",
        "L26-PRIVATE-TARGET",
        "L26-PRIVATE-CONVERSATION",
    ):
        assert forbidden not in rendered
    _queue_manifest,candidates=verify_review_queue(queued["manifest_path"])
    for candidate in candidates:
        assert candidate["candidate_id"] not in rendered


def test_record_decisions_persists_immediately_and_resume_is_exact(tmp_path):
    review,queued,decisions,progress=session(tmp_path)
    candidates=review.pending_candidates()

    review.record_decision(
        candidates[0]["candidate_id"],
        approved=True,
        sensitive=False,
    )
    rows=load_jsonl(decisions)
    assert len(rows)==1
    assert rows[0]["approved"] is True
    assert rows[0]["sensitive"] is False
    assert rows[0]["language"]=="vi"

    review.record_decision(
        candidates[1]["candidate_id"],
        approved=False,
        sensitive=False,
    )
    review.record_decision(
        candidates[2]["candidate_id"],
        approved=False,
        sensitive=True,
    )
    state=review.progress()
    assert state.decided==3
    assert state.approved_non_sensitive==1
    assert state.rejected==2
    assert state.sensitive==1
    assert state.remaining==0
    assert review.verify_progress_manifest()["progress"]==state.to_dict()

    resumed=LocalReviewSession(
        queued["manifest_path"],
        decisions,
        progress,
    )
    assert resumed.pending_candidates()==[]
    assert resumed.progress().to_dict()==state.to_dict()
    assert len(load_jsonl(decisions))==3


def test_explicit_corrected_target_is_frozen_and_feeds_reviewed_source(tmp_path):
    review,queued,decisions,_progress=session(tmp_path)
    candidate=review.pending_candidates()[0]
    corrected="Câu trả lời do người dùng sửa để Nolane học."
    review.record_decision(
        candidate["candidate_id"],
        approved=True,
        sensitive=False,
        corrected_target=corrected,
    )

    rows=load_jsonl(decisions)
    assert rows[0]["corrected_target"]==corrected

    resumed=LocalReviewSession(
        queued["manifest_path"],
        decisions,
        tmp_path/"progress.json",
    )
    assert resumed.decisions[candidate["candidate_id"]][
        "corrected_target"
    ]==corrected

    reviewed=apply_review_decisions(
        queued["manifest_path"],
        decisions,
        tmp_path/"reviewed-corrected",
    )
    evidence=load_jsonl(reviewed["reviewed_source_path"])
    assert evidence[0]["target"]==corrected
    assert evidence[0]["target"]!=candidate["target"]


def test_corrected_target_is_rejected_for_non_approved_or_sensitive_decision(tmp_path):
    review,_queued,_decisions,_progress=session(tmp_path)
    candidates=review.pending_candidates()
    with pytest.raises(ValueError,match="only valid for approved non-sensitive"):
        review.record_decision(
            candidates[0]["candidate_id"],
            approved=False,
            sensitive=False,
            corrected_target="should not be accepted",
        )
    with pytest.raises(ValueError,match="only valid for approved non-sensitive"):
        review.record_decision(
            candidates[1]["candidate_id"],
            approved=True,
            sensitive=True,
            corrected_target="should not be accepted",
        )


def test_exact_frozen_decision_retry_is_idempotent(tmp_path):
    review,_queued,decisions,progress=session(tmp_path)
    candidate=review.pending_candidates()[0]
    first=review.record_decision(
        candidate["candidate_id"],
        approved=True,
        sensitive=False,
        language="en",
        corrected_target="Exact retry target.",
    )
    before=decisions.read_bytes()
    manifest_before=progress.read_bytes()

    second=review.record_decision(
        candidate["candidate_id"],
        approved=True,
        sensitive=False,
        language="en",
        corrected_target="Exact retry target.",
    )

    assert second==first
    assert decisions.read_bytes()==before
    assert progress.read_bytes()==manifest_before
    assert review.progress().decided==1


def test_conflicting_frozen_decision_retry_remains_blocked(tmp_path):
    review,_queued,_decisions,_progress=session(tmp_path)
    candidate=review.pending_candidates()[0]
    review.record_decision(
        candidate["candidate_id"],
        approved=True,
        sensitive=False,
        language="en",
        corrected_target="Frozen target.",
    )
    with pytest.raises(ValueError,match="different frozen decision"):
        review.record_decision(
            candidate["candidate_id"],
            approved=False,
            sensitive=False,
            language="en",
        )


def test_frozen_decision_cannot_be_silently_replaced(tmp_path):
    review,_queued,_decisions,_progress=session(tmp_path)
    candidate=review.pending_candidates()[0]
    review.record_decision(
        candidate["candidate_id"],
        approved=False,
        sensitive=False,
    )
    with pytest.raises(ValueError,match="already has a frozen decision"):
        review.record_decision(
            candidate["candidate_id"],
            approved=True,
            sensitive=False,
            language="vi",
        )


def test_unknown_candidate_and_missing_language_approval_fail_closed(tmp_path):
    review,_queued,decisions,_progress=session(tmp_path,languages=False)
    with pytest.raises(ValueError,match="unknown candidate_id"):
        review.record_decision(
            "not-real",
            approved=False,
            sensitive=False,
        )
    candidate=review.pending_candidates()[0]
    with pytest.raises(ValueError,match="requires vi/en language"):
        review.record_decision(
            candidate["candidate_id"],
            approved=True,
            sensitive=False,
        )
    assert not decisions.exists()
    assert review.progress().decided==0


def test_existing_unknown_or_duplicate_decisions_fail_on_resume(tmp_path):
    review,queued,decisions,progress=session(tmp_path)
    candidates=review.pending_candidates()

    decisions.write_text(json.dumps({
        "candidate_id":"unknown",
        "approved":False,
        "sensitive":False,
        "language":None,
        "weight":1.0,
    })+"\n",encoding="utf-8")
    with pytest.raises(ValueError,match="unknown candidate_id"):
        LocalReviewSession(queued["manifest_path"],decisions,progress)

    row={
        "candidate_id":candidates[0]["candidate_id"],
        "approved":False,
        "sensitive":False,
        "language":"vi",
        "weight":1.0,
    }
    decisions.write_text(
        json.dumps(row)+"\n"+json.dumps(row)+"\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError,match="duplicate candidate decision"):
        LocalReviewSession(queued["manifest_path"],decisions,progress)


def test_progress_manifest_tamper_is_detected(tmp_path):
    review,_queued,_decisions,progress=session(tmp_path)
    payload=json.loads(progress.read_text(encoding="utf-8"))
    payload["progress"]["remaining"]=999
    progress.write_text(json.dumps(payload),encoding="utf-8")
    with pytest.raises(ValueError,match="manifest digest mismatch"):
        review.verify_progress_manifest()


def test_reviewer_decisions_are_directly_consumable_by_l24(tmp_path):
    review,queued,decisions,_progress=session(tmp_path)
    candidates=review.pending_candidates()
    review.record_decision(
        candidates[0]["candidate_id"],
        approved=True,
        sensitive=False,
    )
    review.record_decision(
        candidates[1]["candidate_id"],
        approved=False,
        sensitive=True,
    )
    # Candidate 3 intentionally remains undecided.
    result=apply_review_decisions(
        queued["manifest_path"],
        decisions,
        tmp_path/"reviewed",
    )
    stats=result["manifest"]["stats"]
    assert stats["decisions_supplied"]==2
    assert stats["approved"]==1
    assert stats["sensitive"]==1
    assert stats["undecided"]==1
