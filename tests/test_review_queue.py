import json
from pathlib import Path

import pytest

from nolane_personal.approved_evidence import build_approved_evidence_pack
from nolane_personal.review_queue import (
    apply_review_decisions,
    build_review_queue,
    verify_review_queue,
)


def write_export(path: Path, count: int = 3):
    conversations=[]
    for i in range(count):
        conversations.append({
            "conversation_id":f"conversation-private-{i}",
            "language":"vi" if i%2==0 else "en",
            "messages":[
                {"role":"system","content":"PRIVATE-SYSTEM-INSTRUCTION"},
                {"role":"user","content":f"PRIVATE-USER-{i}"},
                {"role":"assistant","content":f"PRIVATE-ASSISTANT-{i}"},
            ],
        })
    path.write_text(json.dumps(conversations,ensure_ascii=False),encoding="utf-8")


def load_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_import_creates_never_approved_review_queue_and_private_manifest(tmp_path):
    source=tmp_path/"export.json"
    write_export(source,3)
    result=build_review_queue(source,tmp_path/"queue")
    manifest,rows=verify_review_queue(result["manifest_path"])

    assert manifest["authority"]=="LOCAL_REVIEW_QUEUE_NEVER_APPROVED_BY_IMPORT"
    assert manifest["candidate_count"]==3
    assert manifest["stats"]["skipped_non_dialogue_roles"]==3
    assert all(row["approved"] is False for row in rows)
    assert all(row["reviewed"] is False for row in rows)
    assert all(row["sensitive"] is None for row in rows)

    rendered=result["manifest_path"].read_text(encoding="utf-8")
    assert "PRIVATE-USER" not in rendered
    assert "PRIVATE-ASSISTANT" not in rendered
    assert "conversation-private" not in rendered
    assert "PRIVATE-USER-0" in result["queue_path"].read_text(encoding="utf-8")


def test_queue_extracts_only_user_to_assistant_pairs(tmp_path):
    source=tmp_path/"export.json"
    source.write_text(json.dumps([
        {
            "id":"c1",
            "messages":[
                {"role":"assistant","content":"orphan assistant"},
                {"role":"user","content":"first user"},
                {"role":"user","content":"second user replaces pending"},
                {"role":"assistant","content":"paired assistant"},
                {"role":"tool","content":"ignored"},
                {"role":"user","content":[{"text":"multi"},{"text":"part"}]},
                {"role":"assistant","content":{"text":"answer"}},
            ],
        }
    ]),encoding="utf-8")
    result=build_review_queue(source,tmp_path/"queue",default_language="en")
    _manifest,rows=verify_review_queue(result["manifest_path"])
    assert len(rows)==2
    assert rows[0]["prompt"]=="second user replaces pending"
    assert rows[0]["target"]=="paired assistant"
    assert rows[1]["prompt"]=="multi\npart"
    assert rows[1]["target"]=="answer"
    assert rows[0]["language"]=="en"


def test_queue_tamper_cannot_turn_import_into_approval(tmp_path):
    source=tmp_path/"export.json"
    write_export(source,1)
    result=build_review_queue(source,tmp_path/"queue")
    rows=load_jsonl(result["queue_path"])
    rows[0]["approved"]=True
    result["queue_path"].write_text(
        json.dumps(rows[0],ensure_ascii=False)+"\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError,match="review queue digest mismatch"):
        verify_review_queue(result["manifest_path"])


def test_explicit_review_decisions_feed_l23_and_only_eligible_rows_survive(tmp_path):
    source=tmp_path/"export.json"
    write_export(source,9)
    queued=build_review_queue(source,tmp_path/"queue")
    _manifest,candidates=verify_review_queue(queued["manifest_path"])

    decisions=tmp_path/"decisions.jsonl"
    decision_rows=[]
    for index,candidate in enumerate(candidates):
        if index<7:
            decision_rows.append({
                "candidate_id":candidate["candidate_id"],
                "approved":True,
                "sensitive":False,
                "language":candidate["language"],
            })
        elif index==7:
            decision_rows.append({
                "candidate_id":candidate["candidate_id"],
                "approved":True,
                "sensitive":True,
                "language":candidate["language"],
            })
        else:
            decision_rows.append({
                "candidate_id":candidate["candidate_id"],
                "approved":False,
                "sensitive":False,
                "language":candidate["language"],
            })
    decisions.write_text(
        "".join(json.dumps(row)+"\n" for row in decision_rows),
        encoding="utf-8",
    )

    reviewed=apply_review_decisions(
        queued["manifest_path"],
        decisions,
        tmp_path/"reviewed",
    )
    stats=reviewed["manifest"]["stats"]
    assert stats["approved"]==7
    assert stats["sensitive"]==1
    assert stats["rejected"]==2
    assert stats["undecided"]==0

    manifest_rendered=reviewed["manifest_path"].read_text(encoding="utf-8")
    assert "PRIVATE-USER" not in manifest_rendered
    assert "PRIVATE-ASSISTANT" not in manifest_rendered
    assert "conversation-private" not in manifest_rendered

    pack=build_approved_evidence_pack(
        reviewed["reviewed_source_path"],
        tmp_path/"pack",
    )
    assert pack.manifest["stats"]["output_examples"]==7
    dataset=load_jsonl(pack.dataset_path)
    assert len(dataset)==7
    assert all(row["prompt"].startswith("PRIVATE-USER-") for row in dataset)


def test_partial_decisions_leave_candidates_unapproved(tmp_path):
    source=tmp_path/"export.json"
    write_export(source,2)
    queued=build_review_queue(source,tmp_path/"queue")
    _manifest,candidates=verify_review_queue(queued["manifest_path"])
    decisions=tmp_path/"decisions.jsonl"
    decisions.write_text(json.dumps({
        "candidate_id":candidates[0]["candidate_id"],
        "approved":True,
        "sensitive":False,
        "language":candidates[0]["language"],
    })+"\n",encoding="utf-8")
    reviewed=apply_review_decisions(
        queued["manifest_path"],
        decisions,
        tmp_path/"reviewed",
    )
    rows=load_jsonl(reviewed["reviewed_source_path"])
    assert rows[0]["approved"] is True and rows[0]["reviewed"] is True
    assert rows[1]["approved"] is False and rows[1]["reviewed"] is False
    assert rows[1]["sensitive"] is None


def test_unknown_or_duplicate_decisions_fail_closed(tmp_path):
    source=tmp_path/"export.json"
    write_export(source,1)
    queued=build_review_queue(source,tmp_path/"queue")
    _manifest,candidates=verify_review_queue(queued["manifest_path"])

    unknown=tmp_path/"unknown.jsonl"
    unknown.write_text(json.dumps({
        "candidate_id":"not-real",
        "approved":False,
        "sensitive":False,
    })+"\n",encoding="utf-8")
    with pytest.raises(ValueError,match="unknown candidate IDs"):
        apply_review_decisions(queued["manifest_path"],unknown,tmp_path/"out-unknown")

    duplicate=tmp_path/"duplicate.jsonl"
    row={
        "candidate_id":candidates[0]["candidate_id"],
        "approved":False,
        "sensitive":False,
    }
    duplicate.write_text(
        json.dumps(row)+"\n"+json.dumps(row)+"\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError,match="duplicate candidate decision"):
        apply_review_decisions(queued["manifest_path"],duplicate,tmp_path/"out-dup")


def test_approved_decision_requires_explicit_language_if_import_did_not_have_one(tmp_path):
    source=tmp_path/"export.json"
    source.write_text(json.dumps([{
        "id":"c",
        "messages":[
            {"role":"user","content":"hello"},
            {"role":"assistant","content":"hi"},
        ],
    }]),encoding="utf-8")
    queued=build_review_queue(source,tmp_path/"queue")
    _manifest,candidates=verify_review_queue(queued["manifest_path"])
    decisions=tmp_path/"decisions.jsonl"
    decisions.write_text(json.dumps({
        "candidate_id":candidates[0]["candidate_id"],
        "approved":True,
        "sensitive":False,
    })+"\n",encoding="utf-8")
    with pytest.raises(ValueError,match="requires explicit vi/en language"):
        apply_review_decisions(queued["manifest_path"],decisions,tmp_path/"reviewed")


def test_duplicate_conversation_pair_fails_closed(tmp_path):
    source=tmp_path/"export.json"
    source.write_text(json.dumps([
        {"id":"a","messages":[
            {"role":"user","content":"same"},
            {"role":"assistant","content":"same-answer"},
        ]},
        {"id":"b","messages":[
            {"role":"user","content":"same"},
            {"role":"assistant","content":"same-answer"},
        ]},
    ]),encoding="utf-8")
    with pytest.raises(ValueError,match="duplicate prompt/target pair"):
        build_review_queue(source,tmp_path/"queue",default_language="en")
