import hashlib
import json

import pytest

from nolane_personal.approved_evidence import (
    build_approved_evidence_pack,
    verify_approved_evidence_pack,
)
from nolane_personal.evidence_quality import assess_evidence_quality
from nolane_personal.personal_dataset import PersonalizationExample
from nolane_personal.personal_protocol import build_personalization_protocol


def group(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def examples(count=7):
    return [
        PersonalizationExample(
            prompt=f"unique prompt {i} about topic {i}",
            target=f"unique answer {i} with detail {i}",
            language="vi" if i % 2 == 0 else "en",
        )
        for i in range(count)
    ]


def write_source(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def approved_rows(count=7):
    return [
        {
            "prompt": f"L28-PRIVATE-PROMPT-{i} unique subject {i}",
            "target": f"L28-PRIVATE-TARGET-{i} unique response {i}",
            "language": "vi" if i % 2 == 0 else "en",
            "approved": True,
            "sensitive": False,
            "source_id": f"L28-PRIVATE-CONVERSATION-{i}:pair:0",
        }
        for i in range(count)
    ]


def test_quality_court_passes_grouped_distinct_evidence_without_raw_text():
    rows=examples(7)
    protocol=build_personalization_protocol(
        rows,
        dataset_sha256="dataset",
        source_group_sha256=[group(f"conversation-{i}") for i in range(7)],
    )
    receipt=assess_evidence_quality(rows,protocol)
    assert receipt["status"]=="PASS"
    assert receipt["reasons"]==[]
    assert receipt["source_groups"]["coverage"]==7
    assert receipt["source_groups"]["distinct_total"]==7
    assert receipt["source_groups"]["cross_split_leaked_group_count"]==0
    assert receipt["text_leakage"]["near_duplicate_cross_split"]==0
    rendered=json.dumps(receipt,sort_keys=True)
    assert "unique prompt" not in rendered
    assert "unique answer" not in rendered
    assert "conversation-" not in rendered


def test_quality_court_blocks_missing_source_group_lineage():
    rows=examples(7)
    protocol=build_personalization_protocol(rows,dataset_sha256="dataset")
    receipt=assess_evidence_quality(rows,protocol)
    assert receipt["status"]=="BLOCKED"
    assert "incomplete_source_group_lineage" in receipt["reasons"]
    assert "insufficient_distinct_source_groups" in receipt["reasons"]
    assert "non_grouped_split_strategy" in receipt["reasons"]


def test_quality_court_blocks_near_duplicate_across_train_dev():
    rows=examples(7)
    rows[3]=PersonalizationExample(
        prompt="I am really worried about my math exam tomorrow and cannot sleep",
        target="Tell me what part scares you most and we can unpack it slowly together",
        language="en",
    )
    rows[4]=PersonalizationExample(
        prompt="I am really worried about the math exam tomorrow and I cannot sleep",
        target="Tell me what part scares you the most and we can unpack it slowly together",
        language="en",
    )
    protocol=build_personalization_protocol(
        rows,
        dataset_sha256="dataset",
        source_group_sha256=[group(f"conversation-{i}") for i in range(7)],
    )
    assert [row["index"] for row in protocol["splits"]["train"]]==[0,1,2,3]
    assert [row["index"] for row in protocol["splits"]["dev"]]==[4]
    receipt=assess_evidence_quality(rows,protocol)
    assert receipt["status"]=="BLOCKED"
    assert "near_duplicate_cross_split" in receipt["reasons"]
    assert receipt["text_leakage"]["near_duplicate_cross_split"]>=1


def test_l23_pack_contains_verified_quality_receipt_and_private_manifest(tmp_path):
    source=tmp_path/"source.jsonl"
    write_source(source,approved_rows(7))
    result=build_approved_evidence_pack(source,tmp_path/"pack")
    manifest=verify_approved_evidence_pack(result.manifest_path)

    assert manifest["quality_status"]=="PASS"
    assert manifest["quality_court_sha256"]
    assert manifest["quality_receipt_sha256"]
    quality=json.loads(result.quality_path.read_text(encoding="utf-8"))
    assert quality["status"]=="PASS"
    assert quality["court_sha256"]==manifest["quality_court_sha256"]

    for path in (result.manifest_path,result.quality_path):
        rendered=path.read_text(encoding="utf-8")
        assert "L28-PRIVATE-PROMPT" not in rendered
        assert "L28-PRIVATE-TARGET" not in rendered
        assert "L28-PRIVATE-CONVERSATION" not in rendered


def test_l23_pack_blocks_single_conversation_group_before_output(tmp_path):
    source=tmp_path/"source.jsonl"
    rows=approved_rows(7)
    for i,row in enumerate(rows):
        row["source_id"]=f"one-private-conversation:pair:{i}"
    write_source(source,rows)
    output=tmp_path/"pack"
    with pytest.raises(ValueError,match="at least 3 distinct source groups"):
        build_approved_evidence_pack(source,output)
    assert not output.exists()


def test_l23_verification_detects_quality_receipt_tamper(tmp_path):
    source=tmp_path/"source.jsonl"
    write_source(source,approved_rows(7))
    result=build_approved_evidence_pack(source,tmp_path/"pack")
    payload=json.loads(result.quality_path.read_text(encoding="utf-8"))
    payload["status"]="BLOCKED"
    result.quality_path.write_text(json.dumps(payload),encoding="utf-8")
    with pytest.raises(ValueError,match="quality receipt digest mismatch"):
        verify_approved_evidence_pack(result.manifest_path)
