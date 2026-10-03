from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from nolane_personal.interactive_review import LocalReviewSession
from nolane_personal.local_evidence_workbench import (
    finalize_workbench,
    initialize_workbench,
    paths_for,
)
from nolane_personal.product_evidence_bridge import (
    PLAN_SPEC_SCHEMA,
    ProductEvidenceExportPolicy,
    build_product_longitudinal_plan,
    export_product_evidence,
    prepare_product_evidence_window,
    verify_product_evidence_export,
    verify_product_evidence_window,
)
from nolane_personal.review_queue import verify_review_queue


PROMPTS = [
    "Explain why a bicycle is easier to balance while moving.",
    "Nhắc mình một cách đơn giản để nhớ mang áo mưa.",
    "Compare a paper map and GPS for a short hike.",
    "Viết một câu từ chối cuộc hẹn thật lịch sự.",
    "Describe what yeast does while bread dough rises.",
    "Gợi ý một cách ôn từ mới sau ba ngày.",
    "What changes the feel of a mechanical keyboard switch?",
    "Sắp xếp bàn học thế nào để bớt phân tâm?",
]
TARGETS = [
    "Forward motion makes small steering corrections effective before a tilt grows.",
    "Trước khi đi, kiểm tra thời tiết và để sẵn áo mưa gọn trong túi.",
    "Paper maps work without power and show context; GPS gives live position and rerouting.",
    "Mình cảm ơn lời mời, nhưng lần này mình không sắp xếp tham gia được. Hẹn dịp khác nhé.",
    "Yeast consumes sugars and releases carbon dioxide that expands the dough.",
    "Sau ba ngày, tự nhớ nghĩa trước rồi đặt từ đó vào một câu mới.",
    "Spring weight, travel, tactile geometry, housing and materials all change key feel.",
    "Chỉ để đồ đang dùng trên bàn, cất điện thoại xa tay và chừa khoảng trống để viết.",
]


def make_product_db(path: Path, *, turns: int = 12) -> None:
    db = sqlite3.connect(path)
    db.execute(
        """
        CREATE TABLE events (
            event_id TEXT PRIMARY KEY,
            at TEXT NOT NULL,
            kind TEXT NOT NULL,
            source TEXT NOT NULL,
            salience REAL NOT NULL,
            payload_json TEXT NOT NULL,
            after_state_digest TEXT NOT NULL
        )
        """
    )
    start = datetime(2026, 10, 1, 8, 0, tzinfo=timezone.utc)
    rows = []
    for i in range(turns):
        user_at = start + timedelta(minutes=i * 5)
        assistant_at = user_at + timedelta(seconds=3)
        prompt = f"turn-{i}: {PROMPTS[i % len(PROMPTS)]}"
        target = f"reply-{i}: {TARGETS[i % len(TARGETS)]}"
        rows.extend(
            [
                (
                    f"u-{i}",
                    user_at.isoformat(),
                    "user_message",
                    "user",
                    0.5,
                    json.dumps({"text": prompt}, ensure_ascii=False),
                    f"d-{i}-u",
                ),
                (
                    f"a-{i}",
                    assistant_at.isoformat(),
                    "assistant_speech",
                    "assistant",
                    0.5,
                    json.dumps({"text": target}, ensure_ascii=False),
                    f"d-{i}-a",
                ),
            ]
        )
    db.executemany(
        """
        INSERT INTO events(
            event_id,at,kind,source,salience,payload_json,after_state_digest
        ) VALUES(?,?,?,?,?,?,?)
        """,
        rows,
    )
    db.commit()
    db.close()


def conversation_source(
    path: Path,
    seed: str,
    *,
    content_seed: str | None = None,
) -> None:
    content_seed = content_seed or seed
    conversations = []
    for i in range(8):
        conversations.append(
            {
                "conversation_id": f"{seed}-conversation-{i}",
                "language": "vi" if i % 2 else "en",
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"{content_seed}: {PROMPTS[i]}"
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": (
                            f"{content_seed}: {TARGETS[i]}"
                        ),
                    },
                ],
            }
        )
    path.write_text(
        json.dumps({"conversations": conversations}, ensure_ascii=False),
        encoding="utf-8",
    )


def finalized_workbench(
    root: Path,
    seed: str,
    *,
    content_seed: str | None = None,
) -> Path:
    source = root.parent / f"{root.name}-source.json"
    conversation_source(source, seed, content_seed=content_seed)
    initialize_workbench(source, root)
    paths = paths_for(root)
    session = LocalReviewSession(
        paths.queue_manifest,
        paths.decisions,
        paths.review_progress,
    )
    for candidate in session.pending_candidates():
        session.record_decision(
            candidate["candidate_id"],
            approved=True,
            sensitive=False,
            language=str(candidate["language"]),
        )
    finalize_workbench(root)
    return root


def test_product_db_export_creates_review_required_leakage_safe_groups(tmp_path):
    db = tmp_path / "living.db"
    make_product_db(db, turns=12)

    result = export_product_evidence(
        db,
        tmp_path / "export",
        language="vi",
        policy=ProductEvidenceExportPolicy(
            session_gap_minutes=45,
            max_pairs_per_group=4,
        ),
    )
    manifest, export_path = verify_product_evidence_export(
        result.manifest_path
    )
    assert manifest["paired_turns"] == 12
    assert manifest["conversation_groups"] == 3
    assert manifest["privacy"]["approval_created_by_export"] is False
    assert export_path.is_file()

    window = prepare_product_evidence_window(
        db,
        tmp_path / "window",
        language="vi",
        policy=ProductEvidenceExportPolicy(max_pairs_per_group=4),
    )
    assert window["privacy"]["auto_approval"] is False
    verify_product_evidence_window(tmp_path / "window")

    wb = paths_for(tmp_path / "window" / "workbench")
    queue_manifest, candidates = verify_review_queue(wb.queue_manifest)
    assert queue_manifest["candidate_count"] == 12
    assert all(row["approved"] is False for row in candidates)
    assert all(row["reviewed"] is False for row in candidates)


def test_product_export_rejects_range_without_three_source_groups(tmp_path):
    db = tmp_path / "living.db"
    make_product_db(db, turns=6)
    with pytest.raises(ValueError, match="at least 3 leakage-safe"):
        export_product_evidence(
            db,
            tmp_path / "export",
            language="en",
            policy=ProductEvidenceExportPolicy(max_pairs_per_group=4),
        )


def test_product_plan_builder_closes_reviewed_windows_into_l43_plan(tmp_path):
    fixed = finalized_workbench(tmp_path / "fixed", "fixed")
    cycles = []
    for i in range(5):
        retention = finalized_workbench(
            tmp_path / f"ret-{i}",
            f"ret-{i}",
        )
        adaptation = finalized_workbench(
            tmp_path / f"adapt-{i}",
            f"adapt-{i}",
        )
        cycles.append(
            {
                "retention_workbench": str(retention),
                "adaptation_workbench": str(adaptation),
                "training": {
                    "epochs": 1,
                    "learning_rate": 0.0002,
                    "max_length": 128,
                },
            }
        )

    initial = tmp_path / "factorized-nolane.pt"
    initial.write_bytes(b"checkpoint")
    latent = tmp_path / "latent.json"
    latent.write_text('{"values":[0.0]}', encoding="utf-8")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text(
        '{"version":"fixture"}',
        encoding="utf-8",
    )
    (tokenizer / "tokenizer_config.json").write_text(
        '{"chat_template":"fixture"}',
        encoding="utf-8",
    )

    spec = {
        "schema": PLAN_SPEC_SCHEMA,
        "initial_factorized": str(initial),
        "latent": str(latent),
        "tokenizer": str(tokenizer),
        "device": "cpu",
        "fixed_workbench": str(fixed),
        "cycles": cycles,
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    output = tmp_path / "l43-plan.json"

    receipt = build_product_longitudinal_plan(spec_path, output)
    assert receipt["authority"].endswith("NO_TRAINING_AUTHORITY")
    assert receipt["privacy"]["contains_raw_prompt_target"] is False
    assert receipt["privacy"]["contains_individual_prompt_hashes"] is False
    assert len(receipt["cycles"]) == 5
    assert output.is_file()

    plan = json.loads(output.read_text(encoding="utf-8"))
    assert plan["schema"] == "NOLANE-L43-REAL-LONGITUDINAL-PLAN-V1"
    assert len(plan["cycles"]) == 5


def test_product_plan_builder_blocks_content_reuse_even_with_new_source_ids(tmp_path):
    fixed = finalized_workbench(tmp_path / "fixed", "fixed")
    ret = []
    adapt = []
    for i in range(5):
        ret.append(
            finalized_workbench(
                tmp_path / f"ret-{i}",
                f"ret-{i}",
            )
        )
        adapt.append(
            finalized_workbench(
                tmp_path / f"adapt-{i}",
                f"adapt-{i}",
                content_seed=("same-adaptation" if i in {0, 1} else f"adapt-{i}"),
            )
        )

    initial = tmp_path / "factorized-nolane.pt"
    initial.write_bytes(b"checkpoint")
    latent = tmp_path / "latent.json"
    latent.write_text('{"values":[0.0]}', encoding="utf-8")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}", encoding="utf-8")

    spec = {
        "schema": PLAN_SPEC_SCHEMA,
        "initial_factorized": str(initial),
        "latent": str(latent),
        "tokenizer": str(tokenizer),
        "device": "cpu",
        "fixed_workbench": str(fixed),
        "cycles": [
            {
                "retention_workbench": str(ret[i]),
                "adaptation_workbench": str(adapt[i]),
            }
            for i in range(5)
        ],
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    with pytest.raises(ValueError, match="adaptation cycle 1/cycle 2 content leakage"):
        build_product_longitudinal_plan(
            spec_path,
            tmp_path / "l43-plan.json",
        )


def test_product_plan_builder_blocks_fixed_panel_content_leakage(tmp_path):
    fixed = finalized_workbench(
        tmp_path / "fixed",
        "fixed",
        content_seed="shared-fixed",
    )
    cycles = []
    for i in range(5):
        retention = finalized_workbench(
            tmp_path / f"ret-{i}",
            f"ret-{i}",
            content_seed=("shared-fixed" if i == 0 else f"ret-{i}"),
        )
        adaptation = finalized_workbench(
            tmp_path / f"adapt-{i}",
            f"adapt-{i}",
        )
        cycles.append(
            {
                "retention_workbench": str(retention),
                "adaptation_workbench": str(adaptation),
            }
        )

    initial = tmp_path / "factorized-nolane.pt"
    initial.write_bytes(b"checkpoint")
    latent = tmp_path / "latent.json"
    latent.write_text('{"values":[0.0]}', encoding="utf-8")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}", encoding="utf-8")

    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "schema": PLAN_SPEC_SCHEMA,
                "initial_factorized": str(initial),
                "latent": str(latent),
                "tokenizer": str(tokenizer),
                "fixed_workbench": str(fixed),
                "cycles": cycles,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fixed-panel/training content leakage"):
        build_product_longitudinal_plan(
            spec_path,
            tmp_path / "l43-plan.json",
        )
