from __future__ import annotations

import json
import threading
from http.client import HTTPConnection

import pytest

from nolane_personal.cortex import CortexReply
from nolane_personal.product_evidence_bridge import ProductEvidenceExportPolicy
from nolane_personal.product_learning_workspace import ProductLearningWorkspace
from nolane_personal.product_runtime import ProductRuntime
from nolane_personal.product_server import ProductHTTPServer
from nolane_personal.local_evidence_workbench import paths_for


PROMPTS = [
    "Why does a moving bicycle feel easier to balance?",
    "How should I remember an umbrella before leaving home?",
    "What is one advantage of a paper map over GPS?",
    "How can I politely decline an invitation?",
    "What does yeast do while bread dough rises?",
    "What is a good three-day vocabulary review method?",
    "Why do mechanical keyboard switches feel different?",
    "How can I keep my desk less distracting?",
    "What is the difference between a meteor and meteorite?",
    "What should I pack first for a two-day trip?",
    "How can I verify that a downloaded file is unchanged?",
    "Why does a houseplant still need light when watered?",
]


class ReviewCortex:
    checkpoint_sha256 = "a" * 64

    def generate(self, request):
        text = request.user_text or request.intent
        return CortexReply(
            "A concise, concrete answer for: " + str(text),
            intent=request.intent,
        )

    def close(self):
        return None


def factory(_identity_id, _profile_getter):
    return ReviewCortex()


def fill_runtime(runtime: ProductRuntime, *, prefix: str, turns: int = 12):
    if runtime.status()["phase"] != "on":
        runtime.power_on()
    for i in range(turns):
        runtime.send_message(
            f"{prefix}-{i}: {PROMPTS[i % len(PROMPTS)]}"
        )


def approve_all(runtime: ProductRuntime, window_id: str):
    while True:
        candidate = runtime.next_learning_candidate(window_id)
        if candidate is None:
            break
        runtime.record_learning_decision(
            window_id,
            candidate["candidate_id"],
            decision="approve",
            language="en",
        )


def request(server, method, path, payload=None, token=None):
    host, port = server.server_address[:2]
    conn = HTTPConnection(host, port, timeout=3)
    body = None
    headers = {}
    if payload is not None:
        body = json.dumps(payload)
        headers["Content-Type"] = "application/json"
    if token:
        headers["X-Nolane-Token"] = token
    conn.request(method, path, body=body, headers=headers)
    response = conn.getresponse()
    data = json.loads(response.read().decode("utf-8"))
    conn.close()
    return response.status, data


def test_inapp_workspace_never_auto_approves_and_can_finalize(tmp_path):
    runtime = ProductRuntime(tmp_path, cortex_factory=factory)
    try:
        fill_runtime(runtime, prefix="first")
        window = runtime.create_learning_window()
        assert window["window_id"] == "window-0001"
        assert window["phase"] == "QUEUE_READY"
        assert window["review_progress"]["decided"] == 0
        assert window["review_progress"]["approved_non_sensitive"] == 0

        candidate = runtime.next_learning_candidate("window-0001")
        assert candidate is not None
        assert candidate["prompt"].startswith("first-0:")
        assert candidate["privacy"][
            "raw_text_returned_to_local_authenticated_ui"
        ] is True

        approve_all(runtime, "window-0001")
        reviewed = runtime.learning_windows()[0]
        assert reviewed["phase"] == "REVIEW_COMPLETE"
        assert reviewed["review_progress"]["approved_non_sensitive"] == 12

        finalized = runtime.finalize_learning_window("window-0001")
        assert finalized["phase"] == "INTAKE_READY"
        assert finalized["intake_ready"] is True
        assert finalized["quality_status"] == "PASS"
        assert len(finalized["approved_manifest_sha256"]) == 64
    finally:
        runtime.close()


def test_user_correction_becomes_the_approved_training_target(tmp_path):
    runtime = ProductRuntime(tmp_path, cortex_factory=factory)
    corrected = "This is the exact answer I want Nolane to learn."
    try:
        fill_runtime(runtime, prefix="correct")
        runtime.create_learning_window()
        first = runtime.next_learning_candidate("window-0001")
        assert first is not None
        original = first["target"]
        runtime.record_learning_decision(
            "window-0001",
            first["candidate_id"],
            decision="approve",
            language="en",
            corrected_target=corrected,
        )
        approve_all(runtime, "window-0001")
        finalized = runtime.finalize_learning_window("window-0001")
        assert finalized["quality_status"] == "PASS"

        root = (
            tmp_path
            / "learning-evidence"
            / "windows"
            / "window-0001"
            / "workbench"
        )
        wb = paths_for(root)
        approved_manifest = json.loads(
            wb.approved_manifest.read_text(encoding="utf-8")
        )
        dataset = (
            wb.approved_manifest.parent
            / approved_manifest["dataset_filename"]
        )
        rows = [
            json.loads(line)
            for line in dataset.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert rows[0]["target"] == corrected
        assert rows[0]["target"] != original
    finally:
        runtime.close()


def test_inapp_windows_advance_without_reusing_old_product_turns(tmp_path):
    runtime = ProductRuntime(tmp_path, cortex_factory=factory)
    try:
        fill_runtime(runtime, prefix="old")
        first = runtime.create_learning_window()
        first_high_water = first["through_rowid_inclusive"]

        fill_runtime(runtime, prefix="new")
        second = runtime.create_learning_window()
        assert second["window_id"] == "window-0002"
        assert second["through_rowid_inclusive"] > first_high_water

        candidate = runtime.next_learning_candidate("window-0002")
        assert candidate is not None
        assert candidate["prompt"].startswith("new-0:")
        assert "old-" not in candidate["prompt"]

        registry = runtime.learning.verify_registry()
        assert registry["last_exported_rowid"] == second[
            "through_rowid_inclusive"
        ]
        assert [row["window_id"] for row in registry["windows"]] == [
            "window-0001",
            "window-0002",
        ]
        rendered = json.dumps(registry)
        assert "old-0:" not in rendered
        assert "new-0:" not in rendered
    finally:
        runtime.close()


def test_failed_small_window_does_not_advance_registry_cursor(tmp_path):
    runtime = ProductRuntime(tmp_path, cortex_factory=factory)
    runtime.learning = ProductLearningWorkspace(
        data_dir=tmp_path,
        policy=ProductEvidenceExportPolicy(max_pairs_per_group=4),
    )
    try:
        fill_runtime(runtime, prefix="tiny", turns=6)
        with pytest.raises(ValueError, match="at least 3 leakage-safe"):
            runtime.create_learning_window()
        registry = runtime.learning.verify_registry()
        assert registry["last_exported_rowid"] == 0
        assert registry["windows"] == []
        assert registry["next_window_index"] == 1
    finally:
        runtime.close()


def test_learning_review_api_is_authenticated_and_explicit(tmp_path):
    runtime = ProductRuntime(tmp_path, cortex_factory=factory)
    fill_runtime(runtime, prefix="api")
    server = ProductHTTPServer(
        ("127.0.0.1", 0),
        runtime,
        auth_token="learning-secret",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, _ = request(
            server,
            "POST",
            "/v1/learning/windows",
        )
        assert status == 401

        status, window = request(
            server,
            "POST",
            "/v1/learning/windows",
            token="learning-secret",
        )
        assert status == 200
        assert window["review_progress"]["decided"] == 0

        status, pending = request(
            server,
            "GET",
            "/v1/learning/pending?window_id=window-0001",
            token="learning-secret",
        )
        assert status == 200
        candidate = pending["candidate"]
        assert candidate is not None

        corrected = "API-explicit corrected target for learning."
        status, decision = request(
            server,
            "POST",
            "/v1/learning/decision",
            {
                "window_id": "window-0001",
                "candidate_id": candidate["candidate_id"],
                "decision": "approve",
                "language": "en",
                "corrected_target": corrected,
            },
            token="learning-secret",
        )
        assert status == 200
        assert decision["window"]["review_progress"]["decided"] == 1
        assert decision["window"]["review_progress"][
            "approved_non_sensitive"
        ] == 1

        decisions = (
            tmp_path
            / "learning-evidence"
            / "windows"
            / "window-0001"
            / "workbench"
            / "review-decisions.jsonl"
        )
        assert corrected in decisions.read_text(encoding="utf-8")
        registry = runtime.learning.registry_path.read_text(encoding="utf-8")
        progress = (
            tmp_path
            / "learning-evidence"
            / "windows"
            / "window-0001"
            / "workbench"
            / "review-progress-manifest.json"
        ).read_text(encoding="utf-8")
        assert corrected not in registry
        assert corrected not in progress
    finally:
        server.shutdown()
        server.server_close()
        runtime.close()


def test_create_window_retry_returns_same_pending_window(tmp_path):
    runtime = ProductRuntime(tmp_path, cortex_factory=factory)
    try:
        fill_runtime(runtime, prefix="retry-window")
        first = runtime.create_learning_window()
        second = runtime.create_learning_window()
        assert second["window_id"] == first["window_id"] == "window-0001"
        assert second["through_rowid_inclusive"] == first["through_rowid_inclusive"]

        registry = runtime.learning.verify_registry()
        assert len(registry["windows"]) == 1
        assert registry["next_window_index"] == 2
        assert registry["last_exported_rowid"] == first[
            "through_rowid_inclusive"
        ]
    finally:
        runtime.close()


def test_exact_decision_retry_is_idempotent_but_conflicting_retry_is_blocked(tmp_path):
    runtime = ProductRuntime(tmp_path, cortex_factory=factory)
    corrected = "Use this exact corrected answer."
    try:
        fill_runtime(runtime, prefix="retry-decision")
        runtime.create_learning_window()
        candidate = runtime.next_learning_candidate("window-0001")
        assert candidate is not None

        first = runtime.record_learning_decision(
            "window-0001",
            candidate["candidate_id"],
            decision="approve",
            language="en",
            corrected_target=corrected,
        )
        repeated = runtime.record_learning_decision(
            "window-0001",
            candidate["candidate_id"],
            decision="approve",
            language="en",
            corrected_target=corrected,
        )
        assert repeated["window"]["review_progress"] == first["window"][
            "review_progress"
        ]
        assert repeated["window"]["review_progress"]["decided"] == 1

        decisions = (
            tmp_path
            / "learning-evidence"
            / "windows"
            / "window-0001"
            / "workbench"
            / "review-decisions.jsonl"
        )
        rows = [
            line
            for line in decisions.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(rows) == 1

        with pytest.raises(ValueError, match="conflicting retry"):
            runtime.record_learning_decision(
                "window-0001",
                candidate["candidate_id"],
                decision="approve",
                language="en",
                corrected_target="different answer",
            )
        with pytest.raises(ValueError, match="conflicting retry"):
            runtime.record_learning_decision(
                "window-0001",
                candidate["candidate_id"],
                decision="reject",
                language="en",
            )
    finally:
        runtime.close()


def test_partial_review_resumes_exactly_after_runtime_restart(tmp_path):
    runtime = ProductRuntime(tmp_path, cortex_factory=factory)
    fill_runtime(runtime, prefix="restart")
    first = runtime.create_learning_window()
    candidate1 = runtime.next_learning_candidate(first["window_id"])
    runtime.record_learning_decision(
        first["window_id"],
        candidate1["candidate_id"],
        decision="approve",
        language="en",
        corrected_target="Persist this corrected answer through restart.",
    )
    candidate2 = runtime.next_learning_candidate(first["window_id"])
    assert candidate2 is not None
    assert candidate2["candidate_id"] != candidate1["candidate_id"]
    runtime.close()

    reopened = ProductRuntime(tmp_path, cortex_factory=factory)
    try:
        windows = reopened.learning_windows()
        assert len(windows) == 1
        assert windows[0]["review_progress"]["decided"] == 1
        resumed = reopened.next_learning_candidate("window-0001")
        assert resumed is not None
        assert resumed["candidate_id"] == candidate2["candidate_id"]
        assert resumed["prompt"] == candidate2["prompt"]

        exact_retry = reopened.record_learning_decision(
            "window-0001",
            candidate1["candidate_id"],
            decision="approve",
            language="en",
            corrected_target="Persist this corrected answer through restart.",
        )
        assert exact_retry["window"]["review_progress"]["decided"] == 1
    finally:
        reopened.close()


def test_finalize_retry_is_idempotent(tmp_path):
    runtime = ProductRuntime(tmp_path, cortex_factory=factory)
    try:
        fill_runtime(runtime, prefix="finalize-retry")
        runtime.create_learning_window()
        approve_all(runtime, "window-0001")
        first = runtime.finalize_learning_window("window-0001")
        second = runtime.finalize_learning_window("window-0001")
        assert first == second
        assert second["intake_ready"] is True
        assert second["quality_status"] == "PASS"
    finally:
        runtime.close()


def test_sensitive_decision_is_rejected_from_approved_count(tmp_path):
    runtime = ProductRuntime(tmp_path, cortex_factory=factory)
    try:
        fill_runtime(runtime, prefix="sensitive")
        runtime.create_learning_window()
        candidate = runtime.next_learning_candidate("window-0001")
        result = runtime.record_learning_decision(
            "window-0001",
            candidate["candidate_id"],
            decision="sensitive",
            language="en",
        )
        progress = result["window"]["review_progress"]
        assert progress["decided"] == 1
        assert progress["sensitive"] == 1
        assert progress["approved_non_sensitive"] == 0
    finally:
        runtime.close()
