import hashlib
import json
import threading
from http.client import HTTPConnection

import pytest

from nolane_personal.cortex import CortexReply
from nolane_personal.product_profile import ProductProfileStore
from nolane_personal.product_runtime import ProductRuntime
from nolane_personal.product_server import ProductHTTPServer
from nolane_personal.promotion_ceremony import (
    AUTHORITY as CEREMONY_AUTHORITY,
    SCHEMA as CEREMONY_SCHEMA,
)
from nolane_personal.store import payload_digest




def release_ceremony(checkpoint_sha256):
    body = {
        "schema": CEREMONY_SCHEMA,
        "authority": CEREMONY_AUTHORITY,
        "status": "COMPLETE",
        "reasons": [],
        "authorization_sha256": "1" * 64,
        "multicycle_chain_sha256": "2" * 64,
        "long_horizon_retention_court_sha256": "3" * 64,
        "candidate_checkpoint_sha256": checkpoint_sha256,
        "pointer_sha256": "4" * 64,
        "serving_convergence_sha256": "5" * 64,
        "transaction_id": "tx-product-runtime",
        "pointer_generation": 1,
        "authorization_issued_at": "2026-10-02T12:00:00+00:00",
        "authorization_expires_at": "2026-10-02T13:00:00+00:00",
        "transaction_prepared_at": "2026-10-02T12:00:00+00:00",
        "transaction_committed_at": "2026-10-02T12:01:00+00:00",
        "pointer_created_at": "2026-10-02T12:01:00+00:00",
        "serving_convergence_assessed_at": "2026-10-02T12:02:00+00:00",
        "ceremony_at": "2026-10-02T12:03:00+00:00",
    }
    body["ceremony_sha256"] = payload_digest(body)
    return body


class FakeCortex:
    checkpoint_sha256 = "a" * 64

    def generate(self, request):
        return CortexReply(
            f"reply:{request.user_text or request.intent}",
            intent=request.intent,
        )

    def close(self):
        return None


class SmokeFailCortex(FakeCortex):
    def __init__(self):
        self.closed = False

    def self_test(self):
        raise RuntimeError("neural smoke failed")

    def close(self):
        self.closed = True


def fake_factory(_identity_id, _profile_getter):
    return FakeCortex()


def test_product_preflight_checks_core_and_keeps_learning_corruption_advisory(tmp_path):
    runtime = ProductRuntime(tmp_path, cortex_factory=fake_factory)
    try:
        report = runtime.preflight()
        assert report["status"] == "PASS"
        assert report["critical_failures"] == 0
        assert report["advisory_failures"] == 0
        assert {row["name"] for row in report["critical"]} == {
            "data_dir_writable",
            "database",
            "release_assets",
        }

        probe = runtime.store.db.execute(
            "SELECT value FROM meta WHERE key=?",
            ("__product_readiness__",),
        ).fetchone()
        assert probe is None

        runtime.learning.registry_path.write_text(
            '{"schema":"corrupted"}\n',
            encoding="utf-8",
        )
        degraded = runtime.preflight()
        assert degraded["status"] == "PASS"
        assert degraded["critical_failures"] == 0
        assert degraded["advisory_failures"] == 1
        assert degraded["advisory"][0]["name"] == "learning_registry"
        assert degraded["advisory"][0]["status"] == "FAIL"

        powered = runtime.power_on()
        assert powered["phase"] == "on"
        assert powered["readiness"]["status"] == "PASS"
        assert powered["readiness"]["advisory_failures"] == 1
    finally:
        runtime.close()


def test_product_power_never_reports_on_when_live_cortex_smoke_fails(tmp_path):
    holder = {}

    def broken_smoke_factory(_identity, _profile):
        cortex = SmokeFailCortex()
        holder["cortex"] = cortex
        return cortex

    runtime = ProductRuntime(tmp_path, cortex_factory=broken_smoke_factory)
    try:
        status = runtime.power_on()
        assert status["phase"] == "error"
        assert status["powered"] is False
        assert "neural smoke failed" in status["error"]
        assert holder["cortex"].closed is True
        with pytest.raises(RuntimeError, match="not running"):
            runtime.send_message("must stay blocked")
    finally:
        runtime.close()


def test_product_preflight_blocks_default_runtime_without_release_assets(tmp_path):
    runtime = ProductRuntime(tmp_path)
    try:
        report = runtime.preflight()
        assert report["status"] == "BLOCKED"
        assert report["critical_failures"] == 1
        release = next(
            row for row in report["critical"]
            if row["name"] == "release_assets"
        )
        assert release["status"] == "FAIL"

        status = runtime.power_on()
        assert status["phase"] == "error"
        assert "product readiness blocked" in status["error"]
    finally:
        runtime.close()


def test_product_runtime_power_chat_history_and_status(tmp_path):
    runtime = ProductRuntime(
        tmp_path,
        cortex_factory=fake_factory,
    )
    try:
        assert runtime.status()["phase"] == "off"
        with pytest.raises(RuntimeError, match="not running"):
            runtime.send_message("hello")

        status = runtime.power_on()
        assert status["phase"] == "on"
        assert status["model_checkpoint_sha256"] == "a" * 64

        reply = runtime.send_message("hello")
        assert reply["reply"] == "reply:hello"
        history = runtime.history()
        assert [row["role"] for row in history] == ["user", "assistant"]
        assert [row["text"] for row in history] == ["hello", "reply:hello"]

        off = runtime.power_off()
        assert off["phase"] == "off"
    finally:
        runtime.close()


def test_memory_toggle_changes_actual_living_memory_behavior(tmp_path):
    runtime = ProductRuntime(
        tmp_path,
        cortex_factory=fake_factory,
    )
    try:
        runtime.power_on()
        runtime.update_profile({"memory_enabled": False})
        assert runtime.engine.memory_enabled is False
        assert runtime.engine.enable_rest is False
        runtime.send_message("do not memorize this")
        assert runtime.store.memories() == []

        runtime.update_profile({"memory_enabled": True})
        assert runtime.engine.memory_enabled is True
        assert runtime.engine.enable_rest is True
        runtime.send_message("remember this")
        memories = runtime.store.memories()
        assert len(memories) == 1
        assert memories[0].text == "remember this"
    finally:
        runtime.close()


def test_memory_controls_change_real_retrieval_state_and_keep_audit_private(tmp_path):
    runtime = ProductRuntime(
        tmp_path,
        cortex_factory=fake_factory,
    )
    try:
        runtime.power_on()
        runtime.send_message("I prefer concise answers")

        snapshot = runtime.status()["mind"]
        assert len(snapshot["memories"]) == 1
        memory_id = snapshot["memories"][0]["id"]

        kept = runtime.memory_action("keep", memory_id)
        assert kept["memory"]["kept"] is True
        stored = runtime.store.memory_by_ids([memory_id])[memory_id]
        assert stored.metadata["user_kept"] is True
        assert stored.salience >= 0.85

        edited = runtime.memory_action(
            "edit",
            memory_id,
            text="I prefer concise, natural answers",
        )
        assert edited["memory"]["text"] == "I prefer concise, natural answers"
        assert runtime.store.memory_by_ids([memory_id])[memory_id].text == (
            "I prefer concise, natural answers"
        )

        deleted = runtime.memory_action("delete", memory_id)
        assert deleted["memory"]["deleted"] is True
        assert runtime.store.memories() == []
        assert runtime.status()["mind"]["memories"] == []

        rows = runtime.store.db.execute(
            "SELECT action,before_sha256,after_sha256 FROM memory_controls ORDER BY at"
        ).fetchall()
        assert [row["action"] for row in rows] == ["keep", "edit", "delete"]
        receipt_text = json.dumps(
            [dict(row) for row in rows],
            ensure_ascii=False,
        )
        assert "concise" not in receipt_text
    finally:
        runtime.close()


def test_product_profile_persists_and_normalizes(tmp_path):
    store = ProductProfileStore(tmp_path / "profile.json")
    saved = store.update(
        {
            "preferred_name": " Tài ",
            "language": "vi",
            "response_length": "compact",
            "conversation_style": "direct",
            "initiative": "active",
            "memory_enabled": False,
            "personal_instruction": "Nói tự nhiên.",
        }
    )
    assert saved.preferred_name == "Tài"
    loaded = store.load()
    assert loaded.preferred_name == "Tài"
    assert loaded.language == "vi"
    assert loaded.memory_enabled is False
    payload = json.loads((tmp_path / "profile.json").read_text("utf-8"))
    assert len(payload["digest"]) == 64


def test_product_runtime_surfaces_model_start_failure(tmp_path):
    def broken(_identity, _profile):
        raise FileNotFoundError("missing release model")

    runtime = ProductRuntime(tmp_path, cortex_factory=broken)
    try:
        status = runtime.power_on()
        assert status["phase"] == "error"
        assert "missing release model" in status["error"]
        with pytest.raises(RuntimeError, match="not running"):
            runtime.send_message("hello")
    finally:
        runtime.close()


def request(server, method, path, payload=None, token=None):
    host, port = server.server_address[:2]
    conn = HTTPConnection(host, port, timeout=2)
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


def test_product_http_server_requires_configured_auth_token(tmp_path):
    runtime = ProductRuntime(tmp_path, cortex_factory=fake_factory)
    server = ProductHTTPServer(
        ("127.0.0.1", 0),
        runtime,
        auth_token="secret-token",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, _ = request(server, "GET", "/v1/status")
        assert status == 401
        status, data = request(
            server,
            "GET",
            "/v1/status",
            token="secret-token",
        )
        assert status == 200
        assert data["phase"] == "off"

        status, data = request(
            server,
            "POST",
            "/v1/power",
            {"enabled": True},
            token="secret-token",
        )
        assert status == 200
        assert data["phase"] == "on"
    finally:
        server.shutdown()
        server.server_close()
        runtime.close()


def test_authenticated_http_chat_and_history_are_cross_thread_safe(tmp_path):
    runtime = ProductRuntime(tmp_path, cortex_factory=fake_factory)
    server = ProductHTTPServer(
        ("127.0.0.1", 0),
        runtime,
        auth_token="thread-safe-secret",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, powered = request(
            server,
            "POST",
            "/v1/power",
            {"enabled": True},
            token="thread-safe-secret",
        )
        assert status == 200
        assert powered["phase"] == "on"

        status, reply = request(
            server,
            "POST",
            "/v1/chat",
            {"text": "hello through HTTP worker"},
            token="thread-safe-secret",
        )
        assert status == 200
        assert reply["reply"] == "reply:hello through HTTP worker"

        status, history = request(
            server,
            "GET",
            "/v1/history",
            token="thread-safe-secret",
        )
        assert status == 200
        assert [row["role"] for row in history["messages"]] == [
            "user",
            "assistant",
        ]
        assert history["messages"][0]["text"] == "hello through HTTP worker"
    finally:
        server.shutdown()
        server.server_close()
        runtime.close()


def test_authenticated_readiness_endpoint_reports_preflight(tmp_path):
    runtime = ProductRuntime(tmp_path, cortex_factory=fake_factory)
    server = ProductHTTPServer(
        ("127.0.0.1", 0),
        runtime,
        auth_token="readiness-secret",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, _ = request(server, "GET", "/v1/readiness")
        assert status == 401

        status, payload = request(
            server,
            "GET",
            "/v1/readiness",
            token="readiness-secret",
        )
        assert status == 200
        assert payload["status"] == "PASS"
        assert payload["critical_failures"] == 0
        assert payload["schema"] == "NOLANE-PRODUCT-READINESS-V1"
    finally:
        server.shutdown()
        server.server_close()
        runtime.close()


def test_conversation_history_is_separate_from_memory_policy(tmp_path):
    runtime = ProductRuntime(tmp_path, cortex_factory=fake_factory)
    try:
        runtime.power_on()
        runtime.update_profile({"memory_enabled": False})
        runtime.send_message("visible transcript")
        assert runtime.store.memories() == []
        history = runtime.history()
        assert history[0]["text"] == "visible transcript"
        assert history[1]["role"] == "assistant"
    finally:
        runtime.close()



def test_product_preflight_rehashes_checkpoint_when_file_changes(tmp_path):
    checkpoint = tmp_path / "factorized-nolane.pt"
    checkpoint.write_bytes(b"release-checkpoint")
    actual_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    ceremony_path = tmp_path / "promotion-ceremony.json"
    ceremony_path.write_text(
        json.dumps(release_ceremony(actual_sha)),
        encoding="utf-8",
    )
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer_config.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (tokenizer / "tokenizer.json").write_text(
        "{}",
        encoding="utf-8",
    )

    runtime = ProductRuntime(
        tmp_path / "data",
        checkpoint=checkpoint,
        tokenizer_path=tokenizer,
        release_ceremony=ceremony_path,
    )
    try:
        first = runtime.preflight()
        assert first["status"] == "PASS"

        checkpoint.write_bytes(b"release-checkpoint-mutated-and-longer")
        second = runtime.preflight()
        assert second["status"] == "BLOCKED"
        release = next(
            row for row in second["critical"]
            if row["name"] == "release_assets"
        )
        assert release["status"] == "FAIL"
        assert "does not match COMPLETE ceremony" in release["detail"]
    finally:
        runtime.close()


def test_product_runtime_reverifies_release_ceremony_before_startup(tmp_path):
    checkpoint = tmp_path / "factorized-nolane.pt"
    checkpoint.write_bytes(b"release-checkpoint")
    actual_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    ceremony_path = tmp_path / "promotion-ceremony.json"
    ceremony_path.write_text(
        json.dumps(release_ceremony(actual_sha)),
        encoding="utf-8",
    )

    runtime = ProductRuntime(
        tmp_path / "data-ok",
        checkpoint=checkpoint,
        release_ceremony=ceremony_path,
        cortex_factory=fake_factory,
    )
    runtime.close()

    ceremony_path.write_text(
        json.dumps(release_ceremony("f" * 64)),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match="does not match COMPLETE ceremony",
    ):
        ProductRuntime(
            tmp_path / "data-bad",
            checkpoint=checkpoint,
            release_ceremony=ceremony_path,
            cortex_factory=fake_factory,
        )
