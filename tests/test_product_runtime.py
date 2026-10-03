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


def fake_factory(_identity_id, _profile_getter):
    return FakeCortex()


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


def test_product_preflight_is_ready_for_custom_cortex_runtime(tmp_path):
    runtime = ProductRuntime(tmp_path, cortex_factory=fake_factory)
    try:
        receipt = runtime.preflight()
        assert receipt["status"] == "ready"
        assert receipt["reasons"] == []
        assert receipt["checks"]["data_dir"]["status"] == "pass"
        assert receipt["checks"]["conversation_store"]["status"] == "pass"
        assert receipt["checks"]["learning_registry"]["status"] == "pass"
        assert receipt["checks"]["release_binding"]["status"] == "not_applicable"
        assert receipt["checks"]["tokenizer_assets"]["status"] == "not_applicable"
    finally:
        runtime.close()


def test_release_preflight_blocks_incomplete_tokenizer_assets(tmp_path):
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

    runtime = ProductRuntime(
        tmp_path / "data",
        checkpoint=checkpoint,
        tokenizer_path=tokenizer,
        release_ceremony=ceremony_path,
    )
    try:
        receipt = runtime.preflight()
        assert receipt["status"] == "blocked"
        assert receipt["reasons"] == ["tokenizer_assets"]
        assert receipt["checks"]["release_binding"]["status"] == "pass"
        assert receipt["checks"]["tokenizer_assets"]["status"] == "blocked"

        (tokenizer / "tokenizer.json").write_text(
            "{}",
            encoding="utf-8",
        )
        ready = runtime.preflight()
        assert ready["status"] == "ready"
        assert ready["checks"]["tokenizer_assets"]["status"] == "pass"
    finally:
        runtime.close()


def test_preflight_blocks_corrupt_learning_registry_without_crashing_runtime(tmp_path):
    runtime = ProductRuntime(tmp_path, cortex_factory=fake_factory)
    try:
        registry = runtime.learning.registry_path
        payload = json.loads(registry.read_text(encoding="utf-8"))
        payload["next_window_index"] = 99
        registry.write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

        receipt = runtime.preflight()
        assert receipt["status"] == "blocked"
        assert "learning_registry" in receipt["reasons"]
        assert receipt["checks"]["learning_registry"]["status"] == "blocked"
        assert runtime.status()["phase"] == "off"
    finally:
        runtime.close()


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


def test_product_preflight_endpoint_is_authenticated(tmp_path):
    runtime = ProductRuntime(tmp_path, cortex_factory=fake_factory)
    server = ProductHTTPServer(
        ("127.0.0.1", 0),
        runtime,
        auth_token="preflight-secret",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, _ = request(server, "GET", "/v1/preflight")
        assert status == 401
        status, receipt = request(
            server,
            "GET",
            "/v1/preflight",
            token="preflight-secret",
        )
        assert status == 200
        assert receipt["status"] == "ready"
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
