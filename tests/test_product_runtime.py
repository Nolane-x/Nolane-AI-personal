import json
import threading
from http.client import HTTPConnection

import pytest

from nolane_personal.cortex import CortexReply
from nolane_personal.product_profile import ProductProfileStore
from nolane_personal.product_runtime import ProductRuntime
from nolane_personal.product_server import ProductHTTPServer


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
