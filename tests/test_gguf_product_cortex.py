from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import nolane_personal.gguf_product_cortex as gguf
from nolane_personal.cortex import CortexRequest
from nolane_personal.memory import MemoryRecord
from nolane_personal.product_profile import ProductProfile
from nolane_personal.product_runtime import ProductRuntime
from nolane_personal.state import LivingState


class FakeProcess:
    def __init__(self, command):
        self.command = list(command)
        self.returncode = None
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.killed = True
        self.returncode = -9


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gguf_product_cortex_uses_local_authenticated_llama_server(
    tmp_path,
    monkeypatch,
):
    model = tmp_path / "Qwen3-0.6B-Q8_0.gguf"
    model.write_bytes(b"pinned-qwen-gguf")
    server = tmp_path / "llama-server.exe"
    server.write_bytes(b"llama-server")

    spawned = {}

    def fake_popen(command, **kwargs):
        spawned["command"] = list(command)
        spawned["kwargs"] = dict(kwargs)
        spawned["process"] = FakeProcess(command)
        return spawned["process"]

    calls = []

    def fake_json(url, *, api_key, payload=None, timeout=30.0):
        calls.append(
            {
                "url": url,
                "api_key": api_key,
                "payload": payload,
                "timeout": timeout,
            }
        )
        if url.endswith("/health"):
            return {"status": "ok"}
        return {
            "choices": [
                {
                    "message": {
                        "content": "Xin chào từ Nolane.",
                    }
                }
            ]
        }

    monkeypatch.setattr(gguf, "_pick_loopback_port", lambda: 49152)
    monkeypatch.setattr(gguf.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(gguf, "_json_request", fake_json)

    profile = ProductProfile(
        language="vi",
        response_length="compact",
        conversation_style="natural",
    )
    cortex = gguf.GgufProductCortex(
        model_path=model,
        llama_server_path=server,
        profile_getter=lambda: profile,
    )

    command = spawned["command"]
    assert command[0] == str(server.resolve())
    assert command[command.index("-m") + 1] == str(model.resolve())
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--reasoning") + 1] == "off"
    assert "--no-webui" in command
    assert "--api-key" in command

    request = CortexRequest(
        mode="reply",
        intent="respond_to_user",
        user_text="Xin chào",
        state=LivingState(identity_id="gguf-test"),
        memories=[MemoryRecord(text="Người dùng thích câu trả lời ngắn.")],
    )
    reply = cortex.generate(request)
    assert reply.utterance == "Xin chào từ Nolane."

    chat_call = next(
        row
        for row in calls
        if row["url"].endswith("/v1/chat/completions")
    )
    assert chat_call["payload"]["max_tokens"] == 96
    assert chat_call["payload"]["stream"] is False
    assert chat_call["payload"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    assert "Xin chào" in chat_call["payload"]["messages"][1]["content"]
    assert "Người dùng thích câu trả lời ngắn." in chat_call["payload"]["messages"][1]["content"]

    smoke = cortex.self_test()
    assert smoke["status"] == "PASS"
    assert smoke["checkpoint_sha256"] == sha256(model)
    assert smoke["runtime"] == "llama.cpp"

    cortex.close()
    assert spawned["process"].terminated is True


def software_manifest(model: Path, server: Path) -> dict:
    return {
        "schema": "NOLANE-V100-WINDOWS-SOFTWARE-RELEASE-V1",
        "authority": "CI_SOFTWARE_RELEASE_PINNED_UPSTREAM_RUNTIME",
        "product_version": "1.0.0",
        "runtime_channel": "software-v1-gguf",
        "model_repo": "Qwen/Qwen3-0.6B-GGUF",
        "model_revision": "main",
        "model_source_file_commit": "1eaf4d9657fe65ad10a51eab76a8db5b363bddaa",
        "model_filename": model.name,
        "model_sha256": sha256(model),
        "llama_cpp_repo": "ggml-org/llama.cpp",
        "llama_cpp_tag": "b11379",
        "llama_server_sha256": sha256(server),
        "llama_runtime_tree_sha256": "0" * 64,
        "runtime_executable_sha256": "1" * 64,
        "windows_one_click_prerequisites_bundled": True,
        "release_claims": {
            "same_sha_ci_software_release": True,
            "l36_certified": False,
            "physical_device_certified": False,
        },
    }


def test_product_runtime_accepts_hash_bound_software_release_assets(tmp_path):
    model = tmp_path / "Qwen3-0.6B-Q8_0.gguf"
    model.write_bytes(b"software-model")
    server = tmp_path / "llama-server.exe"
    server.write_bytes(b"software-server")
    manifest = tmp_path / "software-release.json"
    manifest.write_text(
        json.dumps(software_manifest(model, server)),
        encoding="utf-8",
    )

    runtime = ProductRuntime(
        tmp_path / "data",
        software_model=model,
        llama_server=server,
        software_manifest=manifest,
    )
    try:
        readiness = runtime.preflight()
        assert readiness["status"] == "PASS"
        status = runtime.status()
        assert status["runtime_channel"] == "software-v1-gguf"

        model.write_bytes(b"tampered-model")
        blocked = runtime.preflight()
        assert blocked["status"] == "BLOCKED"
        release_row = next(
            row
            for row in blocked["critical"]
            if row["name"] == "release_assets"
        )
        assert "digest mismatch" in release_row["detail"]
    finally:
        runtime.close()


def test_product_runtime_rejects_partial_software_mode(tmp_path):
    model = tmp_path / "Qwen3-0.6B-Q8_0.gguf"
    model.write_bytes(b"model")
    with pytest.raises(ValueError, match="requires model, llama-server and manifest"):
        ProductRuntime(
            tmp_path / "data",
            software_model=model,
        )
