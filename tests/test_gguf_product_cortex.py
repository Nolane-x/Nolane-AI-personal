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
    model = tmp_path / "Qwen_Qwen3.5-2B-Q4_K_M.gguf"
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
        messages = (payload or {}).get("messages", [])
        user_text = " ".join(
            str(message.get("content", ""))
            for message in messages
            if isinstance(message, dict) and message.get("role") == "user"
        )
        if "Thủ đô của nước Pháp" in user_text:
            content = "Paris"
        elif "Tên bạn là gì?" in user_text:
            content = "Nolane"
        elif "Chỉ trả lời đúng hai từ: Xin chào" in user_text:
            content = "Xin chào"
        else:
            content = "Xin chào từ Nolane."
        return {
            "choices": [
                {
                    "message": {
                        "content": content,
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
    assert chat_call["payload"]["max_tokens"] == 128
    assert chat_call["payload"]["stream"] is False
    assert chat_call["payload"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    assert "Xin chào" in chat_call["payload"]["messages"][-1]["content"]
    assert (
        "Người dùng thích câu trả lời ngắn."
        in chat_call["payload"]["messages"][0]["content"]
    )
    assert (
        "User message:"
        not in chat_call["payload"]["messages"][0]["content"]
    )

    smoke = cortex.self_test()
    assert smoke["status"] == "PASS"
    assert smoke["basic_fact_probe"] == "PASS"
    assert smoke["identity_probe"] == "PASS"
    assert smoke["vietnamese_probe"] == "PASS"
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
        "model_repo": "bartowski/Qwen_Qwen3.5-2B-GGUF",
        "model_revision": "8de6479d2743924f9dc499e3654d4e51ea0d4b9d",
        "model_source_file_commit": "8de6479d2743924f9dc499e3654d4e51ea0d4b9d",
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
    model = tmp_path / "Qwen_Qwen3.5-2B-Q4_K_M.gguf"
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
    model = tmp_path / "Qwen_Qwen3.5-2B-Q4_K_M.gguf"
    model.write_bytes(b"model")
    with pytest.raises(ValueError, match="requires model, llama-server and manifest"):
        ProductRuntime(
            tmp_path / "data",
            software_model=model,
        )


def test_quality_guard_flags_weak_capability_answer_and_runtime_leak():
    request = CortexRequest(
        mode="reply",
        intent="respond_to_user",
        user_text="Bạn làm được gì?",
        state=LivingState(identity_id="quality-test"),
    )
    profile = ProductProfile(language="vi")
    issues = gguf.GgufProductCortex._quality_issues(
        profile=profile,
        request=request,
        text="Bạn đang cần gì?",
    )
    assert "capability_too_thin" in issues
    assert "capability_askback" in issues

    leaked = gguf.GgufProductCortex._quality_issues(
        profile=profile,
        request=request,
        text="Personalization: preferred_name=Huy assistant_name=Mây",
    )
    assert "runtime_context_leak" in leaked


def test_quality_guard_retries_one_bad_draft_with_low_temperature():
    cortex = object.__new__(gguf.GgufProductCortex)
    profile = ProductProfile(
        preferred_name="Huy",
        assistant_name="Mây",
        language="vi",
        response_length="balanced",
    )
    cortex.profile_getter = lambda: profile
    replies = iter([
        "Bạn đang cần gì?",
        (
            "Mình có thể giúp bạn giải thích kiến thức, phân tích vấn đề, "
            "viết và chỉnh sửa nội dung, dịch, tóm tắt, lên kế hoạch và "
            "trò chuyện theo ngữ cảnh của bạn."
        ),
    ])
    calls = []

    def fake_chat(messages, *, max_tokens, temperature, top_p):
        calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
        return next(replies)

    cortex._chat = fake_chat
    request = CortexRequest(
        mode="reply",
        intent="respond_to_user",
        user_text="Bạn làm được gì?",
        state=LivingState(identity_id="quality-repair"),
    )
    reply = cortex.generate(request)

    assert reply.utterance.startswith("Mình có thể giúp bạn")
    assert len(calls) == 2
    assert calls[0]["temperature"] == 0.45
    assert calls[0]["max_tokens"] == 256
    assert calls[1]["temperature"] == 0.15
    assert "Quality repair is required" in calls[1]["messages"][0]["content"]


def test_quality_guard_respects_explicit_language_override():
    request = CortexRequest(
        mode="reply",
        intent="respond_to_user",
        user_text="Hãy trả lời bằng tiếng Anh: Paris là gì?",
        state=LivingState(identity_id="language-override"),
    )
    profile = ProductProfile(language="vi")
    issues = gguf.GgufProductCortex._quality_issues(
        profile=profile,
        request=request,
        text="Paris is the capital city of France.",
    )
    assert "vietnamese_lock_suspect" not in issues


def test_quality_guard_enforces_explicit_exact_literal_reply():
    request = CortexRequest(
        mode="reply",
        intent="respond_to_user",
        user_text="Chỉ trả lời đúng hai từ: Xin chào",
        state=LivingState(identity_id="exact-reply"),
    )
    profile = ProductProfile(
        preferred_name="Huy",
        assistant_name="Mây",
        language="vi",
        response_length="compact",
    )
    issues = gguf.GgufProductCortex._quality_issues(
        profile=profile,
        request=request,
        text="Xin chào Huy",
    )
    assert "exact_reply_mismatch" in issues

    cortex = object.__new__(gguf.GgufProductCortex)
    cortex.profile_getter = lambda: profile
    replies = iter(["Xin chào Huy", "Xin chào Huy!"])
    cortex._chat = lambda *args, **kwargs: next(replies)

    reply = cortex.generate(request)

    assert reply.utterance == "Xin chào"
