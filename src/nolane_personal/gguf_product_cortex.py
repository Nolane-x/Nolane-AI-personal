from __future__ import annotations

import hashlib
import json
import os
import secrets
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .cortex import CortexReply, CortexRequest
from .product_profile import ProductProfile
from .product_prompt_payload import build_product_messages


_LENGTH_TOKENS = {
    "compact": 96,
    "balanced": 160,
    "expansive": 256,
}

_SERVER_ALIAS = "nolane-qwen35-2b"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pick_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _json_request(
    url: str,
    *,
    api_key: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    data = None
    headers = {"Authorization": f"Bearer {api_key}"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"llama.cpp HTTP {exc.code}: {detail[:1000]}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"llama.cpp request failed: {exc}") from exc
    decoded = json.loads(body)
    if not isinstance(decoded, dict):
        raise RuntimeError("llama.cpp returned a non-object JSON response")
    return decoded


class GgufProductCortex:
    """Pinned local GGUF language cortex for the software release channel.

    Living state, memory, initiative, relationship growth and persistence stay
    inside Nolane. llama.cpp is only the local language-generation boundary.
    """

    def __init__(
        self,
        *,
        model_path: str | Path,
        llama_server_path: str | Path,
        profile_getter: Callable[[], ProductProfile],
        context_size: int = 4096,
        startup_timeout: float = 120.0,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.llama_server_path = Path(llama_server_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"software release GGUF model not found: {self.model_path}"
            )
        if not self.llama_server_path.is_file():
            raise FileNotFoundError(
                "software release llama-server executable not found: "
                f"{self.llama_server_path}"
            )

        self.profile_getter = profile_getter
        self.checkpoint_sha256 = _sha256_file(self.model_path)
        self.port = _pick_loopback_port()
        self.api_key = secrets.token_hex(32)
        self.endpoint = f"http://127.0.0.1:{self.port}"
        self._closed = False

        command = [
            str(self.llama_server_path),
            "-m",
            str(self.model_path),
            "--alias",
            _SERVER_ALIAS,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "-c",
            str(max(2048, int(context_size))),
            "-ngl",
            "0",
            "--reasoning",
            "off",
            "--no-webui",
            "--api-key",
            self.api_key,
            "--sleep-idle-seconds",
            "300",
        ]
        creationflags = 0x08000000 if os.name == "nt" else 0
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        self._wait_until_ready(float(startup_timeout))

    def _wait_until_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    "llama.cpp exited before the software model became ready"
                )
            try:
                health = _json_request(
                    f"{self.endpoint}/health",
                    api_key=self.api_key,
                    timeout=1.0,
                )
                if health.get("status") == "ok":
                    return
            except Exception as exc:
                last_error = exc
            time.sleep(0.2)
        self.close()
        suffix = "" if last_error is None else f": {last_error}"
        raise RuntimeError(
            "timed out waiting for local software model readiness" + suffix
        )

    def _chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        payload = {
            "model": _SERVER_ALIAS,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        response = _json_request(
            f"{self.endpoint}/v1/chat/completions",
            api_key=self.api_key,
            payload=payload,
            timeout=180.0,
        )
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("llama.cpp chat response contains no choices")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise RuntimeError("llama.cpp chat response is missing message")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("llama.cpp chat response is empty")
        return content.strip()

    def generate(self, request: CortexRequest) -> CortexReply:
        profile = self.profile_getter()
        messages = build_product_messages(profile, request)
        text = self._chat(
            messages,
            max_tokens=_LENGTH_TOKENS[profile.response_length],
            temperature=0.55,
            top_p=0.9,
        )
        return CortexReply(text, intent=request.intent)

    def self_test(self) -> dict[str, object]:
        text = self._chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a local readiness probe. Reply with a very "
                        "short acknowledgement and no reasoning."
                    ),
                },
                {"role": "user", "content": "ready?"},
            ],
            max_tokens=8,
            temperature=0.0,
            top_p=1.0,
        )
        factual = self._chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Answer the factual question directly. No reasoning, "
                        "no explanation, no extra text."
                    ),
                },
                {
                    "role": "user",
                    "content": "Thủ đô của nước Pháp là gì?",
                },
            ],
            max_tokens=8,
            temperature=0.0,
            top_p=1.0,
        )
        if "paris" not in factual.casefold():
            raise RuntimeError(
                "semantic smoke failed: basic factual probe did not answer Paris"
            )

        identity = self._chat(
            [
                {
                    "role": "system",
                    "content": (
                        "The user's name is Huy. Your assistant name is Nolane. "
                        "Reply with only your assistant name."
                    ),
                },
                {"role": "user", "content": "Tên bạn là gì?"},
            ],
            max_tokens=8,
            temperature=0.0,
            top_p=1.0,
        )
        identity_folded = identity.casefold()
        if "nolane" not in identity_folded or "huy" in identity_folded:
            raise RuntimeError(
                "semantic smoke failed: assistant/user identity was confused"
            )

        vietnamese = self._chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Reply in Vietnamese. Follow the user's exact wording "
                        "when they request an exact short answer."
                    ),
                },
                {
                    "role": "user",
                    "content": "Chỉ trả lời đúng hai từ: Xin chào",
                },
            ],
            max_tokens=8,
            temperature=0.0,
            top_p=1.0,
        )
        if "xin chào" not in vietnamese.casefold():
            raise RuntimeError(
                "semantic smoke failed: Vietnamese language lock was not obeyed"
            )

        return {
            "status": "PASS",
            "reply_nonempty": bool(text.strip()),
            "basic_fact_probe": "PASS",
            "identity_probe": "PASS",
            "vietnamese_probe": "PASS",
            "checkpoint_sha256": self.checkpoint_sha256,
            "device": "cpu",
            "runtime": "llama.cpp",
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = getattr(self, "process", None)
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
