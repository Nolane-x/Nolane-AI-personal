from __future__ import annotations

import hashlib
import json
import os
import re
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
    "compact": 128,
    "balanced": 256,
    "expansive": 512,
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

    @staticmethod
    def _normalized_text(value: str) -> str:
        return " ".join(str(value).strip().casefold().split())

    @classmethod
    def _requested_exact_reply(cls, user_text: str | None) -> str | None:
        raw = str(user_text or "").strip()
        if not raw:
            return None
        patterns = (
            r"(?is)\bchỉ\s+trả\s+lời\s+đúng(?:\s+[^:\s]+){0,6}\s*:\s*(.+?)\s*$",
            r"(?is)\b(?:reply|answer)\s+(?:with\s+)?exactly(?:\s+[^:\s]+){0,6}\s*:\s*(.+?)\s*$",
        )
        for pattern in patterns:
            match = re.search(pattern, raw)
            if match is None:
                continue
            candidate = match.group(1).strip().strip("\"'“”‘’")
            if candidate and len(candidate) <= 240 and "\n" not in candidate:
                return candidate
        return None

    @classmethod
    def _simple_relation_contract(
        cls,
        user_text: str | None,
    ) -> tuple[str, str, tuple[str, ...]] | None:
        raw = str(user_text or "").strip()
        if not raw:
            return None
        name = r"([^\W\d_][\w-]{0,79})"
        adjectives = (
            "cao", "thấp", "lớn", "nhỏ", "nặng", "nhẹ",
            "nhanh", "chậm", "già", "trẻ", "dài", "ngắn",
            "mạnh", "yếu", "đắt", "rẻ",
        )
        for adjective in adjectives:
            if re.search(
                rf"(?iu)\b{re.escape(adjective)}\s+nhất\b",
                raw,
            ) is None:
                continue
            pairs = re.findall(
                rf"(?iu)\b{name}\s+{re.escape(adjective)}\s+hơn\s+{name}\b",
                raw,
            )
            if len(pairs) < 2:
                continue
            display: dict[str, str] = {}
            sources: set[str] = set()
            targets: set[str] = set()
            for left, right in pairs:
                left_key = left.casefold()
                right_key = right.casefold()
                display.setdefault(left_key, left)
                display.setdefault(right_key, right)
                sources.add(left_key)
                targets.add(right_key)
            roots = sorted(sources - targets)
            if len(roots) != 1:
                continue
            participants = tuple(
                display[key]
                for key in sorted(display)
            )
            return display[roots[0]], adjective, participants
        return None

    @classmethod
    def _relation_reply_is_correct(
        cls,
        *,
        text: str,
        contract: tuple[str, str, tuple[str, ...]],
    ) -> bool:
        expected, adjective, participants = contract
        stripped = text.strip()
        expected_re = re.escape(expected)
        adjective_re = re.escape(adjective)
        bare_expected = (
            re.sub(r"[.!?]+$", "", stripped).strip().casefold()
            == expected.casefold()
        )
        explicit_expected = re.search(
            rf"(?iu)\b{expected_re}\b.{{0,28}}\b{adjective_re}\s+nhất\b",
            stripped,
        ) is not None
        wrong_extreme = any(
            participant.casefold() != expected.casefold()
            and re.search(
                rf"(?iu)\b{re.escape(participant)}\b.{{0,28}}"
                rf"\b{adjective_re}\s+nhất\b",
                stripped,
            ) is not None
            for participant in participants
        )
        expected_negated = re.search(
            rf"(?iu)\b{expected_re}\b.{{0,16}}\bkhông\b.{{0,16}}"
            rf"\b{adjective_re}\s+nhất\b",
            stripped,
        ) is not None
        return (
            (bare_expected or explicit_expected)
            and not wrong_extreme
            and not expected_negated
        )

    @classmethod
    def _quality_issues(
        cls,
        *,
        profile: ProductProfile,
        request: CortexRequest,
        text: str,
    ) -> list[str]:
        issues: list[str] = []
        normalized = cls._normalized_text(text)
        user_text = cls._normalized_text(request.user_text or "")

        if not normalized:
            issues.append("empty")
        if user_text and normalized == user_text:
            issues.append("user_echo")

        exact_reply = cls._requested_exact_reply(request.user_text)
        if exact_reply is not None and text.strip() != exact_reply:
            issues.append("exact_reply_mismatch")

        relation_contract = cls._simple_relation_contract(
            request.user_text
        )
        if (
            relation_contract is not None
            and not cls._relation_reply_is_correct(
                text=text,
                contract=relation_contract,
            )
        ):
            issues.append("simple_relation_inconsistent")

        lowered = text.casefold()
        leaked_markers = (
            "personalization:",
            "runtime state:",
            "relevant memories:",
            "requested_intent=",
            "preferred_name=",
            "assistant_name=",
        )
        if any(marker in lowered for marker in leaked_markers):
            issues.append("runtime_context_leak")

        recall_prompts = (
            "vừa nói",
            "vừa bảo",
            "vừa nhắc",
            "tôi đã nói",
            "mình đã nói",
            "what did i just",
            "what did i say",
            "i just said",
            "i just told",
        )
        if (
            any(marker in user_text for marker in recall_prompts)
            and text.strip().endswith("?")
        ):
            issues.append("recall_askback")

        capability_prompts = (
            "bạn làm được gì",
            "bạn có thể làm gì",
            "what can you do",
            "what are you capable of",
        )
        if any(marker in user_text for marker in capability_prompts):
            weak_backoffs = (
                "bạn đang cần gì",
                "bạn muốn khám phá",
                "có thể nói thêm",
                "what do you need",
            )
            if len(text.strip()) < 45:
                issues.append("capability_too_thin")
            if any(marker in lowered for marker in weak_backoffs):
                issues.append("capability_askback")

        explicit_language_override = any(
            marker in user_text
            for marker in (
                "tiếng anh",
                "tiếng nhật",
                "tiếng hàn",
                "tiếng trung",
                "tiếng pháp",
                "tiếng đức",
                "tiếng tây ban nha",
                "english",
                "japanese",
                "korean",
                "chinese",
                "french",
                "german",
                "spanish",
                "reply in ",
                "answer in ",
                "speak in ",
            )
        )
        vietnamese_signals = (
            " bạn ",
            " mình ",
            " tôi ",
            " có ",
            " là ",
            " và ",
            " được ",
            " không",
            " giúp",
            " của ",
            " với ",
            " cho ",
            " trong ",
        )
        vietnamese_diacritics = set(
            "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệ"
            "íìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
        )
        if (
            profile.language == "vi"
            and not explicit_language_override
            and len(text.strip()) >= 20
            and not any(marker in f" {lowered} " for marker in vietnamese_signals)
            and not any(char in vietnamese_diacritics for char in lowered)
        ):
            issues.append("vietnamese_lock_suspect")

        return issues

    def generate(self, request: CortexRequest) -> CortexReply:
        profile = self.profile_getter()
        messages = build_product_messages(profile, request)
        text = self._chat(
            messages,
            max_tokens=_LENGTH_TOKENS[profile.response_length],
            temperature=0.45,
            top_p=0.9,
        )
        issues = self._quality_issues(
            profile=profile,
            request=request,
            text=text,
        )
        exact_reply = self._requested_exact_reply(request.user_text)
        relation_contract = self._simple_relation_contract(
            request.user_text
        )
        if issues:
            repair_messages = [dict(message) for message in messages]
            repair_messages[0] = {
                "role": "system",
                "content": (
                    repair_messages[0]["content"]
                    + "\n\nQuality repair is required for this turn. "
                    + "The previous draft failed these checks: "
                    + ", ".join(issues)
                    + ". Regenerate the answer from scratch. Answer the "
                    + "actual user request directly, completely, naturally, "
                    + "and without mentioning this quality check."
                    + (
                        ""
                        if exact_reply is None
                        else "\nThe user required an exact literal reply. "
                        + "Output exactly this text and nothing else: "
                        + json.dumps(exact_reply, ensure_ascii=False)
                    )
                    + (
                        ""
                        if "recall_askback" not in issues
                        else "\nThis is a recent-turn recall question. "
                        + "Read the recent role-aware conversation history, "
                        + "recover the information the user already stated, "
                        + "and answer it directly. Do not ask the recall "
                        + "question back to the user."
                    )
                    + (
                        ""
                        if "simple_relation_inconsistent" not in issues
                        else "\nThis is a simple transitive relation problem. "
                        + "Preserve every 'X ... hơn Y' direction exactly, "
                        + "derive the ordering again, and answer the requested "
                        + "extreme without reversing the relation."
                    )
                ),
            }
            repair_temperature = (
                0.0 if "recall_askback" in issues else 0.15
            )
            repaired = self._chat(
                repair_messages,
                max_tokens=_LENGTH_TOKENS[profile.response_length],
                temperature=repair_temperature,
                top_p=0.9,
            )
            repaired_issues = self._quality_issues(
                profile=profile,
                request=request,
                text=repaired,
            )
            if len(repaired_issues) < len(issues):
                text = repaired
            elif exact_reply is not None and "exact_reply_mismatch" in issues:
                # An explicit exact-output instruction is a deterministic
                # formatting contract. Do not let personalization add names,
                # punctuation, explanations or other extra text.
                text = exact_reply
            elif (
                relation_contract is not None
                and "simple_relation_inconsistent" in issues
            ):
                expected, adjective, _participants = relation_contract
                # A tiny deterministic relation kernel is safer than allowing
                # a small language model to reverse a clearly stated chain.
                text = f"{expected} {adjective} nhất."
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
