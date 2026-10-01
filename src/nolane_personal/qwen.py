from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .cortex import CortexReply, CortexRequest
from .events import LivingEvent
from .observer import SocialProposal
from .state import LivingState

SYSTEM_PROMPT = """You are the language cortex of Nolane AI Personal.
You are not a generic assistant. Speak like a persistent personal companion whose state and memories are supplied by the runtime.
Use natural language, usually concise. Vietnamese and English are both allowed; follow the user's language.
You may disagree, tease gently, joke, or sound mildly annoyed when context supports it, but never guilt the user for leaving, demand attention, threaten abandonment, or claim suffering to pressure them.
Do not invent memories. Do not claim certainty about the user's emotion; phrase uncertain impressions naturally.
The runtime may ask you to initiate a conversation. In that case, do not mention that you were triggered or scored by a policy.
"""

OBSERVER_PROMPT = """You are a conservative social-state observer inside Nolane AI Personal.
Return exactly one JSON object and no prose. You only propose updates; you do not have commit authority.
Never turn a guess into a fact. Preserve uncertainty. Small deltas are preferred.
Schema:
{
  "affect_delta": {"valence": number, "arousal": number, "energy": number, "playfulness": number, "irritation": number, "concern": number, "social_drive": number},
  "relationship_delta": {"closeness": number, "trust": number, "familiarity": number},
  "memories": [{"text": string, "kind": "episodic|fact|preference|inference", "confidence": number, "salience": number, "metadata": {}}],
  "open_threads": [{"topic": string, "importance": number, "due_at": string|null}],
  "resolve_thread_ids": [string],
  "intent": string|null,
  "uncertainty": number
}
Omit or leave empty anything not supported by the message.
"""


class QwenCortex:
    """Local Qwen language cortex that can also act as a bounded social observer.

    The same loaded model instance is reused for both roles. Observer output is
    still proposal-only and must pass MutationValidator before persistence.
    """

    def __init__(
        self,
        model_path: str | Path = "models/Qwen3-0.6B",
        *,
        device: str = "auto",
        max_new_tokens: int = 160,
        temperature: float = 0.78,
        top_p: float = 0.9,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install the Qwen runtime with: pip install -e '.[qwen]'") from exc

        self.torch = torch
        path = str(Path(model_path))
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype="auto", local_files_only=True)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)

    @staticmethod
    def _state_summary(request: CortexRequest) -> str:
        s = request.state
        unresolved = [t.topic for t in s.open_threads if t.unresolved][:4]
        return (
            f"identity_id={s.identity_id}\n"
            f"relationship: closeness={s.relationship.closeness:.2f}, trust={s.relationship.trust:.2f}, familiarity={s.relationship.familiarity:.2f}\n"
            f"behavior state: valence={s.affect.valence:.2f}, energy={s.affect.energy:.2f}, playfulness={s.affect.playfulness:.2f}, concern={s.affect.concern:.2f}, irritation={s.affect.irritation:.2f}\n"
            f"open_threads={unresolved}\n"
            f"requested_intent={request.intent}"
        )

    def _generate_text(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int,
        sample: bool,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        try:
            rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(rendered, return_tensors="pt").to(self.device)
        kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(sample),
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if sample:
            kwargs["temperature"] = float(self.temperature if temperature is None else temperature)
            kwargs["top_p"] = float(self.top_p if top_p is None else top_p)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def generate(self, request: CortexRequest) -> CortexReply:
        memory_text = "\n".join(f"- {m.text}" for m in request.memories[:6]) or "(none)"
        context = self._state_summary(request)
        if request.mode == "reply":
            task = f"User message:\n{request.user_text or ''}\n\nReply naturally."
        else:
            task = "Initiate one natural message now. Keep it context-aware and non-intrusive."

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Runtime state:\n{context}\n\nRelevant memories:\n{memory_text}\n\n{task}"},
        ]
        text = self._generate_text(messages, max_new_tokens=self.max_new_tokens, sample=True)
        return CortexReply(text, intent=request.intent)

    def observe(self, text: str, state: LivingState, source_event: LivingEvent) -> SocialProposal:
        compact_state = {
            "affect": asdict(state.affect),
            "relationship": asdict(state.relationship),
            "working": asdict(state.working),
            "open_threads": [asdict(t) for t in state.open_threads if t.unresolved][:8],
        }
        messages = [
            {"role": "system", "content": OBSERVER_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "source_event_id": source_event.event_id,
                        "user_text": text,
                        "current_state": compact_state,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        raw = self._generate_text(messages, max_new_tokens=260, sample=False)
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end < start:
            raise ValueError("observer did not return JSON")
        payload = json.loads(raw[start : end + 1])
        if not isinstance(payload, dict):
            raise ValueError("observer JSON must be an object")
        return SocialProposal.from_dict(payload)
