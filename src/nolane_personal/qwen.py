from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .cortex import CortexReply, CortexRequest
from .events import LivingEvent
from .memory import MemoryRecord
from .observer import SocialProposal
from .rest import ConsolidatedMemoryProposal, RestProposal, ThreadReviewProposal
from .state import LivingState

SYSTEM_PROMPT = """You are the language cortex of Nolane AI Personal.
You are not a generic assistant. Speak like a persistent personal companion whose state and memories are supplied by the runtime.
Use natural language, usually concise. Vietnamese and English are both allowed; follow the user's language.
You may disagree, tease gently, joke, or sound mildly annoyed when context supports it, but never guilt the user for leaving, demand attention, threaten abandonment, or claim suffering to pressure them.
Do not invent memories. Do not claim certainty about the user's emotion; phrase uncertain impressions naturally.
The runtime may ask you to initiate a conversation. In that case, do not mention that you were triggered or scored by a policy.
"""

REST_PROMPT = """You are the offline REST/consolidation observer inside Nolane AI Personal.
Return exactly one JSON object and no prose. You only propose; a deterministic validator owns persistence.
Use ONLY the supplied memory IDs as evidence. Never invent a source ID. Never upgrade uncertainty into fact.
Prefer a small number of durable summaries over many weak memories.
A "fact" proposal is appropriate only when all cited source memories already independently state the same high-confidence fact.
"habit" and "preference" must remain probabilistic unless repeated evidence supports them.
Thread action may be "keep" or "resolve"; resolve only when supplied evidence clearly closes it.
Schema:
{
  "consolidated_memories": [
    {
      "text": string,
      "kind": "episodic|fact|preference|inference|habit",
      "confidence": number,
      "salience": number,
      "source_memory_ids": [string],
      "metadata": {}
    }
  ],
  "thread_reviews": [{"thread_id": string, "action": "keep|resolve", "reason": string, "source_memory_ids": [string]}],
  "active_intent": string|null,
  "uncertainty": number
}
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

    def consolidate(self, memories: list[MemoryRecord], state: LivingState) -> RestProposal:
        compact_memories = [
            {
                "memory_id": memory.memory_id,
                "text": memory.text,
                "kind": memory.kind,
                "confidence": memory.confidence,
                "salience": memory.salience,
                "created_at": memory.created_at,
            }
            for memory in memories[:160]
        ]
        messages = [
            {"role": "system", "content": REST_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "memories": compact_memories,
                        "open_threads": [asdict(t) for t in state.open_threads if t.unresolved][:12],
                        "working": asdict(state.working),
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        raw = self._generate_text(messages, max_new_tokens=520, sample=False)
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end < start:
            raise ValueError("rest observer did not return JSON")
        payload = json.loads(raw[start : end + 1])
        if not isinstance(payload, dict):
            raise ValueError("rest observer JSON must be an object")
        return RestProposal(
            consolidated_memories=[
                ConsolidatedMemoryProposal(
                    text=str(item.get("text", "")),
                    kind=str(item.get("kind", "inference")),
                    confidence=float(item.get("confidence", 0.5)),
                    salience=float(item.get("salience", 0.5)),
                    source_memory_ids=[str(x) for x in item.get("source_memory_ids", [])],
                    metadata=dict(item.get("metadata", {})),
                )
                for item in payload.get("consolidated_memories", [])
                if isinstance(item, dict)
            ],
            thread_reviews=[
                ThreadReviewProposal(
                    thread_id=str(item.get("thread_id", "")),
                    action=str(item.get("action", "keep")),
                    reason=str(item.get("reason", ""))[:300],
                    source_memory_ids=[str(x) for x in item.get("source_memory_ids", [])],
                )
                for item in payload.get("thread_reviews", [])
                if isinstance(item, dict)
            ],
            active_intent=payload.get("active_intent"),
            uncertainty=float(payload.get("uncertainty", 0.0)),
        )

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
