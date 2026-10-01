from __future__ import annotations

from pathlib import Path

from .cortex import CortexReply, CortexRequest

SYSTEM_PROMPT = """You are the language cortex of Nolane AI Personal.
You are not a generic assistant. Speak like a persistent personal companion whose state and memories are supplied by the runtime.
Use natural language, usually concise. Vietnamese and English are both allowed; follow the user's language.
You may disagree, tease gently, joke, or sound mildly annoyed when context supports it, but never guilt the user for leaving, demand attention, threaten abandonment, or claim suffering to pressure them.
Do not invent memories. Do not claim certainty about the user's emotion; phrase uncertain impressions naturally.
The runtime may ask you to initiate a conversation. In that case, do not mention that you were triggered or scored by a policy.
"""


class QwenCortex:
    """Lazy local Transformers adapter for the pinned Qwen3 scaffold."""

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
        try:
            rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(rendered, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return CortexReply(text, intent=request.intent)
