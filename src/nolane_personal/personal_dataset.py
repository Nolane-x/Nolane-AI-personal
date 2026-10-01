from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(slots=True)
class PersonalizationExample:
    prompt: str
    target: str
    latent: list[float] | None = None
    weight: float = 1.0
    language: str | None = None

    def validate(self, *, latent_dim: int = 32) -> None:
        if not self.prompt.strip():
            raise ValueError("personalization prompt is empty")
        if not self.target.strip():
            raise ValueError("personalization target is empty")
        if self.latent is not None and len(self.latent) != latent_dim:
            raise ValueError(f"expected {latent_dim}D latent")
        if self.weight <= 0:
            raise ValueError("example weight must be positive")


def load_jsonl(path: str | Path, *, latent_dim: int = 32) -> list[PersonalizationExample]:
    result: list[PersonalizationExample] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            example = PersonalizationExample(
                prompt=str(payload["prompt"]),
                target=str(payload["target"]),
                latent=(
                    [float(x) for x in payload["latent"]]
                    if payload.get("latent") is not None
                    else None
                ),
                weight=float(payload.get("weight", 1.0)),
                language=payload.get("language"),
            )
            try:
                example.validate(latent_dim=latent_dim)
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
            result.append(example)
    if not result:
        raise ValueError("personalization dataset is empty")
    return result


def encode_chat_example(
    tokenizer,
    example: PersonalizationExample,
    *,
    system_prompt: str,
    max_length: int = 512,
):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example.prompt},
    ]
    try:
        prefix = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prefix = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    eos = tokenizer.eos_token or ""
    target_ids = tokenizer(example.target + eos, add_special_tokens=False)["input_ids"]
    input_ids = (prefix_ids + target_ids)[: max(2, int(max_length))]
    prompt_len = min(len(prefix_ids), len(input_ids))
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    if all(label == -100 for label in labels):
        raise ValueError("max_length removed the entire target")
    return input_ids, labels
