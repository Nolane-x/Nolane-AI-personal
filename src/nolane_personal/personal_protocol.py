from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .personal_dataset import PersonalizationExample
from .store import payload_digest


SCHEMA = "NOLANE-PERSONALIZATION-PROTOCOL-V1"


@dataclass(slots=True)
class PersonalizationSplitPolicy:
    train_fraction: float = 0.70
    dev_fraction: float = 0.15
    min_examples: int = 6

    def validate(self) -> None:
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be in (0,1)")
        if not 0.0 < self.dev_fraction < 1.0:
            raise ValueError("dev_fraction must be in (0,1)")
        if self.train_fraction + self.dev_fraction >= 1.0:
            raise ValueError("train+dev must leave room for test")
        if self.min_examples < 3:
            raise ValueError("min_examples must be >= 3")


def _canonical_example(example: PersonalizationExample, index: int) -> dict[str, Any]:
    example.validate()
    payload = {
        "index": int(index),
        "prompt": example.prompt,
        "target": example.target,
        "latent": example.latent,
        "weight": float(example.weight),
        "language": example.language,
    }
    payload["example_sha256"] = payload_digest(payload)
    return payload


def _split_counts(n: int, policy: PersonalizationSplitPolicy) -> tuple[int, int, int]:
    train = max(1, int(n * policy.train_fraction))
    dev = max(1, int(n * policy.dev_fraction))
    if train + dev >= n:
        dev = 1
        train = n - 2
    test = n - train - dev
    if test < 1:
        raise ValueError("not enough examples for non-empty test split")
    return train, dev, test


def build_personalization_protocol(
    examples: list[PersonalizationExample],
    *,
    dataset_sha256: str,
    policy: PersonalizationSplitPolicy | None = None,
) -> dict[str, Any]:
    policy = policy or PersonalizationSplitPolicy()
    policy.validate()
    if len(examples) < policy.min_examples:
        raise ValueError(f"need at least {policy.min_examples} examples")
    train_n, dev_n, _ = _split_counts(len(examples), policy)
    rows = [_canonical_example(example, i) for i, example in enumerate(examples)]
    protocol = {
        "schema": SCHEMA,
        "dataset_sha256": str(dataset_sha256),
        "policy": asdict(policy),
        "counts": {
            "total": len(rows),
            "train": train_n,
            "dev": dev_n,
            "test": len(rows) - train_n - dev_n,
        },
        "splits": {
            "train": rows[:train_n],
            "dev": rows[train_n:train_n + dev_n],
            "test": rows[train_n + dev_n:],
        },
    }
    protocol["protocol_sha256"] = payload_digest(protocol)
    return protocol


def verify_personalization_protocol(protocol: dict[str, Any], *, dataset_sha256: str) -> None:
    if protocol.get("schema") != SCHEMA:
        raise ValueError("unsupported personalization protocol schema")
    supplied = protocol.get("protocol_sha256")
    body = dict(protocol)
    body.pop("protocol_sha256", None)
    if payload_digest(body) != supplied:
        raise ValueError("personalization protocol digest mismatch")
    if protocol.get("dataset_sha256") != dataset_sha256:
        raise ValueError("personalization dataset digest mismatch")

    rows = (
        list(protocol["splits"]["train"])
        + list(protocol["splits"]["dev"])
        + list(protocol["splits"]["test"])
    )
    for row in rows:
        check = dict(row)
        supplied_row = check.pop("example_sha256", None)
        if payload_digest(check) != supplied_row:
            raise ValueError(f"personalization example drift at index {row.get('index')}")


def examples_for_split(
    examples: list[PersonalizationExample],
    protocol: dict[str, Any],
    split: str,
) -> list[PersonalizationExample]:
    if split not in {"train", "dev", "test"}:
        raise ValueError("split must be train, dev, or test")
    indices = [int(row["index"]) for row in protocol["splits"][split]]
    return [examples[index] for index in indices]


def load_protocol(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
