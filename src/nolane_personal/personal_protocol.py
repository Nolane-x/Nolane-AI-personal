from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .personal_dataset import PersonalizationExample
from .store import payload_digest


SCHEMA = "NOLANE-PERSONALIZATION-PROTOCOL-V1"
GROUPED_SPLIT_STRATEGY = "source_group_chronological_v1"
CHRONOLOGICAL_SPLIT_STRATEGY = "chronological_v1"


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


def _canonical_example(
    example: PersonalizationExample,
    index: int,
    *,
    source_group_sha256: str | None,
) -> dict[str, Any]:
    example.validate()
    payload = {
        "index": int(index),
        "prompt": example.prompt,
        "target": example.target,
        "latent": example.latent,
        "weight": float(example.weight),
        "language": example.language,
        "source_group_sha256": source_group_sha256,
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


def _grouped_split_indices(
    groups: list[str],
    *,
    policy: PersonalizationSplitPolicy,
) -> dict[str, list[int]]:
    if len(set(groups)) < 3:
        raise ValueError("need at least 3 distinct source groups for leakage-safe train/dev/test")

    ordered_groups: list[str] = []
    members: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        if group not in members:
            ordered_groups.append(group)
            members[group] = []
        members[group].append(index)

    target_train, target_dev, target_test = _split_counts(len(groups), policy)
    best: tuple[int, int, int, int, int] | None = None
    # Split only on group boundaries. A source group can never appear in two
    # held-out partitions even if its examples were non-contiguous originally.
    for first_boundary in range(1, len(ordered_groups) - 1):
        for second_boundary in range(first_boundary + 1, len(ordered_groups)):
            train_groups = ordered_groups[:first_boundary]
            dev_groups = ordered_groups[first_boundary:second_boundary]
            test_groups = ordered_groups[second_boundary:]
            train_n = sum(len(members[g]) for g in train_groups)
            dev_n = sum(len(members[g]) for g in dev_groups)
            test_n = sum(len(members[g]) for g in test_groups)
            if min(train_n, dev_n, test_n) < 1:
                continue
            score = (
                abs(train_n - target_train)
                + abs(dev_n - target_dev)
                + abs(test_n - target_test)
            )
            candidate = (score, abs(test_n - target_test), first_boundary, second_boundary, test_n)
            if best is None or candidate < best:
                best = candidate

    if best is None:
        raise ValueError("unable to create non-empty source-group train/dev/test split")

    _, _, first_boundary, second_boundary, _ = best
    split_groups = {
        "train": ordered_groups[:first_boundary],
        "dev": ordered_groups[first_boundary:second_boundary],
        "test": ordered_groups[second_boundary:],
    }
    return {
        split: sorted(index for group in group_names for index in members[group])
        for split, group_names in split_groups.items()
    }


def build_personalization_protocol(
    examples: list[PersonalizationExample],
    *,
    dataset_sha256: str,
    policy: PersonalizationSplitPolicy | None = None,
    source_group_sha256: list[str | None] | None = None,
) -> dict[str, Any]:
    policy = policy or PersonalizationSplitPolicy()
    policy.validate()
    if len(examples) < policy.min_examples:
        raise ValueError(f"need at least {policy.min_examples} examples")

    if source_group_sha256 is None:
        groups: list[str | None] = [None] * len(examples)
    else:
        groups = list(source_group_sha256)
        if len(groups) != len(examples):
            raise ValueError("source-group lineage length does not match examples")

    for value in groups:
        if value is not None and not isinstance(value, str):
            raise ValueError("source-group lineage values must be sha256 strings or null")

    rows = [
        _canonical_example(example, i, source_group_sha256=groups[i])
        for i, example in enumerate(examples)
    ]

    full_group_coverage = bool(rows) and all(
        isinstance(value, str) and value
        for value in groups
    )
    if full_group_coverage:
        split_indices = _grouped_split_indices(
            [str(value) for value in groups],
            policy=policy,
        )
        split_strategy = GROUPED_SPLIT_STRATEGY
    else:
        train_n, dev_n, _ = _split_counts(len(examples), policy)
        split_indices = {
            "train": list(range(0, train_n)),
            "dev": list(range(train_n, train_n + dev_n)),
            "test": list(range(train_n + dev_n, len(examples))),
        }
        split_strategy = CHRONOLOGICAL_SPLIT_STRATEGY

    protocol = {
        "schema": SCHEMA,
        "dataset_sha256": str(dataset_sha256),
        "policy": asdict(policy),
        "split_strategy": split_strategy,
        "counts": {
            "total": len(rows),
            "train": len(split_indices["train"]),
            "dev": len(split_indices["dev"]),
            "test": len(split_indices["test"]),
        },
        "splits": {
            split: [rows[index] for index in split_indices[split]]
            for split in ("train", "dev", "test")
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

    splits = protocol.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "dev", "test"}:
        raise ValueError("personalization split contract invalid")

    rows = (
        list(splits["train"])
        + list(splits["dev"])
        + list(splits["test"])
    )
    counts = protocol.get("counts", {})
    expected_counts = {
        "total": len(rows),
        "train": len(splits["train"]),
        "dev": len(splits["dev"]),
        "test": len(splits["test"]),
    }
    if any(int(counts.get(key, -1)) != value for key, value in expected_counts.items()):
        raise ValueError("personalization split counts mismatch")

    indices: list[int] = []
    for row in rows:
        check = dict(row)
        supplied_row = check.pop("example_sha256", None)
        if payload_digest(check) != supplied_row:
            raise ValueError(f"personalization example drift at index {row.get('index')}")
        index = int(row.get("index", -1))
        indices.append(index)
        group = row.get("source_group_sha256")
        if group is not None:
            if not isinstance(group, str) or len(group) != 64:
                raise ValueError(f"personalization source-group digest invalid at index {index}")
            try:
                int(group, 16)
            except ValueError as exc:
                raise ValueError(
                    f"personalization source-group digest invalid at index {index}"
                ) from exc

    if sorted(indices) != list(range(len(rows))):
        raise ValueError("personalization split indices are not an exact partition")

    strategy = protocol.get("split_strategy", CHRONOLOGICAL_SPLIT_STRATEGY)
    if strategy not in {CHRONOLOGICAL_SPLIT_STRATEGY, GROUPED_SPLIT_STRATEGY}:
        raise ValueError("unsupported personalization split strategy")

    if strategy == GROUPED_SPLIT_STRATEGY:
        groups_by_split = {
            split: {row.get("source_group_sha256") for row in splits[split]}
            for split in ("train", "dev", "test")
        }
        if any(None in values for values in groups_by_split.values()):
            raise ValueError("grouped split contains missing source-group lineage")
        if (
            groups_by_split["train"] & groups_by_split["dev"]
            or groups_by_split["train"] & groups_by_split["test"]
            or groups_by_split["dev"] & groups_by_split["test"]
        ):
            raise ValueError("source-group leakage across personalization splits")


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
