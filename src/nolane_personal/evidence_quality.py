from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .personal_dataset import PersonalizationExample, load_jsonl
from .personal_protocol import (
    GROUPED_SPLIT_STRATEGY,
    load_protocol,
    verify_personalization_protocol,
)
from .store import payload_digest


SCHEMA = "NOLANE-L28-EVIDENCE-QUALITY-COURT-V1"


@dataclass(slots=True)
class EvidenceQualityPolicy:
    min_total_examples: int = 7
    min_train_examples: int = 1
    min_dev_examples: int = 1
    min_test_examples: int = 2
    min_distinct_source_groups: int = 3
    require_full_source_group_coverage: bool = True
    require_grouped_split_strategy: bool = True
    near_duplicate_token_jaccard: float = 0.80
    near_duplicate_sequence_ratio: float = 0.90
    min_tokens_for_near_duplicate: int = 4

    def validate(self) -> None:
        if self.min_total_examples < 3:
            raise ValueError("min_total_examples must be >=3")
        if min(
            self.min_train_examples,
            self.min_dev_examples,
            self.min_test_examples,
        ) < 1:
            raise ValueError("split minimums must be positive")
        if self.min_distinct_source_groups < 3:
            raise ValueError("min_distinct_source_groups must be >=3")
        for value, name in (
            (self.near_duplicate_token_jaccard, "near_duplicate_token_jaccard"),
            (self.near_duplicate_sequence_ratio, "near_duplicate_sequence_ratio"),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0,1]")
        if self.min_tokens_for_near_duplicate < 1:
            raise ValueError("min_tokens_for_near_duplicate must be positive")


def _normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokens(value: str, *, role: str) -> set[str]:
    normalized = _normalize_text(value)
    return {
        f"{role}:{token}"
        for token in re.findall(r"\w+", normalized, flags=re.UNICODE)
    }


def _pair_tokens(example: PersonalizationExample) -> set[str]:
    return _tokens(example.prompt, role="p") | _tokens(example.target, role="t")


def _pair_text(example: PersonalizationExample) -> str:
    return f"p:{_normalize_text(example.prompt)}\nt:{_normalize_text(example.target)}"


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def _split_rows(protocol: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        split: list(protocol["splits"][split])
        for split in ("train", "dev", "test")
    }


def _language_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        language = str(row.get("language") or "unknown")
        counts[language] = counts.get(language, 0) + 1
    return dict(sorted(counts.items()))


def assess_evidence_quality(
    examples: list[PersonalizationExample],
    protocol: dict[str, Any],
    *,
    policy: EvidenceQualityPolicy | None = None,
) -> dict[str, Any]:
    policy = policy or EvidenceQualityPolicy()
    policy.validate()

    verify_personalization_protocol(
        protocol,
        dataset_sha256=str(protocol["dataset_sha256"]),
    )
    splits = _split_rows(protocol)
    reasons: list[str] = []

    counts = {
        "total": len(examples),
        "train": len(splits["train"]),
        "dev": len(splits["dev"]),
        "test": len(splits["test"]),
    }
    if counts["total"] < policy.min_total_examples:
        reasons.append("insufficient_total_examples")
    if counts["train"] < policy.min_train_examples:
        reasons.append("insufficient_train_examples")
    if counts["dev"] < policy.min_dev_examples:
        reasons.append("insufficient_dev_examples")
    if counts["test"] < policy.min_test_examples:
        reasons.append("insufficient_test_examples")

    index_to_split: dict[int, str] = {}
    rows_by_index: dict[int, dict[str, Any]] = {}
    for split, rows in splits.items():
        for row in rows:
            index = int(row["index"])
            index_to_split[index] = split
            rows_by_index[index] = row

    group_values = [
        rows_by_index[index].get("source_group_sha256")
        for index in range(len(examples))
    ]
    group_coverage = sum(
        1 for value in group_values
        if isinstance(value, str) and value
    )
    if (
        policy.require_full_source_group_coverage
        and group_coverage != len(examples)
    ):
        reasons.append("incomplete_source_group_lineage")

    distinct_groups = {
        str(value)
        for value in group_values
        if isinstance(value, str) and value
    }
    if len(distinct_groups) < policy.min_distinct_source_groups:
        reasons.append("insufficient_distinct_source_groups")

    if (
        policy.require_grouped_split_strategy
        and protocol.get("split_strategy") != GROUPED_SPLIT_STRATEGY
    ):
        reasons.append("non_grouped_split_strategy")

    groups_by_split = {
        split: {
            str(row["source_group_sha256"])
            for row in rows
            if isinstance(row.get("source_group_sha256"), str)
            and row.get("source_group_sha256")
        }
        for split, rows in splits.items()
    }
    group_leakage_pairs = {
        "train_dev": sorted(groups_by_split["train"] & groups_by_split["dev"]),
        "train_test": sorted(groups_by_split["train"] & groups_by_split["test"]),
        "dev_test": sorted(groups_by_split["dev"] & groups_by_split["test"]),
    }
    leaked_group_count = sum(len(values) for values in group_leakage_pairs.values())
    if leaked_group_count:
        reasons.append("source_group_leakage")

    normalized_prompts: dict[str, tuple[int, str]] = {}
    exact_prompt_leaks: list[dict[str, Any]] = []
    exact_pair_leaks: list[dict[str, Any]] = []
    normalized_pairs: dict[str, tuple[int, str]] = {}

    pair_tokens = [_pair_tokens(example) for example in examples]
    pair_text = [_pair_text(example) for example in examples]
    near_duplicate_pairs: list[dict[str, Any]] = []
    max_cross_split_jaccard = 0.0
    max_cross_split_sequence = 0.0
    comparisons = 0

    for index, example in enumerate(examples):
        split = index_to_split[index]
        prompt_key = _normalize_text(example.prompt)
        pair_key = _pair_text(example)

        previous = normalized_prompts.get(prompt_key)
        if previous is not None and previous[1] != split:
            exact_prompt_leaks.append({
                "left_index": previous[0],
                "right_index": index,
                "left_split": previous[1],
                "right_split": split,
            })
        else:
            normalized_prompts[prompt_key] = (index, split)

        previous_pair = normalized_pairs.get(pair_key)
        if previous_pair is not None and previous_pair[1] != split:
            exact_pair_leaks.append({
                "left_index": previous_pair[0],
                "right_index": index,
                "left_split": previous_pair[1],
                "right_split": split,
            })
        else:
            normalized_pairs[pair_key] = (index, split)

    if exact_prompt_leaks:
        reasons.append("exact_prompt_leakage")
    if exact_pair_leaks:
        reasons.append("exact_pair_leakage")

    for left in range(len(examples)):
        for right in range(left + 1, len(examples)):
            left_split = index_to_split[left]
            right_split = index_to_split[right]
            if left_split == right_split:
                continue
            comparisons += 1
            left_tokens = pair_tokens[left]
            right_tokens = pair_tokens[right]
            if min(len(left_tokens), len(right_tokens)) < policy.min_tokens_for_near_duplicate:
                continue
            token_jaccard = _jaccard(left_tokens, right_tokens)
            sequence_ratio = SequenceMatcher(
                None,
                pair_text[left],
                pair_text[right],
                autojunk=False,
            ).ratio()
            max_cross_split_jaccard = max(max_cross_split_jaccard, token_jaccard)
            max_cross_split_sequence = max(max_cross_split_sequence, sequence_ratio)

            if (
                token_jaccard >= policy.near_duplicate_token_jaccard
                and sequence_ratio >= policy.near_duplicate_sequence_ratio
            ):
                near_duplicate_pairs.append({
                    "left_index": left,
                    "right_index": right,
                    "left_split": left_split,
                    "right_split": right_split,
                    "token_jaccard": round(token_jaccard, 6),
                    "sequence_ratio": round(sequence_ratio, 6),
                })

    if near_duplicate_pairs:
        reasons.append("near_duplicate_cross_split")

    unique_prompt_ratio = (
        len({_normalize_text(example.prompt) for example in examples}) / len(examples)
        if examples else 0.0
    )
    unique_pair_ratio = (
        len({_pair_text(example) for example in examples}) / len(examples)
        if examples else 0.0
    )
    mean_prompt_chars = (
        sum(len(example.prompt) for example in examples) / len(examples)
        if examples else 0.0
    )
    mean_target_chars = (
        sum(len(example.target) for example in examples) / len(examples)
        if examples else 0.0
    )

    receipt = {
        "schema": SCHEMA,
        "authority": "STRUCTURAL_QUALITY_ONLY_NO_MODEL_AUTHORITY",
        "status": "PASS" if not reasons else "BLOCKED",
        "reasons": sorted(set(reasons)),
        "dataset_sha256": str(protocol["dataset_sha256"]),
        "protocol_sha256": str(protocol["protocol_sha256"]),
        "policy": asdict(policy),
        "counts": counts,
        "split_strategy": protocol.get("split_strategy"),
        "source_groups": {
            "coverage": group_coverage,
            "distinct_total": len(distinct_groups),
            "distinct_by_split": {
                split: len(groups_by_split[split])
                for split in ("train", "dev", "test")
            },
            "cross_split_leaked_group_count": leaked_group_count,
        },
        "text_leakage": {
            "exact_prompt_cross_split": len(exact_prompt_leaks),
            "exact_pair_cross_split": len(exact_pair_leaks),
            "near_duplicate_cross_split": len(near_duplicate_pairs),
            "near_duplicate_pairs": near_duplicate_pairs,
            "comparisons": comparisons,
            "max_cross_split_token_jaccard": round(max_cross_split_jaccard, 6),
            "max_cross_split_sequence_ratio": round(max_cross_split_sequence, 6),
        },
        "diversity": {
            "unique_prompt_ratio": round(unique_prompt_ratio, 6),
            "unique_pair_ratio": round(unique_pair_ratio, 6),
            "mean_prompt_chars": round(mean_prompt_chars, 3),
            "mean_target_chars": round(mean_target_chars, 3),
            "language_counts_by_split": {
                split: _language_counts(splits[split])
                for split in ("train", "dev", "test")
            },
        },
        "privacy": {
            "contains_raw_prompt_target": False,
            "contains_raw_source_id": False,
            "contains_source_group_hash_values": False,
        },
    }
    receipt["court_sha256"] = payload_digest(receipt)
    return receipt


def assess_evidence_quality_files(
    dataset_path: str | Path,
    protocol_path: str | Path,
    *,
    policy: EvidenceQualityPolicy | None = None,
) -> dict[str, Any]:
    examples = load_jsonl(dataset_path)
    protocol = load_protocol(protocol_path)
    verify_personalization_protocol(
        protocol,
        dataset_sha256=str(protocol["dataset_sha256"]),
    )
    return assess_evidence_quality(examples, protocol, policy=policy)


def verify_evidence_quality_receipt(
    receipt: dict[str, Any],
    *,
    dataset_path: str | Path,
    protocol_path: str | Path,
) -> dict[str, Any]:
    if receipt.get("schema") != SCHEMA:
        raise ValueError("unsupported evidence quality receipt schema")
    supplied = receipt.get("court_sha256")
    body = dict(receipt)
    body.pop("court_sha256", None)
    if payload_digest(body) != supplied:
        raise ValueError("evidence quality receipt digest mismatch")
    policy = EvidenceQualityPolicy(**dict(receipt.get("policy", {})))
    computed = assess_evidence_quality_files(
        dataset_path,
        protocol_path,
        policy=policy,
    )
    if computed != receipt:
        raise ValueError("evidence quality receipt does not match current dataset/protocol")
    return receipt


def load_quality_receipt(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
