from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import unicodedata
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .approved_evidence import (
    SCHEMA as APPROVED_PACK_SCHEMA,
    resolve_approved_evidence_pack,
    sha256_file,
    verify_approved_evidence_pack,
)
from .evidence_quality import EvidenceQualityPolicy, assess_evidence_quality
from .local_evidence_workbench import paths_for, verify_workbench_manifest
from .longitudinal_execution import PLAN_SCHEMA, validate_longitudinal_plan
from .personal_dataset import PersonalizationExample, load_jsonl
from .personal_protocol import (
    PersonalizationSplitPolicy,
    build_personalization_protocol,
    load_protocol,
)
from .product_evidence_bridge import verify_product_evidence_window
from .product_learning_workspace import ProductLearningWorkspace
from .store import canonical_json, payload_digest


CAMPAIGN_SPEC_SCHEMA = "NOLANE-L46-REAL-LEARNING-CAMPAIGN-SPEC-V1"
CAMPAIGN_RECEIPT_SCHEMA = "NOLANE-L46-REAL-LEARNING-CAMPAIGN-RECEIPT-V1"
COMPOSITE_AUTHORITY = (
    "COMPOSED_RETENTION_FROM_EXPLICIT_APPROVED_TRAIN_DEV_ONLY"
)
CAMPAIGN_AUTHORITY = (
    "REAL_LEARNING_CAMPAIGN_PLAN_ONLY_NO_TRAINING_PROMOTION_AUTHORITY"
)


@dataclass(slots=True)
class CampaignIsolationPolicy:
    near_duplicate_token_jaccard: float = 0.80
    near_duplicate_sequence_ratio: float = 0.90
    min_tokens_for_near_duplicate: int = 4

    def validate(self) -> None:
        if not 0 < self.near_duplicate_token_jaccard <= 1:
            raise ValueError("near_duplicate_token_jaccard must be in (0,1]")
        if not 0 < self.near_duplicate_sequence_ratio <= 1:
            raise ValueError("near_duplicate_sequence_ratio must be in (0,1]")
        if self.min_tokens_for_near_duplicate < 1:
            raise ValueError("min_tokens_for_near_duplicate must be positive")


@dataclass(slots=True)
class CampaignWindow:
    root: Path
    window_id: str
    ordinal: int
    through_rowid: int
    window_manifest_sha256: str
    workbench_manifest_sha256: str
    approved_manifest_path: Path
    approved_manifest_sha256: str
    dataset_path: Path
    protocol_path: Path
    dataset_sha256: str
    protocol_sha256: str
    quality_court_sha256: str


def _resolve(base: Path, value: Any, *, field: str) -> Path:
    rendered = str(value or "").strip()
    if not rendered:
        raise ValueError(f"{field} is required")
    path = Path(rendered)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def _normalize(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"\s+", " ", text).strip()


def _tokens(example: PersonalizationExample) -> set[str]:
    prompt = {
        "p:" + token
        for token in re.findall(r"\w+", _normalize(example.prompt))
    }
    target = {
        "t:" + token
        for token in re.findall(r"\w+", _normalize(example.target))
    }
    return prompt | target


def _pair_text(example: PersonalizationExample) -> str:
    return "p:" + _normalize(example.prompt) + "\nt:" + _normalize(
        example.target
    )


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def _cross_pack_isolation(
    left_examples: list[PersonalizationExample],
    right_examples: list[PersonalizationExample],
    *,
    label: str,
    policy: CampaignIsolationPolicy,
) -> dict[str, Any]:
    left_prompts = {_normalize(example.prompt) for example in left_examples}
    right_prompts = {_normalize(example.prompt) for example in right_examples}
    left_pairs = {_pair_text(example) for example in left_examples}
    right_pairs = {_pair_text(example) for example in right_examples}

    exact_prompt_overlap = len(left_prompts & right_prompts)
    exact_pair_overlap = len(left_pairs & right_pairs)
    if exact_prompt_overlap or exact_pair_overlap:
        raise ValueError(
            f"{label} exact content leakage: "
            f"prompt_overlap={exact_prompt_overlap} "
            f"pair_overlap={exact_pair_overlap}"
        )

    left_tokens = [_tokens(example) for example in left_examples]
    right_tokens = [_tokens(example) for example in right_examples]
    left_text = [_pair_text(example) for example in left_examples]
    right_text = [_pair_text(example) for example in right_examples]
    comparisons = 0
    near_duplicates = 0
    max_jaccard = 0.0
    max_sequence = 0.0

    for li, left in enumerate(left_examples):
        for ri, right in enumerate(right_examples):
            comparisons += 1
            if min(len(left_tokens[li]), len(right_tokens[ri])) < (
                policy.min_tokens_for_near_duplicate
            ):
                continue
            jaccard = _jaccard(left_tokens[li], right_tokens[ri])
            sequence = SequenceMatcher(
                None,
                left_text[li],
                right_text[ri],
                autojunk=False,
            ).ratio()
            max_jaccard = max(max_jaccard, jaccard)
            max_sequence = max(max_sequence, sequence)
            if (
                jaccard >= policy.near_duplicate_token_jaccard
                and sequence >= policy.near_duplicate_sequence_ratio
            ):
                near_duplicates += 1

    if near_duplicates:
        raise ValueError(
            f"{label} near-duplicate leakage: count={near_duplicates}"
        )
    return {
        "label": label,
        "comparisons": comparisons,
        "exact_prompt_overlap": 0,
        "exact_pair_overlap": 0,
        "near_duplicate_overlap": 0,
        "max_token_jaccard": round(max_jaccard, 6),
        "max_sequence_ratio": round(max_sequence, 6),
    }


def _window_binding(
    root: Path,
    *,
    ordinal_by_window: dict[str, int],
) -> CampaignWindow:
    root = root.resolve()
    window = verify_product_evidence_window(root)
    window_id = root.name
    if window_id not in ordinal_by_window:
        raise ValueError(
            f"campaign window is not present in product learning registry: {window_id}"
        )
    workbench_root = root / "workbench"
    workbench = verify_workbench_manifest(workbench_root)
    if workbench.get("phase") != "INTAKE_READY":
        raise ValueError(f"campaign window is not finalized: {window_id}")
    paths = paths_for(workbench_root)
    manifest, dataset, protocol = resolve_approved_evidence_pack(
        paths.approved_manifest
    )
    return CampaignWindow(
        root=root,
        window_id=window_id,
        ordinal=ordinal_by_window[window_id],
        through_rowid=int(
            window["source_event_range"]["through_rowid_inclusive"]
        ),
        window_manifest_sha256=str(window["manifest_sha256"]),
        workbench_manifest_sha256=str(workbench["manifest_sha256"]),
        approved_manifest_path=paths.approved_manifest.resolve(),
        approved_manifest_sha256=str(manifest["manifest_sha256"]),
        dataset_path=dataset.resolve(),
        protocol_path=protocol.resolve(),
        dataset_sha256=str(manifest["dataset_sha256"]),
        protocol_sha256=str(manifest["protocol_sha256"]),
        quality_court_sha256=str(manifest["quality_court_sha256"]),
    )


def _registry_for_window_root(
    root: Path,
) -> tuple[dict[str, Any], dict[str, int], Path]:
    windows_dir = root.resolve().parent
    if windows_dir.name != "windows":
        raise ValueError(
            "L46 campaign windows must come from one in-app learning-evidence/windows registry"
        )
    learning_root = windows_dir.parent
    if learning_root.name != "learning-evidence":
        raise ValueError(
            "L46 campaign window is outside the in-app learning-evidence registry"
        )
    workspace = ProductLearningWorkspace(data_dir=learning_root.parent)
    registry = workspace.verify_registry()
    order = {
        str(row["window_id"]): index
        for index, row in enumerate(registry["windows"], start=1)
    }
    return registry, order, learning_root


def _examples(binding: CampaignWindow) -> list[PersonalizationExample]:
    return load_jsonl(binding.dataset_path)


def _split_index_groups(
    binding: CampaignWindow,
    splits: tuple[str, ...],
) -> tuple[list[PersonalizationExample], list[str], int]:
    examples = _examples(binding)
    protocol = load_protocol(binding.protocol_path)
    selected_indices: list[int] = []
    group_by_index: dict[int, str] = {}
    for split in ("train", "dev", "test"):
        for row in protocol["splits"][split]:
            index = int(row["index"])
            group = row.get("source_group_sha256")
            if not isinstance(group, str) or len(group) != 64:
                raise ValueError(
                    f"{binding.window_id} contains incomplete source-group lineage"
                )
            group_by_index[index] = group
            if split in splits:
                selected_indices.append(index)

    selected_indices = sorted(set(selected_indices))
    selected = [examples[index] for index in selected_indices]
    groups = [group_by_index[index] for index in selected_indices]
    heldout = len(examples) - len(selected)
    return selected, groups, heldout


def _example_payload(example: PersonalizationExample) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "prompt": example.prompt,
        "target": example.target,
        "language": example.language,
        "weight": float(example.weight),
    }
    if example.latent is not None:
        payload["latent"] = [float(value) for value in example.latent]
    return payload


def _language_counts(examples: list[PersonalizationExample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for example in examples:
        language = str(example.language or "unknown")
        counts[language] = counts.get(language, 0) + 1
    return dict(sorted(counts.items()))


def compose_retention_pack(
    sources: list[CampaignWindow],
    output_dir: str | Path,
) -> dict[str, Any]:
    if len(sources) < 2:
        raise ValueError(
            "composed retention requires baseline plus at least one learned window"
        )
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty composed retention pack: {output}"
        )

    examples: list[PersonalizationExample] = []
    groups: list[str] = []
    source_rows: list[dict[str, Any]] = []
    seen_groups: set[str] = set()
    seen_prompts: set[str] = set()
    seen_pairs: set[str] = set()
    original_test_excluded = 0

    for source in sources:
        selected, selected_groups, heldout = _split_index_groups(
            source,
            ("train", "dev"),
        )
        original_test_excluded += heldout
        if set(selected_groups) & seen_groups:
            raise ValueError(
                "composed retention source-group reuse detected"
            )
        for example in selected:
            prompt_key = _normalize(example.prompt)
            pair_key = _pair_text(example)
            if prompt_key in seen_prompts:
                raise ValueError(
                    "composed retention contains repeated prompt across source packs"
                )
            if pair_key in seen_pairs:
                raise ValueError(
                    "composed retention contains repeated prompt/target pair"
                )
            seen_prompts.add(prompt_key)
            seen_pairs.add(pair_key)
        seen_groups.update(selected_groups)
        examples.extend(selected)
        groups.extend(selected_groups)
        source_rows.append(
            {
                "window_id": source.window_id,
                "approved_manifest_sha256": source.approved_manifest_sha256,
                "dataset_sha256": source.dataset_sha256,
                "protocol_sha256": source.protocol_sha256,
                "quality_court_sha256": source.quality_court_sha256,
                "selected_splits": ["train", "dev"],
                "selected_examples": len(selected),
                "excluded_original_test_examples": heldout,
            }
        )

    if len(examples) < 7:
        raise ValueError(
            "composed retention needs at least 7 train/dev examples"
        )

    split_policy = PersonalizationSplitPolicy(min_examples=7)
    dataset_text = "".join(
        json.dumps(
            _example_payload(example),
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
        for example in examples
    )
    dataset_sha256 = hashlib.sha256(
        dataset_text.encode("utf-8")
    ).hexdigest()
    protocol = build_personalization_protocol(
        examples,
        dataset_sha256=dataset_sha256,
        policy=split_policy,
        source_group_sha256=groups,
    )
    quality_policy = EvidenceQualityPolicy()
    quality = assess_evidence_quality(
        examples,
        protocol,
        policy=quality_policy,
    )
    if quality["status"] != "PASS":
        raise ValueError(
            "L46 composed retention L28 BLOCKED: "
            + ",".join(quality["reasons"])
        )

    output.mkdir(parents=True, exist_ok=True)
    dataset_path = output / "personalization.jsonl"
    protocol_path = output / "personalization-protocol-v1.json"
    quality_path = output / "evidence-quality-receipt.json"
    manifest_path = output / "approved-evidence-manifest.json"

    dataset_path.write_text(dataset_text, encoding="utf-8")
    protocol_path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    quality_path.write_text(
        json.dumps(quality, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    counts = protocol["counts"]
    lineage = {
        "schema": "NOLANE-L46-COMPOSED-RETENTION-LINEAGE-V1",
        "sources": source_rows,
        "original_test_examples_excluded": original_test_excluded,
        "source_count": len(sources),
        "selection_rule": "ORIGINAL_TRAIN_DEV_ONLY",
    }
    lineage["lineage_sha256"] = payload_digest(lineage)

    manifest = {
        "schema": APPROVED_PACK_SCHEMA,
        "authority": COMPOSITE_AUTHORITY,
        "privacy": {
            "raw_prompt_target_in_manifest": False,
            "raw_source_id_in_manifest": False,
            "local_only_recommended": True,
        },
        "source_sha256": lineage["lineage_sha256"],
        "source_id_sha256": [],
        "example_source_group_sha256": groups,
        "source_group_coverage": len(groups),
        "dataset_sha256": dataset_sha256,
        "protocol_sha256": str(protocol["protocol_sha256"]),
        "quality_status": "PASS",
        "quality_court_sha256": str(quality["court_sha256"]),
        "quality_receipt_sha256": sha256_file(quality_path),
        "dataset_filename": dataset_path.name,
        "protocol_filename": protocol_path.name,
        "quality_receipt_filename": quality_path.name,
        "stats": {
            "source_rows": len(examples),
            "approved_rows": len(examples),
            "excluded_unapproved": 0,
            "excluded_sensitive": 0,
            "output_examples": len(examples),
            "duplicate_pairs": 0,
            "language_counts": _language_counts(examples),
            "train_examples": int(counts["train"]),
            "dev_examples": int(counts["dev"]),
            "test_examples": int(counts["test"]),
        },
        "policy": {
            "kind": "L46_COMPOSED_RETENTION",
            "original_heldout_test_never_included": True,
        },
        "split_policy": asdict(split_policy),
        "composite_lineage": lineage,
    }
    manifest["manifest_sha256"] = payload_digest(manifest)
    manifest_path.write_text(
        canonical_json(manifest) + "\n",
        encoding="utf-8",
    )
    verify_approved_evidence_pack(manifest_path)
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "lineage": lineage,
    }


def _window_public(binding: CampaignWindow) -> dict[str, Any]:
    return {
        "window_id": binding.window_id,
        "ordinal": binding.ordinal,
        "through_rowid": binding.through_rowid,
        "window_manifest_sha256": binding.window_manifest_sha256,
        "workbench_manifest_sha256": binding.workbench_manifest_sha256,
        "approved_manifest_sha256": binding.approved_manifest_sha256,
        "dataset_sha256": binding.dataset_sha256,
        "protocol_sha256": binding.protocol_sha256,
        "quality_court_sha256": binding.quality_court_sha256,
    }


def build_real_learning_campaign(
    spec_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    spec_path = Path(spec_path).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("schema") != CAMPAIGN_SPEC_SCHEMA:
        raise ValueError("unsupported real learning campaign spec schema")
    base = spec_path.parent

    fixed_root = _resolve(
        base,
        spec.get("fixed_window"),
        field="fixed_window",
    )
    registry, ordinal_by_window, learning_root = _registry_for_window_root(
        fixed_root
    )
    registry_sha256 = str(registry["registry_sha256"])

    baseline_root = _resolve(
        base,
        spec.get("baseline_retention_window"),
        field="baseline_retention_window",
    )
    adaptation_raw = spec.get("adaptation_windows")
    if not isinstance(adaptation_raw, list) or len(adaptation_raw) < 5:
        raise ValueError(
            "real learning campaign requires at least 5 adaptation windows"
        )

    fixed = _window_binding(
        fixed_root,
        ordinal_by_window=ordinal_by_window,
    )
    if baseline_root.parent.parent.resolve() != learning_root.resolve():
        raise ValueError(
            "all campaign windows must come from one product learning registry"
        )
    baseline = _window_binding(
        baseline_root,
        ordinal_by_window=ordinal_by_window,
    )

    adaptations: list[CampaignWindow] = []
    training_by_cycle: list[dict[str, Any]] = []
    default_training = dict(spec.get("training", {}))
    for index, raw in enumerate(adaptation_raw, start=1):
        if isinstance(raw, str):
            root_value = raw
            training = dict(default_training)
        elif isinstance(raw, dict):
            root_value = raw.get("window")
            training = dict(default_training)
            training.update(dict(raw.get("training", {})))
        else:
            raise ValueError(
                f"adaptation_windows[{index}] must be path or object"
            )
        root = _resolve(
            base,
            root_value,
            field=f"adaptation_windows[{index}]",
        )
        if root.parent.parent.resolve() != learning_root.resolve():
            raise ValueError(
                "all campaign windows must come from one product learning registry"
            )
        adaptations.append(
            _window_binding(
                root,
                ordinal_by_window=ordinal_by_window,
            )
        )
        training_by_cycle.append(training)

    roles = [fixed, baseline, *adaptations]
    if len({item.window_id for item in roles}) != len(roles):
        raise ValueError("campaign window role reused")

    ordinals = [item.ordinal for item in roles]
    if ordinals != sorted(ordinals) or len(set(ordinals)) != len(ordinals):
        raise ValueError(
            "campaign windows must be chronological: fixed < baseline < adaptations"
        )

    policy = CampaignIsolationPolicy(
        **dict(spec.get("isolation_policy", {}))
    )
    policy.validate()
    fixed_examples = _examples(fixed)
    baseline_examples = _examples(baseline)
    adaptation_examples = [_examples(item) for item in adaptations]
    isolation: list[dict[str, Any]] = []

    isolation.append(
        _cross_pack_isolation(
            fixed_examples,
            baseline_examples,
            label="fixed/baseline",
            policy=policy,
        )
    )
    for index, examples in enumerate(adaptation_examples, start=1):
        isolation.append(
            _cross_pack_isolation(
                fixed_examples,
                examples,
                label=f"fixed/adaptation-{index:03d}",
                policy=policy,
            )
        )
        isolation.append(
            _cross_pack_isolation(
                baseline_examples,
                examples,
                label=f"baseline/adaptation-{index:03d}",
                policy=policy,
            )
        )
    for left in range(len(adaptation_examples)):
        for right in range(left + 1, len(adaptation_examples)):
            isolation.append(
                _cross_pack_isolation(
                    adaptation_examples[left],
                    adaptation_examples[right],
                    label=(
                        f"adaptation-{left + 1:03d}/"
                        f"adaptation-{right + 1:03d}"
                    ),
                    policy=policy,
                )
            )

    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite campaign output: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(
        tempfile.mkdtemp(
            dir=output.parent,
            prefix=output.name + ".",
            suffix=".tmp",
        )
    )

    try:
        retention_dir = temp / "retention"
        retention_dir.mkdir(parents=True, exist_ok=True)
        cycles: list[dict[str, Any]] = []
        retention_rows: list[dict[str, Any]] = []

        for index, adaptation in enumerate(adaptations, start=1):
            if index == 1:
                retention_manifest = baseline.approved_manifest_path
                retention_rows.append(
                    {
                        "cycle": index,
                        "kind": "ORIGINAL_BASELINE_PACK",
                        "manifest_sha256": baseline.approved_manifest_sha256,
                        "source_windows": [baseline.window_id],
                        "original_test_examples_excluded_from_future_rehearsal": True,
                    }
                )
            else:
                sources = [baseline, *adaptations[: index - 1]]
                composed = compose_retention_pack(
                    sources,
                    retention_dir / f"cycle-{index:03d}",
                )
                retention_manifest = Path(composed["manifest_path"])
                retention_rows.append(
                    {
                        "cycle": index,
                        "kind": "COMPOSED_PRIOR_TRAIN_DEV_ONLY",
                        "manifest_sha256": composed["manifest"][
                            "manifest_sha256"
                        ],
                        "source_windows": [
                            source.window_id for source in sources
                        ],
                        "lineage_sha256": composed["lineage"][
                            "lineage_sha256"
                        ],
                        "original_test_examples_excluded": composed[
                            "lineage"
                        ]["original_test_examples_excluded"],
                    }
                )

            cycles.append(
                {
                    "retention_manifest": (
                        str(retention_manifest.resolve())
                        if index == 1
                        else str(
                            retention_manifest.relative_to(temp)
                        )
                    ),
                    "adaptation_manifest": str(
                        adaptation.approved_manifest_path.resolve()
                    ),
                    "training": training_by_cycle[index - 1],
                }
            )

        plan = {
            "schema": PLAN_SCHEMA,
            "initial_factorized": str(
                _resolve(
                    base,
                    spec.get("initial_factorized"),
                    field="initial_factorized",
                )
            ),
            "latent": str(
                _resolve(base, spec.get("latent"), field="latent")
            ),
            "tokenizer": str(
                _resolve(base, spec.get("tokenizer"), field="tokenizer")
            ),
            "device": str(spec.get("device", "cpu")),
            "fixed_panel_manifest": str(
                fixed.approved_manifest_path.resolve()
            ),
            "cycles": cycles,
        }
        if "l43_policy" in spec:
            plan["policy"] = dict(spec["l43_policy"])

        plan_path = temp / "l43-plan.json"
        plan_path.write_text(
            canonical_json(plan) + "\n",
            encoding="utf-8",
        )
        validated = validate_longitudinal_plan(plan_path)

        receipt = {
            "schema": CAMPAIGN_RECEIPT_SCHEMA,
            "authority": CAMPAIGN_AUTHORITY,
            "status": "READY_FOR_EXPLICIT_L43_EXECUTION",
            "registry_sha256": registry_sha256,
            "fixed": _window_public(fixed),
            "baseline_retention": _window_public(baseline),
            "adaptations": [
                _window_public(item) for item in adaptations
            ],
            "retention_cycles": retention_rows,
            "isolation_policy": asdict(policy),
            "cross_window_isolation": {
                "comparisons": sum(
                    int(row["comparisons"]) for row in isolation
                ),
                "courts": isolation,
                "status": "PASS",
            },
            "l43_plan_sha256": validated.receipt["plan_sha256"],
            "protected_original_test_windows": [
                baseline.window_id,
                *[item.window_id for item in adaptations],
            ],
            "privacy": {
                "contains_raw_prompt_target": False,
                "contains_candidate_ids": False,
                "contains_individual_prompt_hashes": False,
                "contains_local_paths": False,
                "auto_training": False,
                "auto_promotion": False,
            },
        }
        receipt["receipt_sha256"] = payload_digest(receipt)
        (temp / "campaign-receipt.json").write_text(
            canonical_json(receipt) + "\n",
            encoding="utf-8",
        )

        os.replace(temp, output)
        # Re-validate after the atomic directory move so relative composed
        # retention paths are proven portable inside the final campaign.
        validate_longitudinal_plan(output / "l43-plan.json")
        return receipt
    except Exception:
        if temp.exists():
            shutil.rmtree(temp, ignore_errors=True)
        raise
