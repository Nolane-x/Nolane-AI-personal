from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from nolane_personal.continual_cortex_update import (
    ContinualCortexUpdateConfig,
    train_continual_cortex_update,
)
from nolane_personal.evidence_quality import assess_evidence_quality
from nolane_personal.factorized_artifact import (
    load_factorized_model,
    save_factorized_artifact,
)
from nolane_personal.latent import LatentStore
from nolane_personal.personal_dataset import encode_chat_example, load_jsonl
from nolane_personal.personal_protocol import (
    examples_for_split,
    load_protocol,
    verify_personalization_protocol,
)
from nolane_personal.qwen import SYSTEM_PROMPT
from nolane_personal.store import payload_digest


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode(tokenizer, examples, latent, max_length):
    rows = []
    for example in examples:
        ids, labels = encode_chat_example(
            tokenizer,
            example,
            system_prompt=SYSTEM_PROMPT,
            max_length=max_length,
        )
        rows.append(
            (
                ids,
                labels,
                example.latent or latent,
                example.weight,
            )
        )
    return rows


def groups_for_split(protocol, split: str) -> list[str]:
    groups: list[str] = []
    for row in protocol["splits"][split]:
        value = row.get("source_group_sha256")
        if not isinstance(value, str) or not value:
            raise SystemExit(
                f"{split} contains missing source-group lineage"
            )
        groups.append(value)
    return groups


def verified_dataset(dataset_path: Path, protocol_path: Path):
    examples = load_jsonl(dataset_path)
    protocol = load_protocol(protocol_path)
    verify_personalization_protocol(
        protocol,
        dataset_sha256=sha256_file(dataset_path),
    )
    quality = assess_evidence_quality(examples, protocol)
    if quality["status"] != "PASS":
        raise SystemExit(
            "L28 evidence quality BLOCKED: "
            + ",".join(quality["reasons"])
        )
    return examples, protocol, quality


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--factorized",
        default=(
            "runtime-data/l17-factorized-trained/"
            "factorized-nolane.pt"
        ),
    )
    parser.add_argument(
        "--latent",
        default="runtime-data/living-core-shadow/latent.json",
    )
    parser.add_argument("--retention-dataset", required=True)
    parser.add_argument("--retention-protocol", required=True)
    parser.add_argument("--adaptation-dataset", required=True)
    parser.add_argument("--adaptation-protocol", required=True)
    parser.add_argument("--tokenizer", default="models/Qwen3-0.6B")
    parser.add_argument(
        "--output-dir",
        default="runtime-data/l38-continual-cortex-update",
    )
    parser.add_argument("--run-receipt", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--adaptation-task-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--retention-task-weight",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--retention-distill-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--cortex-anchor-weight",
        type=float,
        default=0.01,
    )
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=384)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"refusing to overwrite non-empty L38 workspace: {output_dir}"
        )

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Install tokenizer support with: pip install -e '.[qwen]'"
        ) from exc

    retention_examples, retention_protocol, retention_quality = (
        verified_dataset(
            Path(args.retention_dataset),
            Path(args.retention_protocol),
        )
    )
    adaptation_examples, adaptation_protocol, adaptation_quality = (
        verified_dataset(
            Path(args.adaptation_dataset),
            Path(args.adaptation_protocol),
        )
    )

    retention_rehearsal_examples = (
        examples_for_split(
            retention_examples,
            retention_protocol,
            "train",
        )
        + examples_for_split(
            retention_examples,
            retention_protocol,
            "dev",
        )
    )
    retention_eval_examples = examples_for_split(
        retention_examples,
        retention_protocol,
        "test",
    )
    adaptation_train_examples = examples_for_split(
        adaptation_examples,
        adaptation_protocol,
        "train",
    )
    adaptation_eval_examples = examples_for_split(
        adaptation_examples,
        adaptation_protocol,
        "test",
    )

    retention_groups = groups_for_split(retention_protocol, "test")
    adaptation_groups = groups_for_split(adaptation_protocol, "test")

    latent = LatentStore(args.latent).load()
    if latent is None:
        raise SystemExit(f"persistent latent not found: {args.latent}")

    reference, reference_meta = load_factorized_model(
        args.factorized,
        latent.values,
        device=args.device,
    )
    candidate, candidate_meta = load_factorized_model(
        args.factorized,
        latent.values,
        device=args.device,
    )
    if (
        reference_meta["checkpoint_sha256"]
        != candidate_meta["checkpoint_sha256"]
    ):
        raise SystemExit("candidate/reference parent checkpoint mismatch")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=True,
    )
    encoded_retention_rehearsal = encode(
        tokenizer,
        retention_rehearsal_examples,
        latent.values,
        args.max_length,
    )
    encoded_retention_eval = encode(
        tokenizer,
        retention_eval_examples,
        latent.values,
        args.max_length,
    )
    encoded_adaptation_train = encode(
        tokenizer,
        adaptation_train_examples,
        latent.values,
        args.max_length,
    )
    encoded_adaptation_eval = encode(
        tokenizer,
        adaptation_eval_examples,
        latent.values,
        args.max_length,
    )

    receipt = train_continual_cortex_update(
        candidate,
        reference,
        encoded_adaptation_train,
        encoded_retention_rehearsal,
        retention_eval_examples=encoded_retention_eval,
        adaptation_eval_examples=encoded_adaptation_eval,
        adaptation_group_sha256=adaptation_groups,
        retention_group_sha256=retention_groups,
        config=ContinualCortexUpdateConfig(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            adaptation_task_weight=args.adaptation_task_weight,
            retention_task_weight=args.retention_task_weight,
            retention_distill_weight=args.retention_distill_weight,
            cortex_anchor_weight=args.cortex_anchor_weight,
            distill_temperature=args.temperature,
            max_grad_norm=args.max_grad_norm,
        ),
    )

    lineage = {
        "schema": "NOLANE-L38-CORTEX-UPDATE-LINEAGE-V1",
        "parent_factorized_checkpoint_sha256": (
            reference_meta["checkpoint_sha256"]
        ),
        "retention_dataset_sha256": retention_protocol["dataset_sha256"],
        "retention_protocol_sha256": retention_protocol["protocol_sha256"],
        "retention_quality_court_sha256": retention_quality["court_sha256"],
        "adaptation_dataset_sha256": adaptation_protocol["dataset_sha256"],
        "adaptation_protocol_sha256": adaptation_protocol["protocol_sha256"],
        "adaptation_quality_court_sha256": adaptation_quality["court_sha256"],
    }
    lineage["lineage_sha256"] = payload_digest(lineage)

    training_receipt = receipt.to_dict()
    training_receipt["lineage"] = lineage

    source_meta = {
        "checkpoint_sha256": (
            reference_meta["source_l16_checkpoint_sha256"]
        ),
        "dataset_fingerprint": lineage["lineage_sha256"],
    }
    manifest = save_factorized_artifact(
        output_dir,
        candidate,
        reference_meta["factorization_receipt"],
        source_meta=source_meta,
        dataset_fingerprint=lineage["lineage_sha256"],
        training_receipt=training_receipt,
    )

    result = {
        "schema": "NOLANE-L38-RECURRENT-CORTEX-UPDATE-RUN-V1",
        "authority": "RECURRENT_CORTEX_UPDATE_EVIDENCE_ONLY_UNPROMOTED",
        "parent_factorized_checkpoint_sha256": (
            reference_meta["checkpoint_sha256"]
        ),
        "training": training_receipt,
        "artifact": manifest,
    }
    rendered = json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    receipt_path = (
        Path(args.run_receipt)
        if args.run_receipt
        else output_dir / "l38-run-receipt.json"
    )
    if receipt_path.exists():
        raise SystemExit(
            f"refusing to overwrite L38 run receipt: {receipt_path}"
        )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)

    passed = (
        receipt.candidate_cortex_changed
        and receipt.candidate_boundary_unchanged
        and receipt.reference_boundary_unchanged
        and receipt.reference_cortex_unchanged
        and receipt.candidate_boundary_gradients_seen == 0
        and receipt.candidate_cortex_gradients_seen > 0
        and receipt.continual_learning["status"] == "PASS"
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
