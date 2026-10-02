from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from nolane_personal.factorized_artifact import save_factorized_artifact
from nolane_personal.latent import LatentStore
from nolane_personal.personal_dataset import encode_chat_example, load_jsonl
from nolane_personal.personal_protocol import examples_for_split, load_protocol, verify_personalization_protocol
from nolane_personal.qwen import SYSTEM_PROMPT
from nolane_personal.rank_frontier import RankFrontierConfig, search_rank_frontier
from nolane_personal.standalone_artifact import load_standalone_model
from nolane_personal.store import canonical_json, payload_digest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_ranks(raw: str) -> tuple[int, ...]:
    ranks = tuple(int(x.strip()) for x in raw.split(",") if x.strip())
    if not ranks:
        raise ValueError("rank schedule cannot be empty")
    return ranks


def encode(tokenizer, examples, latent, max_length):
    rows = []
    for example in examples:
        ids, labels = encode_chat_example(
            tokenizer,
            example,
            system_prompt=SYSTEM_PROMPT,
            max_length=max_length,
        )
        rows.append((ids, labels, example.latent or latent, example.weight))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--l16", default="runtime-data/l16-standalone/standalone-nolane.pt")
    parser.add_argument("--latent", default="runtime-data/living-core-shadow/latent.json")
    parser.add_argument("--dataset", default="runtime-data/personalization.jsonl")
    parser.add_argument("--protocol", default="runtime-data/personalization-protocol-v1.json")
    parser.add_argument("--tokenizer", default="models/Qwen3-0.6B")
    parser.add_argument("--output-dir", default="runtime-data/l18-rank-frontier")
    parser.add_argument("--ranks", default="256,192,128,96,64")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--distill-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--max-step-regression", type=float, default=0.02)
    parser.add_argument("--max-l16-regression", type=float, default=0.05)
    parser.add_argument("--min-agreement", type=float, default=0.93)
    parser.add_argument("--max-boundary-ratio", type=float, default=0.60)
    parser.add_argument("--max-length", type=int, default=384)
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install tokenizer support with: pip install -e '.[qwen]'") from exc

    dataset_path = Path(args.dataset)
    examples = load_jsonl(dataset_path)
    protocol = load_protocol(args.protocol)
    verify_personalization_protocol(
        protocol,
        dataset_sha256=sha256_file(dataset_path),
    )
    train_examples = examples_for_split(examples, protocol, "train")
    dev_examples = examples_for_split(examples, protocol, "dev")
    latent = LatentStore(args.latent).load()
    if latent is None:
        raise SystemExit(f"persistent latent not found: {args.latent}")

    source, source_meta = load_standalone_model(
        args.l16,
        latent.values,
        device=args.device,
        expected_dataset_fingerprint=protocol["protocol_sha256"],
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=True,
    )
    train = encode(tokenizer, train_examples, latent.values, args.max_length)
    dev = encode(tokenizer, dev_examples, latent.values, args.max_length)

    selected, receipt = search_rank_frontier(
        source,
        train,
        dev,
        config=RankFrontierConfig(
            ranks=parse_ranks(args.ranks),
            max_dev_nll_regression_per_step=args.max_step_regression,
            max_dev_nll_regression_vs_l16=args.max_l16_regression,
            min_greedy_token_agreement=args.min_agreement,
            max_boundary_parameter_ratio=args.max_boundary_ratio,
            distill_epochs=args.distill_epochs,
            learning_rate=args.learning_rate,
            distill_weight=args.distill_weight,
        ),
        device=args.device,
    )
    selected_stage = next(
        row for row in receipt.stages
        if row["accepted"] and int(row["rank"]) == receipt.selected_rank
    )
    factorization_receipt = {
        "schema": "NOLANE-L18-SELECTED-FACTORIZATION-V1",
        "rank": receipt.selected_rank,
        "input_relative_frobenius_error": selected_stage["reconstruction_input_error"],
        "output_relative_frobenius_error": selected_stage["reconstruction_output_error"],
        "dense_parameters": receipt.source_boundary_parameters,
        "factorized_parameters": receipt.selected_boundary_parameters,
        "parameter_ratio": receipt.selected_boundary_parameter_ratio,
        "tie_word_embeddings": selected.boundary.config.tie_word_embeddings,
        "config": asdict(selected.boundary.config),
    }
    output_dir = Path(args.output_dir)
    manifest = save_factorized_artifact(
        output_dir,
        selected,
        factorization_receipt,
        source_meta=source_meta,
        dataset_fingerprint=protocol["protocol_sha256"],
        training_receipt=receipt.to_dict(),
    )
    frontier = {
        "schema": "NOLANE-L18-RANK-FRONTIER-RECEIPT-V1",
        "authority": "SELECTED_CANDIDATE_UNPROMOTED",
        "source_l16_checkpoint_sha256": source_meta["checkpoint_sha256"],
        "selected_checkpoint_sha256": manifest["checkpoint_sha256"],
        "selected_rank": receipt.selected_rank,
        "dataset_fingerprint": protocol["protocol_sha256"],
        "frontier": receipt.to_dict(),
    }
    frontier["receipt_sha256"] = payload_digest(frontier)
    (output_dir / "rank-frontier-receipt.json").write_text(
        canonical_json(frontier) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "frontier": frontier,
        "artifact": manifest,
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
