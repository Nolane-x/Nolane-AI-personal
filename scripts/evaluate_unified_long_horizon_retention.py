from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from nolane_personal.evidence_quality import assess_evidence_quality
from nolane_personal.factorized_artifact import load_factorized_model
from nolane_personal.latent import LatentStore
from nolane_personal.long_horizon_retention import (
    LongHorizonRetentionPolicy,
    assess_long_horizon_retention,
)
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


def per_example_nll(model, examples) -> list[float]:
    torch = model.cortex.torch
    device = next(model.cortex.module.parameters()).device
    values: list[float] = []
    model.eval()
    with torch.no_grad():
        for ids, labels, latent, weight in examples:
            x = torch.tensor([ids], dtype=torch.long, device=device)
            y = torch.tensor([labels], dtype=torch.long, device=device)
            if latent is not None:
                model.set_latent(latent)
            loss = model.forward(
                input_ids=x,
                labels=y,
                state=None,
            ).loss
            values.append(float(loss.detach().cpu()) * float(weight))
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", required=True)
    parser.add_argument("--final", required=True)
    parser.add_argument(
        "--latent",
        default="runtime-data/living-core-shadow/latent.json",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--tokenizer", default="models/Qwen3-0.6B")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument(
        "--max-overall-regression",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--max-worst-group-regression",
        type=float,
        default=0.03,
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise SystemExit(
            f"refusing to overwrite L39 long-horizon receipt: {output}"
        )

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Install tokenizer support with: pip install -e '.[qwen]'"
        ) from exc

    dataset_path = Path(args.dataset)
    examples = load_jsonl(dataset_path)
    protocol = load_protocol(args.protocol)
    verify_personalization_protocol(
        protocol,
        dataset_sha256=sha256_file(dataset_path),
    )
    quality = assess_evidence_quality(examples, protocol)
    if quality["status"] != "PASS":
        raise SystemExit(
            "L28 sentinel evidence quality BLOCKED: "
            + ",".join(quality["reasons"])
        )
    test_examples = examples_for_split(
        examples,
        protocol,
        "test",
    )

    latent = LatentStore(args.latent).load()
    if latent is None:
        raise SystemExit(f"persistent latent not found: {args.latent}")

    initial, initial_meta = load_factorized_model(
        args.initial,
        latent.values,
        device=args.device,
    )
    final, final_meta = load_factorized_model(
        args.final,
        latent.values,
        device=args.device,
    )
    if (
        initial_meta["source_l16_checkpoint_sha256"]
        != final_meta["source_l16_checkpoint_sha256"]
    ):
        raise SystemExit(
            "initial/final checkpoints do not share the same L16 ancestry"
        )
    if initial.boundary.config != final.boundary.config:
        raise SystemExit(
            "initial/final factorized boundary configs differ"
        )
    if initial.cortex.config != final.cortex.config:
        raise SystemExit(
            "initial/final recurrent cortex configs differ"
        )
    if (
        bool(initial.carry_recurrent_state)
        != bool(final.carry_recurrent_state)
    ):
        raise SystemExit(
            "initial/final recurrent-state carry policy differs"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=True,
    )
    encoded = encode(
        tokenizer,
        test_examples,
        latent.values,
        args.max_length,
    )
    initial_values = per_example_nll(initial, encoded)
    final_values = per_example_nll(final, encoded)

    receipt = assess_long_horizon_retention(
        protocol,
        initial_checkpoint_sha256=initial_meta["checkpoint_sha256"],
        final_checkpoint_sha256=final_meta["checkpoint_sha256"],
        initial_values=initial_values,
        final_values=final_values,
        policy=LongHorizonRetentionPolicy(
            max_overall_regression=args.max_overall_regression,
            max_worst_group_regression=args.max_worst_group_regression,
        ),
    )
    receipt["evidence_quality_court_sha256"] = quality["court_sha256"]
    receipt["endpoint_state"] = {
        "initial_boundary_state_digest": initial_meta[
            "boundary_state_digest"
        ],
        "initial_cortex_state_digest": initial_meta[
            "cortex_state_digest"
        ],
        "final_boundary_state_digest": final_meta[
            "boundary_state_digest"
        ],
        "final_cortex_state_digest": final_meta[
            "cortex_state_digest"
        ],
    }
    body = dict(receipt)
    body.pop("court_sha256", None)
    receipt["court_sha256"] = payload_digest(body)

    rendered = json.dumps(
        receipt,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
