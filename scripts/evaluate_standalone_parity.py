from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from nolane_personal.latent import LatentStore
from nolane_personal.native_artifact import build_native_boundary_model
from nolane_personal.personal_dataset import encode_chat_example, load_jsonl
from nolane_personal.personal_protocol import examples_for_split, load_protocol, verify_personalization_protocol
from nolane_personal.qwen import QwenCortex, SYSTEM_PROMPT
from nolane_personal.standalone_artifact import load_standalone_model
from nolane_personal.standalone_evaluation import StandaloneParityEvidence, decide_standalone_parity
from nolane_personal.surgery import module_parameter_digest
from nolane_personal.surgery_candidate import load_model_lock, model_lock_fingerprint


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode(tokenizer, examples, max_length):
    rows = []
    for example in examples:
        ids, _labels = encode_chat_example(
            tokenizer,
            example,
            system_prompt=SYSTEM_PROMPT,
            max_length=max_length,
        )
        rows.append(ids)
    return rows


def checkpoint_has_decoder_keys(path: Path) -> bool:
    import torch
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    state_sections = [
        payload.get("boundary_state", {}),
        payload.get("cortex_state", {}),
    ]
    forbidden = ("model.layers.", ".self_attn.", ".mlp.")
    return any(
        any(token in str(key) for token in forbidden)
        for section in state_sections
        for key in section.keys()
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-lock", default=str(ROOT / "model.lock.json"))
    parser.add_argument("--model", default=str(ROOT / "models/Qwen3-0.6B"))
    parser.add_argument("--l15", default="runtime-data/l15-native-boundary/native-nolane-boundary.pt")
    parser.add_argument("--standalone", default="runtime-data/l16-standalone/standalone-nolane.pt")
    parser.add_argument("--latent", default="runtime-data/living-core-shadow/latent.json")
    parser.add_argument("--dataset", default="runtime-data/personalization.jsonl")
    parser.add_argument("--protocol", default="runtime-data/personalization-protocol-v1.json")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-examples", type=int, default=8)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    import torch

    dataset_path = Path(args.dataset)
    examples = load_jsonl(dataset_path)
    protocol = load_protocol(args.protocol)
    verify_personalization_protocol(
        protocol,
        dataset_sha256=sha256_file(dataset_path),
    )
    test_examples = examples_for_split(examples, protocol, "test")
    test_examples = test_examples[: max(1, args.max_examples)]

    latent = LatentStore(args.latent).load()
    if latent is None:
        raise SystemExit(f"persistent latent not found: {args.latent}")
    fingerprint = model_lock_fingerprint(load_model_lock(args.model_lock))
    qwen = QwenCortex(args.model, device=args.device)
    l15, l15_meta = build_native_boundary_model(
        qwen.model,
        args.l15,
        latent.values,
        expected_base_model_fingerprint=fingerprint,
        expected_dataset_fingerprint=protocol["protocol_sha256"],
    )
    standalone, standalone_meta = load_standalone_model(
        args.standalone,
        latent.values,
        device=qwen.device,
        expected_source_base_model_fingerprint=fingerprint,
        expected_dataset_fingerprint=protocol["protocol_sha256"],
    )

    encoded = encode(qwen.tokenizer, test_examples, args.max_length)
    max_logit = 0.0
    max_state = 0.0
    generation_equal = True
    scan_equal = True
    with torch.no_grad():
        for ids in encoded:
            tensor = torch.tensor([ids], dtype=torch.long, device=qwen.device)
            a = l15.forward(input_ids=tensor, state=None)
            b = standalone.forward(input_ids=tensor, state=None)
            max_logit = max(
                max_logit,
                float((a.logits-b.logits).abs().max().cpu()),
            )
            max_state = max(
                max_state,
                float((a.state-b.state).abs().max().cpu()),
            )
            generation_equal = generation_equal and bool(torch.equal(
                l15.generate(
                    input_ids=tensor,
                    max_new_tokens=3,
                    do_sample=False,
                    eos_token_id=None,
                ),
                standalone.generate(
                    input_ids=tensor,
                    max_new_tokens=3,
                    do_sample=False,
                    eos_token_id=None,
                ),
            ))
            scan_equal = scan_equal and standalone.prompt_scan_equivalent(tensor)

    probe = torch.tensor([encoded[0]], dtype=torch.long, device=qwen.device)
    before_mutation = standalone.forward(input_ids=probe, state=None).logits.detach().clone()
    with torch.no_grad():
        probe_token = int(encoded[0][0])
        source_row = qwen.model.model.embed_tokens.weight[probe_token].detach().clone()
        qwen.model.model.embed_tokens.weight[probe_token].add_(1.0)
        after_mutation = standalone.forward(input_ids=probe, state=None).logits.detach().clone()
        qwen.model.model.embed_tokens.weight[probe_token].copy_(source_row)
    isolated = bool(torch.equal(before_mutation, after_mutation))

    cortex_equal = (
        module_parameter_digest(standalone.cortex.module)
        == l15_meta["cortex_state_digest"]
    )
    runtime_has_qwen = hasattr(standalone, "qwen_model")
    decision = decide_standalone_parity(
        StandaloneParityEvidence(
            examples=len(encoded),
            max_logit_abs_error=max_logit,
            max_state_abs_error=max_state,
            greedy_generation_equal=generation_equal,
            prompt_scan_equivalence_passed=scan_equal,
            cortex_digest_equal_to_l15=cortex_equal,
            source_mutation_isolated=isolated,
            runtime_has_qwen_model_reference=runtime_has_qwen,
            runtime_requires_transformers=bool(
                standalone_meta["runtime_requires_transformers"]
            ),
            checkpoint_contains_decoder_keys=checkpoint_has_decoder_keys(
                Path(args.standalone)
            ),
            cortex_parameters=standalone.cortex_parameter_count(),
        )
    )
    result = {
        "schema": "NOLANE-L16-STANDALONE-PARITY-EVAL-V1",
        "standalone_checkpoint_sha256": standalone_meta["checkpoint_sha256"],
        "source_l15_checkpoint_sha256": standalone_meta[
            "source_l15_checkpoint_sha256"
        ],
        "decision": asdict(decision),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        if path.exists():
            raise SystemExit(f"refusing to overwrite parity evidence: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered+"\n", encoding="utf-8")
    print(rendered)
    return 0 if decision.status == "STANDALONE_PARITY_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
