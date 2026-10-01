from __future__ import annotations

import argparse
from pathlib import Path

from nolane_personal.hybrid_artifact import load_hybrid_artifact
from nolane_personal.latent import LatentStore
from nolane_personal.qwen import QwenCortex, SYSTEM_PROMPT
from nolane_personal.recurrent_state import RecurrentStateBundle, RecurrentStateStore
from nolane_personal.surgery import module_parameter_digest
from nolane_personal.surgery_candidate import load_model_lock, model_lock_fingerprint


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-lock", default=str(ROOT / "model.lock.json"))
    parser.add_argument("--model", default=str(ROOT / "models/Qwen3-0.6B"))
    parser.add_argument("--hybrid", default="runtime-data/l7-hybrid-recurrent/hybrid-recurrent.pt")
    parser.add_argument("--latent", default="runtime-data/living-core-shadow/latent.json")
    parser.add_argument("--state", default="runtime-data/l7-hybrid-recurrent/recurrent-state.json")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--reset-state", action="store_true")
    args = parser.parse_args()

    latent = LatentStore(args.latent).load()
    if latent is None:
        raise SystemExit(f"persistent latent not found: {args.latent}")

    lock = load_model_lock(args.model_lock)
    fingerprint = model_lock_fingerprint(lock)
    qwen = QwenCortex(args.model, device=args.device)
    cortex, metadata = load_hybrid_artifact(
        args.hybrid,
        expected_base_model_fingerprint=fingerprint,
        expected_hidden_size=int(qwen.model.config.hidden_size),
        latent=latent.values,
        model=qwen.model,
    )
    cortex.mixer.to(qwen.device).eval()
    mixer_digest = module_parameter_digest(cortex.mixer.module)
    state_store = RecurrentStateStore(args.state)

    if args.reset_state and state_store.path.exists():
        state_store.path.unlink()

    saved = state_store.load_bound(
        identity_id=latent.identity_id,
        base_model_fingerprint=fingerprint,
        mixer_digest=mixer_digest,
        latent_digest=latent.digest,
        recurrent_dim=cortex.mixer.config.recurrent_dim,
    )
    if saved is not None:
        torch = cortex.mixer.torch
        cortex.recurrent_states = {
            int(layer): torch.tensor([values], dtype=torch.float32, device=qwen.device)
            for layer, values in saved.layer_states.items()
        }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": args.prompt},
    ]
    try:
        rendered = qwen.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        rendered = qwen.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    inputs = qwen.tokenizer(rendered, return_tensors="pt").to(qwen.device)
    output = cortex.generate(
        **dict(inputs),
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=qwen.tokenizer.eos_token_id,
    )
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    response = qwen.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    layer_states = {
        str(layer): state.detach().float().cpu()[0].tolist()
        for layer, state in cortex.recurrent_states.items()
    }
    sequence = (saved.sequence + 1) if saved is not None else 1
    state_store.save(
        RecurrentStateBundle(
            identity_id=latent.identity_id,
            base_model_fingerprint=fingerprint,
            mixer_digest=mixer_digest,
            latent_digest=latent.digest,
            recurrent_dim=cortex.mixer.config.recurrent_dim,
            layer_states=layer_states,
            sequence=sequence,
        )
    )

    print(f"[UNPROMOTED HYBRID RECURRENT CORTEX | {metadata['checkpoint_sha256'][:12]} | recurrent-turn={sequence}]")
    print(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
