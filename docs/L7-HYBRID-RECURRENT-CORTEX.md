# L7 Hybrid Recurrent Cortex

## Goal

L7 adds a recurrent neural pathway **inside the Qwen decoder computation**.

This is more than a static residual adapter. Selected Qwen decoder layers share one small recurrent mixer, while each selected layer carries its own recurrent state.

The state can continue:

- from token to token during one forward/generation;
- from one generation call to the next;
- across process restarts when sealed to disk.

## Architecture

```text
Qwen decoder hidden state
          |
          v
      LayerNorm
          |
    hidden -> R
          |
          v
      GRUCell
      ^      |
      |      +---- previous recurrent state
      |
persistent Living latent 32D
      |
   32 -> R initializes recurrent state
          |
          v
      R -> hidden
          |
     bounded gate
          |
          v
Qwen hidden + recurrent residual
          |
          v
next Qwen decoder layer
```

The recurrent mixer weights are shared across selected Qwen layers to keep the footprint small. Each selected layer keeps a separate recurrent state.

With Qwen hidden size 1024 and the default recurrent width `R=24`, the exact analytical footprint is **56,641 parameters**.

## Why this differs from ordinary Transformer inference

A normal decoder layer is driven by token context/KV-cache and its fixed weights.

The L7 path adds another transition:

```text
r[t+1] = GRU(phi(h[t]), r[t])
h'[t]  = h[t] + gate * W_out(r[t+1])
```

and the initial recurrent state is conditioned by the persistent Living latent:

```text
r[0] = tanh(W_latent * z_living)
```

Therefore the same token sequence can produce a different path when the recurrent state entering the call differs.

This does not remove Transformer attention yet. It creates a hybrid Transformer + recurrent architecture and establishes the court infrastructure needed before any attention/block replacement experiment.

## Base-Qwen boundary

Every base Qwen parameter is frozen.

Only recurrent mixer parameters are trainable.

Training aborts if any base Qwen parameter receives a gradient.

The recurrent artifact remains separate from pinned Qwen weights and is marked:

`HYBRID_RECURRENT_CANDIDATE_UNPROMOTED`

## Training

L7 reuses the frozen personalization protocol from L6:

```bash
python scripts/train_hybrid_recurrent_cortex.py \
  --dataset runtime-data/personalization.jsonl \
  --protocol runtime-data/personalization-protocol-v1.json \
  --recurrent-dim 24
```

The trainer consumes the train split only and rejects mixer configurations above the 100K parameter cap.

## Held-out court

```bash
python scripts/evaluate_hybrid_recurrent_cortex.py
```

The same frozen test split is scored with:

1. untouched Qwen;
2. Hybrid Recurrent Cortex.

The frozen Vietnamese/English general anchor is also measured.

Default gates require:

- sufficient held-out personal examples;
- sufficient anchor examples;
- personal NLL improvement >= 0.01;
- general-anchor NLL regression <= 0.05;
- mixer <= 100K parameters;
- unchanged base-Qwen parameter guard;
- zero gradients on Qwen.

## Persistent neural state across calls

Experimental generation:

```bash
python scripts/generate_hybrid_recurrent_cortex.py \
  --prompt "Nay tôi hơi mệt."
```

After generation, per-layer recurrent states are sealed into:

```text
runtime-data/l7-hybrid-recurrent/recurrent-state.json
```

The state artifact binds to:

- AI identity;
- exact base-model fingerprint;
- exact mixer digest;
- exact persistent-latent digest;
- recurrent dimension;
- monotonic turn sequence;
- self-digest.

On the next call, the state is loaded and the model continues from it.

If any binding changes, loading fails closed. `--reset-state` is the explicit reset path.

## Real Qwen3 neural court

CI instantiates a real tiny `Qwen3ForCausalLM` and proves:

- exact recurrent parameter accounting;
- the same input changes on a second call when state is carried;
- resetting recurrent state restores the original trajectory;
- mixer gradients exist;
- base Qwen gradients remain zero;
- mixer digest changes under training;
- base Qwen guard remains unchanged;
- training loss decreases;
- held-out loss on different token sequences improves;
- recurrent generation works;
- forward hooks are removed after generation;
- recurrent-state persistence/tamper/binding checks work.

## Scientific boundary

L7 is a genuine recurrent model path inside Qwen, but it is still **hybrid**:

```text
Transformer blocks + recurrent mixer
```

It is not yet:

```text
Transformer attention removed/replaced
```

A later wave may replace or bypass selected attention/decoder blocks only if matched courts demonstrate better quality/continuity/compute tradeoffs.
