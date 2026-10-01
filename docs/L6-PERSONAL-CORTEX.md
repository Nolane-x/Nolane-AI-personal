# L6 Trainable Personal Cortex

## Goal

L6 is the first stage where Nolane AI Personal trains a neural path that directly changes Qwen hidden-state computation.

This is not prompt tuning.

The base Qwen checkpoint is loaded normally, then every base parameter is frozen:

```text
Qwen parameter.requires_grad = False
```

Only the persistent-latent adapter receives gradients.

## Architecture

```text
persistent Living latent (32D)
          |
          v
      LayerNorm
          |
        32 -> 16
          |
         SiLU
          |
      16 -> hidden_size
          |
      bounded gate
          |
          +--------------------+
                               v
Qwen block -> hidden state + latent residual -> later Qwen blocks -> logits
```

The adapter remains a separate artifact. It does not overwrite the pinned Qwen safetensors.

For Qwen3-0.6B, the exact adapter parameter count is derived from the actual local model hidden size at candidate creation time. The parameter cap remains 100K.

## Gradient boundary

Training uses normal autograd through Qwen operations after the injection points, but Qwen weights remain frozen.

The trainer records:

- frozen Qwen parameter count;
- trainable adapter parameter count;
- number of Qwen parameters that ever received a gradient;
- number of adapter parameters that received gradients;
- base-model parameter guard before/after;
- adapter digest before/after;
- loss before/after;
- effective gate before/after.

If a Qwen parameter receives a gradient, training aborts.

## Personalization dataset

The local dataset is JSONL:

```json
{"prompt":"Nay mệt quá.","target":"Thế nghỉ tí đi, đừng cố quá :))","language":"vi"}
```

Optionally an example may carry its own 32D latent and weight.

Private training data belongs under `runtime-data/` and is ignored by Git.

## Frozen protocol

Freeze the chronological train/dev/test split before training:

```bash
python scripts/freeze_personalization_protocol.py \
  --dataset runtime-data/personalization.jsonl \
  --output runtime-data/personalization-protocol-v1.json
```

The protocol binds every example to:

- chronological index;
- prompt;
- target;
- latent, if supplied;
- weight;
- language;
- example SHA-256;
- full dataset SHA-256;
- protocol SHA-256.

Training consumes the train split only.

## Train

First create the L5 base adapter candidate, then:

```bash
python scripts/train_personal_cortex.py \
  --candidate runtime-data/l5-adapter-candidate/latent-adapter.pt \
  --latent runtime-data/living-core-shadow/latent.json \
  --dataset runtime-data/personalization.jsonl \
  --protocol runtime-data/personalization-protocol-v1.json \
  --output-dir runtime-data/l6-personal-cortex
```

The artifact is marked:

`TRAINED_CANDIDATE_UNPROMOTED`

It is not silently enabled in the normal runtime.

## Held-out quality court

```bash
python scripts/evaluate_personal_cortex.py \
  --adapter runtime-data/l6-personal-cortex/personal-cortex-adapter.pt
```

The court compares the exact same held-out targets under:

1. untouched Qwen;
2. latent-conditioned Personal Cortex.

It also evaluates a frozen Vietnamese/English general-language anchor suite.

Default quality gates require:

- at least 2 personal held-out examples;
- at least 4 general anchor examples;
- held-out personal NLL improvement >= 0.01;
- general-anchor NLL regression <= 0.05;
- adapter <= 100K parameters;
- Qwen base parameter guard unchanged;
- zero Qwen gradients.

Passing this court still does not automatically enable production generation. Real user data must be sufficient and representative.

## Experimental generation

A trained candidate can be exercised explicitly:

```bash
python scripts/generate_personal_cortex.py \
  --adapter runtime-data/l6-personal-cortex/personal-cortex-adapter.pt \
  --prompt "Nay tôi hơi mệt."
```

The command labels output as `UNPROMOTED PERSONAL CORTEX`.

## Neural CI

The repository court instantiates a real tiny `Qwen3ForCausalLM`, freezes its complete base model, trains the latent adapter through actual Qwen decoder blocks, and requires:

- adapter gradients exist;
- Qwen gradients remain zero;
- adapter digest changes;
- Qwen parameter guard remains unchanged;
- training loss decreases;
- held-out loss on a different token sequence improves;
- generation through the trained path works;
- hooks are removed after generation.

This proves the trainable model-level mechanism. It does not prove the real 0.6B adapter has learned a high-quality personality until the real frozen-data court is run.
