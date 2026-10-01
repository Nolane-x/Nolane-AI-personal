# L9 Recurrent Transformer-Block Replacement

## Purpose

L5-L8 alter Qwen hidden-state computation while still executing every Transformer decoder block.

L9 crosses that boundary.

Selected frozen Qwen decoder blocks are **not called**. A compact recurrent module replaces their forward computation.

This is the first Nolane AI Personal wave that can remove Transformer attention/MLP computation from selected depth positions at inference time.

## Replacement architecture

```text
incoming hidden sequence
        |
        +--> hidden norm -> projection -----+
        |                                   |
Living latent 32D -> RMS norm -> projection |
        |                                   |
layer identity -----------------------------+
                                            |
                                            v
                                  vectorized GRU sequence
                                            |
                                  residual + sigmoid gate
                                            |
                                            v
                                 replacement hidden sequence
                                            |
                         original Qwen block is NOT executed
```

A single replacement module is shared by all selected layers; learned layer embeddings let it specialize by depth.

Default Qwen-hidden-1024 footprint: **84,289 trainable parameters**.

## Why two-stage training

Training from random replacement weights directly under task NLL can destabilize the network.

L9 first uses the original frozen Qwen blocks as teachers:

1. run the untouched Qwen path;
2. capture each selected block's input/output hidden tensors;
3. train the recurrent replacement to approximate teacher output with Smooth-L1 + cosine loss;
4. switch to true bypass mode;
5. fine-tune the replacement end-to-end on the frozen personalization train split.

Qwen weights remain frozen in both phases.

```bash
python scripts/train_block_replacement.py
```

By default the trainer replaces two internal layers near one-third and two-thirds depth. Layer indices can be supplied explicitly.

## Real bypass court

CI uses a real tiny `Qwen3ForCausalLM` and instruments the original selected decoder block.

Inside `BlockReplacementSession` the original block forward-call count must remain **zero**.

The court also verifies:

- exact 17,825 parameters at hidden size 32;
- exact 84,289 parameters at hidden size 1024;
- exception-safe restoration of original layer forwards;
- teacher hidden-state capture;
- distillation gradients on replacement only;
- zero Qwen gradients;
- unchanged Qwen parameter guard;
- task loss reduction;
- synthetic held-out improvement;
- artifact roundtrip and lineage;
- generation with selected blocks bypassed.

## Five-way quality court

```bash
python scripts/evaluate_block_replacement.py
```

The exact same frozen held-out examples are scored by:

1. untouched Qwen;
2. L6 Personal Cortex;
3. L7 Hybrid Recurrent Cortex;
4. L8 Depth-Recurrent Living Bridge;
5. L9 Recurrent Block Replacement.

Default quality gates require:

- at least 2 held-out personalization examples;
- at least 4 Vietnamese/English anchor examples;
- at least one decoder block genuinely skipped;
- L9 improves NLL over untouched Qwen by >= **0.005**;
- L9 is no worse than the best of L6/L7/L8 by more than **0.010 NLL**;
- general anchor regression <= **0.05 NLL**;
- <=100K trainable replacement parameters;
- Qwen weights unchanged;
- Qwen gradient count = 0.

L9 is allowed a tiny non-inferiority margin against the best prior architecture because its purpose includes real compute removal.

## Resource court

```bash
python scripts/benchmark_block_replacement_resources.py
```

Default resource gates require:

- at least 4 prompts;
- at least one but not all Qwen blocks skipped;
- median forward latency <= **0.95×** untouched Qwen;
- artifact <= **2 MB**;
- <=100K trainable parameters.

Thus a replacement that saves no real wall-clock time is rejected even if it bypasses a block structurally.

## Promotion ceremony

Quality and resource evidence must refer to the exact same L9 checkpoint:

```bash
python scripts/decide_block_replacement_promotion.py \
  --quality runtime-data/l9-quality.json \
  --resources runtime-data/l9-resources.json
```

Only then can `BLOCK_REPLACEMENT_PROMOTION_PASS` be emitted.

## Generation

```bash
python scripts/generate_block_replacement.py \
  --prompt "Nay tôi hơi mệt."
```

The experimental path uses `use_cache=False` so skipped attention blocks do not need to emulate Qwen KV-cache semantics.

## Scientific boundary

L9 replaces selected Transformer blocks, not the entire model. It remains an unpromoted candidate until real Qwen3-0.6B personalization and resource courts pass.

A later wave may investigate cache-compatible replacement or progressively replacing a larger fraction of Transformer depth, but only after L9 demonstrates a real quality/latency tradeoff.
