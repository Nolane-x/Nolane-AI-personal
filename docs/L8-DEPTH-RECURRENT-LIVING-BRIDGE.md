# L8 Depth-Recurrent Living Bridge

## Purpose

L7 Hybrid Recurrent Cortex recurs **through tokens** and can persist per-layer state across calls.

L8 explores an orthogonal recurrence axis: **through model depth**.

During one Qwen forward, one small bridge state moves across selected decoder layers:

```text
persistent Living latent 32D
          |
       RMS norm
          |
          v
     latent feature
          |
          +------------------------------+
                                         |
Qwen layer i hidden -> hidden summary    |
          |                              |
          v                              |
     hidden feature                      |
          |                              |
layer identity embedding ----------------+
          |
          v
      GRU state
          |
     +----+----+
     |         |
 residual   bounded gate
     |         |
     +----*----+
          |
          v
 modified hidden -> next selected Qwen layer
          |
          v
      GRU state continues across depth
```

This is different from L7:

- **L7:** token-recurrent, separate state per selected layer, state may persist across calls.
- **L8:** depth-recurrent, one state passes from selected layer to selected layer inside a forward.

The two architectures are kept separate so they can be compared scientifically rather than mixed before evidence exists.

## Latent normalization finding

The first L8 court found a real architecture bug: ordinary LayerNorm erased constant/common-mode Living latents such as `[+0.8,...]` and `[-0.8,...]`.

L8 therefore uses RMS-style latent normalization, preserving sign/direction while controlling scale. The court requires the same prompt to produce a measurably different neural path for opposite Living latents.

## Footprint

Default configuration:

- Living latent: 32D
- depth bridge: 32D
- layer identities: up to 64
- Qwen hidden size: 1024

Exact trainable footprint: **84,289 parameters**.

The hard promotion cap remains **100,000 parameters**.

## Training

L8 reuses the exact frozen personalization protocol used by L6 and L7.

```bash
python scripts/train_depth_bridge.py \
  --dataset runtime-data/personalization.jsonl \
  --protocol runtime-data/personalization-protocol-v1.json
```

Base Qwen is frozen. Only the depth bridge receives gradients.

## Four-way held-out court

```bash
python scripts/evaluate_depth_bridge.py
```

The same held-out examples are scored by:

1. untouched Qwen;
2. L6 Personal Cortex;
3. L7 Hybrid Recurrent Cortex;
4. L8 Depth-Recurrent Living Bridge.

L8 is blocked unless it improves held-out NLL by at least:

- **0.010 vs Qwen base**;
- **0.005 vs L6**;
- **0.005 vs L7 Hybrid**.

It must also keep Vietnamese/English anchor regression <= 0.05, remain <=100K parameters, preserve Qwen weights, and produce zero Qwen gradients.

All compared artifacts must carry the **same personalization protocol SHA-256**.

## Artifact

Training creates:

```text
runtime-data/l8-depth-bridge/depth-bridge.pt
runtime-data/l8-depth-bridge/depth-bridge-manifest.json
```

The artifact is bound to the base-model fingerprint, hidden size and frozen dataset protocol.

## Generation

```bash
python scripts/generate_depth_bridge.py \
  --prompt "Nay tôi hơi mệt."
```

This remains experimental/unpromoted until the real four-way court passes.

## Scientific boundary

L8 still uses Qwen Transformer blocks. It adds a recurrent path across their depth; it does not yet replace attention.

The next deeper surgery should only replace or bypass Transformer computation after L6/L7/L8 have been compared under the same real evidence.
