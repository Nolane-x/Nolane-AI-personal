# L11 Recurrent Transformer Islands

## Purpose

L10 can progressively replace many individual Qwen decoder blocks, but each selected block still invokes the recurrent substitute separately.

L11 compresses **whole contiguous Transformer regions** into single recurrent islands.

An island `[start..end]` executes exactly one recurrent replacement call at `start`. Every later Qwen block inside that region becomes an identity mapping. The original attention and MLP computation of all blocks in the island is absent from that forward.

```text
Qwen block before island
        |
        v
hidden entering block 4
        |
        v
  recurrent island
     ONE call
        |
        +---- replaces original Qwen blocks 4,5,6,7
        |
        v
Qwen block after island
```

Thus a width-4 island compresses four Transformer blocks to one recurrent computation.

## Region calibration

L11 does not simply group L10's individually ranked layers. It calibrates candidate **contiguous spans directly**.

For every candidate span, default width 2–4 blocks, untouched Qwen provides:

- hidden state before the first block;
- hidden state after the last block.

The plan measures:

- residual RMS ratio across the full region;
- cosine change across the full region;
- composite transformation score;
- transformation score per removed Transformer block.

Low-difficulty non-overlapping regions are selected first. Edge decoder regions remain protected and islands have a configurable minimum gap.

```bash
python scripts/freeze_recurrent_island_plan.py
```

The frozen plan is bound to exact Qwen fingerprint, personalization protocol, island selection order, stages and plan SHA-256.

## Region-level teacher distillation

The teacher is the **entire original Transformer region**.

For island `[4..7]`:

```text
teacher input  = hidden before Qwen block 4
teacher target = hidden after  Qwen block 7
student        = one recurrent island call
```

Training then switches to the genuine island path where blocks 4–7 no longer execute.

Stages add islands monotonically. After each stage, dev NLL is evaluated. If the new island set regresses beyond budget, replacement weights roll back to the previous accepted stage and progression stops.

```bash
python scripts/train_recurrent_islands.py
```

## Quality court

```bash
python scripts/evaluate_recurrent_islands.py
```

L11 compares three paths on the exact same frozen held-out examples:

1. untouched Qwen;
2. L10 Progressive Replacement;
3. L11 Recurrent Islands.

Default quality gates require:

- enough held-out personal and Vietnamese/English anchor evidence;
- at least 25% of Qwen decoder depth genuinely replaced;
- region compression ratio >=2 Transformer blocks per island;
- improvement over untouched Qwen >=0.005 NLL;
- no more than 0.020 NLL degradation versus L10;
- general anchor regression <=0.05 NLL;
- cached/replay-safe generation equality;
- <=100K trainable replacement parameters;
- Qwen weights unchanged and zero Qwen gradients.

## Resource court

```bash
python scripts/benchmark_recurrent_island_resources.py
```

L11 must turn region compression into real compute benefit. Default resource gates require:

- at least 25% depth replaced;
- region compression ratio >=2;
- forward latency <=0.90x untouched Qwen;
- forward latency <=0.98x L10;
- cached generation >=1.05x replay-safe generation;
- artifact <=2 MB;
- <=100K trainable parameters.

## Promotion identity

Quality and resource evidence must refer to the same:

- island checkpoint SHA-256;
- frozen island-plan SHA-256.

```bash
python scripts/decide_recurrent_island_promotion.py \
  --quality runtime-data/l11-quality.json \
  --resources runtime-data/l11-resources.json
```

Any mismatch fails closed.

## Neural court evidence

CI instantiates a real tiny `Qwen3ForCausalLM` and verifies:

- a width-3 island causes original Qwen blocks 2,3,4 to execute zero times;
- that width-3 region invokes the recurrent replacement exactly once;
- trailing blocks inside the region execute only identity paths;
- island overlap and minimum-gap violations fail closed;
- candidate regions are calibrated directly on real Qwen hidden states;
- frozen plan digest/model/dataset lineage is enforced;
- cached and replay-safe deterministic generation match;
- multi-stage full-region teacher distillation trains the recurrent substitute;
- Qwen remains frozen and receives zero gradients;
- island artifact roundtrip preserves plan SHA and replacement-state digest.

## Scientific boundary

L11 is still a hybrid architecture because untouched Qwen Transformer regions remain before, between and after recurrent islands.

The tiny-Qwen3 neural court proves the mechanism is genuine; it does **not** prove that production Qwen3-0.6B can already lose 25–35% of its Transformer depth without quality loss.

Real promotion requires sufficient approved personal history, a frozen real held-out protocol, and measured Qwen3-0.6B quality plus CPU/GPU resource evidence.
