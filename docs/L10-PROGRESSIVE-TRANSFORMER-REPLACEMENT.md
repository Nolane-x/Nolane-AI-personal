# L10 Progressive Transformer-Depth Replacement

## Purpose

L9 proved that selected Qwen decoder blocks can be bypassed and replaced by a compact recurrent module.

L10 turns that proof into a controlled **depth-reduction process**.

The goal is not to maximize the number of removed Transformer blocks. The goal is to remove progressively more Transformer depth only while held-out quality and real resource evidence continue to justify it.

## 1. Layer calibration

Before training, L10 runs untouched Qwen on frozen training evidence and measures how strongly each internal decoder block transforms its input:

- residual RMS relative to output RMS;
- cosine change between block input and output.

Blocks with smaller transformation scores are ranked as safer initial replacement candidates.

The first and last decoder regions are protected by default.

```bash
python scripts/freeze_progressive_replacement_plan.py
```

The resulting plan is immutable and bound to:

- exact Qwen fingerprint;
- exact personalization protocol;
- total decoder layer count;
- calibration scores;
- target replacement fraction;
- ordered target layers;
- every curriculum stage;
- plan SHA-256.

## 2. Progressive curriculum

Default target: **50% of Transformer depth**, capped at 12 selected blocks.

The curriculum grows approximately:

```text
1 block
  |
  v
2 blocks
  |
  v
4 blocks
  |
  v
8 blocks
  |
  v
target
```

Every later stage must be a strict superset of accepted earlier stages.

At each stage:

1. original frozen Qwen blocks act as hidden-state teachers;
2. the shared recurrent replacement is distilled on all currently selected layers;
3. the selected blocks are genuinely bypassed;
4. task NLL training runs on the frozen personalization train split;
5. dev NLL is measured;
6. if dev regression exceeds the configured budget, replacement weights are rolled back to the previous stage and progression stops.

```bash
python scripts/train_progressive_replacement.py
```

Thus L10 cannot claim progress merely by deleting more Transformer blocks.

## 3. Correct autoregressive recurrent state

L9 generation used `use_cache=False`.

That exposed a subtle state problem: no-cache generation replays the full prefix on every token, while a recurrent replacement state inside one generation session could otherwise carry across those repeated forwards and count old history more than once.

L10 adds explicit model-forward boundaries:

- **replay-safe / no-cache:** recurrent replacement state resets at the beginning of every model forward;
- **cached generation:** recurrent state persists across incremental token forwards.

The first Qwen decoder layer must remain untouched in cached mode.

```text
cached:
prompt forward -> recurrent state S_t
new token      -> continue S_t -> S_t+1

replay-safe:
prompt forward          -> reset -> recompute
prompt + token forward  -> reset -> recompute
```

Neural CI verifies both paths produce the same deterministic generated sequence on a real tiny `Qwen3ForCausalLM`.

## 4. Quality court

```bash
python scripts/evaluate_progressive_replacement.py
```

Default quality gates require:

- held-out personalization evidence;
- Vietnamese/English general anchor evidence;
- >=25% of Qwen decoder depth genuinely replaced;
- >=2 accepted curriculum stages;
- NLL improvement over untouched Qwen >= 0.005;
- no more than 0.015 NLL degradation versus the L9 checkpoint;
- general anchor regression <=0.05 NLL;
- <=100K trainable replacement parameters;
- cached generation contract PASS;
- base Qwen unchanged;
- base Qwen gradient count = 0.

## 5. Resource court

```bash
python scripts/benchmark_progressive_replacement_resources.py
```

Default gates require:

- >=25% decoder depth replaced;
- median forward latency <=0.90x untouched Qwen;
- cached generation >=1.05x faster than replay-safe generation;
- artifact <=2 MB;
- <=100K trainable parameters.

L10 therefore has to convert removed Transformer depth into measurable compute benefit.

## 6. Promotion identity

Quality and resource courts must refer to both:

- the exact same L10 checkpoint SHA-256;
- the exact same frozen progressive plan SHA-256.

```bash
python scripts/decide_progressive_replacement_promotion.py \
  --quality runtime-data/l10-quality.json \
  --resources runtime-data/l10-resources.json
```

Mismatched checkpoints or plans fail closed.

## Engineering evidence

The CI court instantiates real tiny Qwen3 models and proves:

- layer calibration covers only eligible internal blocks;
- edge layers stay protected;
- progressive stages are monotonic;
- plan/model/dataset drift fails closed;
- no-cache recurrent history is reset per model forward;
- cached recurrent state persists across incremental forwards;
- cached and replay-safe deterministic outputs match;
- progressive stage training changes only replacement weights;
- all Qwen parameters stay frozen and receive zero gradients;
- multi-stage training reaches the frozen target plan under permissive test conditions;
- L10 artifact roundtrip preserves plan SHA and replacement digest.

## Scientific boundary

L10 is still a hybrid model: untouched Qwen Transformer blocks remain around the recurrent replacements.

Engineering closure does **not** mean 50% of Qwen3-0.6B can already be removed at production quality. Real promotion still requires frozen real-user quality evidence and measured Qwen3-0.6B latency/resource evidence.

The next architectural boundary is to determine whether larger contiguous Transformer regions can be collapsed into recurrent/state-space islands without losing the language competence inherited from Qwen.
