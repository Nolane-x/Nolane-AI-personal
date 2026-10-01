# L13 Shrinking Qwen Scaffold

## Purpose

L12 can replace one wide middle region of Qwen with a selective state-space cortex. L13 makes the dependency on Qwen itself an explicit optimization target.

Instead of asking only how many Transformer blocks are replaced, L13 asks:

**How few Qwen decoder blocks can remain while the Nolane cortex preserves held-out quality and earns real compute savings?**

The surviving Transformer blocks form a thin language scaffold at the head and tail of the model. The middle is replaced by one compact multi-timescale state-space cortex.

## Multi-timescale cortex

L13 maintains two causal recurrent states per token:

- **fast state**: free token-conditioned decay, optimized for local changes;
- **slow state**: token-conditioned decay with a retention floor, optimized for longer-lived context.

Both states are conditioned by the persistent 32D Living latent and jointly project back into Qwen hidden space.

```text
hidden_t
   |
   v
low-rank token feature
   |
   +-------------------------+
   |                         |
   v                         v
fast proposal/decay     slow proposal/decay
   |                         |
fast state_t ----------> slow state_t
   |                         |
   +-----------+-------------+
               |
               v
       gated joint projection
               |
               v
          hidden'_t
```

The slow-state decay is constrained to remain at or above the configured retention floor, so the two states cannot collapse into the same nominal time scale merely by naming.

## Exact footprint

At Qwen hidden size 1024 with fast/slow state width 24:

**81,249 trainable parameters**

The production parameter cap remains **100,000**.

Parameter count does not grow with the number of Qwen blocks removed.

## Scaffold shrinking

L13 keeps Qwen only around the outside of the cortex:

```text
Qwen head anchors
      |
      v
Multi-Timescale State-Space Cortex
      |
      +--> central Qwen attention/MLP blocks: not executed
      |
      v
Qwen tail anchors
```

The frozen plan progressively shrinks the remaining Qwen scaffold. Default real-Qwen schedule:

```text
~50% Qwen remains
       |
       v
~40% remains
       |
       v
~32% remains
       |
       v
~25% remains
```

That corresponds to a default target of roughly **75% Transformer-depth removal**.

Each stage calibrates all valid head/tail splits for the requested remaining-layer count. Region transformation difficulty dominates selection; a small balance penalty only breaks ties, so L13 may keep an asymmetric shell when Qwen evidence supports it.

At least two head and two tail decoder blocks are protected by default.

```bash
python scripts/freeze_scaffold_plan.py
```

The frozen plan is bound to:

- exact Qwen fingerprint;
- exact personalization protocol;
- every calibrated region score;
- head/tail anchor counts at each stage;
- plan SHA-256.

## Training

L13 reuses the whole-region distillation discipline from L12:

1. untouched Qwen maps hidden-before-region to hidden-after-region;
2. the multi-timescale cortex learns that mapping;
3. original central Qwen blocks are removed from the actual training forward;
4. personalization task training runs through the thin scaffold;
5. dev NLL decides whether the shell may shrink further;
6. regression beyond budget rolls cortex weights back to the prior accepted stage.

```bash
python scripts/train_shrinking_scaffold.py
```

## Sequence-state contracts

Neural court requires:

- full-sequence scan equals token-by-token scan with carried packed fast/slow state;
- fast and slow final states are not numerically identical;
- measured slow-state decay respects the retention floor;
- cached autoregressive generation equals replay-safe no-cache generation.

These contracts prevent the multi-timescale design from degenerating into a cosmetic second state.

## Quality court

```bash
python scripts/evaluate_shrinking_scaffold.py
```

The evaluator compares:

1. untouched Qwen;
2. L12 Selective State-Space Cortex;
3. L13 Shrinking Qwen Scaffold.

Default promotion gates include:

- Qwen scaffold <=35% of total decoder depth;
- at least 2 Qwen head anchors and 2 tail anchors;
- at least two accepted shrink stages;
- >=0.005 NLL improvement over untouched Qwen;
- <=0.020 NLL degradation versus L12;
- Vietnamese/English anchor regression <=0.05 NLL;
- cached-generation contract PASS;
- scan-equivalence contract PASS;
- fast/slow separation contract PASS;
- <=100K trainable cortex parameters;
- base Qwen unchanged and zero Qwen gradients.

## Resource court

```bash
python scripts/benchmark_scaffold_resources.py
```

Default resource gates require:

- Qwen scaffold <=35%;
- forward latency <=0.75x untouched Qwen;
- forward latency <=0.93x L12;
- cached generation >=1.05x replay-safe generation;
- artifact <=2 MB;
- <=100K trainable parameters.

## Promotion identity

Quality and resource courts must use the exact same checkpoint SHA-256 and frozen scaffold-plan SHA-256.

```bash
python scripts/decide_scaffold_promotion.py \
  --quality runtime-data/l13-quality.json \
  --resources runtime-data/l13-resources.json
```

Identity mismatch fails closed.

## Neural court evidence

The engineering court uses a real tiny 12-layer `Qwen3ForCausalLM` and proves:

- exact 81,249-parameter contract at hidden size 1024;
- full scan equals incremental packed-state scan;
- fast and slow states separate dynamically;
- Living-latent sign changes the neural path;
- a final test shell retains only 2 head + 2 tail Qwen blocks;
- the central 8/12 Qwen blocks execute zero original attention/MLP forwards;
- one multi-timescale cortex call replaces that central region;
- scaffold calibration respects anchor boundaries;
- scaffold shrinking is monotonic;
- cached and replay-safe generations match;
- staged whole-region distillation/training changes only cortex weights;
- base Qwen receives zero gradients;
- artifact roundtrip preserves checkpoint/plan/state lineage.

## Scientific boundary

L13 is much closer to a Nolane-native cortex, but Qwen still supplies embeddings, the language-model head, and a thin decoder shell.

The tiny-Qwen3 court proves architecture mechanics. It does **not** prove that real Qwen3-0.6B can already discard roughly 75% of decoder depth at equal quality.

Real promotion still requires sufficient approved personalization history, frozen held-out quality evidence, and real CPU/GPU resource measurements.
