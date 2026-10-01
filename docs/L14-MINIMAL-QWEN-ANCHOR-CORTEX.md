# L14 Minimal Qwen Anchor Cortex

## Purpose

L13 shrinks Qwen to a thin head/tail scaffold around a Nolane multi-timescale cortex.

L14 pushes that boundary to the minimum cache-safe Qwen shell used by the current architecture: **one decoder block at the head and one at the tail**.

For a 28-layer Qwen scaffold this means 26/28 decoder blocks can be absent from the experimental forward path.

## Architecture

L14 adds a deep-recurrent state-space cortex with three packed states:

- fast temporal state;
- slow temporal state;
- virtual-depth state.

The fast and slow states carry information across tokens. For every token, the depth state then performs several recurrent micro-steps with shared parameters and learned step embeddings.

```text
Qwen block 0
    |
    v
token hidden
    |
    v
fast state ----+
slow state ----+--> virtual-depth recurrent micro-steps
               |         x 8 default
               v
          depth state
               |
               v
        bounded residual
               |
               v
Qwen final block
```

The virtual-depth recurrence is intended to approximate the composition of many removed Transformer blocks without adding parameters proportional to removed Qwen depth.

## Default footprint

For hidden size 1024, 32D Living latent, state dimension 20, eight active virtual-depth steps and sixteen allocated step identities:

**89,805 trainable parameters**

The hard architecture cap remains **100,000 parameters**.

## Minimal-anchor plan

The frozen plan shrinks remaining Qwen depth through calibrated nested regions and ends at exactly two decoder anchors by default:

- at least one head block;
- at least one tail block;
- all decoder blocks between them replaced by the deep recurrent cortex.

For Qwen3-0.6B's 28-layer-style scaffold, the intended experimental endpoint is 2/28 Qwen decoder blocks remaining.

```bash
python scripts/freeze_anchor_plan.py
```

The plan is bound to exact base-model fingerprint, personalization protocol and plan SHA-256.

## Training

```bash
python scripts/train_minimal_anchor_cortex.py
```

Training reuses the rollback-safe whole-region court:

1. untouched Qwen provides the full removed-region teacher target;
2. the deep-recurrent cortex distills that mapping;
3. the central Qwen region is genuinely bypassed;
4. personalization loss trains the cortex;
5. dev NLL decides whether the thinner shell is accepted;
6. failed shrink stages restore the previous cortex weights.

Qwen remains frozen and receives zero gradients.

## Quality court

```bash
python scripts/evaluate_minimal_anchor_cortex.py
```

The evaluator compares untouched Qwen, L13 and L14 on the same frozen held-out protocol.

Default production gates require:

- Qwen decoder anchors <=15% of total depth;
- at least one head and one tail anchor;
- >=2 accepted shrink stages;
- >=4 active virtual-depth micro-steps;
- held-out improvement over untouched Qwen >=0.005 NLL;
- no more than 0.020 NLL degradation versus L13;
- Vietnamese/English anchor regression <=0.05 NLL;
- cached/replay-safe generation equality;
- full-sequence/incremental recurrent scan equality;
- measurable virtual-depth effect;
- <=100K cortex parameters;
- unchanged and zero-gradient Qwen weights.

## Resource court

```bash
python scripts/benchmark_anchor_resources.py
```

Default resource gates require:

- Qwen anchors <=15% of decoder depth;
- median forward latency <=0.65x untouched Qwen;
- median forward latency <=0.85x L13;
- cached generation >=1.05x replay-safe generation;
- artifact <=2 MB;
- cortex <=100K parameters.

## Promotion

```bash
python scripts/decide_anchor_promotion.py \
  --quality runtime-data/l14-quality.json \
  --resources runtime-data/l14-resources.json
```

Quality and resource courts must refer to the exact same checkpoint SHA-256 and frozen anchor-plan SHA-256.

## Neural court evidence

CI instantiates a real tiny 12-layer `Qwen3ForCausalLM` and proves:

- a region covering layers 1..10 executes all ten original Qwen decoder blocks zero times;
- the deep recurrent cortex executes once for that whole region;
- only Qwen layer 0 and layer 11 remain outside the replacement region;
- exact analytical parameter count equals runtime parameter count;
- full-sequence scan equals token-by-token carried-state scan;
- eight-step virtual depth produces different hidden/state trajectories from one-step depth;
- cached and replay-safe deterministic generation match;
- staged whole-region distillation trains the cortex;
- all Qwen parameters remain frozen with zero gradients;
- artifact roundtrip preserves checkpoint, plan and cortex-state lineage.

## Scientific boundary

L14 is still a hybrid model because Qwen embeddings, one head decoder block, one tail decoder block, final normalization and LM head remain.

The court proves the **mechanism** can run with an extremely thin Qwen decoder shell. It does not prove that real Qwen3-0.6B can already discard roughly 90%+ of decoder depth at equal language quality.

Production authority remains blocked until sufficient approved real personalization history and real 0.6B quality/resource evidence pass the frozen courts.
