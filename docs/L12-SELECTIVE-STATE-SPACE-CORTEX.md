# L12 Selective State-Space Cortex

## Purpose

L11 proved that one recurrent computation can replace several contiguous Transformer blocks. L12 changes the recurrence axis again: the replacement now carries a compact **state across tokens**, making the substitute a selective state-space sequence model rather than a depth-only GRU.

The target is a single wide Qwen region that can progressively grow toward most of the internal Transformer depth while the outer Qwen layers remain as language scaffolding.

## Core recurrence

For each token `t`, a 32D state is updated without self-attention:

```text
hidden_t -> low-rank feature x_t
Living latent 32D -> latent feature l

proposal_t = tanh(P(x_t) + l)
decay_t    = sigmoid(D(x_t))
state_t    = decay_t * state_(t-1) + (1-decay_t) * proposal_t
out_gate_t = sigmoid(G(x_t))

residual_t = Up(state_t * out_gate_t)
hidden'_t  = hidden_t + bounded_gate * residual_t
```

`decay_t`, proposal and output gate all depend on the current token representation. No attention matrix is created inside the L12 cortex.

## Default footprint

For Qwen hidden size 1024 and state dimension 32:

**72,897 trainable parameters**

The production court keeps a hard **<=100K parameter cap**.

## Wide-region collapse

L12 replaces one contiguous region `[start..end]`:

```text
Qwen blocks before region
        |
        v
Selective State-Space Cortex  ---- one sequence scan
        |
        +---- original Qwen blocks start..end are not called
        |
        v
Qwen blocks after region
```

Every Transformer block after the first position inside the region becomes an identity path. Thus a 17-block region still invokes one L12 cortex module rather than 17 replacement modules.

## Full scan and cached recurrence

The SSM contract is tested in two equivalent forms:

```text
full sequence:       scan(x_1 ... x_T)

incremental cache:   scan(x_1, s_0)
                     scan(x_2, s_1)
                     ...
                     scan(x_T, s_(T-1))
```

Neural CI requires both the emitted hidden sequence and final recurrent state to match numerically.

For autoregressive generation:

- cached mode scans the prompt once, then carries the SSM state across one-token forwards;
- replay-safe no-cache mode resets SSM state at each full-prefix replay.

Cached and replay-safe deterministic generations must match.

## Region calibration and widening curriculum

L12 calibrates candidate wide spans directly from untouched Qwen hidden states. For each span it measures:

- residual RMS ratio from region input to region output;
- cosine change;
- transformation score;
- transformation score per removed Transformer block.

The frozen plan contains nested regions. Default real-Qwen target:

```text
~20% depth
    |
    v
~35% depth
    |
    v
~50% depth
    |
    v
~60% depth
```

with protected edge layers and a maximum default region width of 20 blocks.

```bash
python scripts/freeze_state_space_plan.py
```

Each later region must strictly contain the previous accepted region.

## Whole-region teacher distillation

At every stage, untouched Qwen acts as teacher:

```text
teacher input  = hidden before region start
teacher target = hidden after region end
student        = one Selective State-Space Cortex scan
```

After teacher distillation, training switches to the genuine replacement path with the whole Transformer region absent.

```bash
python scripts/train_state_space_cortex.py
```

If widening causes dev NLL to regress beyond budget, cortex weights roll back to the previous accepted stage and expansion stops.

## Quality court

```bash
python scripts/evaluate_state_space_cortex.py
```

The evaluator compares on the exact same frozen held-out protocol:

1. untouched Qwen;
2. L11 Recurrent Transformer Islands;
3. L12 Selective State-Space Cortex.

Default quality gates require:

- at least 40% of decoder depth genuinely replaced;
- at least two accepted widening stages;
- >=0.005 NLL improvement over untouched Qwen;
- <=0.020 NLL degradation versus L11;
- Vietnamese/English anchor regression <=0.05 NLL;
- full-scan == incremental-scan contract PASS;
- cached == replay-safe deterministic generation PASS;
- <=100K cortex parameters;
- base Qwen unchanged;
- base Qwen gradient count = 0.

## Resource court

```bash
python scripts/benchmark_state_space_resources.py
```

Default resource gates require:

- at least 40% decoder depth replaced;
- median forward latency <=0.85x untouched Qwen;
- median forward latency <=0.95x L11;
- cached generation >=1.05x replay-safe generation;
- artifact <=2 MB;
- <=100K trainable parameters.

## Promotion identity

Quality and resource evidence must refer to the same:

- exact L12 checkpoint SHA-256;
- exact frozen L12 plan SHA-256.

```bash
python scripts/decide_state_space_promotion.py \
  --quality runtime-data/l12-quality.json \
  --resources runtime-data/l12-resources.json
```

Any mismatch fails closed.

## Neural court evidence

CI uses a real tiny `Qwen3ForCausalLM` and proves:

- exact 72,897-parameter analytical contract at hidden size 1024;
- latent sign/direction survives normalization;
- full-sequence scan equals token-by-token carried-state scan;
- a wide Qwen region executes zero original attention/MLP blocks inside the region;
- one L12 cortex call replaces that region;
- real Qwen hidden-state calibration produces nested widening candidates;
- cached and replay-safe deterministic generation match;
- multi-stage whole-region distillation trains the SSM cortex;
- Qwen remains frozen with zero gradients;
- artifact roundtrip preserves plan SHA and cortex-state digest;
- wrong base-model lineage fails closed.

## Scientific boundary

L12 is substantially less Transformer-like than earlier waves, but it is still a hybrid model because outer Qwen layers remain before and after the state-space region.

The tiny-Qwen3 court proves architecture mechanics and training ownership. It does **not** prove that production Qwen3-0.6B can already replace 40–60% of its depth at equal or better quality.

Real promotion requires sufficient approved personalization history, a frozen real held-out protocol, and measured Qwen3-0.6B quality plus CPU/GPU resource evidence.
