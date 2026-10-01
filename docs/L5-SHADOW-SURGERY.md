# L5 Shadow Architecture Surgery

## Purpose

L5 is the first step where the persistent 32D Living latent can touch a Qwen hidden state.

That does **not** mean the modified path is allowed to answer the user.

The current authority model is:

```text
same prompt
   |
   +--> baseline Qwen forward --------------------> served path
   |
   +--> Qwen + latent residual adapter forward --> counterfactual metrics only
```

The baseline output remains authoritative until later held-out quality courts demonstrate a measurable benefit.

## Candidate architecture

The default latent adapter is deliberately tiny:

```text
persistent latent 32D
      |
   LayerNorm
      |
  Linear 32 -> 16
      |
     SiLU
      |
 Linear 16 -> Qwen hidden size
      |
 bounded scalar gate
      |
 residual injection into selected decoder layers
```

The exact parameter count is derived from the local model hidden size and audited analytically against the instantiated PyTorch module.

The default gate is bounded to ±0.10.

## Candidate lineage

A surgery candidate is a standalone artifact. It does not rewrite Qwen weights.

Each candidate binds to:

- the pinned model-lock identity;
- exact upstream model revision;
- Qwen hidden size;
- adapter configuration;
- deterministic initialization seed;
- adapter state digest;
- checkpoint SHA-256;
- candidate ID.

A mismatched base-model fingerprint or hidden size fails closed.

## Counterfactual probe

`CounterfactualSurgeryProbe` runs two forwards against the same input.

It records:

- baseline latency;
- counterfactual latency;
- overhead ratio;
- KL divergence from baseline;
- mean/max absolute logit shift;
- cosine similarity;
- top-1 token change;
- injected layer indices;
- gate and token scope;
- model/latent/adapter lineage;
- whether base-model parameter identity/version guards remained unchanged.

Hooks are context-managed and removed even when the injected path throws an exception.

## Authority

Every counterfactual receipt declares:

`COUNTERFACTUAL_ONLY_BASELINE_OUTPUT`

Shadow admission declares:

`SHADOW_ONLY_NO_PROMOTION`

Admission proves only that the candidate is structurally bounded enough to study. It does **not** prove that it improves conversation quality, memory, personality, reasoning, or user preference.

## Create a candidate

First bootstrap the pinned Qwen checkpoint and a persistent L3 latent.

Then:

```bash
python scripts/init_latent_adapter.py \
  --model models/Qwen3-0.6B \
  --output-dir runtime-data/l5-adapter-candidate
```

The command reads the local Qwen config instead of hard-coding hidden size.

## Probe one prompt

```bash
python scripts/probe_latent_adapter.py \
  --prompt "Nói chuyện với tôi tự nhiên một chút nhé." \
  --adapter runtime-data/l5-adapter-candidate/latent-adapter.pt \
  --latent runtime-data/living-core-shadow/latent.json \
  --output runtime-data/l5-probe.json
```

By default, approximately the 25%, 50% and 75% decoder-depth positions are injected. Explicit layer indices may be supplied with `--layers`.

## Run the frozen prompt suite

The repository contains a frozen multilingual structural suite at:

`research/L5-SHADOW-PROMPTS.json`

Run all prompts while loading Qwen only once:

```bash
python scripts/run_qwen_surgery_suite.py \
  --adapter runtime-data/l5-adapter-candidate/latent-adapter.pt \
  --latent runtime-data/living-core-shadow/latent.json
```

The result contains all paired receipts and one aggregate shadow-admission decision.

## Shadow admission

Default admission blocks candidates when any of the following is true:

- candidate/base/adapter/latent lineage is mixed;
- counterfactual authority is violated;
- base-model parameter guards change;
- adapter exceeds 100,000 parameters;
- gate exceeds ±0.10;
- no decoder layer is injected;
- a metric is non-finite;
- KL exceeds the structural ceiling;
- cosine similarity collapses;
- median forward overhead exceeds the ceiling.

These are structural safety/research gates, not quality promotion gates.

## Courts

The neural CI now verifies L2, L3 and L5 together.

In addition to toy-module failure-path tests, it instantiates a real random `Qwen3ForCausalLM` from Hugging Face Transformers with a tiny configuration and verifies that:

- Qwen3 decoder layers are resolved correctly;
- counterfactual injection changes logits;
- baseline logits remain the served result;
- base-model parameters remain unchanged;
- hooks are removed after the forward;
- the previous persistent-latent restart court still passes.

This tests the Qwen3 architecture contract without downloading the full 0.6B checkpoint in CI.

## What is still required before promotion

L5 is not eligible for production behavior yet.

The next scientific work must:

1. produce/train adapter candidates only from frozen train evidence;
2. freeze matched held-out dialogue/replay cases;
3. compare untouched Qwen and latent-conditioned Qwen under identical contexts;
4. measure whether personalization/continuity improves without general quality regressions;
5. include latency and resource gates;
6. promote only a candidate that wins the preregistered court.

Until then the adapter remains counterfactual-only.
