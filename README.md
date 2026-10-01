# Nolane AI Personal

**Nolane AI Personal** is an experimental always-on personal AI designed to persist through time instead of behaving as a stateless prompt-response chatbot.

The initial language cortex is **Qwen3-0.6B**, but identity, time, memory, initiative and relationship continuity belong to the Living Runtime rather than to prompt history.

## Current executable milestone: Living Runtime v0.10.0

The runtime now contains two very different compute scales:

- **Qwen3-0.6B**: language cortex, used only when language inference is needed.
- **Tiny Living Core**: a recurrent 32D-latent model with only **14,515 parameters** by default.

L0 persistent runtime is complete. L1 validated social-observer engineering is complete. L2 has a **frozen held-out promotion court** and remains **UNPROMOTED** until real replay evidence passes it. L3 has persistent neural latent continuity in **shadow-only mode** across restarts. L4 adds audited REST/consolidation. L5 provides counterfactual Qwen3 hidden-state surgery. L6 adds a trainable Personal Cortex. L7 adds a **Hybrid Recurrent Cortex** whose state recurs through tokens and can persist across calls. L8 adds a separate **Depth-Recurrent Living Bridge** whose state recurs across selected Qwen decoder layers inside each forward. L9 crosses the boundary where selected Qwen decoder blocks can be genuinely bypassed. L10 turns that into **Progressive Transformer-Depth Replacement**: Qwen blocks are calibrated, ranked and replaced through a rollback-safe curriculum with cached autoregressive generation. All learned architecture paths remain unpromoted until matched real evidence decides whether they earn production authority.

### Persistent runtime

- identity survives restart;
- heartbeat runs without invoking Qwen;
- SQLite event timeline and full state snapshots;
- SHA-256 state digests;
- auditable rollback;
- episodic memory and bounded retrieval;
- relationship and affect-control state;
- unresolved conversation threads;
- initiative where silence is a first-class action.

### L1 social observer

Qwen may optionally inspect a user message and propose structured updates. It never writes persistent state directly. Every proposal is clipped and validated by a deterministic authority boundary.

```bash
nolane-personal run --social-observer
```

Invalid observer output is recorded as a rejection and the existing state is preserved.

### L2 tiny recurrent core

The development core receives:

```text
(previous observed state, event features, delta-time, previous latent)
                              |
                              v
                    14,515-param GRU core
                       /       |        \
                      v        v         v
                next latent  state Δ  future-return
```

The future-return head predicts whether the user returns inside the configured horizon. Tail examples that do not expose the full future horizon are censored rather than treated as negatives.

Audit the neural core:

```bash
python -m pip install -e '.[neural]'
python scripts/audit_living_core.py
```

Freeze chronological replay evidence:

```bash
python scripts/freeze_replay_protocol.py \
  --db runtime-data/living.db \
  --output runtime-data/replay-protocol-v1.json
```

Train **only** on the frozen train split:

```bash
python scripts/train_living_core.py \
  --db runtime-data/living.db \
  --protocol runtime-data/replay-protocol-v1.json
```

Evaluate on the frozen test split:

```bash
python scripts/evaluate_living_core.py \
  --db runtime-data/living.db \
  --protocol runtime-data/replay-protocol-v1.json \
  --checkpoint runtime-data/living-core-dev/living-core.pt
```

Default promotion requires protocol verification, checkpoint/protocol identity, enough held-out samples, state-fidelity gates, a future-return Brier improvement over the train base-rate baseline, and <=100K parameters.

See `docs/L2-PROMOTION-COURT.md` for the exact gates.

### L3 persistent latent shadow

The recurrent latent can now survive process restarts independently of Qwen context. It is atomically sealed with identity, exact checkpoint SHA-256, protocol SHA-256, latent dimension and the last processed state version.

Run one shadow catch-up:

```bash
python scripts/run_shadow_living_core.py \
  --db runtime-data/living.db \
  --checkpoint runtime-data/living-core-dev/living-core.pt \
  --protocol runtime-data/replay-protocol-v1.json
```

Continuously follow committed transitions:

```bash
python scripts/run_shadow_living_core.py \
  --db runtime-data/living.db \
  --checkpoint runtime-data/living-core-dev/living-core.pt \
  --protocol runtime-data/replay-protocol-v1.json \
  --follow
```

Shadow mode emits neural predictions and advances the latent but has **zero authority** over LivingState, memory, initiative or Qwen. See `docs/L3-PERSISTENT-LATENT.md`.

### L4 REST / consolidation

After at least 30 minutes of user inactivity, and no more than once every 45 minutes, the runtime may enter a bounded REST cycle.

The default path is cheap and deterministic:

```bash
nolane-personal run --no-model
```

For semantic consolidation using the already-loaded Qwen instance:

```bash
nolane-personal run --deep-rest
```

REST can derive durable memories only from existing evidence. A derived memory must cite at least two source memories; confidence cannot exceed its weakest source. Original memories remain intact and are connected through an append-only `memory_links` provenance graph.

Manual development cycle:

```text
/rest
```

Audit the graph:

```bash
python scripts/audit_rest_state.py --db runtime-data/living.db
```

REST cannot directly modify affect, relationship or personality state and does not update model weights. See `docs/L4-REST-CONSOLIDATION.md`.

### L5 counterfactual latent adapter

L5 can now project the persistent 32D Living latent into selected Qwen decoder hidden states through a tiny bounded residual adapter.

The critical authority rule is:

```text
same prompt
   |
   +--> untouched Qwen --------------------> served baseline
   |
   +--> Qwen + latent residual adapter ----> metrics only
```

A candidate is a separate artifact bound to the pinned base-model identity, revision, hidden size, deterministic seed and adapter digest. It never rewrites Qwen weights.

Create a candidate from the actual local Qwen config:

```bash
python scripts/init_latent_adapter.py \
  --model models/Qwen3-0.6B \
  --output-dir runtime-data/l5-adapter-candidate
```

Probe a single prompt:

```bash
python scripts/probe_latent_adapter.py \
  --prompt "Nói chuyện với tôi tự nhiên một chút nhé." \
  --adapter runtime-data/l5-adapter-candidate/latent-adapter.pt \
  --latent runtime-data/living-core-shadow/latent.json \
  --output runtime-data/l5-probe.json
```

Or run the frozen multilingual structural suite while loading Qwen once:

```bash
python scripts/run_qwen_surgery_suite.py \
  --adapter runtime-data/l5-adapter-candidate/latent-adapter.pt \
  --latent runtime-data/living-core-shadow/latent.json
```

The court measures KL divergence, logit shift, cosine similarity, latency overhead, top-token changes, hook cleanup and base-parameter mutation guards.

Neural CI also instantiates a real tiny `Qwen3ForCausalLM` and proves that the hook contract works against the actual Qwen3 architecture class. **This is structural shadow evidence, not evidence that the adapter improves quality.** Production use remains blocked until matched held-out quality courts pass.

See `docs/L5-SHADOW-SURGERY.md`.

### L6 trainable Personal Cortex

L6 moves beyond counterfactual probing. Qwen remains the pinned language foundation, but a latent-conditioned neural path can now be trained through actual Qwen decoder blocks.

The ownership boundary is explicit:

```text
Qwen3-0.6B parameters  -> frozen, no gradients
Persistent latent 32D  -> adapter input
Latent adapter         -> trainable
Qwen hidden states     -> modified by trained residual
Output logits          -> Personal Cortex path
```

Freeze a local personalization dataset before training:

```bash
python scripts/freeze_personalization_protocol.py \
  --dataset runtime-data/personalization.jsonl
```

Train only the frozen train split:

```bash
python scripts/train_personal_cortex.py \
  --dataset runtime-data/personalization.jsonl \
  --protocol runtime-data/personalization-protocol-v1.json
```

Evaluate the same trained artifact on the held-out personal split plus the frozen Vietnamese/English general anchor:

```bash
python scripts/evaluate_personal_cortex.py
```

Exercise the neural path explicitly:

```bash
python scripts/generate_personal_cortex.py \
  --prompt "Nay tôi hơi mệt."
```

The trained artifact stays `TRAINED_CANDIDATE_UNPROMOTED` until real held-out data passes the quality and regression courts. Neural CI trains a real tiny `Qwen3ForCausalLM` and verifies that adapter weights learn while every base-Qwen parameter remains frozen.

See `docs/L6-PERSONAL-CORTEX.md`.

### L7 Hybrid Recurrent Cortex

L7 changes the model path again: the personalization component is no longer only a latent-to-hidden residual. A small recurrent mixer now evolves its own state **inside selected Qwen decoder layers**.

```text
Qwen hidden
    |
    v
hidden -> recurrent width
    |
    v
GRU(previous recurrent state)
    |
    v
recurrent -> Qwen hidden
    |
    v
bounded residual -> next Qwen layer
```

The persistent Living latent initializes the recurrent state. Selected Qwen layers share mixer weights but maintain separate recurrent states.

For Qwen hidden size 1024 and the default recurrent width 24, the mixer is only **56,641 parameters**.

Train it on the same frozen personalization train split:

```bash
python scripts/train_hybrid_recurrent_cortex.py \
  --dataset runtime-data/personalization.jsonl \
  --protocol runtime-data/personalization-protocol-v1.json \
  --recurrent-dim 24
```

Run the held-out personal/general court:

```bash
python scripts/evaluate_hybrid_recurrent_cortex.py
```

Exercise the recurrent model path:

```bash
python scripts/generate_hybrid_recurrent_cortex.py \
  --prompt "Nay tôi hơi mệt."
```

After generation, per-layer recurrent neural state is sealed locally and loaded on the next call. The state is bound to the AI identity, Qwen fingerprint, mixer digest and persistent-latent digest; incompatible lineage fails closed.

Neural CI uses a real tiny `Qwen3ForCausalLM` and proves that recurrent state changes the second call on the same input, reset restores the first trajectory, the mixer can learn held-out targets, and Qwen base weights receive zero gradients.

L7 is **Transformer + recurrent**, not yet a full Transformer replacement. See `docs/L7-HYBRID-RECURRENT-CORTEX.md`.

### L8 Depth-Recurrent Living Bridge

L8 explores a different recurrence axis from L7. Instead of carrying a separate recurrent state through tokens at each selected layer, one bridge state moves **across selected decoder layers** during a forward.

```text
Living latent 32D
      |
      v
RMS-normalized latent feature
      |
hidden summary + layer identity
      |
      v
 depth GRU state -----> next selected layer
      |
  residual + bounded gate
      |
      v
   Qwen hidden
```

For Qwen hidden size 1024, the default bridge has **84,289 trainable parameters**. Qwen remains frozen.

Train on the same frozen protocol used by L6/L7:

```bash
python scripts/train_depth_bridge.py
```

Run the matched four-way court:

```bash
python scripts/evaluate_depth_bridge.py
```

The evaluator compares untouched Qwen, L6 Personal Cortex, L7 Hybrid Recurrent Cortex and L8 Depth Bridge on the same held-out examples. L8 is blocked unless it beats all three required baselines while staying within the Vietnamese/English regression and <=100K parameter gates.

A separate resource court also requires median forward latency <=1.50× Qwen and artifact size <=2 MB:

```bash
python scripts/benchmark_depth_bridge_resources.py
```

Final promotion requires both quality and resource PASS on the exact same checkpoint:

```bash
python scripts/decide_depth_bridge_promotion.py \
  --quality runtime-data/l8-quality.json \
  --resources runtime-data/l8-resources.json
```

Experimental generation:

```bash
python scripts/generate_depth_bridge.py \
  --prompt "Nay tôi hơi mệt."
```

See `docs/L8-DEPTH-RECURRENT-LIVING-BRIDGE.md`.

### L9 Recurrent Transformer-Block Replacement

L9 is the first wave where a selected Qwen decoder block can disappear from the actual forward computation.

```text
incoming Qwen hidden
       |
       +---- Living latent 32D
       +---- layer identity
       |
       v
vectorized recurrent replacement
       |
 residual + sigmoid gate
       |
       v
next Qwen layer

(original selected attention + MLP block: NOT CALLED)
```

The shared replacement is **84,289 trainable parameters** for hidden size 1024.

Training has two phases:

1. distill the frozen original Qwen block's hidden output;
2. bypass that block and fine-tune end-to-end on the frozen personalization train split.

```bash
python scripts/train_block_replacement.py
```

Run the five-way quality court:

```bash
python scripts/evaluate_block_replacement.py
```

It compares untouched Qwen, L6, L7, L8 and L9 on the same held-out protocol. L9 must improve over Qwen and stay within **0.010 NLL** of the best prior learned architecture.

Real compute removal is mandatory. The resource court requires median forward latency <= **0.95x Qwen**:

```bash
python scripts/benchmark_block_replacement_resources.py
```

Final promotion requires quality + speed/resource PASS on the exact same checkpoint:

```bash
python scripts/decide_block_replacement_promotion.py \
  --quality runtime-data/l9-quality.json \
  --resources runtime-data/l9-resources.json
```

Experimental generation currently forces `use_cache=False`; cache-compatible replacement remains a production gate.

See `docs/L9-RECURRENT-BLOCK-REPLACEMENT.md`.

### L10 Progressive Transformer-Depth Replacement

L10 no longer chooses a fixed pair of Qwen blocks manually. It freezes a layer-sensitivity plan, then attempts progressively larger block-replacement stages:

```text
Qwen calibration
      |
      v
rank internal blocks by transformation sensitivity
      |
      v
1 block -> 2 -> 4 -> ... -> target
      |
      +--> teacher distillation
      +--> personalization training
      +--> dev court
      |
      +--> PASS: continue
      `--> FAIL: rollback weights and stop
```

The default target is **50% of Qwen decoder depth**, capped at 12 blocks. Promotion still requires at least 25% real depth replacement.

Freeze the plan:

```bash
python scripts/freeze_progressive_replacement_plan.py
```

Train stage by stage:

```bash
python scripts/train_progressive_replacement.py
```

Evaluate held-out quality and cache correctness:

```bash
python scripts/evaluate_progressive_replacement.py
```

Benchmark real forward and generation resources:

```bash
python scripts/benchmark_progressive_replacement_resources.py
```

L10 also fixes recurrent-state semantics during generation. In no-cache mode the replacement state resets on every full-prefix replay; in cached mode the state carries only across incremental token forwards. Neural CI proves cached and replay-safe deterministic generations match on a real tiny `Qwen3ForCausalLM`.

Quality/resource promotion is bound to both the exact checkpoint SHA and frozen progressive-plan SHA. See `docs/L10-PROGRESSIVE-TRANSFORMER-REPLACEMENT.md`.

## Bootstrap Qwen

Model weights are intentionally **not committed to GitHub**. A pinned downloader reproduces the exact upstream checkpoint locally.

```bash
python -m pip install -r requirements-model.txt
python scripts/download_model.py
```

The model is downloaded into:

```text
models/Qwen3-0.6B/
```

## Install runtime

Core state/memory/heartbeat remains standard-library-only.

```bash
python -m pip install -e .
nolane-personal init
nolane-personal status
```

For local Qwen conversation:

```bash
python -m pip install -e '.[qwen]'
nolane-personal run
```

For the state/memory/heartbeat loop without loading any language model:

```bash
nolane-personal run --no-model --tick-seconds 5
```

## Pinned upstream

- Model: `Qwen/Qwen3-0.6B`
- Revision: `c1899de289a04d12100db370d81485cdf75e47ca`
- Architecture: Qwen3 causal LM
- License: Apache-2.0
- Role: initial language cortex, not the final architecture

## Direction

```text
events + time
     |
     v
persistent runtime
     |
     +--> validated social observer
     +--> local memory
     +--> initiative / silence
     |
     v
tiny recurrent living core
     |
     +--> held-out state fidelity
     +--> future-return prediction
     +--> frozen promotion court
     |
     v
persistent 32D latent (shadow)
     |
     +--> restart continuity
     +--> checkpoint/protocol binding
     +--> no production authority yet
     |
     v
REST / consolidation
     |
     +--> evidence graph
     +--> durable memory proposals
     +--> thread review
     |
     v
L5 latent adapter (counterfactual shadow)
     |
     +--> paired baseline/counterfactual forward
     +--> Qwen3 structural court
     +--> no production authority
     |
     v
L6 trained Personal Cortex candidate
     |
     v
L7 Hybrid Recurrent Cortex
     |
     +--> token recurrence + persistent per-layer state
     |
     v
L8 Depth-Recurrent Living Bridge
     |
     +--> recurrent state across decoder depth
     |
     v
L9 Recurrent Block Replacement
     |
     +--> selected Qwen attention/MLP blocks skipped
     |
     v
L10 Progressive Transformer-Depth Replacement
     |
     +--> calibrated block ranking
     +--> rollback-safe 1 -> 2 -> 4 -> ... curriculum
     +--> cache-correct recurrent generation
     +--> quality + real speed + plan-lineage court
     |
     v
future recurrent/state-space islands replacing larger Transformer regions
```

See `docs/ARCHITECTURE.md`, `docs/ROADMAP.md`, `docs/L2-PROMOTION-COURT.md`, `docs/L3-PERSISTENT-LATENT.md`, `docs/L4-REST-CONSOLIDATION.md`, `docs/L5-SHADOW-SURGERY.md`, `docs/L6-PERSONAL-CORTEX.md`, `docs/L7-HYBRID-RECURRENT-CORTEX.md`, `docs/L8-DEPTH-RECURRENT-LIVING-BRIDGE.md`, `docs/L9-RECURRENT-BLOCK-REPLACEMENT.md`, and `docs/L10-PROGRESSIVE-TRANSFORMER-REPLACEMENT.md`.
