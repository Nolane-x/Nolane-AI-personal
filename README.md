# Nolane AI Personal

**Nolane AI Personal** is an experimental always-on personal AI designed to persist through time instead of behaving as a stateless prompt-response chatbot.

The initial language cortex is **Qwen3-0.6B**, but identity, time, memory, initiative and relationship continuity belong to the Living Runtime rather than to prompt history.

## Current executable milestone: Living Runtime v0.7.0

The runtime now contains two very different compute scales:

- **Qwen3-0.6B**: language cortex, used only when language inference is needed.
- **Tiny Living Core**: a recurrent 32D-latent model with only **14,515 parameters** by default.

L0 persistent runtime is complete. L1 validated social-observer engineering is complete. L2 has a **frozen held-out promotion court** and remains **UNPROMOTED** until real replay evidence passes it. L3 has persistent neural latent continuity in **shadow-only mode** across restarts. L4 adds audited REST/consolidation. L5 provides counterfactual Qwen3 hidden-state surgery. L6 adds a trainable Personal Cortex. L7 adds a **Cross-Layer Living Bridge**: a small GRU-based recurrent pathway that carries Living state across selected Qwen decoder layers.

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

### L7 Cross-Layer Living Bridge

L7 replaces the fixed per-layer residual idea with a learned recurrent pathway across Transformer depth.

```text
Living latent 32D
      |
      v
hidden summary + layer identity
      |
      v
   GRU state 32D  ---- recurs across selected Qwen layers
      |
   +--+--+
   |     |
residual gate
   \     /
    Qwen hidden
```

For a Qwen hidden size of 1024, the default bridge has **84,289 trainable parameters**. Qwen itself remains frozen.

Train on the same frozen personalization train split used by L6:

```bash
python scripts/train_living_bridge.py
```

The quality court compares **three paths on the same held-out examples**:

1. untouched Qwen;
2. L6 residual Personal Cortex;
3. L7 recurrent Living Bridge.

```bash
python scripts/evaluate_living_bridge.py
```

L7 is blocked unless it beats both untouched Qwen and L6, stays under 100K parameters, preserves base-Qwen weights, receives no base-Qwen gradients, and stays within the Vietnamese/English general-regression budget.

Experimental generation:

```bash
python scripts/generate_living_bridge.py \
  --prompt "Nay tôi hơi mệt."
```

See `docs/L7-CROSS-LAYER-LIVING-BRIDGE.md`.

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
L7 Cross-Layer Living Bridge
     |
     +--> recurrent state across Qwen depth
     +--> latent + hidden + layer identity fusion
     +--> must beat Qwen base AND L6 held-out
     |
     v
future promoted personal architecture / deeper block replacement
```

See `docs/ARCHITECTURE.md`, `docs/ROADMAP.md`, `docs/L2-PROMOTION-COURT.md`, `docs/L3-PERSISTENT-LATENT.md`, `docs/L4-REST-CONSOLIDATION.md`, `docs/L5-SHADOW-SURGERY.md`, `docs/L6-PERSONAL-CORTEX.md`, and `docs/L7-CROSS-LAYER-LIVING-BRIDGE.md`.
