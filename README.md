# Nolane AI Personal

**Nolane AI Personal** is an experimental always-on personal AI designed to persist through time instead of behaving as a stateless prompt-response chatbot.

The initial language cortex is **Qwen3-0.6B**, but identity, time, memory, initiative and relationship continuity belong to the Living Runtime rather than to prompt history.

## Current executable milestone: Living Runtime v0.29.0

The runtime now contains two very different compute scales:

- **Standalone Nolane language model path**: owned recurrent cortex plus compressed language-boundary weights; Qwen remains an upstream teacher/provenance source for experimental training, not a required model object in the standalone runtime.
- **Tiny Living Core**: a recurrent 32D-latent model with only **14,515 parameters** by default.

L0 persistent runtime is complete. L1 validated social-observer engineering is complete. L2 has a **frozen held-out promotion court** and remains **UNPROMOTED** until real replay evidence passes it. L3 has persistent neural latent continuity in **shadow-only mode** across restarts. L4 adds audited REST/consolidation. L5 provides counterfactual Qwen3 hidden-state surgery. L6 adds a trainable Personal Cortex. L7 adds a **Hybrid Recurrent Cortex** whose state recurs through tokens and can persist across calls. L8 adds a separate **Depth-Recurrent Living Bridge** whose state recurs across selected Qwen decoder layers inside each forward. L9 crosses the boundary where selected Qwen decoder blocks can be genuinely bypassed. L10 turns that into Progressive Transformer-Depth Replacement. L11 collapses contiguous Transformer regions into recurrent islands. L12 adds a Selective State-Space Cortex. L13 adds a **Shrinking Qwen Scaffold**. L14 pushes the decoder down to a minimal 1+1 anchor shell. L15 removes the Transformer decoder entirely from native inference. L16 exports every remaining inference tensor into a standalone Nolane-owned checkpoint. L17 factorizes and distills the inherited dense language boundary. L18 adds an **Adaptive Rank Frontier** that searches progressively smaller ranks and keeps only ranks that pass development compression/quality gates before the selected artifact is exposed to held-out promotion courts. L19 adds an **int8 Quantized Factor Runtime** that stores the selected low-rank factors as row-wise int8 + scales and dequantizes vocabulary chunks only when computing logits. L20 adds a **Real Qwen3-0.6B Weight Court** that downloads and executes the exact pinned checkpoint, catches model-lock drift, and runs the factorization/int8 mechanics on learned weights from the full model. L21 adds a **Real Candidate Evidence Pipeline** that fail-closes the complete L15→L16→L18→L19 training/evaluation/promotion chain and refuses to fabricate missing personal evidence. L22 adds a **shared End-to-End Evidence Harness**: CI executes the same canonical 17-stage contract with synthetic non-authority artifacts, verifies artifact SHA-256 coverage, proves first-failure stop behavior, and rejects stage-order drift. L23 adds an **Approved Evidence Pack Builder** that accepts only explicitly approved, non-sensitive local VI/EN examples, freezes their train/dev/test protocol, and binds the pack into L21 without copying raw private text into audit receipts. L24 adds a **Local Review Queue** so conversation exports become review candidates with `approved:false` by construction; a separate explicit decisions file is required before L23 can accept anything. L25 adds a **Local Evidence Intake Pipeline** that cryptographically binds the queue, review decisions, reviewed source, approved pack, dataset and frozen protocol into one privacy-preserving lineage receipt that L21 can consume. L26 adds an **Interactive Local Reviewer** so a human can review candidates one-by-one, resume safely, and persist immutable explicit decisions without any model/network call or automatic approval. L27 adds a **Local Evidence Workbench** that unifies queue import, review progress, L25 finalization and L21 readiness into one hash-bound local workspace without moving the human-consent boundary. L28 adds an **Evidence Quality & Leakage Court**: source-group-aware train/dev/test splitting, cross-split source leakage checks, exact/near-duplicate detection, and a quality receipt that L23/L25/L27/L21 all bind and reverify. L29 adds a **Held-out Group Robustness Court** so global held-out averages cannot hide a severe regression on one independent conversation group; L15/L17-L18/L19 quality and promotion now require that worst-group court. All learned architecture paths remain unpromoted until matched real evidence decides whether they earn production authority.

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

### L11 Recurrent Transformer Islands

L11 stops treating every replaced Transformer block as a separate recurrent call. A contiguous region can now collapse into one recurrent island:

```text
Qwen block 3
    |
    v
 island [4..7] ---- one recurrent call
    |               blocks 4,5,6,7 original attention/MLP: skipped
    v
Qwen block 8
```

Candidate regions are calibrated directly from the hidden state before the first block to the hidden state after the last block. Default candidate widths are 2–4 blocks; edge regions are protected and islands cannot overlap.

Freeze a region plan:

```bash
python scripts/freeze_recurrent_island_plan.py
```

Train with whole-region teacher targets and rollback-safe stages:

```bash
python scripts/train_recurrent_islands.py
```

Evaluate against untouched Qwen and L10 on the same frozen held-out protocol:

```bash
python scripts/evaluate_recurrent_islands.py
```

Benchmark whether region collapse is actually faster than both Qwen and L10:

```bash
python scripts/benchmark_recurrent_island_resources.py
```

Neural CI proves on real tiny Qwen3 that a width-3 island causes all three original decoder blocks to execute zero times while the recurrent substitute executes once. Promotion still requires real Qwen3-0.6B quality and resource evidence.

See `docs/L11-RECURRENT-TRANSFORMER-ISLANDS.md`.

### L12 Selective State-Space Cortex

L12 changes the replacement itself. Instead of a depth-only recurrent substitute, one compact **state-space sequence model** carries a 32D neural state across tokens:

```text
Qwen outer layers
      |
      v
wide region [start..end]
      |
      v
Selective State-Space Cortex
  token-conditioned decay/proposal/output gates
  one sequence scan
      |
      +--> all original attention/MLP blocks inside region skipped
      |
      v
Qwen outer layers
```

For hidden size 1024 and state dimension 32, the cortex has **72,897 trainable parameters**.

Freeze nested region widening:

```bash
python scripts/freeze_state_space_plan.py
```

Train whole-region teacher distillation plus true replacement stages:

```bash
python scripts/train_state_space_cortex.py
```

Evaluate untouched Qwen vs L11 vs L12 on the same frozen held-out evidence:

```bash
python scripts/evaluate_state_space_cortex.py
```

Benchmark whether replacing a much larger share of Transformer depth is actually faster:

```bash
python scripts/benchmark_state_space_resources.py
```

Neural CI proves full-sequence scan equals token-by-token carried-state scan, cached and replay-safe generations match, a wide Qwen region executes zero original attention/MLP blocks, and all Qwen parameters stay frozen with zero gradients.

Default real promotion requires at least **40% of Qwen decoder depth** genuinely replaced, non-inferiority against L11, <=100K cortex parameters, and real speed/resource gains.

See `docs/L12-SELECTIVE-STATE-SPACE-CORTEX.md`.

### L13 Shrinking Qwen Scaffold

L13 treats **remaining Qwen depth** as a quantity that must shrink, not just replaced depth as a quantity that must grow.

```text
Qwen head anchors
      |
      v
fast + slow multi-timescale state-space cortex
      |
      +--> central Transformer attention/MLP blocks skipped
      |
      v
Qwen tail anchors
```

The default cortex is **81,249 trainable parameters** at hidden size 1024. It keeps a fast recurrent state for local changes and a slow recurrent state with a learned retention floor for longer-lived context.

The frozen scaffold curriculum targets approximately:

```text
50% Qwen remains -> 40% -> 32% -> 25%
```

so the default destination is roughly **75% decoder-depth removal**. Head/tail splits are calibrated rather than forced symmetric.

Freeze the scaffold plan:

```bash
python scripts/freeze_scaffold_plan.py
```

Train with whole-region teacher distillation and rollback-safe shrink stages:

```bash
python scripts/train_shrinking_scaffold.py
```

Evaluate Qwen vs L12 vs L13:

```bash
python scripts/evaluate_shrinking_scaffold.py
```

Benchmark whether the thinner shell actually beats L12 on compute:

```bash
python scripts/benchmark_scaffold_resources.py
```

Neural CI uses a real tiny 12-layer Qwen3 and proves a final test shell with only **2 head + 2 tail blocks** while the central **8/12 blocks execute zero original attention/MLP forwards**.

Promotion is blocked if Qwen still occupies >35% of decoder depth, if fast/slow states collapse to the same dynamics, or if L13 fails held-out/resource gates against L12.

See `docs/L13-SHRINKING-QWEN-SCAFFOLD.md`.

### L14 Minimal Qwen Anchor Cortex

L14 targets the smallest cache-safe Qwen decoder shell currently supported by the architecture:

```text
Qwen first block
      |
      v
deep-recurrent Nolane cortex
 fast + slow temporal state
 + shared virtual depth x8
      |
      v
Qwen final block
```

All decoder blocks between those two anchors can be absent from the experimental forward path. The default deep cortex has **89,805 trainable parameters** at Qwen hidden size 1024 and does not grow with the number of removed Qwen blocks.

```bash
python scripts/freeze_anchor_plan.py
python scripts/train_minimal_anchor_cortex.py
python scripts/evaluate_minimal_anchor_cortex.py
python scripts/benchmark_anchor_resources.py
```

Neural CI proves this mechanism on a real tiny 12-layer Qwen3 model with layers 1..10 removed from the original Transformer path. Promotion remains blocked pending real Qwen3-0.6B held-out quality and resource evidence.

See `docs/L14-MINIMAL-QWEN-ANCHOR-CORTEX.md`.

### L15 Native Nolane Boundary

L15 removes every Qwen Transformer decoder block from the experimental inference path:

```text
Qwen token embedding
        |
        v
Nolane deep recurrent cortex
 fast + slow + virtual depth
        |
        v
Qwen final norm -> Qwen LM head
```

Native generation is a custom recurrent loop. It does **not** call Hugging Face `generate()` and does not use Transformer KV cache.

```bash
python scripts/freeze_native_boundary_spec.py
python scripts/train_native_boundary.py
python scripts/evaluate_native_boundary.py
python scripts/benchmark_native_boundary_resources.py
```

Neural CI verifies on a real tiny Qwen3 model that native forward and generation both produce **zero decoder-layer calls**. Full Qwen is used only as a frozen teacher during training.

L15 is decoder-free, not fully Qwen-free: embedding, final norm and LM head are still Qwen boundary weights. See `docs/L15-NATIVE-NOLANE-BOUNDARY.md`.

### L16 Standalone Nolane Weights

L16 materializes the remaining L15 boundary tensors into an owned standalone checkpoint:

```text
owned token embedding
        |
        v
Nolane deep recurrent cortex
        |
        v
owned RMSNorm -> owned output projection
```

The standalone loader accepts only the checkpoint, Living latent and device. It does not accept or construct `QwenForCausalLM`.

```bash
python scripts/export_standalone_nolane.py
python scripts/evaluate_standalone_parity.py
python scripts/benchmark_standalone_resources.py
```

Neural CI proves L15/L16 logit-state-generation parity, independent tensor storage, source-mutation isolation, source-object deletion survival, and absence of decoder tensors in the standalone artifact.

The first L16 export deliberately preserves inherited vocabulary/output matrices exactly. It is runtime-independent from Qwen weights, but those matrices still have Qwen provenance. See `docs/L16-STANDALONE-NOLANE-WEIGHTS.md`.

### L17 Factorized Language Boundary

L17 targets the largest remaining tensors in the standalone model: the inherited dense token embedding and output projection.

```text
dense vocab x hidden
      |
      v
vocab x rank  +  rank x hidden
```

With rank 128 and a representative 151,936 x 1,024 tied boundary, the analytical parameter ratio is about **12.6%** of the dense boundary.

```bash
python scripts/export_factorized_boundary.py --rank 128
python scripts/train_factorized_boundary.py
python scripts/evaluate_factorized_boundary.py
python scripts/benchmark_factorized_resources.py
```

Full-rank SVD is a numerical parity court; low-rank candidates must pass separate compression, held-out quality and resource gates. Boundary distillation updates only low-rank factors while the deep recurrent cortex stays frozen with the same digest.

The core L17 runtime needs owned tensors + PyTorch. Tokenization remains vocabulary-compatible with the inherited frontend. See `docs/L17-FACTORIZED-LANGUAGE-BOUNDARY.md`.

### L18 Adaptive Rank Frontier

L18 removes the fixed-rank assumption. It tries a descending schedule such as:

```text
256 -> 192 -> 128 -> 96 -> 64
```

Every rank is independently initialized from the same dense L16 boundary, distilled on train evidence, and judged on dev evidence. The frontier stops at the first rejected smaller rank; the smallest previously accepted rank becomes the selected candidate.

```bash
python scripts/search_rank_frontier.py --ranks 256,192,128,96,64
```

Selection never consumes the frozen test split. The selected checkpoint must subsequently pass the L17 held-out quality/resource courts, and L18 promotion binds the exact selected rank, selected checkpoint and source L16 checkpoint.

See `docs/L18-ADAPTIVE-RANK-FRONTIER.md`.

### L19 Quantized Factor Runtime

L19 compresses the factorized boundary chosen by L18 without changing rank, vocabulary or the recurrent cortex.

```text
L18 selected float factors
          |
          v
row-wise int8 + per-row scales
          |
          v
chunked dequantized logits
```

The persistent code/basis tensors stay int8. Token embedding dequantizes only requested rows, while output logits process vocabulary codes in bounded chunks instead of keeping a full float `vocab x rank` matrix.

```bash
python scripts/export_quantized_boundary.py
python scripts/evaluate_quantized_boundary.py
python scripts/benchmark_quantized_resources.py
```

CI includes exact byte-level storage accounting. For a representative 151,936 × 1,024 tied rank-128 boundary, the reference gate requires int8 storage below **30% of FP32** and below **60% of BF16/FP16** factorized storage.

This is a pure-PyTorch chunked-dequant runtime, not a claim of native int8 GEMM acceleration. Promotion still requires held-out quality plus measured target-device latency/resource evidence.

See `docs/L19-QUANTIZED-FACTOR-RUNTIME.md`.

### L20 Real Qwen3-0.6B Weight Court

L20 moves the architecture evidence from tiny structural surrogates onto the exact pinned upstream checkpoint.

Successful authority run **36977448315** on head `bb2ff92e5af170f7e618bf69ff910d615f622b3f`:

- exact pinned revision verified;
- **596,049,920** real model parameters;
- 28 decoder layers;
- hidden size 1,024;
- vocabulary 151,936;
- tied input/output embeddings;
- full-model forward produced finite logits;
- rank-128 factorization executed on 512 deterministic rows of the learned boundary;
- row-wise int8 factor error ~1.13%;
- factorized-vs-int8 probe-logit relative error ~1.11%;
- measured-shape rank-128 boundary ratio ~12.58% of dense parameters;
- int8+scale footprint **20,191,232 bytes**, ~25.78% of FP32-factorized and ~51.56% of BF16-factorized storage.

The successful JSON evidence is uploaded as workflow artifact **11214325349** with digest
`sha256:9754324c4be62c5445445fc2ba19a1ab20234bd1d978195c8508e67a5642896c`.

This is mechanical evidence, **not language-quality promotion**. The sampled rank-128 SVD is deliberately not treated as proof that a full undistilled rank-128 boundary preserves Vietnamese/English quality.

See `docs/L20-REAL-QWEN06-WEIGHT-COURT.md`.

### L21 Real Candidate Evidence Pipeline

L21 turns the separate L15-L19 scripts into one reproducible, fail-closed evidence chain.

```bash
# metadata/readiness only
python scripts/run_real_candidate_pipeline.py

# explicitly authorize expensive execution
python scripts/run_real_candidate_pipeline.py --execute --device cpu
```

Readiness verifies dataset/protocol SHA lineage, split counts, general anchor size, persistent latent validity, exact pinned model revision and the required L14 comparison candidate. The receipt stores counts/digests/paths only; it does not copy personalization prompts or targets.

After readiness, 17 stages run in authority order: L15 train/quality/resource/promotion → L16 export/parity/resource/promotion → L18 rank search/held-out/resource/promotion → L19 int8 export/held-out/resource/promotion. Any non-zero stage stops the chain immediately.

A clean Git checkout is expected to report BLOCKED because personal `runtime-data`, trained checkpoints and model weights are intentionally not committed.

See `docs/L21-REAL-CANDIDATE-EVIDENCE-PIPELINE.md`.

### L22 End-to-End Evidence Harness

L22 makes the L21 authority chain testable end-to-end without fabricating model evidence.

```bash
python scripts/run_evidence_chain_fixture.py
python scripts/audit_evidence_chain_contract.py
```

The synthetic fixture runs the exact canonical 17-stage contract used by the real L21 executor. Every stage must appear in order, exit zero, create every declared output, and receive SHA-256 coverage.

Failure injection is supported:

```bash
python scripts/run_evidence_chain_fixture.py \
  --fail-stage evaluate_l16_parity
```

The fixture is permanently labeled `SYNTHETIC_NON_AUTHORITY_NEVER_PROMOTABLE`. CI also verifies synthetic private prompt/target sentinels never leak into the audit receipt.

See `docs/L22-END-TO-END-EVIDENCE-HARNESS.md`.

### L23 Approved Evidence Pack Builder

L23 provides the local-only consent boundary needed to feed real personalization evidence into L21.

```bash
python scripts/build_approved_evidence_pack.py \
  --source /private/path/approved-conversations.jsonl

python scripts/run_real_candidate_pipeline.py \
  --evidence-pack runtime-data/approved-evidence/approved-evidence-manifest.json
```

Only rows with `approved:true` are eligible, and `sensitive:true` rows are always excluded. The frozen dataset/protocol remain under gitignored `runtime-data/`; the manifest stores hashes/counts rather than raw prompts, targets or source IDs.

The pack is `USER_APPROVED_LOCAL_EVIDENCE_UNPROMOTED`: user approval permits empirical use but never grants model promotion authority.

See `docs/L23-APPROVED-EVIDENCE-PACK.md`.

### L24 Local Review Queue

L24 makes importing existing local conversations safe-by-default.

```text
conversation export
       |
       v
review queue
approved=false
reviewed=false
       |
       v
explicit decisions file
       |
       v
reviewed source
       |
       v
L23 approved evidence pack
```

```bash
python scripts/build_review_queue.py \
  --source /private/path/conversations.json

python scripts/apply_review_decisions.py \
  --queue-manifest runtime-data/review-queue/review-queue-manifest.json \
  --decisions /private/path/review-decisions.jsonl
```

Import never implies consent. Editing a frozen queue to `approved:true` invalidates its digest; valid approval must come from a separate explicit decision record.

See `docs/L24-LOCAL-REVIEW-QUEUE.md`, and `docs/L25-LOCAL-EVIDENCE-INTAKE.md`, and `docs/L26-INTERACTIVE-LOCAL-REVIEWER.md`, and `docs/L27-LOCAL-EVIDENCE-WORKBENCH.md`, and `docs/L28-EVIDENCE-QUALITY-LEAKAGE-COURT.md`, and `docs/L29-HELDOUT-GROUP-ROBUSTNESS-COURT.md`.

### L25 Local Evidence Intake Pipeline

L25 turns the separate L24 review and L23 pack-building steps into one verified local lineage.

```text
review queue
   + review decisions
          |
          v
reviewed source
          |
          v
approved L23 pack
          |
          v
L21-ready dataset/protocol
```

```bash
python scripts/finalize_local_evidence_intake.py \
  --queue-manifest runtime-data/review-queue/review-queue-manifest.json \
  --decisions /private/path/review-decisions.jsonl
```

The final L25 manifest stores only hashes, counts and lineage. Raw prompt/target text remains local in the reviewed source and approved dataset. Any post-freeze change to the queue, decisions, reviewed source, approved pack, dataset or protocol fails verification.

See `docs/L25-LOCAL-EVIDENCE-INTAKE.md`.

### L26 Interactive Local Reviewer

L26 provides a local-only terminal workflow for reviewing L24 candidates.

```bash
python scripts/review_local_queue.py \
  --queue-manifest runtime-data/review-queue/review-queue-manifest.json \
  --decisions runtime-data/review-decisions.jsonl
```

Each candidate can be explicitly approved, rejected, marked sensitive, skipped, or left for a later resumed session. Opening the reviewer never creates consent. Completed decisions are written atomically and cannot be silently replaced through the review session.

The progress manifest stores only hashes and counts; raw conversation text is shown only in the local terminal.

See `docs/L26-INTERACTIVE-LOCAL-REVIEWER.md`.

### L27 Local Evidence Workbench

L27 gives the local evidence workflow one stateful workspace instead of separate manual paths.

```text
QUEUE_READY
    -> REVIEW_IN_PROGRESS
    -> REVIEW_COMPLETE
    -> INTAKE_READY
    -> L21 readiness
```

```bash
python scripts/local_evidence_workbench.py init --source /private/path/conversations.json
python scripts/local_evidence_workbench.py review
python scripts/local_evidence_workbench.py status
python scripts/local_evidence_workbench.py finalize
python scripts/local_evidence_workbench.py readiness
```

Every phase reuses the real L24/L26/L25/L21 verification code. The workbench never synthesizes review decisions, defaults to refusing finalize while candidates remain undecided, and stores no raw conversation text in its manifest.

See `docs/L27-LOCAL-EVIDENCE-WORKBENCH.md`.

### L28 Evidence Quality & Leakage Court

L28 protects the held-out court from looking better than it really is.

Approved examples now carry only a hashed source-group lineage into the frozen protocol. When lineage is complete, train/dev/test splitting keeps every conversation group inside exactly one split.

```bash
python scripts/assess_evidence_quality.py \
  --dataset runtime-data/personalization.jsonl \
  --protocol runtime-data/personalization-protocol-v1.json
```

The court blocks incomplete source lineage, fewer than three source groups, source-group overlap, exact prompt/pair leakage and high-similarity cross-split near-duplicates. L23 will not produce a complete approved pack unless the court passes, and L21 recomputes the same court even when dataset/protocol are supplied directly.

The quality receipt contains only hashes, counts, indices and similarity metrics—no raw private conversation text or raw conversation IDs.

See `docs/L28-EVIDENCE-QUALITY-LEAKAGE-COURT.md`.

### L29 Held-out Group Robustness Court

L29 changes held-out evaluation from “one average number” into a source-group-aware non-inferiority court.

For L15, L17/L18 and L19, the evaluator now records per-example NLL, groups the frozen test examples by their privacy-preserving source lineage, and checks candidate-vs-reference regression separately for every independent group.

The worst-group tolerance is not a new arbitrary constant: each stage reuses its existing global non-inferiority threshold.

A global average can therefore no longer pass while one held-out conversation group degrades beyond the same quality limit.

Real-candidate readiness also requires at least two independent held-out source groups, so two test examples from one conversation are not treated as independent evidence.

Promotion scripts require a valid, self-digested L29 PASS receipt; missing, blocked, or tampered group evidence fails closed.

See `docs/L29-HELDOUT-GROUP-ROBUSTNESS-COURT.md`.

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
     +--> calibrated individual block removal
     |
     v
L11 Recurrent Transformer Islands
     |
     +--> multiple contiguous regions collapse to recurrent calls
     |
     v
L12 Selective State-Space Cortex
     |
     +--> one wide region replaced by token-recurrent SSM
     |
     v
L13 Shrinking Qwen Scaffold
     |
     v
L14 Minimal Qwen Anchor Cortex
     |
     v
L15 Native Nolane Boundary
     |
     +--> zero Transformer decoder blocks in native inference
     |
     v
L16 Standalone Nolane Weights
     |
     +--> no Qwen model object required at runtime
     |
     v
L17 Factorized Language Boundary
     |
     +--> low-rank owned token boundary
     |
     v
L18 Adaptive Rank Frontier
     |
     +--> evidence-driven smallest accepted rank
     |
     v
L19 Quantized Factor Runtime
     |
     +--> persistent int8 factors + row scales
     +--> chunked dequantized logits
     |
     v
L20 Real Qwen3-0.6B Weight Court
     |
     +--> exact pinned full-model forward
     +--> learned-weight factorization/int8 evidence
     |
     v
L21 Real Candidate Evidence Pipeline
     |
     +--> fail-closed L15 -> L16 -> L18 -> L19 authority chain
     +--> no fabricated personal evidence
     |
     v
L22 End-to-End Evidence Harness
     |
     +--> same canonical 17-stage executor
     +--> synthetic non-authority full-chain CI
     +--> artifact SHA-256 + privacy + fail-stop court
     |
     v
L23 Approved Evidence Pack Builder
     |
     +--> explicit approved:true intake
     +--> sensitive-row exclusion
     +--> local frozen train/dev/test protocol
     +--> L21 manifest-SHA lineage without receipt text leakage
     ^
     |
L24 Local Review Queue
     |
     +--> conversation import defaults approved=false
     +--> explicit separate review decisions
     +--> queue/reviewed-source SHA lineage
     |
     v
L25 Local Evidence Intake Pipeline
     |
     +--> queue + decisions + reviewed source + L23 pack
     +--> dataset/protocol lineage bound into one receipt
     +--> direct L21 evidence-pack bridge
     ^
     |
L26 Interactive Local Reviewer
     |
     +--> local-only human review
     +--> resumable immutable explicit decisions
     +--> no model/network or auto-approval
     |
     v
L27 Local Evidence Workbench
     |
     +--> unified import/review/finalize/readiness workspace
     +--> phase + SHA lineage
     +--> still no automatic consent or training
     |
     v
L28 Evidence Quality & Leakage Court
     |
     +--> source-group-aware train/dev/test split
     +--> exact + near-duplicate leakage court
     +--> quality SHA lineage required by L21 readiness
     |
     v
L29 Held-out Group Robustness Court
     |
     +--> per-source-group held-out NLL
     +--> worst-group non-inferiority gate
     +--> promotion requires valid L29 PASS evidence
```

See `docs/ARCHITECTURE.md`, `docs/ROADMAP.md`, `docs/L2-PROMOTION-COURT.md`, `docs/L3-PERSISTENT-LATENT.md`, `docs/L4-REST-CONSOLIDATION.md`, `docs/L5-SHADOW-SURGERY.md`, `docs/L6-PERSONAL-CORTEX.md`, `docs/L7-HYBRID-RECURRENT-CORTEX.md`, `docs/L8-DEPTH-RECURRENT-LIVING-BRIDGE.md`, `docs/L9-RECURRENT-BLOCK-REPLACEMENT.md`, `docs/L10-PROGRESSIVE-TRANSFORMER-REPLACEMENT.md`, `docs/L11-RECURRENT-TRANSFORMER-ISLANDS.md`, `docs/L12-SELECTIVE-STATE-SPACE-CORTEX.md`, `docs/L13-SHRINKING-QWEN-SCAFFOLD.md`, `docs/L14-MINIMAL-QWEN-ANCHOR-CORTEX.md`, `docs/L15-NATIVE-NOLANE-BOUNDARY.md`, `docs/L16-STANDALONE-NOLANE-WEIGHTS.md`, `docs/L17-FACTORIZED-LANGUAGE-BOUNDARY.md`, `docs/L18-ADAPTIVE-RANK-FRONTIER.md`, `docs/L19-QUANTIZED-FACTOR-RUNTIME.md`, `docs/L20-REAL-QWEN06-WEIGHT-COURT.md`, `docs/L21-REAL-CANDIDATE-EVIDENCE-PIPELINE.md`, `docs/L22-END-TO-END-EVIDENCE-HARNESS.md`, `docs/L23-APPROVED-EVIDENCE-PACK.md`, and `docs/L24-LOCAL-REVIEW-QUEUE.md`.
