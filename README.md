# Nolane AI Personal

**Nolane AI Personal** is an experimental always-on personal AI designed to persist through time instead of behaving as a stateless prompt-response chatbot.

The initial language cortex is **Qwen3-0.6B**, but identity, time, memory, initiative and relationship continuity belong to the Living Runtime rather than to prompt history.

## Current executable milestone: Living Runtime v0.56.0

The runtime now contains two very different compute scales:

- **Standalone Nolane language model path**: owned recurrent cortex plus compressed language-boundary weights; Qwen remains an upstream teacher/provenance source for experimental training, not a required model object in the standalone runtime.
- **Tiny Living Core**: a recurrent 32D-latent model with only **14,515 parameters** by default.

L0 persistent runtime is complete. L1 validated social-observer engineering is complete. L2 has a **frozen held-out promotion court** and remains **UNPROMOTED** until real replay evidence passes it. L3 has persistent neural latent continuity in **shadow-only mode** across restarts. L4 adds audited REST/consolidation. L5 provides counterfactual Qwen3 hidden-state surgery. L6 adds a trainable Personal Cortex. L7 adds a **Hybrid Recurrent Cortex** whose state recurs through tokens and can persist across calls. L8 adds a separate **Depth-Recurrent Living Bridge** whose state recurs across selected Qwen decoder layers inside each forward. L9 crosses the boundary where selected Qwen decoder blocks can be genuinely bypassed. L10 turns that into Progressive Transformer-Depth Replacement. L11 collapses contiguous Transformer regions into recurrent islands. L12 adds a Selective State-Space Cortex. L13 adds a **Shrinking Qwen Scaffold**. L14 pushes the decoder down to a minimal 1+1 anchor shell. L15 removes the Transformer decoder entirely from native inference. L16 exports every remaining inference tensor into a standalone Nolane-owned checkpoint. L17 factorizes and distills the inherited dense language boundary. L18 adds an **Adaptive Rank Frontier** that searches progressively smaller ranks and keeps only ranks that pass development compression/quality gates before the selected artifact is exposed to held-out promotion courts. L19 adds an **int8 Quantized Factor Runtime** that stores the selected low-rank factors as row-wise int8 + scales and dequantizes vocabulary chunks only when computing logits. L20 adds a **Real Qwen3-0.6B Weight Court** that downloads and executes the exact pinned checkpoint, catches model-lock drift, and runs the factorization/int8 mechanics on learned weights from the full model. L21 adds a **Real Candidate Evidence Pipeline** that fail-closes the complete L15→L16→L18→L19 training/evaluation/promotion chain and refuses to fabricate missing personal evidence. L22 adds a **shared End-to-End Evidence Harness**: CI executes the same canonical 17-stage contract with synthetic non-authority artifacts, verifies artifact SHA-256 coverage, proves first-failure stop behavior, and rejects stage-order drift. L23 adds an **Approved Evidence Pack Builder** that accepts only explicitly approved, non-sensitive local VI/EN examples, freezes their train/dev/test protocol, and binds the pack into L21 without copying raw private text into audit receipts. L24 adds a **Local Review Queue** so conversation exports become review candidates with `approved:false` by construction; a separate explicit decisions file is required before L23 can accept anything. L25 adds a **Local Evidence Intake Pipeline** that cryptographically binds the queue, review decisions, reviewed source, approved pack, dataset and frozen protocol into one privacy-preserving lineage receipt that L21 can consume. L26 adds an **Interactive Local Reviewer** so a human can review candidates one-by-one, resume safely, and persist immutable explicit decisions without any model/network call or automatic approval. L27 adds a **Local Evidence Workbench** that unifies queue import, review progress, L25 finalization and L21 readiness into one hash-bound local workspace without moving the human-consent boundary. L28 adds an **Evidence Quality & Leakage Court**: source-group-aware train/dev/test splitting, cross-split source leakage checks, exact/near-duplicate detection, and a quality receipt that L23/L25/L27/L21 all bind and reverify. L29 adds a **Held-out Group Robustness Court** so global held-out averages cannot hide a severe regression on one independent conversation group; L15/L17-L18/L19 quality and promotion now require that worst-group court. All learned architecture paths remain unpromoted until matched real evidence decides whether they earn production authority.

### v0.42 Windows + Android product client

v0.42 starts turning the research/runtime system into a deliberately small personal product surface.

The product UI follows the repository's Nolane UI Intelligence (NUI) lifecycle rather than treating a successful build as UI completion. The selected **Ember Quiet** direction keeps only four things permanently visible: identity, AI power state, transcript and composer. Orange is reserved for living/primary state rather than decorative chrome.

The client lives in `apps/product-client/` and uses a shared Tauri v2 host for Windows and Android.

**Windows path**

- the installer owns the app UI, Python sidecar runtime, approved factorized Nolane checkpoint and tokenizer assets;
- the bundled sidecar binds only to a random loopback port;
- the UI reads power truth from the runtime instead of simulating an ON state;
- AI OFF unloads the model while keeping local transcript/state readable;
- WebView2 uses Tauri's offline installer mode so first installation does not require a second runtime download;
- release staging verifies the exact model SHA-256 and refuses incomplete runtime/tokenizer assets.

**Android path**

- the same responsive chat surface and Rust host compile as an Android APK target;
- remote pairing is HTTPS-only except loopback and the pairing token is not persisted in plaintext;
- Android does **not** claim local inference yet. Until the mobile inference or secure pairing court closes, an unpaired Android client is visibly unavailable rather than pretending to contain the desktop model.

**Personalization**

The chat remains visually quiet while behavior can change deeply through preferred name, language, response length, conversational style, initiative, local memory and one bounded personal instruction. Disabling memory changes the actual LivingEngine retrieval/write policy; it is not a cosmetic toggle.

The product release court is `.github/workflows/product-client.yml`:

- product runtime/privacy/API tests;
- Playwright desktop + mobile viewport court;
- reduced-motion and minimum touch-target checks;
- Windows native Tauri build;
- Android `tauri android init/build` APK court;
- fail-closed release asset staging.

See `docs/V042-PRODUCT-CLIENT-NUI-CONTRACT.md`.

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

### L30 Long-Horizon Continual-Learning Court

L30 adds a sequential stability/plasticity court for future continual updates. It compares one checkpoint before an approved update with the checkpoint after that update on two frozen, disjoint evidence sets: old retention groups and newly introduced adaptation groups.

A candidate update cannot pass merely because the average improves. The court gates mean and worst-group forgetting on old evidence, mean adaptation gain on new evidence, and worst-group regression on new evidence. Same-checkpoint evidence, old/new source overlap, non-finite metrics and tampered receipts fail closed.

```bash
python scripts/assess_continual_learning.py \
  --evidence runtime-data/l30/continual-evidence.json \
  --output runtime-data/l30/continual-learning-receipt.json
```

The receipt exposes checkpoint digests, counts, local aliases and metrics, but never raw prompt/target text or source-group hashes. L30 is court infrastructure for real sequential updates; it is not yet evidence that indefinite lifelong learning has been achieved.

See `docs/L30-LONG-HORIZON-CONTINUAL-LEARNING-COURT.md`.

### L31 Factorized Continual Neural Update

L31 turns the L30 stability/plasticity court into a real sequential neural-update path for the standalone factorized Nolane language boundary.

One parent checkpoint is loaded twice: a frozen reference and an initially identical trainable candidate. Only the candidate low-rank boundary may receive gradients. The recurrent cortex and the entire reference remain immutable.

The evidence path is deliberately split:

```text
old train+dev -> rehearsal during update
old test      -> held-out retention court
new train     -> adaptation update
new test      -> held-out adaptation court
```

Both old and new protocols must independently pass L28 before training. After training, L30 decides whether the candidate learned the new held-out groups without unacceptable mean or worst-group forgetting on old held-out groups. A completed optimizer run can therefore still finish BLOCKED.

```bash
python scripts/train_continual_factorized_update.py \
  --retention-dataset /private/old/personalization.jsonl \
  --retention-protocol /private/old/personalization-protocol-v1.json \
  --adaptation-dataset /private/new/personalization.jsonl \
  --adaptation-protocol /private/new/personalization-protocol-v1.json
```

The saved artifact binds the parent checkpoint plus old/new dataset, protocol, L28 and L30 lineage. It remains unpromoted.

See `docs/L31-FACTORIZED-CONTINUAL-NEURAL-UPDATE.md`.

### L32 Multi-Cycle Continual Learning Ledger

L32 extends the real L31 neural updater from one checkpoint transition into an auditable chain of updates.

Each cycle must preserve exact artifact ancestry and neural boundary-state continuity, must carry a valid PASS L30 court, and must save the exact L31 training receipt inside the produced artifact. Reusing the same adaptation protocol does not count as a fresh learning cycle by default.

Per-cycle PASS is still not enough. L32 also evaluates the final factorized checkpoint directly against the checkpoint that existed before cycle 1 on one fixed L28-approved held-out retention panel. This catches cumulative forgetting that can remain hidden when every small update is considered separately.

```bash
python scripts/evaluate_long_horizon_retention.py \
  --initial /path/to/checkpoint-0/factorized-nolane.pt \
  --final /path/to/checkpoint-N/factorized-nolane.pt \
  --dataset /private/fixed-panel/personalization.jsonl \
  --protocol /private/fixed-panel/personalization-protocol-v1.json \
  --output runtime-data/l32/long-horizon-retention.json

python scripts/assess_multicycle_continual.py \
  --cycle runtime-data/cycle-1/l31-run-receipt.json \
  --cycle runtime-data/cycle-2/l31-run-receipt.json \
  --long-horizon runtime-data/l32/long-horizon-retention.json \
  --output runtime-data/l32/multicycle-chain.json
```

This remains evidence-only and unpromoted. Production authority still requires real multi-cycle runs plus transactional update/rollback safety.

See `docs/L32-MULTICYCLE-CONTINUAL-LEARNING-LEDGER.md`.

### L33 Transactional Checkpoint Recovery

L33 protects the actual checkpoint switch after L31/L32 evidence exists.

Continual candidates are no longer meant to replace a serving file in place. A local registry stores checkpoint bundles immutably and gives serving authority to one small self-digested `active.json` pointer. Candidate L31 evidence is reverified, its parent must equal the currently active checkpoint, and the bundle is staged before any authority change.

The pointer swap is atomic. If the process dies before the swap, recovery leaves the parent active and records the transaction aborted. If it dies after the swap but before commit bookkeeping finishes, recovery recognizes that the durable active pointer already references the candidate and records the transaction recovered-committed.

Rollback is also forward-only:

```text
generation 0 -> model A
generation 1 -> model B
generation 2 -> model A  (audited rollback)
```

Old pointer history is never rewritten.

```bash
python scripts/manage_checkpoint_registry.py \
  --registry runtime-data/continual-checkpoint-registry \
  init --bundle /path/to/initial-bundle

python scripts/manage_checkpoint_registry.py \
  --registry runtime-data/continual-checkpoint-registry \
  begin --candidate runtime-data/l31-cycle-next

python scripts/manage_checkpoint_registry.py \
  --registry runtime-data/continual-checkpoint-registry \
  verify --transaction <transaction-id>

python scripts/manage_checkpoint_registry.py \
  --registry runtime-data/continual-checkpoint-registry \
  commit --transaction <transaction-id>
```

The registry remains an experimental transactional authority substrate, not permission for autonomous production updates.

See `docs/L33-TRANSACTIONAL-CHECKPOINT-RECOVERY.md`.

### L34 Serving Reload Convergence

L34 closes the next failure mode after atomic checkpoint switching: multiple live serving processes temporarily holding different generations.

Each process records a self-digested lease for the exact checkpoint it has loaded. When the L33 active pointer changes, an old process becomes `DRAIN_RELOAD_REQUIRED`. A process that has loaded the new checkpoint becomes `WAITING_FOR_PEERS` until every still-live serving lease has converged. Only then does its request gate return `SERVE`.

The coordinator also double-reads the active pointer while computing convergence, so a second promotion landing during the court cannot produce a stale PASS.

```text
gen N active
  workers A/B -> SERVE

gen N+1 promoted
  A/B old -> DRAIN

A reloads
  A new -> WAITING_FOR_PEERS
  B old -> DRAIN
  convergence -> BLOCKED / split-brain

B reloads
  A/B new -> SERVE
```

`ServingSession.model_for_request()` enforces this barrier directly; a model object is not returned while the process is drained, expired, or waiting for peers.

v0.35.1 hardens request admission further: a second fence rechecks the active pointer after a gate PASS, and `ServingSession.request_model()` holds the process-local model generation stable for the lifetime of one in-flight inference. Already-admitted work may drain, but later requests are blocked as soon as checkpoint authority moves.

See `docs/L34-SERVING-RELOAD-CONVERGENCE.md`.

### L35 Evidence-Bound Promotion Authority

L35 closes the authority gap between offline continual-learning evidence and the transactional checkpoint registry.

A valid L31 candidate is no longer sufficient for the intended registry CLI. Promotion requires a short-lived authorization derived from the complete L32 multicycle evidence, the fixed initial-vs-final retention court, and an explicit local operator request that binds the exact production parent and final candidate.

This also handles an important multicycle case correctly: production can remain on checkpoint 0 while experiments learn `0 -> 1 -> 2` offline. If the full chain passes and checkpoint 2 remains good on the fixed checkpoint-0 retention panel, L35 can authorize one atomic `0 -> 2` L33 transaction. Intermediate checkpoint 1 never needs production authority.

The chain is recomputed from its cycle receipts; a standalone chain hash is not trusted. Authorization is rechecked at begin, verify and commit, expires by default after one hour, and cannot be replayed into a second transaction.

```bash
python scripts/authorize_continual_promotion.py request ... --approve
python scripts/authorize_continual_promotion.py authorize ...
python scripts/manage_checkpoint_registry.py begin \
  --candidate runtime-data/final-l31-candidate \
  --authorization runtime-data/l35/promotion-authorization.json
```

CI still uses synthetic evidence, so v0.35.0 provides the authority mechanism without claiming real-data production promotion has been earned.

See `docs/L35-EVIDENCE-BOUND-PROMOTION-AUTHORITY.md`.

### L36 Final Promotion Ceremony

L36 closes the evidence and serving chain for one checkpoint promotion:

```text
L32 multicycle evidence
  -> L35 explicit short-lived authorization
  -> L33 atomic committed UPDATE
  -> L34 serving convergence
  -> L36 immutable COMPLETE ceremony
```

A pointer swap alone is not a completed release. The ceremony re-verifies the stored authorization, committed transaction, checkpoint pointer, serving convergence receipt and their chronology. It also requires the promoted pointer to still be active at finalization.

Only a `COMPLETE` ceremony is persisted under `<registry>/ceremonies/<generation>.json`. A BLOCKED attempt remains diagnostic and does not become final authority evidence. Persisted receipts remain historically verifiable after later rollback, and registry audit detects ceremony tampering.

```bash
python scripts/finalize_promotion_ceremony.py \
  --registry runtime-data/continual-checkpoint-registry \
  finalize \
  --transaction <transaction-id> \
  --convergence runtime-data/serving-convergence.json
```

This implements the release ceremony mechanism; it does not claim that synthetic CI evidence is sufficient for autonomous real-world self-promotion.

See `docs/L36-FINAL-PROMOTION-CEREMONY.md`.

### L37 Cross-Platform Hard-Kill Court

L37 replaces same-process exception simulation with **real child-process death**. Test children call `os._exit()` at exact L33 commit and L34 reload durability boundaries, so Python cleanup does not run and stale locks/disk state must be recovered by a new process.

This court discovered and closes a real crash window: if generation N+1 pointer history was written but the process died before `active.json` moved from N, recovery could previously leave an orphan N+1 snapshot. L37 removes that snapshot only when its transaction, candidate checkpoint and parent pointer exactly match the interrupted transaction; otherwise recovery records a conflict.

A dedicated GitHub Actions matrix executes the hard-kill court on **Ubuntu and Windows**. It also proves reload death before ACK leaves an old drain-required lease, while death after ACK preserves the new-generation lease.

These are genuine process-kill/filesystem courts, but CI still uses synthetic checkpoint bundles rather than large real trained model files.

See `docs/L37-CROSS-PLATFORM-HARD-KILL-COURT.md`.

### L38 Recurrent Cortex Continual Plasticity

L38 moves continual learning inside the Nolane-owned recurrent cognition. Unlike L31, which updates only the low-rank language boundary, L38 freezes the entire language boundary and trains only the `DeepRecurrentStateSpaceCortex`.

The update still uses strict evidence separation:

```text
old train+dev -> rehearsal + frozen-reference distillation
old test      -> held-out retention
new train     -> recurrent-cortex adaptation
new test      -> held-out adaptation
```

A small parameter anchor discourages unnecessary cortex drift, but the real decision remains the held-out L30 stability/plasticity court. The candidate boundary must have zero gradients and an identical digest after training; the recurrent cortex must receive finite gradients and actually change.

```bash
python scripts/train_continual_cortex_update.py \
  --factorized runtime-data/l17-factorized-trained/factorized-nolane.pt \
  --retention-dataset /private/old/personalization.jsonl \
  --retention-protocol /private/old/personalization-protocol-v1.json \
  --adaptation-dataset /private/new/personalization.jsonl \
  --adaptation-protocol /private/new/personalization-protocol-v1.json
```

L38 artifacts remain explicitly unpromoted. The current L32/L35 production evidence chain understands L31 boundary-update cycles only; L38 will not masquerade as an L31 cycle to bypass that authority boundary.

See `docs/L38-RECURRENT-CORTEX-CONTINUAL-PLASTICITY.md`.

### L39 Unified Continual Model Ledger

L39 connects the two real continual-learning surfaces without pretending they are the same update type.

```text
L31 -> factorized language-boundary plasticity
L38 -> recurrent-cortex plasticity
```

Every adjacent cycle must preserve exact artifact ancestry plus boundary, cortex and composite model-state continuity. The default ledger requires at least one real recurrent-cortex cycle, so an L31-only chain cannot claim L39 recurrent-plasticity evidence.

A new endpoint evaluator compares the final mixed-plasticity checkpoint directly with the checkpoint before cycle 1 on one L28-approved held-out panel. Boundary and cortex are both allowed to change, but the endpoints must share L16 ancestry, architecture configuration and recurrent-state policy.

```bash
python scripts/evaluate_unified_long_horizon_retention.py \
  --initial /path/to/checkpoint-0/factorized-nolane.pt \
  --final /path/to/checkpoint-N/factorized-nolane.pt \
  --dataset /private/fixed-panel/personalization.jsonl \
  --protocol /private/fixed-panel/personalization-protocol-v1.json \
  --output runtime-data/l39/long-horizon-retention.json

python scripts/assess_unified_continual.py \
  --cycle runtime-data/cycle-1/l31-run-receipt.json \
  --cycle runtime-data/cycle-2/l38-run-receipt.json \
  --long-horizon runtime-data/l39/long-horizon-retention.json \
  --output runtime-data/l39/unified-chain.json
```

L39 remains unpromoted. L35 understands L32/L31 evidence only; a later authority layer must explicitly recompute and authorize the L39 mixed model-state chain.

See `docs/L39-UNIFIED-CONTINUAL-MODEL-LEDGER.md`.

### L40 Unified Promotion Authority

L40 adds a separate authorization path for the L39 mixed continual-learning ledger. It does not reinterpret L38 as L31 and does not weaken the older L35 schema.

Before issuing authorization, L40 re-verifies the fixed long-horizon receipt and recomputes the complete L39 chain from the ordered raw L31/L38 cycle receipts. The authorization binds the expected production parent, final candidate artifact, L39 chain SHA, long-horizon court, boundary/cortex cycle counts and initial/final composite model-state identities.

Operator intent remains explicit and short-lived: approval is a boolean, the raw nonce is never persisted, and authorization expires according to policy.

```bash
python scripts/authorize_unified_promotion.py \
  --cycle runtime-data/cycle-1/l31-run-receipt.json \
  --cycle runtime-data/cycle-2/l38-run-receipt.json \
  --long-horizon runtime-data/l39/long-horizon-retention.json \
  --unified-chain runtime-data/l39/unified-chain.json \
  --active-parent-checkpoint-sha256 <parent-sha> \
  --candidate-checkpoint-sha256 <candidate-sha> \
  --nonce "<one-time-secret>" \
  --approve \
  --output runtime-data/l40/unified-authorization.json
```

L40 is intentionally not consumed by L33 yet. A later integration wave must preserve transaction recovery, serving convergence and hard-kill courts before recurrent-cortex candidates can change production authority.

See `docs/L40-UNIFIED-PROMOTION-AUTHORITY.md`.

### L41 Unified Promotion Integration

L41 connects L40 mixed-plasticity authorization to the existing crash-recoverable production path without replacing the older L35/L31 mechanism.

The L33 registry now dispatches candidate bundles by their native receipt: L31 boundary updates remain valid, while L38 recurrent-cortex candidates are independently reverified and must match the L40 authorization's final composite model-state SHA. The production parent is likewise checked against the authorization's initial model-state SHA.

Authorization and staged candidate evidence are rechecked at begin, verify and commit. L36 final ceremony now records which authorization schema/kind justified the release and can bind either an L32/L35 chain or an L39/L40 chain.

The Platform Crash Court also exercises L40-authorized L38 transactions with real child-process `os._exit()` termination on Linux and Windows, including both pre-swap abort recovery and post-swap committed recovery.

```text
L39 mixed chain
 -> L40 AUTHORIZED
 -> L33 atomic transaction
 -> L34 serving convergence
 -> L36 COMPLETE ceremony
```

This closes the mechanical release path. A real recurrent-cortex production claim still requires approved real multi-window evidence and a real ceremony on trained checkpoints.

See `docs/L41-UNIFIED-PROMOTION-INTEGRATION.md`.

### L43 Real Longitudinal Learning Execution

L43 turns the real recurrent-cortex update path into one resumable longitudinal experiment instead of requiring manual orchestration for every window.

A valid plan contains at least five L23-approved learning windows plus one fixed held-out panel that is source-group isolated from all training/rehearsal data. Adaptation protocols and adaptation source groups must be fresh across cycles.

The executor runs the exact L38 checkpoint chain and then asks two different retention questions:

```text
initial checkpoint -> final checkpoint
    fixed old-capability panel

cycle 1 learned state -> final checkpoint
cycle 2 learned state -> final checkpoint
...
cycle N-1 learned state -> final checkpoint
    each cycle's own held-out adaptation panel
```

This means a final model cannot PASS merely by preserving pre-existing skills while forgetting things it learned during earlier cycles.

```bash
python scripts/run_real_longitudinal_learning.py \
  --plan /private/l43-plan.json \
  --validate-only

python scripts/run_real_longitudinal_learning.py \
  --plan /private/l43-plan.json \
  --output-dir runtime-data/l43-real-longitudinal \
  --resume
```

L43 never auto-authorizes promotion. A PASS produces evidence for later human review; L40 remains a separate explicit authority step.

See `docs/L43-REAL-LONGITUDINAL-EXECUTION.md`.

### L44 Product Experience Evidence Bridge

L44 connects normal Nolane product usage to the real longitudinal-learning evidence path without auto-approving or auto-training on private conversations.

A product chat window can now be exported directly from the local `living.db`, split into leakage-safe source groups, and initialized as an L24/L26/L27 local review workbench. Every imported candidate starts unreviewed and unapproved.

```bash
python scripts/product_evidence.py prepare-window \
  --db "/path/to/Nolane/living.db" \
  --profile "/path/to/Nolane/personalization.json" \
  --workspace runtime-data/real-window-001

python scripts/product_evidence.py review \
  --workspace runtime-data/real-window-001

python scripts/product_evidence.py finalize \
  --workspace runtime-data/real-window-001
```

After at least five explicitly reviewed windows exist, L44 can bind their finalized workbenches into one L43 plan. The operator still explicitly chooses which workbench is the isolated fixed panel, retention evidence and new adaptation evidence.

Before emitting the L43 plan, L44 adds a cross-window exact-content leakage court: the fixed panel may not repeat training prompts/pairs, adaptation windows may not repeat older adaptation content, and one cycle's retention/adaptation sets may not overlap even under different source IDs.

L44 authority remains `PRODUCT_REVIEWED_EVIDENCE_TO_L43_PLAN_NO_TRAINING_AUTHORITY`. It cannot train, call L40 or promote a checkpoint.

See `docs/L44-PRODUCT-EXPERIENCE-EVIDENCE-BRIDGE.md`.

### L45 In-App Explicit Evidence Review

L45 removes the terminal as a requirement for reviewing real product evidence while keeping the main chat surface unchanged.

Under Personalization -> Advanced, the user can prepare one local learning window from new product conversations and review one real USER/NOLANE pair at a time. Every candidate remains unapproved until the user explicitly chooses Approve, Reject or Sensitive. Before approval, the assistant target is editable, so the user can supply the exact corrected behavior Nolane should learn instead of merely reinforcing an answer the current model already produced. There is no approve-all or background training action.

The local learning registry forms a contiguous SQLite high-water chain, so later windows cannot silently reuse earlier product turns. Failed window preparation does not advance the cursor. Raw candidate text is returned only to the authenticated local review dialog; window/registry metadata remains text-free.

Finalization still reuses the existing L25/L27/L28 evidence courts. L45 cannot train or promote a model.

See `docs/L45-INAPP-EXPLICIT-EVIDENCE-REVIEW.md`.

### v0.47 Product Reliability Freeze

v0.47 intentionally adds no new primary UI surface. It hardens the existing product actions that are most likely to be retried or interrupted in real use.

- exact review-decision retries are idempotent;
- conflicting retries remain blocked by the frozen-decision rule;
- repeated Create Learning Window while a window is still pending returns the same window instead of consuming more transcript;
- a crash after an evidence window directory is atomically installed but before the registry pointer is written can recover the exact next self-verified contiguous window;
- registry cursor/index corruption remains fail-closed and cannot be hidden by recovery;
- unregistered window gaps remain blocked.

The v1 product-surface rule is now explicit: prefer reliability and depth in chat, personalization and reviewed learning over adding more permanent controls.

See `docs/V047-PRODUCT-RELIABILITY-FREEZE.md`.

### v0.48 Startup Readiness Self-Test

v0.48 keeps the v1 UI feature-frozen and makes the existing power state truthful.

Before the sidecar listens, the runtime now verifies writable local storage, SQLite integrity/write rollback and production release assets. Reviewed-learning registry health is advisory so an advanced learning feature cannot take down core chat.

When the user presses the existing power control, the production factorized cortex must additionally generate one real local token before phase may become `on`. A model that loads but cannot infer leaves the runtime in `error`.

The detailed readiness report is available only through authenticated `GET /v1/readiness`; normal status exposes only PASS/BLOCKED and failure counts.

See `docs/V048-STARTUP-READINESS-SELFTEST.md`.

### v0.49 Windows Clean-Install Court

v0.49 keeps the product surface frozen and upgrades Windows release evidence from “installer built” to “installed product works”.

The manual Windows release workflow now installs the generated NSIS package into an empty directory, verifies the installed model/runtime/tokenizer/ceremony/manifest hashes, runs the installed sidecar through authenticated readiness -> AI ON -> real local chat/history, then launches the installed Tauri app and requires it to spawn its bundled runtime.

The clean-install receipt contains hashes/status only, not chat text, auth tokens or user data paths. The court still requires a real promoted checkpoint; source CI cannot fake that evidence.

See `docs/V049-WINDOWS-CLEAN-INSTALL-COURT.md`.

### v0.50 Android Local Inference Foundation

v0.50 keeps the Ember Quiet product surface unchanged and moves the local Android path below the UI.

The factorized Nolane model now has an explicit one-token mobile contract, a verified `contract.json + weights.safetensors` package format and a pure-Rust token-step kernel. Python golden courts require the exported mobile path to match the existing standalone Nolane logits and recurrent state across multiple sequential tokens. A separate cross-language court runs the same package through Rust and requires matching logits/state before the crate may compile for `aarch64-linux-android`.

The mobile package contains no user latent, tokenizer, chat text or promotion authority. It is bound to the source checkpoint SHA-256 and fails closed on contract/weights tamper.

This is **not yet end-to-end Android local chat**. Native tokenizer/chat-template handling, sampling loop, persistent product-state bridge, Tauri local-target wiring and emulator/device courts remain explicit gates.

See `docs/V050-ANDROID-LOCAL-INFERENCE-FOUNDATION.md`.

### v0.51 Native Tokenizer & Generation Host

v0.51 keeps the existing Ember Quiet product UI unchanged and moves another inference layer into native Rust.

The mobile runtime now loads a frozen `tokenizer.json` without Python or Transformers, pre-fills every prompt token through the v0.50 Rust neural kernel, and performs bounded deterministic greedy autoregressive generation with EOS handling. CI freezes tokenization and generation outputs in Python and requires Rust to reproduce prompt token IDs, generated token IDs, decoded text and final recurrent state before both native crates compile for `aarch64-linux-android`.

This still does **not** claim end-to-end Android product chat. Exact product chat-template rendering, sampling, persistent product-state/latent wiring, Tauri LocalMobile routing, release-bound promoted assets and emulator/device courts remain explicit gates.

See `docs/V051-NATIVE-TOKENIZER-GENERATION.md`.

### v0.52 Frozen Product Prompt Contract

v0.52 keeps the product UI frozen and closes another Android backend parity boundary.

Instead of embedding a Jinja chat-template runtime or manually copying Qwen markup into Rust, Python Transformers renders unique system/user sentinels through the exact pinned tokenizer template at export/court time. Nolane freezes the resulting prefix/between/suffix segments into an integrity-bound prompt contract. Rust verifies the contract-file SHA plus tokenizer.json/tokenizer_config.json hashes, reproduces the exact prompt by concatenation, and can feed that prompt directly into the v0.51 native generation host.

CI requires both a synthetic end-to-end prompt→generation trajectory and an exact pinned Qwen3 tokenizer/template court. Android still is not called product-complete until dynamic product state/profile/memory payloads, sampling, LocalMobile Tauri routing, authorized release assets and emulator/device courts close.

See `docs/V052-FROZEN-PRODUCT-PROMPT-CONTRACT.md`.

### v0.53 Product Payload Parity

v0.53 keeps the product surface unchanged and removes a desktop/Android behavior split below the UI.

The product personalization/state/memory context is now first represented as a structured `NOLANE-V053-PRODUCT-PAYLOAD-INPUT-V1` payload, then rendered by both desktop Python and native Rust. The native runtime must match desktop on the user payload byte-for-byte, the full frozen Qwen chat prompt byte-for-byte, exact token IDs and response-length token budget.

The old Python-specific `repr(list)` formatting for open threads has been replaced with compact UTF-8 JSON so quotes, backslashes and Unicode have one cross-language representation. Reply/initiative modes, 4-thread and 8-memory bounds, language/style guidance and compact/balanced/expansive generation budgets are all explicit and fail-closed.

This does not yet claim end-to-end Android local chat. Seeded sampling parity, persistent local product state, Tauri LocalMobile wiring and emulator/device release courts remain open.

See `docs/V053-PRODUCT-PAYLOAD-PARITY.md`.

### v0.54 Seeded Sampling Parity

v0.54 keeps the product UI unchanged and closes the stochastic decoding split between the desktop product path and the native mobile runtime.

Both sides now share one frozen seeded nucleus-sampling contract: SplitMix64 RNG, millilogit quantization, Q40/Q32 probability arithmetic, deterministic token-ID tie breaking and minimal top-p nucleus semantics. Known-answer RNG vectors, top-p boundary courts, invalid-logit bounds and a cross-language sampled-generation fixture prevent “same seed” from being only a best-effort claim.

With the same prompt, product payload, latent, seed, temperature and top-p, Python and Rust must reproduce the same sampled token IDs, decoded text and final recurrent state. Production desktop generation still defaults to a fresh cryptographic 64-bit seed per response.

This still does not make Android product-complete: persistent local state/latent binding, Tauri LocalMobile routing, L36-authorized release assets and emulator/device courts remain open.

See `docs/V054-SEEDED-SAMPLING-PARITY.md`.

### v0.55 Persistent Mobile State Bridge

v0.55 keeps the product UI unchanged and closes the persistent state split between the desktop Living Runtime projection and the native Android inference host.

Python and Rust now share `NOLANE-V055-MOBILE-PERSISTENT-STATE-V1`: one checkpoint-bound, integrity-checked local state artifact carries the persistent latent, product profile, identity, relationship/affect state, bounded unresolved threads and bounded relevant memories. Writes are fsync + atomic replace; corruption, schema drift, checkpoint mismatch, invalid latent shape and non-finite state fail closed instead of silently creating a fresh identity.

The deterministic mobile fixture is now a real cross-language bridge court: Python emits the state artifact, Rust verifies it before f32 narrowing, attaches it to `MobileRuntime`, rebuilds the v0.53 product payload, executes seeded native generation, writes the state back and reloads identical semantics.

This still does not make Android product-complete: Tauri `LocalMobile` routing, L36-authorized mobile release assets and emulator/device courts remain open.

See `docs/V055-PERSISTENT-MOBILE-STATE-BRIDGE.md`.

### v0.56 Tauri LocalMobile Target

v0.56 closes the Android app-routing split. The existing product UI can now dispatch directly through Tauri into a native Rust LocalMobile product host instead of requiring a paired HTTP runtime.

The LocalMobile host owns product-compatible status, readiness, power, profile, bounded history and chat routes. First launch derives a fresh device-local identity from a checkpoint-bound bootstrap state; later launches keep the same identity, profile, interaction count, persistent state and conversation history. Chat uses the same seeded native generation path established in v0.54/v0.55.

Android now has a dedicated `RuntimeTarget::LocalMobile`. Remote pairing remains an explicit development/fallback override, and clearing that override returns to LocalMobile when native assets are available. No Python process, loopback HTTP server or Transformers runtime is required by the local Android route.

This is still not a production Android release claim. L36-authorized LocalMobile assets, emulator/device local-chat evidence, full mobile LivingEngine transition parity and latency/memory/battery courts remain open.

See `docs/V056-TAURI-LOCALMOBILE-TARGET.md`.

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
