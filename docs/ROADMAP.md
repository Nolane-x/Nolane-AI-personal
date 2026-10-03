# Living Architecture Roadmap

## L0 — Persistent organism shell

- [x] identity survives restart
- [x] heartbeat advances state without an LLM call
- [x] append-only event timeline
- [x] snapshot + digest after every transition
- [x] rollback primitive
- [x] local episodic memory
- [x] bounded memory retrieval
- [x] unresolved conversation threads
- [x] initiative and silence decision
- [x] Qwen3-0.6B local cortex adapter
- [x] always-on CLI event loop
- [x] core replay/continuity tests

## L1 — Social observer

Engineering closure: **COMPLETE**.

- [x] structured `SocialProposal`
- [x] Qwen can reuse the already-loaded model as observer
- [x] proposal-only authority boundary
- [x] deterministic `MutationValidator`
- [x] bounded affect delta
- [x] bounded relationship delta
- [x] source-event provenance on memories
- [x] low-confidence facts downgraded to inference
- [x] bounded thread creation/resolution
- [x] observer uncertainty persisted explicitly
- [x] observer failure is fail-closed and audited
- [x] mutation receipt stored as its own event
- [x] CLI opt-in via `--social-observer`

L1 does not claim the observer infers human emotion perfectly. It gives model inference a constrained, reversible persistence boundary.

## L2 — Recurrent living core

Engineering substrate: **COURT-READY / UNPROMOTED**.

- [x] deterministic replay records from real runtime history
- [x] exact state vector contract
- [x] deterministic event featurizer
- [x] explicit time features
- [x] 32D recurrent latent state
- [x] tiny GRU transition model
- [x] bounded state-delta head
- [x] action-prior head
- [x] future-return prediction head
- [x] exact analytical parameter audit
- [x] default core footprint: **14,515 parameters**
- [x] replay trainer with truncated BPTT
- [x] gradient clipping
- [x] checkpoint + manifest + DB/checkpoint SHA-256
- [x] chronological train/dev/test freeze protocol
- [x] per-record before/after state digests
- [x] protocol self-digest and drift verification
- [x] censored future-return labels to prevent tail leakage
- [x] held-out evaluator
- [x] state-fidelity gates
- [x] Brier improvement gate against train base-rate
- [x] parameter-cap gate
- [x] checkpoint/protocol SHA match gate
- [ ] train on a sufficiently large real interaction history
- [ ] obtain >=50 held-out state cases and >=20 uncensored return cases
- [ ] pass the frozen promotion court
- [ ] promote recurrent core to production state authority

A neural core that only imitates the deterministic reducer has not earned replacement authority. It must preserve state continuity and add held-out predictive value.

## L3 — Persistent latent continuity

Engineering substrate: **SHADOW-READY / NO PRODUCTION AUTHORITY**.

- [x] 32D latent checkpoint independent from Qwen KV cache
- [x] atomic latent persistence across restart
- [x] latent self-digest verification
- [x] bind latent to Living Runtime identity
- [x] bind latent to exact neural checkpoint SHA-256
- [x] bind latent to replay protocol SHA-256 when present
- [x] persist source-state version cursor
- [x] scalable replay resume via `after_version`
- [x] shadow prediction receipts
- [x] explicit `SHADOW_ONLY_NO_STATE_MUTATION` authority
- [x] explicit reset required for incompatible latent lineage
- [ ] pass L2 promotion court on real replay history
- [ ] evaluate latent stability across long restarts
- [ ] inject promoted latent into selected Qwen adapters
- [ ] permit promoted latent to influence wake policy

L3 may run in shadow before L2 promotion because it cannot mutate state, memory, initiative or Qwen behavior.

## L4 — Rest/consolidation

Engineering closure: **COMPLETE**.

- [x] idle-window REST scheduler
- [x] REST cooldown
- [x] deterministic zero-model consolidation baseline
- [x] optional Qwen deep-rest using the already-loaded cortex
- [x] proposal-only REST authority
- [x] deterministic ConsolidationValidator
- [x] minimum two-source evidence for derived memory
- [x] source IDs verified against the local DB
- [x] derived confidence capped by the weakest source
- [x] unsupported fact claims downgraded to inference
- [x] evidence required before REST can resolve a thread
- [x] append-only memory provenance graph
- [x] original memories preserved after consolidation
- [x] persistent REST cycle state
- [x] schema-v1 -> schema-v2 migration
- [x] manual `/rest` command
- [x] `--deep-rest` and `--no-rest` runtime controls
- [x] provenance audit script
- [x] automated REST courts

Live or REST conversations never update model weights directly. REST may form evidence-backed summaries, preferences, habits, inferences and intentions, but it cannot directly edit affect, relationship or personality state.

## L5 — Architecture surgery

Engineering substrate: **SHADOW-COURT-READY / UNPROMOTED**.

- [x] isolated 32D-latent residual adapter candidate
- [x] exact analytical adapter parameter audit
- [x] bounded residual gate
- [x] deterministic candidate initialization
- [x] bind candidate to pinned base-model identity/revision
- [x] hidden-size compatibility gate
- [x] paired baseline/counterfactual forwards
- [x] baseline-only served authority
- [x] counterfactual KL/logit/cosine/latency metrics
- [x] lightweight base-parameter mutation guard
- [x] exception-safe decoder hooks
- [x] structural shadow-admission court
- [x] one-load multi-prompt Qwen surgery suite
- [x] frozen multilingual structural prompt suite
- [x] real tiny `Qwen3ForCausalLM` compatibility court
- [x] L3 persistent-latent regression remains in neural court
- [ ] train candidate adapters only on frozen train evidence
- [ ] freeze matched held-out personalization/continuity cases
- [ ] compare untouched Qwen vs latent-conditioned Qwen under identical contexts
- [ ] prove held-out quality benefit without unacceptable general regression
- [ ] pass latency/resource promotion gates
- [ ] allow a promoted adapter to influence production generation
- [ ] evaluate later recurrent/state-space replacement blocks and dynamic-depth routing

A structural shadow-admission pass is not a quality promotion. No surgery is promoted without measured held-out benefit.


## L6 — Trainable Personal Cortex

Engineering substrate: **TRAINING-COURT-READY / UNPROMOTED**.

- [x] freeze every base-Qwen parameter
- [x] adapter-only optimizer ownership
- [x] latent-conditioned Qwen forward path
- [x] latent-conditioned Qwen generation path
- [x] explicit Qwen gradient-boundary abort
- [x] warm-start bounded residual gate for trainability
- [x] training receipts with loss/grad/digest evidence
- [x] trained adapter artifact separate from Qwen weights
- [x] artifact bound to base-model fingerprint and hidden size
- [x] local JSONL personalization dataset format
- [x] frozen chronological personalization train/dev/test protocol
- [x] dataset and per-example SHA-256 drift detection
- [x] train split only consumed by trainer
- [x] matched untouched-Qwen vs Personal-Cortex held-out NLL evaluation
- [x] frozen Vietnamese/English general-language regression anchor
- [x] parameter, gradient and base-mutation quality gates
- [x] explicit experimental generation command
- [x] real tiny Qwen3 gradient-training court
- [ ] collect sufficient real approved personalization examples
- [ ] freeze a real user personalization protocol
- [ ] train the real Qwen3-0.6B adapter
- [ ] pass held-out personal-quality + general-regression court
- [ ] measure real 0.6B latency/RAM overhead
- [ ] allow a promoted Personal Cortex artifact in the normal runtime

L6 changes Qwen hidden-state computation through trained neural parameters. It is no longer prompt-only personalization, but production authority remains blocked until real held-out evidence passes.


## L7 — Hybrid Recurrent Cortex

Engineering substrate: **REAL-QWEN3-COURT-READY / UNPROMOTED**.

- [x] recurrent neural mixer inside selected Qwen decoder layers
- [x] token-by-token recurrent state transition
- [x] persistent Living latent initializes recurrent state
- [x] selected layers share mixer weights but keep separate states
- [x] exact analytical parameter audit
- [x] default Qwen-hidden-1024 / recurrent-24 footprint: **56,641 parameters**
- [x] <=100K parameter cap in real trainer
- [x] frozen base-Qwen gradient boundary
- [x] recurrent-mixer-only optimizer ownership
- [x] same-input carry-state divergence court
- [x] reset restores original recurrent trajectory court
- [x] real tiny Qwen3 recurrent training court
- [x] held-out different-token-sequence gain court
- [x] trained hybrid artifact with base-model fingerprint binding
- [x] held-out personal NLL evaluator
- [x] Vietnamese/English general-regression anchor
- [x] persistent per-layer recurrent state across generation calls
- [x] recurrent-state self-digest
- [x] recurrent-state binding to AI identity, base model, mixer and Living latent
- [x] explicit recurrent-state reset path
- [x] experimental real local-Qwen training command
- [x] experimental real local-Qwen generation command
- [ ] collect enough real approved personalization data
- [ ] train the real Qwen3-0.6B recurrent mixer
- [ ] pass real held-out personal/general court
- [ ] measure real Qwen3-0.6B latency/RAM overhead
- [ ] compare static L6 adapter vs L7 recurrent mixer under the same frozen protocol
- [ ] promote only if L7 earns a better quality/continuity/resource tradeoff
- [ ] experiment with replacing/bypassing selected Transformer blocks only after L7 promotion evidence

L7 is a genuine recurrent neural path inside Qwen. It does **not** yet remove Transformer attention; it establishes the hybrid architecture and evidence boundary required before deeper block replacement.


## L8 — Depth-Recurrent Living Bridge

Engineering substrate: **FOUR-WAY-COURT-READY / UNPROMOTED**.

- [x] one recurrent bridge state travels across selected Qwen decoder depth
- [x] persistent 32D Living latent initializes the depth path
- [x] RMS latent normalization preserves sign/common-mode information
- [x] current hidden-state summary feeds every depth transition
- [x] learned decoder-layer identity embedding
- [x] learned bounded gate at each selected depth
- [x] exact analytical parameter audit
- [x] default Qwen-hidden-1024 footprint: **84,289 parameters**
- [x] hard <=100K parameter cap
- [x] frozen base-Qwen gradient boundary
- [x] real tiny Qwen3 depth-recurrent court
- [x] same-prompt opposite-latent sensitivity court
- [x] training loss + synthetic held-out gain court
- [x] generation and exception-safe hook cleanup
- [x] artifact save/load + base-model/hidden-size lineage gates
- [x] reuse the exact L6/L7 frozen personalization protocol
- [x] four-way evaluator: Qwen vs L6 vs L7 Hybrid vs L8
- [x] mandatory held-out improvement over Qwen, L6 and L7 Hybrid
- [x] Vietnamese/English general-regression gate
- [x] latency-overhead resource court
- [x] artifact-size resource court
- [x] checkpoint-bound quality+resource promotion ceremony
- [ ] train L6/L7/L8 on sufficient real approved personalization history
- [ ] freeze real held-out protocol
- [ ] prove L8 beats L7 Hybrid by >=0.005 held-out NLL
- [ ] measure real Qwen3-0.6B latency/RAM overhead
- [ ] promote only if depth recurrence earns its extra parameters/compute
- [ ] consider attention/block bypass only after matched L6/L7/L8 evidence

L7 and L8 explore different recurrence axes. Neither is assumed superior before the same held-out evidence decides.


## L9 — Recurrent Transformer-Block Replacement

Engineering substrate: **REAL-BYPASS-COURT-READY / UNPROMOTED**.

- [x] selected Qwen decoder blocks can be genuinely bypassed
- [x] shared recurrent replacement conditioned on Living latent + layer identity
- [x] vectorized GRU token processing
- [x] original decoder block forward count = 0 under replacement session
- [x] exception-safe restoration of original Qwen forwards
- [x] exact analytical parameter audit
- [x] default Qwen-hidden-1024 footprint: **84,289 parameters**
- [x] hard <=100K replacement parameter cap
- [x] frozen Qwen teacher hidden-state capture
- [x] teacher distillation before block bypass fine-tuning
- [x] end-to-end personalization fine-tuning with selected blocks absent
- [x] zero-gradient boundary on all Qwen weights
- [x] trained replacement artifact + base-model/hidden-size/protocol lineage
- [x] experimental generation with skipped blocks
- [x] five-way quality evaluator: Qwen/L6/L7/L8/L9
- [x] best-prior non-inferiority gate
- [x] Vietnamese/English general-regression gate
- [x] real wall-clock speedup resource gate
- [x] checkpoint-bound quality+resource promotion ceremony
- [ ] train L6/L7/L8/L9 on sufficient real approved personalization history
- [ ] freeze matched real held-out protocol
- [ ] prove L9 stays within 0.010 NLL of the best prior architecture
- [ ] prove L9 improves over untouched Qwen by >=0.005 NLL
- [ ] prove real Qwen3-0.6B median forward latency <=0.95x baseline
- [ ] evaluate how many decoder blocks can be replaced before the quality frontier breaks
- [ ] implement cache-compatible replacement before production autoregressive serving
- [ ] promote only if real compute removal survives both quality and resource courts

L9 is the first architecture wave where selected Transformer attention/MLP blocks can be absent from the forward path. It remains experimental until the matched real court demonstrates a worthwhile quality/speed frontier.

## L10 — Progressive Transformer-depth replacement

Engineering substrate: **PROGRESSIVE-COURT-READY / UNPROMOTED**.

- [x] calibration of eligible internal Qwen blocks using hidden residual RMS + cosine change
- [x] default protection for first/last decoder regions
- [x] immutable replacement ranking and plan SHA-256
- [x] plan bound to exact Qwen fingerprint and personalization protocol
- [x] monotonic 1 -> 2 -> 4 -> ... replacement curriculum
- [x] default target 50% decoder depth, capped at 12 blocks
- [x] shared <=100K recurrent replacement across all selected layers
- [x] per-stage teacher distillation on all currently selected layers
- [x] per-stage task fine-tuning on frozen train evidence
- [x] dev-regression rollback to previous accepted replacement weights
- [x] explicit model-forward boundary for recurrent replacement state
- [x] replay-safe no-cache generation state reset
- [x] incremental cached generation with recurrent state carry
- [x] cached vs replay-safe deterministic sequence court
- [x] progressive artifact bound to plan + dataset + base-model identity
- [x] quality court requires >=25% real Transformer depth replacement
- [x] quality court requires non-inferiority against L9
- [x] resource court requires <=0.90x baseline forward latency
- [x] resource court requires cached generation speedup
- [x] promotion binds quality/resource evidence to identical checkpoint and plan SHA
- [ ] freeze a real Qwen3-0.6B progressive plan from sufficient approved history
- [ ] train enough stages to replace >=25% of real Qwen depth
- [ ] pass real held-out personalization/general quality court
- [ ] pass real CPU/GPU latency and cached-generation resource court
- [ ] promote only if progressive depth removal earns measurable quality/compute value

L10 turns block replacement into a rollback-safe curriculum rather than a fixed proof-of-concept. It remains hybrid and unpromoted until real evidence passes.

## L11 — Recurrent Transformer Islands

Engineering substrate: **REGION-COLLAPSE-COURT-READY / UNPROMOTED**.

- [x] contiguous Transformer region abstraction with width >=2 blocks
- [x] one recurrent call replaces an entire island
- [x] all later Qwen blocks inside an island become identity mappings
- [x] original attention/MLP calls inside an island are absent from the forward path
- [x] direct whole-region calibration from pre-region to post-region hidden states
- [x] default candidate widths 2–4 blocks
- [x] edge-layer protection, non-overlap and minimum-gap constraints
- [x] frozen island selection order, stages and plan SHA-256
- [x] plan bound to Qwen fingerprint + personalization protocol
- [x] whole-region teacher distillation
- [x] staged island addition with dev-regression rollback
- [x] cached and replay-safe generation paths
- [x] region compression gate >=2 removed Transformer blocks per island
- [x] quality court compares untouched Qwen vs L10 vs L11
- [x] resource court requires speedup versus both Qwen and L10
- [x] artifact bound to checkpoint, plan, dataset and base-model identity
- [x] real tiny Qwen3 court proves width-3 region = 0 original block calls + 1 recurrent call
- [x] zero-gradient boundary on all Qwen parameters
- [ ] freeze real Qwen3-0.6B island plan from sufficient approved history
- [ ] replace >=25% of real Qwen depth with accepted islands
- [ ] pass real held-out quality/non-inferiority court against L10
- [ ] pass real CPU/GPU speed and cached-generation court
- [ ] promote islands only when region compression yields measurable quality/compute value

L11 is the first wave where one Nolane recurrent computation can stand in for multiple contiguous Transformer blocks. It remains hybrid and unpromoted until real Qwen3-0.6B evidence passes.

## L12 — Selective State-Space Cortex

Engineering substrate: **WIDE-REGION-STATE-SPACE-COURT-READY / UNPROMOTED**.

- [x] non-attentional token-recurrent state-space cortex
- [x] token-conditioned proposal, decay and output gates
- [x] persistent 32D Living latent conditions the state-space transition
- [x] RMS latent normalization preserves sign/common-mode information
- [x] exact analytical parameter audit
- [x] default Qwen-hidden-1024 / state-32 footprint: **72,897 parameters**
- [x] hard <=100K cortex parameter cap
- [x] full-sequence scan == incremental carried-state scan court
- [x] one wide contiguous Qwen region collapses to one state-space cortex call
- [x] every original Qwen attention/MLP block inside the region is absent
- [x] direct whole-region hidden-state calibration
- [x] nested widening curriculum with protected edge layers
- [x] default target ~60% decoder depth when dev evidence allows
- [x] whole-region Qwen teacher distillation at every widening stage
- [x] true replacement task training with Qwen region absent
- [x] dev-regression rollback to previous accepted cortex weights
- [x] cached autoregressive state carry
- [x] replay-safe no-cache state reset
- [x] cached/replay deterministic sequence equivalence court
- [x] quality court compares Qwen vs L11 vs L12
- [x] quality court requires >=40% real Transformer depth replacement
- [x] resource court requires speedup versus Qwen and L11
- [x] checkpoint + frozen-plan identity binding for promotion
- [x] real tiny Qwen3 court proves wide-region bypass and state-space semantics
- [x] zero-gradient boundary on all Qwen parameters
- [ ] freeze a real Qwen3-0.6B state-space widening plan from sufficient approved history
- [ ] train enough accepted stages to replace >=40% of real Qwen depth
- [ ] pass real held-out non-inferiority court against L11
- [ ] pass real CPU/GPU forward and cached-generation resource court
- [ ] promote only if the SSM cortex earns its larger architectural departure from Qwen

L12 is the first wave where a compact token-recurrent state-space sequence model can replace a single large contiguous share of Qwen depth. It remains hybrid and unpromoted until real Qwen3-0.6B evidence passes.

## L13 — Shrinking Qwen Scaffold

Engineering substrate: **THIN-SCAFFOLD-COURT-READY / UNPROMOTED**.

- [x] fast/slow multi-timescale non-attentional cortex
- [x] token-conditioned fast proposal + decay
- [x] slow-state retention floor for longer time scale
- [x] persistent 32D Living latent conditions both time scales
- [x] packed fast/slow recurrent state compatible with cached generation
- [x] exact analytical parameter audit
- [x] default hidden-1024 / state-24 footprint: **81,249 parameters**
- [x] hard <=100K cortex parameter cap
- [x] full-sequence == incremental packed-state scan court
- [x] fast/slow dynamic-separation court
- [x] Qwen scaffold size is an explicit optimization target
- [x] calibrated head/tail anchor selection
- [x] default shrink schedule ~50% -> 40% -> 32% -> 25% Qwen remaining
- [x] default target ~75% Qwen decoder-depth removal
- [x] minimum head/tail anchor protection
- [x] monotonic frozen scaffold plan + SHA-256
- [x] whole-region Qwen teacher distillation at every shrink stage
- [x] true central-region replacement during task training
- [x] dev-regression rollback to previous accepted shell
- [x] cached/replay-safe generation equivalence
- [x] quality court compares Qwen vs L12 vs L13
- [x] production gate blocks if Qwen scaffold >35%
- [x] resource court requires speedup versus Qwen and L12
- [x] checkpoint + scaffold-plan identity binding
- [x] tiny 12-layer Qwen3 court proves 2-head + 2-tail shell with central 8/12 blocks absent
- [x] zero-gradient boundary on all Qwen parameters
- [ ] freeze a real Qwen3-0.6B scaffold plan from sufficient approved history
- [ ] train accepted stages until real Qwen scaffold <=35%
- [ ] pass real held-out non-inferiority court against L12
- [ ] pass real CPU/GPU forward and cached-generation resource court
- [ ] test whether the shell can shrink below 25% without losing language competence
- [ ] promote only if reduced Qwen dependence earns measurable quality/compute value

L13 makes Qwen dependency itself measurable. The target is no longer merely a larger replacement region; it is a progressively thinner Qwen language scaffold around a Nolane-native multi-timescale cortex.


## L14 — Minimal Qwen Anchor Cortex

Engineering substrate: **MINIMAL-ANCHOR-COURT-READY / UNPROMOTED**.

- [x] deep recurrent state-space cortex with fast, slow and virtual-depth packed states
- [x] shared virtual-depth micro-steps per token
- [x] learned virtual-depth step embeddings
- [x] parameter count independent of removed Qwen depth
- [x] exact analytical parameter audit
- [x] default hidden-1024 / state-20 footprint: **89,805 parameters**
- [x] hard <=100K cortex cap
- [x] calibrated nested minimal-anchor plan
- [x] default endpoint: exactly one Qwen head block + one Qwen tail block
- [x] whole central Qwen region genuinely absent from the forward path
- [x] rollback-safe region distillation + task training
- [x] full-sequence == incremental packed-state scan
- [x] virtual-depth effect court
- [x] cached == replay-safe deterministic generation
- [x] quality court compares untouched Qwen vs L13 vs L14
- [x] quality gate requires Qwen anchors <=15% of decoder depth
- [x] resource court requires speedup versus Qwen and L13
- [x] checkpoint + anchor-plan identity binding
- [x] tiny 12-layer Qwen3 court proves layers 1..10 absent while only layer 0 + layer 11 remain
- [x] zero-gradient boundary on all Qwen parameters
- [ ] freeze a real Qwen3-0.6B minimal-anchor plan from sufficient approved history
- [ ] train accepted stages down to the 1+1 anchor shell
- [ ] pass real held-out non-inferiority court against L13
- [ ] pass real CPU/GPU forward and cached-generation resource court
- [ ] test whether the final Qwen decoder anchors themselves can be replaced by a native input/output boundary
- [ ] promote only if minimal anchors preserve language competence and earn measurable compute value

L14 makes the remaining Qwen decoder depth almost a boundary condition rather than the main reasoning substrate. It remains experimental until real Qwen3-0.6B evidence passes.


## L15 — Native Nolane Boundary

Engineering substrate: **DECODER-FREE-COURT-READY / UNPROMOTED**.

- [x] native forward bypasses all Qwen Transformer decoder blocks
- [x] native custom generation does not call Hugging Face generate()
- [x] Transformer KV cache removed from the native generation contract
- [x] autoregressive continuity owned by Nolane recurrent state
- [x] frozen Qwen token embeddings retained as input boundary
- [x] frozen Qwen final norm + LM head retained as output boundary
- [x] deep recurrent fast/slow/virtual-depth cortex reused under <=100K cap
- [x] frozen native-boundary spec with exact base-model + dataset lineage
- [x] teacher-logit distillation from full frozen Qwen
- [x] task loss + teacher KL training
- [x] zero gradients and zero mutation on Qwen weights
- [x] forward court requires total Qwen decoder calls = 0
- [x] generation court requires total Qwen decoder calls = 0
- [x] prompt full-scan == incremental recurrent scan
- [x] quality court compares untouched Qwen vs L14 vs L15
- [x] resource court independently counts decoder execution
- [x] quality/resource evidence bound to identical checkpoint + boundary spec SHA
- [x] real tiny Qwen3 neural court passes decoder-free forward, generation and training
- [ ] train real Qwen3-0.6B teacher -> native candidate on sufficient approved history
- [ ] pass held-out non-inferiority court against L14
- [ ] pass real CPU/GPU native-forward and native-generation resource court
- [ ] distill/replace Qwen embedding matrix
- [ ] distill/replace Qwen final norm + LM head
- [ ] export a standalone Nolane checkpoint with no Qwen weights required at inference

L15 removes the Transformer decoder from inference. It still retains frozen Qwen input/output boundary weights and therefore is decoder-free, not fully Qwen-free.


## L16 — Standalone Nolane Weights

Engineering substrate: **STANDALONE-OWNERSHIP-COURT-READY / UNPROMOTED**.

- [x] export Qwen-derived embedding/final-norm/output weights into Nolane-owned artifact
- [x] preserve tied and untied output-weight semantics
- [x] preserve boundary FP32/FP16/BF16 dtype
- [x] export exact L15 deep-recurrent cortex state
- [x] standalone loader takes no Qwen model object or model path
- [x] core standalone model imports no Transformers model implementation
- [x] standalone forward/generation run from owned tensors + PyTorch
- [x] artifact contains no Qwen decoder-layer, attention or MLP tensors
- [x] tight L15 -> L16 logit/state parity court
- [x] greedy generation parity court
- [x] source mutation isolation court
- [x] standalone survives deletion of source Qwen object
- [x] boundary and cortex digest roundtrip
- [x] quality/resource promotion binds standalone checkpoint to exact source L15 checkpoint
- [ ] export from a promoted real Qwen3-0.6B/L15 candidate
- [ ] pass full real held-out L15/L16 parity court
- [ ] pass real CPU/GPU standalone resource court
- [ ] factorize/distill inherited token embedding/output projection
- [ ] evaluate Vietnamese/English-focused native tokenizer without quality regression
- [ ] remove remaining provenance dependence on inherited Qwen language-boundary matrices

L16 owns every inference tensor inside the standalone artifact. It inherits the language boundary matrices from Qwen and therefore is runtime-independent, not provenance-independent.
\n## L17 — Factorized Language Boundary

Engineering substrate: **LOW-RANK-BOUNDARY-COURT-READY / UNPROMOTED**.

- [x] low-rank token embedding and output projection
- [x] tied input/output factor reuse
- [x] exact/full-rank numerical reconstruction court
- [x] randomized low-rank decomposition for large matrices
- [x] boundary-only L16 teacher distillation
- [x] cortex zero-gradient + exact digest preservation
- [x] standalone PyTorch runtime with no Qwen model object
- [x] held-out L16/L17 quality, general-anchor and resource courts
- [x] exact source-L16/checkpoint promotion lineage
- [ ] export/train/evaluate on a promoted real L16 checkpoint
- [ ] pass real held-out quality and CPU/GPU resource courts

L17 establishes the low-rank boundary mechanism but does not assert a universally correct rank.

## L18 — Adaptive Rank Frontier

Engineering substrate: **RANK-SELECTION-COURT-READY / UNPROMOTED**.

- [x] strictly descending configurable rank schedule
- [x] default search order 256 -> 192 -> 128 -> 96 -> 64
- [x] every rank independently initialized from the same dense L16 source
- [x] per-rank SVD/randomized factorization
- [x] per-rank boundary-only teacher distillation
- [x] real parameter-compression gate at every accepted rank
- [x] dev NLL regression gate versus L16
- [x] incremental dev-regression gate versus previous accepted rank
- [x] greedy-token agreement gate
- [x] first rejected smaller rank stops the frontier
- [x] smallest accepted rank becomes selected candidate
- [x] source cortex digest must remain identical through all candidates
- [x] immutable frontier receipt binds attempted/accepted/selected ranks
- [x] receipt binds exact source L16 checkpoint and selected checkpoint
- [x] frozen test split excluded from rank selection
- [x] selected candidate reuses L17 held-out quality/resource courts
- [x] promotion requires quality evidence rank == frontier-selected rank
- [x] promotion binds frontier/quality/resource to identical selected checkpoint and source L16
- [ ] run the full frontier on a promoted real L16 checkpoint
- [ ] identify the smallest real rank passing dev gates
- [ ] pass held-out quality/resource courts on that selected rank
- [ ] only then evaluate tokenizer/vocabulary migration

L18 turns low-rank compression from a fixed hyperparameter into an evidence-driven model-selection process. It does not yet claim that rank 64, 96, 128, or any other rank is the production optimum.


## L19 — Quantized Factor Runtime

Engineering substrate: **INT8-BOUNDARY-COURT-READY / UNPROMOTED**.

- [x] symmetric row-wise int8 quantization for token-code factors
- [x] symmetric row-wise int8 quantization for rank-to-hidden basis factors
- [x] per-row float scales
- [x] tied input/output path reuses the same quantized factors
- [x] untied output factors quantized independently
- [x] zero-row-safe quantization
- [x] int8 factors remain persistent runtime buffers
- [x] embedding dequantizes only requested token rows
- [x] output projection dequantizes vocabulary codes in bounded chunks
- [x] configurable logit chunk size
- [x] no full persistent float vocab x rank reconstruction
- [x] quantized runtime is inference-only
- [x] source factorized checkpoint SHA-256 binding
- [x] source L16 checkpoint SHA-256 binding
- [x] dataset/protocol fingerprint binding
- [x] boundary and cortex state digests
- [x] exact artifact roundtrip
- [x] cortex digest unchanged from source factorized candidate
- [x] rank unchanged from source factorized candidate
- [x] held-out NLL non-inferiority court versus float factorized source
- [x] Vietnamese/English anchor regression court
- [x] greedy-token agreement court
- [x] prompt full-scan == incremental scan court
- [x] checkpoint-size and tensor-byte resource court
- [x] target-device latency and generation-throughput court
- [x] exact byte audit for representative 151,936 x 1,024 rank-128 tied boundary
- [x] analytical gate: <30% of FP32 factorized storage
- [x] analytical gate: <60% of BF16/FP16 factorized storage
- [x] promotion binds quantized/factorized/L16 checkpoint lineage
- [ ] run L18 rank frontier on a promoted real L16 checkpoint
- [ ] quantize the actual selected L18 artifact
- [ ] pass real held-out quality court after quantization
- [ ] pass real target CPU/GPU latency and checkpoint/RAM resource court
- [ ] consider native int8 GEMM/packing only if pure-PyTorch dequant latency is insufficient
- [ ] evaluate lower-bit formats only with separate quality/resource evidence
- [ ] evaluate tokenizer/vocabulary migration only after real rank + quantization evidence

L19 is a real storage-format change, not a parameter-count relabel. It intentionally does not claim native int8 compute acceleration: the reference path dequantizes bounded chunks before floating-point matrix multiplication.


## L20 — Real Qwen3-0.6B Weight Court

Engineering evidence: **REAL-PINNED-WEIGHT-COURT-PASS / NO QUALITY PROMOTION**.

- [x] dedicated heavy GitHub Actions workflow separate from ordinary CI
- [x] exact pinned Hugging Face revision download and verification
- [x] full real Qwen3-0.6B model load
- [x] full-model forward with finite logits
- [x] actual model parameter-count court
- [x] stale model-lock metadata fails closed
- [x] corrected real parameter count: 596,049,920
- [x] measured real decoder depth: 28
- [x] measured real hidden size: 1,024
- [x] measured real vocabulary: 151,936
- [x] measured tied input/output embeddings
- [x] deterministic 512-row sample from learned vocabulary weights
- [x] rank-128 L17 factorization code executed on real learned weights
- [x] L19 row-wise int8 code executed on real learned factorized weights
- [x] sampled factor quantization error ~1.13%
- [x] sampled factorized-vs-int8 probe-logit error ~1.11%
- [x] full-shape dense boundary parameters: 155,583,488
- [x] full-shape rank-128 factorized parameters: 19,579,904 (~12.58%)
- [x] full-shape int8+scale footprint: 20,191,232 bytes
- [x] int8 footprint ~25.78% of FP32-factorized storage
- [x] int8 footprint ~51.56% of BF16/FP16-factorized storage
- [x] successful authority run 36977448315
- [x] evidence artifact 11214325349 with SHA-256 digest
- [ ] produce a trained real L16 standalone candidate from approved evidence
- [ ] run the real L18 rank frontier using train/dev only
- [ ] run frozen held-out personal + Vietnamese/English quality on the selected real rank
- [ ] quantize that selected real artifact and pass L19 held-out quality/resource courts
- [ ] compare target-device latency/RAM before any native int8-kernel work

L20 deliberately upgrades **evidence realism**, not production authority. A real full-model forward plus real-weight factorization/int8 fidelity does not substitute for trained held-out language-quality evidence.


## L21 — Real Candidate Evidence Pipeline

Engineering substrate: **FULL-EVIDENCE-CHAIN-READY / NO FABRICATED PROMOTION**.

- [x] privacy-preserving readiness receipt
- [x] dataset SHA-256 verification
- [x] frozen personalization protocol digest verification
- [x] protocol dataset-count consistency check
- [x] train split must be non-empty
- [x] dev split must be non-empty
- [x] held-out test split must contain at least 2 examples
- [x] frozen general anchor must contain at least 4 examples
- [x] persistent latent must verify its own digest and remain 32D
- [x] pinned Qwen weights must exist locally
- [x] local model revision marker must match `model.lock.json`
- [x] L14 comparison candidate required explicitly
- [x] clean workspace required to prevent evidence mixing
- [x] check-only mode is default
- [x] expensive execution requires explicit `--execute`
- [x] L15 spec freeze stage
- [x] L15 real native training stage
- [x] L15 held-out quality court
- [x] L15 resource court
- [x] L15 promotion gate
- [x] L16 standalone export
- [x] L16 parity court
- [x] L16 resource court
- [x] L16 promotion gate
- [x] L18 adaptive rank frontier uses train/dev only
- [x] L18 selected candidate receives frozen held-out quality court
- [x] L18 selected candidate receives resource court
- [x] L18 promotion binds selected rank/checkpoint/source
- [x] L19 quantized export from exact selected factorized source
- [x] L19 held-out quantized quality court
- [x] L19 target-device resource court
- [x] L19 promotion binds quantized/factorized/L16 lineage
- [x] each stage requires both zero exit code and expected artifact creation
- [x] first failing stage stops all later stages
- [x] stage receipt records artifact SHA-256 rather than prompt/target contents
- [x] final success status only after all 17 stages pass
- [ ] supply a sufficient user-approved local personalization dataset
- [ ] freeze its real train/dev/test protocol
- [ ] provide a matching real L14 comparator
- [ ] execute L21 on the real local evidence workspace
- [ ] inspect where the first empirical gate actually passes or blocks
- [ ] only after a complete PASS consider normal-runtime promotion

L21 does not make the model stronger by declaration. It makes the evidence chain reproducible and prevents later compression or runtime wins from masking an earlier quality failure.


## L22 — End-to-End Evidence Harness

Engineering substrate: **FULL-CHAIN-ORCHESTRATION-COURT-READY / SYNTHETIC-NON-AUTHORITY**.

- [x] canonical 17-stage evidence-chain contract in shared source module
- [x] stable contract SHA-256
- [x] real L21 pipeline imports the shared contract
- [x] stage-order enforcement before subprocess execution
- [x] extra-stage rejection after stage 17
- [x] zero exit code required for every stage
- [x] every declared output file must exist
- [x] every declared output file must receive SHA-256 coverage
- [x] final PASS requires exact observed order == frozen contract
- [x] final PASS requires all 17 stage indices to be contiguous
- [x] synthetic external subprocess stage runner
- [x] full 17-stage success fixture in ordinary CI
- [x] injected first-failure fixture
- [x] later stages provably do not execute after first failure
- [x] synthetic private prompt sentinel excluded from receipt
- [x] synthetic private target sentinel excluded from receipt
- [x] fixture authority permanently marked NEVER_PROMOTABLE
- [x] L21 readiness receipt binds the same stage-contract SHA-256
- [x] explicit CI contract audit on Python 3.10 and 3.12
- [ ] supply real user-approved personalization data
- [ ] supply matching real L14 comparator
- [ ] run the real L21 17-stage chain
- [ ] inspect the first empirical quality/resource blocker, if any
- [ ] only promote after real evidence completes the exact same contract

L22 proves orchestration correctness, not model quality. Synthetic fixture artifacts cannot satisfy any production promotion court.


## L23 — Approved Evidence Pack Builder

Engineering substrate: **CONSENT-BOUND-EVIDENCE-INTAKE-READY / UNPROMOTED**.

- [x] local JSONL evidence-source intake
- [x] explicit `approved:true` required for every eligible row
- [x] `sensitive:true` always excludes a row even when approved
- [x] Vietnamese/English allowlist by default
- [x] minimum seven eligible examples for downstream held-out readiness
- [x] prompt/target length bounds
- [x] bounded example weights
- [x] exact prompt/target duplicate rejection
- [x] NFC normalization
- [x] optional raw source IDs converted to SHA-256 only
- [x] raw source IDs absent from manifest
- [x] raw prompts/targets absent from manifest
- [x] output dataset contains only approved non-sensitive examples
- [x] frozen train/dev/test personalization protocol
- [x] at least two held-out test examples required
- [x] dataset SHA-256 binding
- [x] protocol SHA-256 binding
- [x] manifest SHA-256 binding
- [x] tampered dataset fails verification
- [x] non-empty output directory refuses overwrite
- [x] `runtime-data/` remains gitignored/local by default
- [x] generated pack passes L21 readiness when other prerequisites are valid
- [x] L21 `--evidence-pack` resolves verified dataset/protocol automatically
- [x] readiness CLI accepts `--evidence-pack`
- [x] L21 receipt binds exact evidence-manifest SHA-256
- [x] L21 receipt never copies approved prompt/target text
- [x] evidence authority explicitly remains UNPROMOTED
- [ ] supply a real user-approved local source file
- [ ] build and freeze the first real evidence pack
- [ ] provide matching persistent latent + L14 comparator
- [ ] execute L21 `--execute` on the verified pack
- [ ] inspect the first empirical model-quality/resource blocker

L23 closes the consent/intake gap without fabricating personal evidence. It does not automatically ingest chats and does not infer consent from conversation history.


## L24 — Local Review Queue

Engineering substrate: **LOCAL-HUMAN-REVIEW-BOUNDARY-READY / NO AUTOMATIC CONSENT**.

- [x] generic JSON/JSONL conversation export intake
- [x] only user and assistant roles considered
- [x] system/tool roles skipped
- [x] user→assistant pair extraction
- [x] imported candidates always `approved:false`
- [x] imported candidates always `reviewed:false`
- [x] imported sensitivity starts unknown
- [x] queue manifest stores hashes/counts rather than text
- [x] queue SHA-256 binding
- [x] source export SHA-256 binding
- [x] editing queue to self-approve invalidates verification
- [x] explicit separate decisions JSONL
- [x] decision requires boolean `approved`
- [x] decision requires boolean `sensitive`
- [x] unknown candidate decisions fail closed
- [x] duplicate candidate decisions fail closed
- [x] approved candidate requires VI/EN language
- [x] partial decisions leave undecided rows unapproved
- [x] reviewed-source manifest binds queue manifest + decisions file + output SHA
- [x] reviewed-source manifest contains no raw prompt/target/conversation ID
- [x] exact duplicate imported pairs fail closed
- [x] reviewed source feeds L23 without bypassing L23 filters
- [x] L23 court proves only approved non-sensitive rows survive
- [ ] import a real local conversation export
- [ ] manually review enough candidate pairs
- [ ] build first real L23 approved evidence pack
- [ ] execute L21 on that pack with valid model/latent/L14 prerequisites

L24 deliberately refuses to infer approval, sensitivity, usefulness or representativeness from imported conversations.


## L25 — Local Evidence Intake Pipeline

Engineering substrate: **LOCAL-EVIDENCE-LINEAGE-READY / NO MODEL PROMOTION AUTHORITY**.

- [x] L24 queue manifest verification before intake
- [x] explicit decisions-file SHA-256 binding
- [x] reviewed-source manifest self-digest verification
- [x] reviewed-source file SHA-256 verification
- [x] reviewed approved/rejected/sensitive/undecided count verification
- [x] eligible reviewed rows require VI/EN
- [x] L23 approved-pack build from verified reviewed source
- [x] L23 approved-pack independent verification
- [x] approved-pack manifest SHA-256 binding
- [x] dataset SHA-256 binding
- [x] frozen personalization protocol SHA-256 binding
- [x] complete queue -> decisions -> reviewed -> approved-pack lineage receipt
- [x] final intake manifest contains no raw prompt/target/source IDs
- [x] non-empty output workspace never overwritten
- [x] decisions tamper after freeze fails closed
- [x] reviewed-source tamper after freeze fails closed
- [x] fewer than seven eligible approved rows fail closed
- [x] default held-out reserve remains >=2 test examples
- [x] final approved pack directly drives L21 check-only
- [x] L21 receipt remains free of private prompt/target sentinels
- [ ] finalize an intake from a real local review queue
- [ ] manually approve enough representative non-sensitive examples
- [ ] verify the real L25 intake receipt
- [ ] run L21 --execute using the L25 approved-pack manifest
- [ ] treat downstream quality/resource failure as empirical evidence, not an orchestration error

L25 closes the local evidence lineage from human review to the exact frozen dataset/protocol consumed by L21. It does not infer consent, sensitivity, usefulness, balance or representativeness.


## L26 — Interactive Local Reviewer

Engineering substrate: **LOCAL-HUMAN-REVIEW-UX-READY / NO AUTOMATIC CONSENT**.

- [x] local terminal reviewer over verified L24 queue
- [x] no model dependency
- [x] no network dependency
- [x] opening a queue creates no decisions and no approvals
- [x] explicit approve/reject/sensitive/skip/quit actions
- [x] approved non-sensitive decision requires VI/EN
- [x] inherited VI/EN metadata reused when available
- [x] missing approval language requested explicitly
- [x] decisions persisted immediately after each reviewed candidate
- [x] atomic decisions-file replacement
- [x] existing decisions verified on resume
- [x] unknown existing candidate IDs fail closed
- [x] duplicate existing decisions fail closed
- [x] frozen decision cannot be silently overwritten
- [x] decided candidates skipped on resume
- [x] progress manifest bound to queue manifest + queue file + decisions SHA
- [x] progress manifest contains counts only, no candidate IDs
- [x] progress manifest contains no raw prompt/target/conversation ID
- [x] progress-manifest tamper detection
- [x] partial review remains valid with undecided candidates
- [x] generated decisions directly accepted by L24 apply-review path
- [ ] import the user's real conversation export locally
- [ ] manually review enough representative candidates
- [ ] finalize and verify a real L25 evidence intake
- [ ] execute L21 using that real approved pack

L26 reduces the manual-review friction without moving the consent boundary. It never infers approval, sensitivity, usefulness, or representativeness.


## L27 — Local Evidence Workbench

Engineering substrate: **LOCAL-EVIDENCE-WORKSPACE-READY / NO TRAINING AUTHORITY**.

- [x] one local workspace for L24 queue + L26 decisions + L25 intake
- [x] explicit phase state: QUEUE_READY
- [x] explicit phase state: REVIEW_IN_PROGRESS
- [x] explicit phase state: REVIEW_COMPLETE
- [x] explicit phase state: INTAKE_READY
- [x] initialization reuses real L24 queue builder
- [x] review command reuses real L26 interactive reviewer
- [x] status independently verifies queue and decisions
- [x] finalized status independently verifies complete L25 lineage
- [x] default finalize refuses any undecided candidates
- [x] explicit allow-undecided override never creates approval
- [x] finalization still requires >=7 approved non-sensitive examples
- [x] finalization reuses real L25 pack builder
- [x] readiness resolves exact L23/L25 approved pack
- [x] readiness reuses real L21 readiness assessor
- [x] workbench manifest binds source/queue/decisions/intake/dataset/protocol SHA lineage
- [x] manifest contains no raw prompt/target text
- [x] manifest contains no candidate IDs
- [x] workbench-manifest self-digest
- [x] workbench manifest tamper court
- [x] decisions tamper after intake invalidates workbench
- [x] private sentinels absent from readiness receipt
- [ ] initialize a workbench from the user's real local conversation export
- [ ] complete enough human review to reach >=7 representative approvals
- [ ] finalize a real L25 intake through the workbench
- [ ] pass real L21 readiness with the finalized pack
- [ ] run L21 --execute and let held-out quality/resource evidence decide promotion

L27 removes operational fragmentation from the real-data path. It does not fabricate evidence or infer consent, sensitivity, usefulness, representativeness or model quality.


## L28 — Evidence Quality & Leakage Court

Engineering substrate: **HELD-OUT-LEAKAGE-COURT-READY / NO MODEL PROMOTION AUTHORITY**.

- [x] preserve per-example hashed source-group lineage from reviewed source IDs
- [x] raw conversation/source IDs never copied into quality receipts
- [x] grouped personalization split strategy
- [x] one source group can belong to only one of train/dev/test
- [x] deterministic grouped boundary search
- [x] fewer than three distinct source groups fail closed
- [x] complete source-group coverage required for real-candidate readiness
- [x] exact normalized prompt cross-split leakage detection
- [x] exact prompt+target cross-split leakage detection
- [x] token-Jaccard + sequence-ratio near-duplicate cross-split detection
- [x] minimum train/dev/test structural gates
- [x] privacy-preserving evidence-quality receipt
- [x] quality court self-digest
- [x] standalone quality-court CLI
- [x] L23 builds court before creating a complete pack
- [x] L23 manifest binds quality receipt SHA + court SHA
- [x] L23 verification recomputes quality court
- [x] quality-receipt tamper detection
- [x] L25 intake binds quality court lineage
- [x] L27 workbench surfaces quality status + court SHA
- [x] L21 readiness recomputes quality court from dataset + protocol
- [x] direct dataset/protocol path cannot bypass source-group quality gate
- [x] L21 pipeline receipt exposes quality status + court SHA
- [x] legacy/no-group protocol remains usable only outside real-candidate authority
- [ ] run L28 on a real user-reviewed workbench
- [ ] inspect any near-duplicate/source-group blocks manually
- [ ] finalize a real L25 pack with L28 PASS
- [ ] execute L21 only after L28 PASS

L28 protects held-out validity. It does not prove that the approved evidence is representative enough, that the model learned personality correctly, or that a candidate deserves promotion.


## L29 — Held-out Group Robustness Court

Engineering substrate: **WORST-GROUP-HELDOUT-COURT-READY / NO MODEL PROMOTION BY AVERAGE ONLY**.

- [x] generic held-out source-group robustness module
- [x] privacy-preserving local group aliases
- [x] per-group reference mean
- [x] per-group candidate mean
- [x] per-group regression
- [x] overall regression retained for context
- [x] worst/best/mean group regression
- [x] group-regression dispersion
- [x] minimum two held-out source groups
- [x] real-candidate readiness blocks one-group test evidence before training
- [x] L15 per-example native/L14 NLL capture
- [x] L15 worst-group threshold reuses existing L14 non-inferiority tolerance
- [x] L17/L18 per-example factorized/L16 NLL capture
- [x] L17/L18 worst-group threshold reuses existing L16 non-inferiority tolerance
- [x] L19 per-example quantized/source-factorized NLL capture
- [x] L19 worst-group threshold reuses existing factorized non-inferiority tolerance
- [x] L15/L17/L19 quality evaluator exit status requires group PASS
- [x] robustness receipt self-digest
- [x] robustness receipt tamper court
- [x] promotion quality status requires valid L29 PASS receipt
- [x] promotion fails closed on missing group evidence
- [x] promotion fails closed on blocked group evidence
- [x] promotion fails closed on tampered group evidence
- [x] promotion output carries robustness status + court SHA
- [x] regression test where global average passes but one group fails
- [x] receipts omit raw prompt/target and source-group hashes
- [ ] run L29 on a real user-reviewed held-out set
- [ ] inspect worst-group failures before any promotion
- [ ] increase independent test-group coverage if readiness blocks
- [ ] execute full L21 only when L28 + L29 evidence requirements are satisfied

L29 is not cross-validation. It evaluates the frozen held-out source groups that the candidate has not trained on, and deliberately avoids pretending alternate already-seen groups are independent folds.

## L30 — Long-Horizon Continual-Learning Court

Engineering substrate: **STABILITY-PLASTICITY-COURT-READY / NO CONTINUAL PROMOTION AUTHORITY**.

- [x] explicit pre-update and post-update checkpoint lineage
- [x] frozen retention evidence for previously learned behavior
- [x] frozen adaptation evidence for newly introduced behavior
- [x] retention/adaptation source groups must be disjoint
- [x] minimum independent source-group counts
- [x] mean retention-regression gate
- [x] worst-group retention-forgetting gate
- [x] mean adaptation-gain gate
- [x] worst-group adaptation-regression gate
- [x] privacy-preserving local group aliases only
- [x] source-group hashes excluded from receipts
- [x] self-digested court receipt
- [x] tamper detection
- [x] fail-closed promotion-status helper
- [x] standalone local CLI
- [x] CI courts for hidden forgetting and hidden adaptation failure
- [ ] train a real sequential candidate update from approved new evidence
- [ ] freeze real old/new source-group evidence across multiple time windows
- [ ] pass L30 with real checkpoints
- [ ] repeat over multiple update cycles to measure long-horizon forgetting
- [ ] bind any future continual-learning production updater to valid L30 receipts

L30 proves stability/plasticity for one measured sequential checkpoint update. It does not yet prove indefinite lifelong learning.
\n

## L31 — Factorized Continual Neural Update

Engineering substrate: **REAL-SEQUENTIAL-UPDATE-COURT-READY / UNPROMOTED**.

- [x] load one parent factorized checkpoint as frozen reference and trainable candidate
- [x] exact parent boundary-state equality required before update
- [x] exact parent recurrent-cortex equality required before update
- [x] candidate/reference object aliasing rejected
- [x] old train+dev evidence used only for retention rehearsal
- [x] old test evidence reserved for held-out retention court
- [x] new train evidence used for adaptation update
- [x] new test evidence reserved for held-out adaptation court
- [x] frozen-parent KL rehearsal during adaptation
- [x] candidate low-rank boundary is the only trainable neural ownership surface
- [x] candidate recurrent cortex receives zero gradients
- [x] frozen reference receives zero mutation
- [x] old and new evidence independently require L28 PASS
- [x] post-update candidate is judged by the L30 stability/plasticity court
- [x] a trained candidate can remain BLOCKED
- [x] parent/dataset/protocol/L28/L30 lineage is bound into the output artifact
- [x] non-empty output workspace fails closed
- [x] real local-data runner
- [x] tiny real-PyTorch regression courts
- [ ] execute L31 on real approved old/new evidence windows
- [ ] pass L30 with a real factorized checkpoint transition
- [ ] repeat sequential update cycles and quantify cumulative forgetting
- [ ] add interruption-safe transactional update/rollback
- [ ] bind production model promotion to multi-cycle evidence rather than one update

L31 makes continual learning a real neural update path rather than only a court definition. It remains unpromoted until real held-out old/new evidence passes.

## L32 — Multi-Cycle Continual Learning Ledger

Engineering substrate: **MULTICYCLE-CHAIN-AND-FIXED-PANEL-READY / NO PRODUCTION AUTHORITY**.

- [x] L31 run receipts persist beside each candidate artifact
- [x] exact parent artifact SHA -> next parent SHA continuity
- [x] exact candidate boundary-after -> next boundary-before continuity
- [x] every embedded L30 receipt must be self-digest valid and PASS
- [x] every saved artifact must contain the exact claimed L31 training receipt
- [x] saved boundary/cortex state digests must match L31 training evidence
- [x] L31 lineage digest verified for every cycle
- [x] fresh adaptation protocol required per cycle by default
- [x] moving-window retention/adaptation statistics recorded across cycles
- [x] fixed held-out retention panel compares final checkpoint directly to initial checkpoint
- [x] fixed panel requires valid grouped personalization protocol
- [x] fixed panel requires L28 structural quality PASS in the real evaluator
- [x] fixed panel gates overall and worst-group cumulative regression
- [x] initial/final checkpoints must share L16 ancestry
- [x] recurrent cortex must remain identical across boundary-only update chain
- [x] fixed-panel endpoints must exactly match multicycle chain endpoints
- [x] self-digested multicycle chain receipt
- [x] CLI for real checkpoint fixed-panel evaluation
- [x] CLI for ordered multicycle verification
- [x] CI courts for artifact discontinuity, neural-state discontinuity, replayed adaptation windows and fixed-panel failure
- [ ] execute at least two real approved L31 update cycles
- [ ] pass a real initial-vs-final fixed retention panel
- [ ] extend real run to 5+ update windows
- [ ] add transactional interruption recovery and atomic rollback
- [ ] define production promotion authority only after repeated real multicycle PASS

L32 prevents individually acceptable updates from being mistaken for lifelong learning when their checkpoint history is broken or their cumulative endpoint forgets the original held-out panel.

## L33 — Transactional Checkpoint Recovery

Engineering substrate: **CRASH-RECOVERABLE-POINTER-REGISTRY-READY / NO AUTONOMOUS PROD AUTHORITY**.

- [x] immutable checkpoint artifact store
- [x] self-digested active checkpoint pointer
- [x] immutable pointer history by generation
- [x] atomic pointer replacement
- [x] temporary-file flush + fsync before authority swap
- [x] directory fsync where supported
- [x] exclusive registry writer lock
- [x] stale lock requires explicit recovery action
- [x] L31 candidate receipt fully reverified before staging
- [x] candidate parent checkpoint must equal currently active checkpoint
- [x] complete candidate bundle digest bound into transaction ID
- [x] staged candidate reverified after copy
- [x] immutable PREPARED and VERIFIED transaction events
- [x] active parent rechecked before commit
- [x] crash before pointer swap recovers as aborted
- [x] crash after pointer swap recovers as committed
- [x] ambiguous recovery records RECOVERY_CONFLICT instead of guessing
- [x] rollback creates a new forward pointer generation
- [x] rollback receipts are self-digested and audited
- [x] registry audit covers pointer ancestry and artifact digests
- [x] active-pointer tamper court
- [x] rollback-receipt tamper court
- [x] local registry management CLI
- [ ] execute L33 around real L31 candidate artifacts
- [ ] simulate hard process kill during real artifact copy/commit on Windows and Linux
- [x] add serving-process checkpoint reload handshake
- [x] prevent split-brain when multiple serving processes observe pointer changes
- [x] define real L32 evidence -> promotion authorization policy
- [ ] promote only after transactional reload/rollback court passes on real checkpoints

L33 makes a model update crash-recoverable without rewriting history. It deliberately does not grant an autonomous updater production authority.

## L34 — Serving Reload Convergence

Engineering substrate: **SPLIT-BRAIN-SAFE-SERVING-BARRIER-READY / NO PROMOTION DECISION AUTHORITY**.

- [x] process-local serving leases
- [x] lease self-digest
- [x] loaded generation/pointer/checkpoint binding
- [x] hashed process identity in receipts
- [x] heartbeat + lease expiration
- [x] minimum-live-process policy
- [x] reload plan bound to one process and one target pointer
- [x] reload ACK rejects stale target after active changes
- [x] wrong-checkpoint loader rejected
- [x] split-brain detection across live leases
- [x] old-generation process enters DRAIN_RELOAD_REQUIRED
- [x] early new-generation process enters WAITING_FOR_PEERS
- [x] SERVE granted only after full live-process convergence
- [x] active pointer double-read during convergence
- [x] active-change race fails closed
- [x] process-local model swap lock
- [x] model_for_request enforces serving gate
- [x] post-gate request fence rechecks active authority before admission
- [x] request context holds process-local generation stable for in-flight inference
- [x] admitted old-generation request may drain; later requests fail closed
- [x] poll_reload cannot swap local model inside request context
- [x] real factorized checkpoint loader helper
- [x] local serving coordination CLI
- [x] CI multi-worker convergence court
- [ ] run L34 with real factorized serving processes
- [ ] test process kill/restart during real reload
- [ ] test Windows/Linux filesystem + process timing differences
- [x] bind L32 evidence into explicit promotion authorization
- [x] add promotion policy that L33 must verify before begin/commit
- [ ] expose production-ready reload metrics only after real multi-process court passes

L34 prevents mixed live checkpoint generations from serving simultaneously when all serving requests obey the gate.

## L35 — Evidence-Bound Promotion Authority

Engineering substrate: **EXPLICIT-EVIDENCE-PROMOTION-AUTHORITY-READY / REAL-DATA PROMOTION UNPROVEN**.

- [x] operator promotion request is explicit and self-digested
- [x] raw operator nonce is never persisted
- [x] request binds active parent checkpoint
- [x] request binds exact final candidate checkpoint
- [x] request binds exact L32 multicycle chain
- [x] request binds exact fixed long-horizon court
- [x] full L32 chain is recomputed from ordered L31 cycle receipts
- [x] fixed-panel receipt is reverified
- [x] adaptation-protocol reuse is forbidden for promotion
- [x] minimum multicycle evidence policy
- [x] authorization TTL
- [x] authorization self-digest
- [x] authorization checked at begin
- [x] authorization checked at transaction verify
- [x] authorization checked immediately before commit
- [x] expiration between verify/commit blocks pointer swap
- [x] one authorization cannot begin two transactions
- [x] authorization file is included in registry audit
- [x] offline multicycle final artifact may jump from chain start -> chain end
- [x] direct L33 immediate-parent path still rejects that un-authorized jump
- [x] operational registry CLI begin requires authorization
- [x] local request + authorization CLI
- [x] courts for denial, mismatch, tamper, expiry, replay and evidence recomputation
- [ ] run L35 on approved real multicycle evidence
- [ ] execute authorized transaction with a real final factorized artifact
- [ ] complete real L34 multi-process convergence after that transaction
- [x] build final promotion ceremony receipt spanning L35 -> L33 -> L34
- [ ] run hard-kill tests during authorized commit/reload on Windows and Linux
- [ ] only then consider enabling autonomous promotion policy

L35 decides whether a proven offline learning chain may request an atomic checkpoint transition. Synthetic CI proves mechanics, not real production fitness.

## L36 — Final Promotion Ceremony

Engineering substrate: **END-TO-END-RELEASE-CEREMONY-READY / REAL PRODUCTION EVIDENCE STILL REQUIRED**.

- [x] L34 convergence receipts carry self-digested assessed_at timestamp
- [x] public convergence receipt verifier
- [x] registry pointer lookup by verified pointer SHA
- [x] registry promotion-authorization lookup bound to PREPARED transaction
- [x] require exactly one COMMITTED or RECOVERED_COMMITTED terminal event
- [x] committed pointer transaction binding
- [x] committed pointer must be UPDATE transition
- [x] authorized candidate must equal committed checkpoint
- [x] L35 authorization SHA bound into ceremony
- [x] L32 multicycle chain SHA bound into ceremony
- [x] long-horizon retention court SHA bound into ceremony
- [x] L34 serving convergence SHA bound into ceremony
- [x] pointer generation/checkpoint/SHA must match convergence
- [x] authorization -> prepare -> pointer -> commit -> convergence -> ceremony timeline court
- [x] pointer swap must occur before authorization expiry
- [x] active pointer must still equal promoted pointer at finalization
- [x] COMPLETE and BLOCKED ceremony states
- [x] only COMPLETE ceremonies persist as final authority evidence
- [x] immutable per-generation ceremony file
- [x] historical ceremony remains verifiable after later rollback
- [x] ceremony self-digest and privacy boundary
- [x] registry audit covers persisted ceremony receipts
- [x] dedicated ceremony audit
- [x] local finalization/verify/audit CLI
- [x] end-to-end synthetic L32 -> L35 -> L33 -> L34 -> L36 court
- [x] courts for blocked convergence, pointer move, bad chronology and tamper
- [ ] execute L36 on approved real multicycle evidence
- [ ] execute real authorized L33 checkpoint swap
- [ ] converge real multi-process serving workers on that checkpoint
- [ ] run hard-kill commit/reload tests on Windows and Linux
- [ ] repeat real promotion then rollback and verify historical ceremony
- [ ] define policy for autonomous promotion only after repeated real COMPLETE ceremonies

L36 closes the release-evidence chain mechanically. Synthetic CI proves the ceremony protocol, not that any learned candidate has earned real production authority.

## L37 — Cross-Platform Hard-Kill Court

Engineering substrate: **REAL-PROCESS-CRASH-RECOVERY-COURT-READY / REAL LARGE-CHECKPOINT DEPLOYMENT UNPROVEN**.

- [x] commit fault hook at durable authority boundaries
- [x] child process terminated with os._exit rather than Python exception
- [x] stale registry lock survives hard process death
- [x] explicit stale-lock recovery court
- [x] crash after artifact install is representable
- [x] crash after pointer snapshot is representable
- [x] crash after active pointer swap is representable
- [x] crash after pointer audit event is representable
- [x] crash after COMMITTED event is representable
- [x] detect orphan N+1 pointer snapshot when active remains N
- [x] remove orphan pointer only when transaction/checkpoint/parent lineage all match
- [x] unexpected orphan snapshot becomes RECOVERY_CONFLICT
- [x] hard kill after pointer snapshot recovers RECOVERED_ABORTED
- [x] hard kill after active swap recovers RECOVERED_COMMITTED
- [x] hard kill after COMMITTED preserves one terminal state
- [x] authorized L35 transaction path used by hard-kill checkpoint courts
- [x] serving reload process kill before lease ACK
- [x] serving reload process kill after lease ACK
- [x] old lease remains drain-required after pre-ACK crash
- [x] restarted worker must register active checkpoint before serving
- [x] dedicated Linux + Windows GitHub Actions crash matrix
- [ ] repeat process-kill court with real trained factorized checkpoint files
- [ ] stress very large artifact copy/install interruption
- [ ] run real multi-worker service restart under hard kill
- [ ] test accelerator/GPU teardown and reload where available
- [ ] exercise real L36 promotion -> hard kill -> convergence -> ceremony path
- [ ] define power-loss expectations beyond process-crash/fsync guarantees

L37 replaces exception-only crash simulation with real child-process death and closes the orphan-pointer window discovered between pointer-history write and active authority swap.

## L38 — Recurrent Cortex Continual Plasticity

Engineering substrate: **INTERNAL-RECURRENT-PLASTICITY-READY / UNPROMOTED**.

- [x] load one factorized parent as frozen reference + identical candidate
- [x] exact candidate/reference boundary equality required before training
- [x] exact candidate/reference recurrent-cortex equality required before training
- [x] composite model-state digest binds boundary + cortex
- [x] candidate language boundary fully frozen
- [x] candidate boundary gradients must remain zero
- [x] candidate deep recurrent cortex is the only trainable surface
- [x] finite cortex gradients required every optimizer step
- [x] candidate cortex digest must change
- [x] frozen reference boundary/cortex must remain unchanged
- [x] new train evidence drives cortex adaptation
- [x] old train+dev evidence provides rehearsal
- [x] frozen-reference KL protects old behavior
- [x] recurrent-cortex parameter anchor limits unnecessary neural drift
- [x] old test remains held-out retention court
- [x] new test remains held-out adaptation court
- [x] old/new evidence independently require L28 PASS in the real runner
- [x] post-update L30 stability/plasticity court
- [x] optimizer success may still finish BLOCKED
- [x] self-digested L38 training receipt
- [x] embedded L30 receipt reverified
- [x] real local-data cortex update runner
- [x] output remains standalone factorized Nolane artifact with no Qwen runtime dependency
- [x] Neural Shadow runs real PyTorch L38 courts
- [x] L38 deliberately excluded from current L32/L35 L31-only promotion schema
- [ ] execute L38 on approved real old/new evidence windows
- [ ] compare L38 adaptation/forgetting against L31 boundary-only updates
- [ ] repeat L38 across multiple real time windows
- [ ] build mixed-cycle ledger that understands boundary and cortex update types explicitly
- [ ] add identity/relationship-specific retention panels before any cortex promotion
- [ ] only then consider L38 artifacts eligible for L35/L36 promotion

L38 makes the Nolane-owned recurrent cognition itself plastic. It remains unpromoted because the existing production evidence chain intentionally understands only L31 boundary-update cycles.

## L39 — Unified Continual Model Ledger

Engineering substrate: **MIXED-PLASTICITY-LONG-HORIZON-READY / NO PRODUCTION AUTHORITY**.

- [x] explicit L31 boundary-update normalization
- [x] explicit L38 recurrent-cortex-update normalization
- [x] native L31 verifier reused without schema weakening
- [x] L38 run receipt + lineage verifier
- [x] L38 composite model-state digest reverified
- [x] L38 L30 pre/post identity bound to composite state
- [x] artifact SHA continuity across mixed update types
- [x] boundary-state continuity across mixed update types
- [x] recurrent-cortex-state continuity across mixed update types
- [x] composite model-state continuity across mixed update types
- [x] unique adaptation protocol required across all update types
- [x] at least one recurrent-cortex cycle required by default
- [x] real initial-vs-final fixed-panel evaluator permits both boundary and cortex plasticity
- [x] endpoint evaluator requires same L16 ancestry
- [x] endpoint evaluator requires stable boundary/cortex architecture configs
- [x] endpoint evaluator requires stable recurrent-state carry policy
- [x] L28-approved fixed held-out panel remains mandatory
- [x] unified chain self-digest
- [x] mixed L31 -> L38 CI court
- [x] cortex discontinuity court
- [x] model-state discontinuity court
- [x] cross-update adaptation replay court
- [x] L31-only rejection court for recurrent-plasticity evidence
- [ ] execute real mixed L31/L38 multi-window learning chain
- [ ] pass real initial-vs-final fixed panel after 5+ mixed cycles
- [ ] extend promotion authorization to L39 evidence
- [ ] bind recurrent-cortex plasticity into L33/L36 production ceremony
- [ ] run real serving + hard-kill court after recurrent-cortex promotion

L39 makes the continual-learning ledger understand the whole factorized Nolane model state. It does not grant L38 artifacts production authority.

## L40 — Unified Promotion Authority

Engineering substrate: **MIXED-PLASTICITY-AUTHORIZATION-READY / TRANSACTION-INTEGRATION-PENDING**.

- [x] separate promotion schema for L39 mixed model-state evidence
- [x] explicit operator approval request
- [x] one-time nonce hashed; raw nonce never persisted
- [x] explicit production-parent checkpoint binding
- [x] explicit final-candidate checkpoint binding
- [x] L39 chain SHA binding
- [x] fixed long-horizon court SHA binding
- [x] recompute L39 from ordered raw L31/L38 cycle receipts
- [x] refuse chain policy that allowed adaptation replay
- [x] minimum total cycle policy
- [x] minimum recurrent-cortex cycle policy
- [x] initial/final composite model-state SHA bound into authorization
- [x] short-lived authorization TTL
- [x] authorization self-digest
- [x] operator deny court
- [x] candidate mismatch court
- [x] raw-cycle evidence drift court
- [x] cortex-evidence floor court
- [x] authorization expiry court
- [x] authorization tamper court
- [x] local authorization CLI
- [x] teach L33 transaction registry to consume L40 authorization
- [x] verify final candidate bundle according to native L31/L38 schema
- [x] extend L36 ceremony to record unified authorization schema
- [x] rerun Linux/Windows hard-kill court on unified promotion path
- [ ] execute real mixed-plasticity production ceremony

L40 can authorize L39 evidence but deliberately cannot yet change the active checkpoint. Transaction and ceremony integration remain fail-closed.

## L41 — Unified Promotion Integration

Engineering substrate: **MIXED-PLASTICITY-RELEASE-PATH-READY / REAL-DATA-CLOSURE-PENDING**.

- [x] authorization dispatcher preserves L35 and adds L40
- [x] candidate-bundle dispatcher preserves L31 and adds L38
- [x] ambiguous L31+L38 candidate bundles rejected
- [x] L40 active-parent composite model-state binding
- [x] L40 final-candidate composite model-state binding
- [x] candidate run schema bound into PREPARED transaction evidence
- [x] candidate run receipt SHA bound into PREPARED transaction evidence
- [x] L40 authorization reverified at begin
- [x] L40 authorization reverified at transaction verify
- [x] L40 authorization reverified at commit
- [x] L38 candidate reverified after staging
- [x] L38 candidate reverified immediately before commit
- [x] staged L38 receipt tamper court
- [x] L36 ceremony accepts L35 or L40 through dispatcher
- [x] ceremony records authorization schema + kind
- [x] ceremony binds L39 evidence-chain SHA for L40
- [x] registry audit accepts and reverifies L40 authorization
- [x] end-to-end L39 -> L40 -> L33 -> L34 -> L36 court
- [x] L40/L38 hard-kill pointer-snapshot court on Linux + Windows
- [x] L40/L38 hard-kill active-swap court on Linux + Windows
- [ ] execute L41 with real approved L39 evidence
- [ ] run 5+ real mixed plasticity cycles
- [ ] run actual trained checkpoint serving convergence
- [ ] run resource/GPU teardown court when applicable
- [ ] complete a real immutable L36 ceremony for recurrent-cortex promotion

L41 closes the mechanical release path for recurrent-cortex plasticity while preserving the older L35/L31 path. The remaining closure is empirical, not another synthetic authority shortcut.

## L42 — Product Client & Distribution Surface

Engineering target: **WINDOWS-ONE-CLICK + ANDROID-SHELL / COURT-PENDING UNTIL CI CLOSES**.

- [x] bind product UI work to NUI ARTIFACT_WORK lifecycle
- [x] task-profile checksum and routed UI obligations
- [x] NUI V12.1 reference execution capsule
- [x] three materially different visual directions recorded
- [x] select restrained Ember Quiet visual system
- [x] orange semantic living/action accent
- [x] responsive Windows + Android chat shell
- [x] persistent transcript surface
- [x] explicit AI OFF / STARTING / ON / THINKING / ERROR semantics
- [x] backend-truth power control
- [x] deep personalization sheet hidden from primary chat surface
- [x] preferred name/language/length/style/initiative profile
- [x] memory toggle changes real LivingEngine memory retrieval/write policy
- [x] product history API separated from memory policy
- [x] standalone factorized Nolane product cortex adapter
- [x] product inference does not require a Qwen model object
- [x] local Python sidecar API with loopback default
- [x] non-loopback sidecar requires explicit auth token
- [x] Tauri v2 native host
- [x] Windows hidden sidecar process lifecycle
- [x] Windows random loopback runtime port
- [x] 256-bit per-launch loopback API authentication
- [x] WebView never receives the loopback auth token
- [x] Windows resource checks for sidecar/model/tokenizer/ceremony
- [x] Windows offline WebView2 installer mode
- [x] PyInstaller onedir product-runtime specification
- [x] fail-closed release asset staging with exact model SHA-256
- [x] release staging requires COMPLETE L36 promotion ceremony
- [x] product sidecar reverifies L36 ceremony at startup
- [x] release ceremony bundled beside model
- [x] Windows Product Release workflow
- [x] release workflow performs real power-on + chat inference smoke
- [x] release workflow builds NSIS and uploads installer artifact
- [x] Android Tauri target source path
- [x] remote Android endpoint requires HTTPS except loopback
- [x] pairing token remains memory-only in current Android shell
- [x] browser desktop/mobile viewport court source
- [x] rendered desktop/mobile screenshot evidence upload
- [x] reduced-motion court source
- [x] >=44px primary touch-target court source
- [x] Windows native build CI job
- [x] Android APK init/build CI job
- [ ] Product Client Court PASS on branch head
- [x] record two NUI critique/correction cycles with re-observation obligations
- [ ] close both cycles with final-head rendered CI evidence
- [ ] build release Windows sidecar with actual torch/transformers runtime
- [ ] stage an approved real factorized checkpoint + tokenizer into installer
- [ ] produce installable Windows NSIS/MSI release artifact
- [ ] smoke test installed Windows product on a clean VM
- [ ] implement Android local inference OR cryptographically secure pairing transport
- [ ] run Android IME/safe-area court on emulator/device
- [ ] only then call Android chat production-complete

L42 deliberately refuses to equate a responsive mobile shell with mobile AI inference. Windows distribution can close independently once real release assets and a clean-install court pass.

## L43 — Real Longitudinal Learning Execution

Scientific execution target: **5+-WINDOW REAL CORTEX LEARNING / NO SELF-PROMOTION AUTHORITY**.

- [x] require at least five recurrent-cortex learning cycles
- [x] accept only verified L23 approved-evidence packs
- [x] require complete source-group lineage for every real pack
- [x] retention/adaptation source groups disjoint within each cycle
- [x] adaptation protocols unique across cycles
- [x] adaptation source groups fresh across cycles
- [x] fixed long-horizon panel isolated from every train/rehearsal group
- [x] evidence thresholds may be tightened but never loosened
- [x] per-cycle L38 update uses exact previous artifact as parent
- [x] resumable immutable cycle journal
- [x] resumed cycle must reverify native L38 receipt + checkpoint SHA + parent
- [x] initial-vs-final fixed-panel retention court
- [x] learned-window forgetting matrix for every cycle with future updates
- [x] final cycle excluded from self-comparison and protected by immediate L30
- [x] recompute complete L39 chain after all cycles
- [x] privacy-preserving longitudinal report
- [x] training executor cannot call L40 or grant itself promotion authority
- [x] validate-only mode before expensive training
- [ ] Product v0.42 merge becomes the base for L43 PR
- [ ] L43 CI PASS
- [ ] collect at least five real approved adaptation windows
- [ ] freeze one completely isolated real fixed panel
- [ ] execute 5+ real L38 cycles
- [ ] obtain real L43 PASS
- [ ] human review of longitudinal evidence
- [ ] separately issue L40 authorization only if evidence merits promotion
- [ ] complete real L33/L34/L36 production ceremony on that trained checkpoint

L43 is the point where “Nolane learns over time” becomes an empirical claim rather than an architectural capability. A synthetic fixture cannot close these unchecked items.

## L44 — Product Experience Evidence Bridge

Empirical-input target: **REAL PRODUCT USE -> HUMAN REVIEW -> L43 PLAN / NO AUTO-TRAINING AUTHORITY**.

- [x] read product `living.db` through a read-only SQLite snapshot
- [x] export only complete user -> assistant supervised turns
- [x] ignore autonomous assistant messages without a pending user turn
- [x] configurable session-gap source grouping
- [x] bounded pairs per source group to keep grouped held-out splitting possible
- [x] require at least three leakage-safe source groups before review intake
- [x] product language hint may come from explicit vi/en profile only
- [x] raw transcript stays in local export/workbench files
- [x] public export/window receipts exclude raw prompt/target and raw event IDs
- [x] imported review candidates remain approved=false and reviewed=false
- [x] one command initializes L24/L26/L27 workbench from product DB
- [x] reuse explicit immutable human-review decisions
- [x] reuse L25/L27 finalization and L28 quality/leakage court
- [x] incremental SQLite row cursor for later real evidence windows
- [x] explicit operator assignment of fixed/retention/adaptation workbench roles
- [x] require at least five cycles before product->L43 plan handoff
- [x] fixed-panel exact prompt/pair leakage blocked across all training packs
- [x] adaptation exact prompt/pair reuse blocked across longitudinal windows
- [x] within-cycle retention/adaptation exact content overlap blocked
- [x] individual prompt hashes excluded from bridge receipt
- [x] emitted plan is immediately revalidated by native L43 validator
- [x] bridge cannot train, authorize or promote a checkpoint
- [ ] L44 CI PASS
- [ ] collect first real product window from actual Nolane usage
- [ ] complete explicit local human review of that window
- [ ] accumulate at least five distinct real adaptation windows
- [ ] freeze one real fixed panel that never enters training/rehearsal
- [ ] build a real L43 plan from finalized workbenches
- [ ] execute real L43 and measure adaptation + learned-window retention
- [ ] only if L43 PASS: human review -> separate L40 authorization
- [ ] run real L33/L34/L36 ceremony and product release on that trained checkpoint

L44 does not make synthetic evidence more authoritative. Its job is to make genuine product experience usable without weakening consent, privacy or held-out isolation.

