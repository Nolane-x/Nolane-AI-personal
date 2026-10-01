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
