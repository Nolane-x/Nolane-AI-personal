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
