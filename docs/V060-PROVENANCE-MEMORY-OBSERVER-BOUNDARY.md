# v0.60 Provenance Memory + Social Mutation Boundary

## Goal

v0.60 closes the largest remaining semantic gap between the desktop Living
Engine and LocalMobile memory handling without changing the frozen v0.55
persistent neural/product state schema.

The v0.55 state continues to carry only a bounded text projection used by the
product prompt contract. The authoritative mobile memory evidence now lives in
an integrity-checked sidecar with stable IDs, source-event provenance and
consolidation links.

## Provenance memory store

Schema:

`NOLANE-V060-MOBILE-PROVENANCE-MEMORY-V1`

Each record carries:

- memory ID;
- text;
- kind;
- salience;
- confidence;
- source event ID;
- creation timestamp;
- bounded metadata.

REST-derived memories additionally retain parent memory IDs through explicit
`consolidated_into` links. Source memories are not deleted merely because a
derived memory exists.

The store is bounded and integrity checked. Invalid kinds, non-finite
confidence/salience, duplicate IDs, broken provenance links or digest drift
fail closed.

## v0.55 migration

An existing LocalMobile installation may only have the old `Vec<String>`
prompt-memory projection.

On first v0.60 load, those strings are migrated into deterministic legacy
records. Their IDs are stable for the same ordered projection and metadata marks
them as `migrated_from_v055_projection`.

The old persistent state schema is not rewritten into an incompatible format.
Instead, the richer store projects a bounded set of texts back into the frozen
state/prompt contract.

## User-event ordering

Desktop LivingEngine ordering is:

```text
user event
 -> base relationship/affect dynamics
 -> episodic memory commit
 -> validated social observer mutation
 -> cortex generation
```

v0.60 preserves that causal order on LocalMobile. A generation failure cannot
erase a real user interaction that was already accepted into persistent state
and memory evidence.

New episodic memory salience uses the same length-derived desktop baseline.

## Retrieval

Native retrieval ports the desktop score structure:

- 0.52 lexical query overlap;
- 0.30 salience;
- 0.18 recency with a 30-day exponential scale.

Only the bounded text projection is sent to the language cortex. Provenance and
metadata remain local and are not injected into the prompt.

When memory is disabled, stored evidence remains on disk so it can be restored
later, but retrieval is empty and new user/observer memories are not written.

## Provenance-rich deterministic REST

The v0.59 REST scheduler thresholds remain unchanged.

v0.60 upgrades the deterministic near-duplicate baseline from destructive text
compaction into provenance-preserving consolidation:

- source records remain intact;
- already-consolidated parents are not repeatedly reconsidered;
- clusters require at least two sources;
- at most eight sources feed one derived memory;
- confidence cannot exceed the weakest source;
- fact status cannot be created from non-fact sources;
- each derived memory records its source IDs;
- explicit parent -> child links are persisted.

This is now provenance-rich deterministic consolidation. It still does not
claim parity with a model-generated REST proposal that resolves threads or
changes active intent.

## Social mutation boundary

v0.60 adds the mobile equivalent of the desktop proposal/validator boundary:

- affect deltas are allow-listed and clamped;
- relationship deltas are allow-listed and clamped;
- memories per event are bounded;
- memory text length is bounded;
- confidence/salience are normalized;
- low-confidence fact proposals are downgraded to inference;
- observer uncertainty and provenance are persisted;
- unknown fields do not gain mutation authority.

Production proposal generation remains deliberately conservative. The built-in
deterministic observer recognizes only explicit preference statements such as
`Tôi thích ...`, `I like ...` or `I prefer ...`. It may propose a
preference memory, but it cannot turn an ambiguous inference into a fact.

This proves the mutation boundary while avoiding the false claim that a
full model-driven social observer has already been ported to mobile.

## Court

The v0.60 native court proves:

1. legacy v0.55 memory projection migration;
2. new episodic memories are source-event bound;
3. explicit preferences pass through the observer validator;
4. retrieval/status exposes the richer store only through bounded projections;
5. deterministic REST creates derived memories plus parent links;
6. source memories survive consolidation;
7. memory-disabled chat does not create new memory records;
8. provenance store survives product-host restart byte-semantically;
9. all previous lifecycle, prompt, tokenizer, sampling and Android courts remain green.

## Explicit non-claims

v0.60 does not by itself prove:

- model-generated mobile social-observer proposal quality;
- rich open-thread ID/importance provenance on mobile;
- desktop reviewed-learning UI/training parity on Android;
- a real L43 longitudinal-learning PASS;
- a real COMPLETE-L36 Android release campaign;
- physical-device acceptance;
- final latency/RAM/thermal/battery acceptance.

Those remain real v1 closure evidence, not boxes to mark synthetically.
