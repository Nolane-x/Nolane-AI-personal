# L4 REST / Consolidation

## Purpose

Nolane AI Personal should not only react while the user is speaking. During sufficiently long idle periods, it can enter a bounded REST cycle and reorganize existing evidence into more durable memory.

REST is not hidden retraining. Live and REST cycles do **not** update Qwen or Tiny Living Core weights.

## Default lifecycle

A REST cycle becomes eligible when:

- the runtime has real user history;
- the user has been idle for at least 30 minutes;
- the previous REST cycle was at least 45 minutes ago.

The low-cost default observer is deterministic and only merges near-duplicate memories.

```bash
nolane-personal run --no-model
```

For deeper semantic consolidation, the already-loaded Qwen instance can be reused:

```bash
nolane-personal run --deep-rest
```

No second model copy is loaded.

## Authority boundary

REST observers have proposal authority only.

A deterministic `ConsolidationValidator` owns persistence and enforces:

- every derived memory cites at least two existing source memories;
- source IDs must exist in the current candidate set;
- confidence cannot exceed the weakest cited source;
- a proposed fact is downgraded to inference unless all cited sources are already high-confidence facts;
- thread resolution requires cited memory evidence;
- at most four new consolidated memories per cycle;
- at most eight source memories per derived memory;
- at most two thread resolutions per cycle;
- relationship/affect/personality state cannot be edited by REST proposals.

## Append-only provenance graph

Original memories are never deleted when consolidated.

Instead, the runtime writes:

```text
source memory A ─┐
                 ├── consolidated_into ──> durable memory C
source memory B ─┘
```

This graph is stored in `memory_links`.

It preserves the ability to answer:

- which observations produced this preference/habit?
- was a durable memory derived from one event or repeated evidence?
- can a later court audit the source lineage?
- can a future correction supersede a derived belief without erasing history?

## State schema v2

The persistent state now includes:

- REST cycle count;
- last cycle timestamp;
- number of source memories considered;
- number of memories produced by the last cycle.

Legacy schema-v1 state loads are upgraded to schema v2 in memory and become v2 on the next committed transition.

## Manual REST

```text
/rest
```

This is useful for development and court replay. Automatic REST can be disabled with `--no-rest`.

## Audit

```bash
python scripts/audit_rest_state.py --db runtime-data/living.db
```

The audit fails if a provenance edge references a missing parent or child memory.

## Scientific boundary

REST may summarize, derive hypotheses, and maintain evidence structure. It does not prove that a preference, habit, or inferred emotional state is objectively true. Uncertainty and provenance remain first-class data.
