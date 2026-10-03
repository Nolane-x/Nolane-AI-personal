# L44 Product Experience Evidence Bridge

## Purpose

L43 can run a real 5+ window longitudinal recurrent-cortex experiment, but it
deliberately does not invent evidence.

L44 closes the practical gap between normal product use and that empirical
pipeline:

```text
Nolane product chat
  -> local living.db
  -> L44 transcript export
  -> L24 review queue
  -> L26 explicit human review
  -> L25/L27 approved evidence pack
  -> L44 longitudinal plan bridge
  -> L43 5+ cycle real execution
```

The bridge is local-only by design. Exporting a chat never means approval.

## Safety and authority boundary

L44 authority is:

`PRODUCT_REVIEWED_EVIDENCE_TO_L43_PLAN_NO_TRAINING_AUTHORITY`.

It cannot:

- approve a conversation automatically;
- mark sensitive content safe;
- train a model;
- invoke L40;
- promote a checkpoint;
- upload product conversations to a server.

The raw transcript exists only in the local export/workbench files. Public
receipts contain hashes, counts and lineage, not raw prompts or targets.

## Prepare one real product window

The product database is opened read-only.

```bash
python scripts/product_evidence.py prepare-window \
  --db "/path/to/Nolane/living.db" \
  --profile "/path/to/Nolane/personalization.json" \
  --workspace "runtime-data/real-window-001"
```

The exporter reads only complete user-message -> assistant-speech pairs.

Autonomous assistant messages without a pending user turn are not converted
into supervised evidence.

By default:

- a gap above 45 minutes starts a new source group;
- a source group is capped at 4 pairs;
- at least 3 source groups are required;
- every imported candidate starts with `approved=false` and
  `reviewed=false`.

The group cap is intentional. One very long conversation must not become one
source group that makes leakage-safe train/dev/test splitting impossible.

It is also intentionally not one group per turn. Nearby turns may share context,
so splitting every turn independently could leak conversational context across
held-out partitions.

## Incremental windows

A prepared window receipt records its SQLite event-row high-water mark.

A later window can start after that row:

```bash
python scripts/product_evidence.py prepare-window \
  --db "/path/to/Nolane/living.db" \
  --workspace "runtime-data/real-window-002" \
  --after-rowid 1842
```

The row number is only a local cursor. Event IDs and raw text are not copied
into the public window manifest.

## Human review

Review happens with the existing L26 local reviewer:

```bash
python scripts/product_evidence.py review \
  --workspace runtime-data/real-window-001
```

Each candidate is shown as USER / ASSISTANT and requires one explicit decision:

- approve;
- reject;
- reject as sensitive;
- skip;
- quit.

Existing decisions remain immutable through the reviewer. Reconsideration
requires a new decision file/workbench rather than silently rewriting old
approval history.

## Finalize a window

```bash
python scripts/product_evidence.py finalize \
  --workspace runtime-data/real-window-001
```

Finalization still enforces all L23-L28 constraints:

- >=7 approved non-sensitive examples;
- vi/en language on every approved row;
- grouped train/dev/test protocol;
- at least three distinct source groups;
- held-out test reserve;
- exact/near-duplicate leakage court;
- full source-group lineage.

A window that fails L28 never becomes an approved pack.

## Building a real L43 plan

L44 does not guess which evidence is fixed-panel, retention or adaptation.

The operator explicitly assigns those roles in one spec.

Example:

```json
{
  "schema": "NOLANE-L44-PRODUCT-LONGITUDINAL-SPEC-V1",
  "initial_factorized": "/private/checkpoints/factorized-nolane.pt",
  "latent": "/private/runtime/latent.json",
  "tokenizer": "/private/models/Qwen3-0.6B",
  "device": "cuda",
  "fixed_workbench": "/private/evidence/fixed/workbench",
  "cycles": [
    {
      "retention_workbench": "/private/evidence/c1-old/workbench",
      "adaptation_workbench": "/private/evidence/c1-new/workbench"
    },
    {
      "retention_workbench": "/private/evidence/c2-old/workbench",
      "adaptation_workbench": "/private/evidence/c2-new/workbench"
    },
    {
      "retention_workbench": "/private/evidence/c3-old/workbench",
      "adaptation_workbench": "/private/evidence/c3-new/workbench"
    },
    {
      "retention_workbench": "/private/evidence/c4-old/workbench",
      "adaptation_workbench": "/private/evidence/c4-new/workbench"
    },
    {
      "retention_workbench": "/private/evidence/c5-old/workbench",
      "adaptation_workbench": "/private/evidence/c5-new/workbench"
    }
  ]
}
```

Build and immediately revalidate the exact L43 plan:

```bash
python scripts/product_evidence.py build-plan \
  --spec /private/evidence/l44-spec.json \
  --output /private/evidence/l43-plan.json
```

## Cross-window leakage court

Source-group hashes alone are not enough.

The same prompt/target could be copied into another window under a different
conversation ID. L43 would correctly see different source-group identities, but
the fixed panel would no longer be genuinely held out.

L44 therefore adds content-level checks before handing a plan to L43.

It blocks:

1. exact normalized prompt or prompt/target overlap between the fixed panel and
   any retention/adaptation training pack;
2. exact normalized prompt or pair reuse across adaptation windows;
3. exact normalized prompt or pair overlap between retention and adaptation
   inside one cycle.

The bridge receipt does not retain individual prompt hashes. It records only
aggregate set digests, counts and overlap-free lineage.

L28 remains responsible for exact and near-duplicate leakage *inside* each
approved pack.

## Then run the real experiment

After L44 produces the validated plan:

```bash
python scripts/run_real_longitudinal_learning.py \
  --plan /private/evidence/l43-plan.json \
  --validate-only

python scripts/run_real_longitudinal_learning.py \
  --plan /private/evidence/l43-plan.json \
  --output-dir runtime-data/l43-real-longitudinal \
  --resume
```

A PASS still does not auto-promote the model.

The next authority step remains separate human review followed by L40.

## What L44 does not prove

L44 makes real evidence collection practical and leakage-aware.

It does not prove that Nolane learns well.

That claim still requires:

- at least five genuinely new reviewed adaptation windows;
- one isolated fixed panel;
- real L38 updates;
- L43 PASS;
- human review;
- explicit L40 authorization;
- real L33/L34/L36 ceremony.

The key difference is that the missing input can now come directly from normal
product usage without weakening consent or evidence boundaries.
