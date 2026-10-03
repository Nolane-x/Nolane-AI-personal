# L45 In-App Explicit Evidence Review

## Purpose

L44 makes real product conversations eligible for the existing local evidence
pipeline, but the first interface is a CLI.

L45 moves the same explicit human-review boundary into the product without
putting research controls on the primary chat surface.

The normal chat remains:

```text
identity + power
conversation
composer
```

Evidence review exists only under Personalization -> Advanced ->
**Learn from conversations**.

## Non-negotiable consent boundary

Opening, exporting or preparing a learning window does not approve anything.

Every candidate begins:

```text
reviewed=false
approved=false
```

The product deliberately has no:

- approve-all action;
- silent auto-approval;
- model-based approval;
- background training trigger;
- cloud upload;
- promotion action.

For every candidate the user must choose exactly one explicit action:

- Approve
- Reject
- Sensitive

A sensitive decision is rejected from the approved count.

## Local evidence registry

The product stores a self-digested registry under:

```text
<product-data>/learning-evidence/learning-registry.json
```

The registry contains hashes, window IDs, progress lineage and SQLite row
cursors. It does not contain raw prompt/target text or candidate IDs.

Each window is exported into its own local workspace.

New windows form a contiguous event-row chain:

```text
window-0001  (row 0, row N]
window-0002  (row N, row M]
window-0003  (row M, row K]
```

A failed window preparation does not advance the cursor.

Therefore an old product turn cannot silently re-enter a later longitudinal
window.

## Atomic preparation

A new evidence window is prepared in a temporary directory.

Only after L44 export + review-workbench initialization succeed is that
directory atomically moved into the numbered window path and the registry
cursor advanced.

If preparation fails, the temporary directory is removed and the registry is
unchanged.

## Authenticated product API

The existing product sidecar authentication also covers the review API.

```text
GET  /v1/learning/windows
GET  /v1/learning/pending?window_id=...
POST /v1/learning/windows
POST /v1/learning/decision
POST /v1/learning/finalize
```

Raw prompt/target text is returned only by the pending-candidate endpoint to the
authenticated product UI so the user can review it.

The registry/list endpoints remain metadata-only.

## UI behavior

The review dialog displays one pair at a time:

```text
YOU
<real user message>

NOLANE
<real assistant response>

[ Sensitive ] [ Reject ] [ Approve ]
```

There is no batch approval.

When no candidates remain, the decision buttons disappear and the user may
choose **Package reviewed data**.

Finalization still goes through L25/L27/L28. It can fail if too few approved
examples remain or if the evidence-quality/leakage court blocks the pack.

## Remote/older runtimes

The learning surface is fail-soft.

If a paired Android client talks to an older runtime without L45 endpoints,
normal chat remains usable. The hidden learning section reports that the
feature is unavailable instead of turning the whole product into an error
state.

## Authority

L45 authority is:

`LOCAL_EXPLICIT_REVIEW_ONLY_NO_AUTO_TRAINING_AUTHORITY`.

L45 can prepare and approve evidence only through explicit user decisions.

It cannot execute L43, call L40, swap checkpoints or promote a model.

The empirical path remains:

```text
normal product use
 -> L45 explicit in-app review
 -> L44 approved reviewed windows
 -> L43 5+ real longitudinal learning
 -> human evidence review
 -> L40 explicit authorization
 -> L33/L34/L36 production ceremony
```

## Production courts

L45 closure requires:

- registry digest and contiguous-window verification;
- failed window must not advance event cursor;
- explicit approve/reject/sensitive behavior;
- sensitive decisions excluded from approved count;
- authenticated learning API court;
- real product history -> evidence window -> L28 finalized pack;
- rendered browser court proving one-at-a-time review with three explicit
  decisions and no approve-all action;
- existing Windows and Android product courts remain green.
