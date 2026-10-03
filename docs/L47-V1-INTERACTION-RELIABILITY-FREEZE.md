# L47 v1 Interaction Reliability Freeze

## Goal

L47 does not add a product feature.

It freezes the v1 interaction surface and hardens the existing learning-review
flow against the failures users actually cause accidentally:

- retry after a lost response;
- double click;
- app restart during review;
- repeated finalize;
- stale/conflicting retry.

## Frozen v1 surface

The primary product surface is intentionally limited to:

1. chat;
2. AI power;
3. personalization;
4. explicit learning review/correction under Advanced.

L46 remains backend-only.

No sidebar, model picker, dashboard, plugin marketplace, prompt library or
additional permanent toolbar is part of the v1 scope.

A new UI surface requires evidence that a core user job cannot be completed
safely through the existing surface.

## Window creation retry

Only one unfinished learning window may exist.

If a create-window request is retried after the server already committed the
window, the runtime returns the same pending window.

It does not:

- advance the SQLite high-water cursor again;
- create another review queue;
- allocate another window ID.

## Decision retry

Review decisions remain immutable.

An exact retry of the already frozen decision is idempotent and returns current
window/progress state. This covers the case where the server committed the
decision but the client did not receive the response.

A retry that changes any frozen field is blocked:

- approve -> reject;
- corrected target A -> target B;
- sensitive flag change;
- language/weight change.

The existing decision is never silently overwritten.

## Restart continuity

A partially reviewed window must survive a full ProductRuntime shutdown and
restart.

After reopening:

- decided count is unchanged;
- the next candidate is exactly the first undecided candidate;
- an exact retry of the pre-restart decision remains idempotent.

## Finalize retry

Finalization is idempotent.

Once a window is `INTAKE_READY`, repeating finalize returns the same verified
status instead of rebuilding or replacing the evidence pack.

## Authority

L47 changes reliability only.

It does not:

- auto-approve evidence;
- auto-train;
- run L43;
- grant L40 authorization;
- promote a checkpoint.

## v1 closure principle

From L47 onward, feature count is not a progress metric.

Progress toward v1 means:

- fewer ambiguous states;
- clean install/start/restart;
- real reviewed evidence;
- real 5+ cycle longitudinal result;
- stable Windows/Android client courts;
- real promoted checkpoint only if empirical evidence passes.
