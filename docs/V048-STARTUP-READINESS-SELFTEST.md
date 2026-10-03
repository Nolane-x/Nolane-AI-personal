# v0.48 Startup Readiness Self-Test

## Goal

v0.48 adds no new primary product surface.

It makes **AI ON** a truthful state.

A listening sidecar process is not enough. Before the product is considered
usable, the runtime must prove that critical local storage/release assets are
healthy and, on power-on, that the neural runtime can actually generate output.

## Two readiness stages

### Stage 1 — before sidecar listen

The product runtime runs a critical preflight before binding its HTTP port.

Critical checks:

1. product data directory accepts a durable local write probe;
2. SQLite `PRAGMA quick_check` returns `ok`;
3. a rollback write probe succeeds without leaving a metadata row;
4. production release assets are complete:
   - factorized checkpoint;
   - tokenizer directory;
   - `tokenizer_config.json`;
   - `tokenizer.json`;
   - COMPLETE promotion ceremony;
   - ceremony checkpoint SHA matches the model.

If any critical check fails, the sidecar exits before listening.

That means the native host cannot mistake "TCP port opened" for "runtime ready".

### Stage 2 — before phase becomes ON

When the user presses the existing AI power control, ProductRuntime:

1. re-runs the critical preflight;
2. constructs the cortex;
3. requires the production cortex to run a one-token local generation smoke;
4. only then changes phase to `on`.

A cortex that loads but cannot produce a token leaves the product in
`error`, never a false ON state.

The smoke is intentionally tiny. It is a readiness probe, not a benchmark and
not a second visible chat message.

## Advisory isolation

Reviewed-learning metadata is useful but not required for basic chat.

The learning registry is therefore an advisory check.

If that registry is corrupted:

- readiness still reports core PASS when model/storage are healthy;
- chat/power remain available;
- the learning feature remains independently diagnosable.

This prevents an optional advanced feature from taking down the primary product
surface.

## Authenticated readiness endpoint

The sidecar exposes:

```text
GET /v1/readiness
```

behind the same authentication token as other private runtime endpoints.

The normal unauthenticated health endpoint remains deliberately minimal.

Detailed readiness information is not exposed to unauthenticated local
processes.

## SQLite write probe

The database readiness check uses a savepoint:

```text
SAVEPOINT
  -> temporary metadata write
  -> ROLLBACK TO
  -> RELEASE
  -> verify temporary row is absent
```

It tests actual write capability without permanently changing user state.

## Product-state reporting

`/v1/status` carries only a compact readiness summary:

- PASS/BLOCKED;
- critical failure count;
- advisory failure count.

Raw diagnostic details remain in the authenticated readiness response.

## v1 principle

Readiness is automatic.

No new dashboard, startup wizard or permanent status panel is added.

The existing power control remains the user's single operational action:
if Nolane says ON, the local model has already passed its smoke test.
