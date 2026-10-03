# v0.47 Product Reliability Freeze

## Goal

v0.47 is a feature-freeze reliability wave.

It does not add a new primary screen, dashboard, model picker, sidebar or
background-learning switch.

The supported user surface remains deliberately small:

- chat;
- AI power;
- personalization;
- explicit one-by-one reviewed learning/correction.

The goal is to make those flows resilient to retries and narrow crash windows.

## Exact decision retries

A review decision is immutable once frozen.

That rule remains unchanged.

However, an exact retry can happen legitimately:

1. the app sends Approve;
2. the decision file is durably written;
3. the HTTP response is lost;
4. the UI retries the same request.

v0.47 treats the second request as an idempotent replay only when every frozen
decision field is identical.

The existing decision is returned and no second row is written.

A retry with a different decision, language, weight or corrected target remains
blocked.

## Pending-window deduplication

While the newest learning window is not finalized, another create-window call
returns that same window.

This covers double-clicks and lost-response retries without consuming a second
slice of transcript.

After the window reaches INTAKE_READY, a later create request may advance to
new product events.

## Crash recovery after atomic window install

Window preparation already builds into a temporary directory and atomically
moves the completed workspace into its numbered path.

One narrow crash remained:

```text
temporary build
  -> verified window
  -> atomic move to window-000N
  -> CRASH
  -> registry write never happens
```

On restart, v0.47 may adopt that orphan only when all of the following hold:

- it is exactly the next numbered window;
- its own L44/L27 manifests verify;
- its source range begins at the current registry high-water mark;
- its high-water mark advances;
- the pre-existing registry cursor and next-index are already internally
  consistent.

A gap, non-contiguous orphan or inconsistent registry remains fail-closed.

This is forward recovery, not heuristic repair.

## No silent metadata repair

The registry is self-digested, but a self-consistent bad cursor should still be
treated as corruption.

v0.47 therefore verifies the registered high-water mark and next-window index
before orphan recovery.

Recovery cannot be used to hide unrelated metadata damage.

## Threaded sidecar SQLite safety

The product sidecar uses a threaded HTTP server. Core LivingStore instances
remain same-thread by default, but ProductRuntime explicitly opts its one
store connection into cross-thread use because every engine/store operation is
serialized by the ProductRuntime re-entrant lock.

This closes a production-only class of failure where power/status requests
could work while the first real `/v1/chat` request failed when SQLite was
touched from an HTTP worker thread.

The product runtime court now performs authenticated:

```text
power on -> HTTP chat -> HTTP history
```

through the actual ThreadingHTTPServer.

## v1 product rule

New permanent UI features are frozen unless they are required to complete a
core user job that cannot be expressed safely through the current surfaces.

The priority order is now:

1. correctness;
2. recovery;
3. privacy/consent;
4. real longitudinal evidence;
5. distribution/installation quality;
6. only then additional product surface.
