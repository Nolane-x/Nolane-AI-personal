# L34 Serving Reload Convergence

## Purpose

L33 can switch checkpoint authority atomically.

L34 makes serving processes converge on that authority without allowing two live
checkpoint generations to serve traffic at the same time.

## Safety rule

Every request must pass a local serving gate.

A process may serve only when all of the following are true:

- its lease is alive;
- its loaded pointer equals the current active pointer;
- the convergence court is PASS;
- the active pointer did not change while convergence was being assessed.

Otherwise the process returns one of:

- `DRAIN_RELOAD_REQUIRED`;
- `WAITING_FOR_PEERS`;
- `LEASE_EXPIRED`.

## Reload barrier

When generation N+1 becomes active:

1. all generation-N processes immediately fail their request gate;
2. each process loads N+1 into memory;
3. it ACKs only after the loaded checkpoint SHA matches the target;
4. early reloaders remain WAITING_FOR_PEERS;
5. only after every live lease points to N+1 does the court PASS;
6. then request gates return SERVE.

This intentionally prefers temporary unavailability over mixed-generation
answers.

## Leases

Each process owns one atomically replaced, self-digested lease containing:

- hashed process identity;
- loaded generation;
- loaded pointer SHA;
- loaded checkpoint SHA;
- heartbeat time;
- lease lifetime.

Raw process identifiers are not emitted in convergence receipts.

Expired processes do not count as live. A minimum-live-process policy can still
require more than one survivor before serving resumes.

## Reload race

Reload plans are bound to one process and one target pointer.

If the active pointer changes while a model is loading, ACK fails. The process
must reload the newer active checkpoint instead.

## Convergence race

The coordinator reads the active pointer both before and after evaluating live
leases.

If those reads differ, the court adds:

```text
active_pointer_changed_during_convergence
```

and blocks.

This prevents a stale convergence PASS from authorizing traffic after a newer
transaction landed.

## Process-local session

`ServingSession` owns the loaded model reference and a process-local lock.

`poll_reload()`:

- creates/validates a reload plan;
- loads the target artifact;
- verifies checkpoint SHA;
- swaps the local model reference;
- ACKs the new lease.

`model_for_request()` returns the model only when the gate status is SERVE.
All other states raise and must drain/block traffic.

A real factorized loader is provided through `factorized_loader()`.

## CLI

Operational inspection is available through:

```bash
python scripts/manage_serving_coordination.py \
  --registry runtime-data/continual-checkpoint-registry \
  convergence

python scripts/manage_serving_coordination.py \
  --registry runtime-data/continual-checkpoint-registry \
  gate --process-id worker-1
```

The CLI also supports register-active, heartbeat, reload-plan creation and ACK.

## Courts

CI proves:

- two workers both serve generation 0 before update;
- after active changes, both old workers drain;
- after only one worker reloads, split-brain is detected and nobody may serve;
- after all live workers reload, both may serve;
- stale reload ACK is rejected if active changes again;
- expired old workers do not block a permitted surviving quorum;
- minimum-live-process policy can keep traffic blocked;
- wrong-checkpoint loaders never register;
- lease tampering is detected;
- reload plans cannot be reused by another process;
- active-pointer changes during convergence fail closed;
- raw process IDs stay out of convergence receipts.

## Authority boundary

L34 coordinates serving only.

It does not decide whether a candidate deserves promotion. Promotion still
depends on the L30-L32 evidence path plus the L33 transaction.

The next authority layer must bind those evidence receipts into an explicit
promotion decision instead of allowing any valid L31 candidate to reach the
transactional registry.

## v0.35.1 request-admission hardening

The original L34 gate blocked mixed live generations, but one smaller race
remained possible: `active.json` could change immediately after a gate returned
SERVE and immediately before the caller took the model reference.

v0.35.1 adds a second **request fence** at admission time.

The fence re-verifies:

- the gate self-digest;
- the process binding;
- SERVE status;
- the current active pointer;
- the process lease still matching that pointer.

If authority moved after gate evaluation, admission fails closed.

Production inference should use:

```python
with session.request_model() as model:
    # inference
    ...
```

The session lock remains held for the full request. This gives explicit drain
semantics:

- a request admitted before a pointer transition may finish;
- `poll_reload()` cannot replace its local model while that request is active;
- once it exits, every later request must pass a fresh gate and request fence;
- if authority moved, later requests drain/block until reload and convergence.

This does not claim that an already-running inference can be retroactively
moved to a new checkpoint. The guarantee is that no **new** request is admitted
using stale authority after the transition is observed.

## L36 evidence timestamp

L36 requires every convergence receipt to carry a self-digested `assessed_at`
timestamp. The public `verify_serving_convergence_receipt()` verifier checks
schema, digest, authority, policy, pointer/checkpoint SHA values, timestamp and
PASS/BLOCKED status.

This lets the final promotion ceremony prove that serving convergence was
measured only after the committed checkpoint pointer existed.

