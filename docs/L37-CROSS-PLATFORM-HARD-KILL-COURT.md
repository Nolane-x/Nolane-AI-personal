# L37 Cross-Platform Hard-Kill Court

## Purpose

Earlier L33 crash tests simulated interruption by raising exceptions inside the
same Python process.

That proves state-machine logic, but it does not prove what survives when the
process disappears without running Python cleanup.

L37 uses real child processes and `os._exit()`.

The child dies immediately:

- context-manager `finally` blocks do not run;
- the registry lock remains stale;
- only bytes already written to disk survive;
- the parent process must recover from disk state alone.

The hard-kill court runs on both Linux and Windows.

## Crash phases

L33 commit now exposes an optional test/fault hook at durable boundaries:

```text
ARTIFACT_INSTALLED
POINTER_SNAPSHOT_WRITTEN
ACTIVE_POINTER_SWAPPED
POINTER_EVENT_RECORDED
COMMIT_EVENT_RECORDED
```

Normal production callers do not provide a hook.

The hook exists so a separate test process can terminate at an exact durability
boundary without changing the normal commit algorithm.

## Newly discovered crash window

The strongest new finding was the interval:

```text
pointer generation N+1 snapshot written
        |
        |  process dies here
        v
active.json still generation N
```

Before L37, recovery correctly recognized that active authority never moved and
marked the transaction aborted, but the N+1 pointer snapshot could remain in
pointer history.

That produced an inconsistent registry:

```text
pointer history latest = N+1
active.json             = N
```

and the registry audit failed.

## Orphan pointer recovery

L37 makes this case explicit.

If active authority still equals the transaction parent, recovery examines the
next pointer generation.

It removes the snapshot only when all of these are true:

- generation is exactly parent generation + 1;
- transaction ID equals the interrupted transaction;
- checkpoint SHA equals the staged candidate;
- previous pointer SHA equals the still-active parent.

Anything else is a recovery conflict and is not deleted automatically.

After removing the proven orphan snapshot, recovery records:

```text
RECOVERED_ABORTED
orphan_pointer_snapshot_removed=true
```

and pointer history is consistent again.

## Crash after active pointer swap

For:

```text
pointer snapshot N+1 written
active.json -> N+1
process dies before audit events finish
```

recovery observes that durable serving authority already moved to the candidate.

It records:

```text
RECOVERED_COMMITTED
```

rather than rolling back a real atomic swap.

## Crash after COMMITTED

A process may also die after the COMMITTED event was durably appended but
before the registry lock is cleaned up.

In that case:

- transaction history is already terminal;
- active pointer is the candidate;
- only the lock is stale.

Explicit stale-lock recovery removes the abandoned lock, does not manufacture a
second terminal event, and registry audit remains PASS.

## Authorized path

The hard-kill tests do not bypass L35.

Each checkpoint crash court creates an L35-compatible authorized transaction,
verifies it, and only then launches the child process that calls L33 commit.

This proves that the same transaction path used by evidence-bound promotion is
the path being killed.

## Serving reload hard kill

L37 also kills a serving reload process at two persistent boundaries.

### Before lease ACK

The child:

1. reads the reload plan;
2. resolves and verifies the target registry artifact;
3. dies before writing the new lease.

The old lease remains on disk.

Therefore the restarted coordinator still sees the process as old-generation
and its serving gate is:

```text
DRAIN_RELOAD_REQUIRED
```

A restarted worker must load the active checkpoint and replace its lease before
it can serve.

### After lease ACK

The child verifies the target and atomically ACKs the new loaded generation,
then dies immediately.

The new-generation lease survives.

The process itself is gone, so the lease will eventually expire, but no stale
old-generation lease is resurrected.

## Crash probes

L37 provides two intentionally narrow test executables:

```text
scripts/crash_probe_checkpoint_commit.py
scripts/crash_probe_serving_reload.py
```

They are test tools.

The checkpoint probe exits with a configured non-zero code at one commit
durability phase.

The reload probe exits either:

- after target verification but before ACK; or
- immediately after lease ACK.

## Cross-platform CI

The dedicated workflow:

```text
.github/workflows/platform-crash-court.yml
```

runs the hard-kill tests on:

- `ubuntu-latest`;
- `windows-latest`.

The tests use the platform's actual filesystem, subprocess implementation and
process termination semantics.

No PyTorch installation is required for this court because it tests authority
durability rather than neural numerics.

## Courts

L37 proves mechanically:

- a real process death leaves the registry lock stale;
- explicit recovery is required to break that lock;
- crash after pointer snapshot but before active swap aborts safely;
- the proven orphan pointer snapshot is removed;
- crash after active swap recovers committed authority;
- crash after COMMITTED does not duplicate terminal state;
- registry audit passes after each recovered state;
- reload death before ACK preserves the old lease;
- old lease cannot serve after active authority moved;
- restart on the active pointer restores convergence;
- reload death after ACK preserves the new-generation lease;
- the same tests run on Linux and Windows.

## Boundary

These are real process kills but the CI model artifacts are synthetic
factorized bundles.

L37 therefore proves filesystem/process recovery mechanics across two target
operating systems.

It still does not prove:

- hard-kill recovery while loading a full real trained checkpoint;
- GPU/accelerator teardown behavior;
- very large artifact copy durability;
- power-loss guarantees below the operating-system/fsync boundary;
- real production service orchestration.

Those remain real deployment courts.
