# L41 Unified Promotion Integration

## Purpose

L40 can authorize a mixed L31/L38 continual-learning chain, but deliberately
cannot change production authority.

L41 connects that authorization to the existing transactional and serving
release stack without forking the reliability implementation.

The path becomes:

```text
L39 mixed continual chain
        |
        v
L40 explicit short-lived authorization
        |
        v
L33 atomic checkpoint transaction
        |
        v
L34 serving convergence + request fences
        |
        v
L36 immutable COMPLETE ceremony
```

## Authorization dispatch

L41 adds one dispatcher for supported authorization schemas:

- L35 boundary-chain authorization;
- L40 unified model-chain authorization.

Existing L35 receipts remain valid.

The registry and ceremony do not silently reinterpret one schema as the other.

## Candidate dispatch

An authorized candidate bundle may contain exactly one supported continual run
receipt:

- `l31-run-receipt.json`;
- `l38-run-receipt.json`.

A bundle containing both is rejected as ambiguous.

The native L31 or L38 verifier is applied before staging.

For L38, this includes the recurrent-cortex ownership invariants and embedded
L30 court.

## L40 model-state binding

For L40 transactions, checkpoint SHA alone is insufficient.

Before staging, the registry computes the active production model state from:

```text
SHA256(boundary_state_digest, cortex_state_digest)
```

and requires it to equal the authorization's
`initial_model_state_sha256`.

The final candidate artifact is checked the same way against
`final_model_state_sha256`.

This closes a gap where the right file identity could otherwise be paired with
the wrong neural-state evidence.

## Reverification through transaction phases

The authorization and candidate bundle are reverified at:

- transaction begin;
- transaction verify;
- transaction commit.

The staged run schema and run-receipt SHA are bound into the PREPARED event.

Changing the L38 receipt after begin causes verification to fail before the
pointer can move.

Authorization expiry continues to be checked at verify and commit.

## Backward compatibility

Direct non-authorized L31 transactions retain their old path.

L35 authorized L31 release remains supported.

L40 uses the generalized candidate path.

The hard-kill/recovery code itself is shared rather than duplicated.

## Ceremony

L36 now obtains authorization through the dispatcher.

A ceremony records:

- authorization SHA;
- authorization schema;
- authorization kind;
- evidence-chain SHA;
- long-horizon court SHA;
- committed checkpoint and pointer;
- serving convergence receipt.

For backward compatibility, the historical
`multicycle_chain_sha256` field remains the ceremony's generic evidence-chain
slot. For L35 it contains the L32 chain SHA. For L40 it contains the L39 chain
SHA.

Registry-backed ceremony verification uses the authorization dispatcher to
interpret that field correctly.

## End-to-end court

L41 creates a synthetic recurrent-cortex release:

```text
production parent
  -> offline L38 cycle 1
  -> offline L38 cycle 2
  -> L39 PASS
  -> L40 AUTHORIZED
  -> L33 commit final L38 artifact
  -> L34 convergence PASS
  -> L36 COMPLETE
```

The final registry audit must report both one authorization and one ceremony.

## Cross-platform hard-kill court

L41 extends the L37 platform workflow on:

- Ubuntu;
- Windows.

It runs real child-process termination for an L40-authorized L38 transaction.

Two durable phases are tested:

### Pointer snapshot written, active pointer not swapped

The child exits with `os._exit()`.

Recovery must:

- detect the stale lock;
- remove the expected orphan pointer snapshot;
- record RECOVERED_ABORTED;
- keep the production parent active.

### Active pointer swapped

The child exits after `active.json` changed.

Recovery must:

- preserve the L38 candidate as active;
- record RECOVERED_COMMITTED;
- keep registry audit PASS.

## Fail-closed courts

L41 also proves:

- L40 final model-state mismatch blocks at begin;
- staged L38 receipt tampering blocks before commit;
- L40 ceremony binds the L39 chain SHA;
- registry audit verifies L40 authorization;
- L35/L31 regression tests remain green.

## Boundary

L41 closes the **mechanical production path** for L38 evidence.

It still does not prove that a real recurrent-cortex candidate deserves
production.

The remaining evidence gap is empirical:

- real approved multi-window data;
- real 5+ cycle L39 chain;
- real fixed-panel long-horizon PASS;
- real L40 approval;
- real checkpoint serving processes;
- hard-kill/resource courts on the actual trained artifact.

Only that real-data ceremony can close the scientific production claim.
