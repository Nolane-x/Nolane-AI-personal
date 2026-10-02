# L36 Final Promotion Ceremony

## Purpose

L30-L32 prove learning evidence.

L33 makes checkpoint switching transactional.

L34 makes serving processes converge on one active checkpoint.

L35 decides whether an offline multicycle candidate is explicitly authorized to
request production checkpoint authority.

L36 closes the chain.

A promotion is not called complete merely because `active.json` changed.

It is complete only when one immutable receipt proves:

```text
L32 evidence
    |
    v
L35 authorization
    |
    v
L33 atomic committed UPDATE pointer
    |
    v
L34 serving convergence PASS
    |
    v
L36 immutable final ceremony receipt
```

## What L36 verifies

The ceremony re-verifies the L35 authorization stored inside the L33
transaction and requires it to remain self-digest valid and AUTHORIZED.

It then requires exactly one terminal committed transaction event:

- `COMMITTED`, or
- `RECOVERED_COMMITTED`.

The committed event must identify an immutable registry pointer whose:

- transaction ID equals the promotion transaction;
- transition is `UPDATE`;
- checkpoint SHA equals the L35 authorized candidate;
- generation equals the L34 convergence generation.

The L34 convergence receipt is independently self-digest verified.

## Timeline court

L36 treats event order as evidence.

The following order must hold:

```text
authorization issued
        <=
transaction prepared
        <=
pointer created
        <=
transaction committed
        <=
serving convergence assessed
        <=
ceremony finalized
```

The pointer must also have been created before authorization expiry.

Violations do not create a COMPLETE ceremony.

Examples include:

- transaction predates authorization;
- pointer predates transaction;
- commit predates pointer swap;
- pointer swap occurs after authorization expiry;
- serving convergence predates pointer swap;
- serving convergence predates commit;
- ceremony timestamp predates convergence.

## Active authority at finalization

At finalization time, the committed promotion pointer must still be the active
registry pointer.

If another promotion or rollback has already moved authority, L36 records:

```text
active_pointer_moved_before_ceremony
```

and returns BLOCKED.

This prevents a delayed ceremony from declaring an already-superseded
checkpoint as the current successful promotion.

## Serving evidence

The convergence receipt must be PASS and must match the committed pointer on:

- active pointer SHA-256;
- active checkpoint SHA-256;
- active generation.

L34 receipts now include `assessed_at` and expose a public verifier.

This timestamp is part of the self-digested convergence receipt and is used by
the L36 timeline court.

## Immutable ceremony receipt

A COMPLETE receipt binds:

- transaction ID;
- L35 authorization SHA-256;
- L32 multicycle chain SHA-256;
- long-horizon retention court SHA-256;
- candidate checkpoint SHA-256;
- committed pointer generation and SHA-256;
- L34 serving convergence SHA-256;
- authorization issue/expiry timestamps;
- transaction prepare/commit timestamps;
- pointer creation timestamp;
- serving convergence timestamp;
- ceremony timestamp.

It contains no raw prompt/target text, raw operator nonce or raw process ID.

The receipt has its own `ceremony_sha256`.

## Persistence

Only COMPLETE ceremony receipts are persisted.

The default path is:

```text
<registry>/ceremonies/<generation>.json
```

A BLOCKED ceremony is returned for diagnosis but is not written as final
promotion authority.

A generation cannot silently receive different ceremony evidence. If the file
already exists, the existing receipt must verify and must be byte-equivalent at
the semantic JSON level.

## Historical validity

A valid ceremony remains historical evidence even if a later rollback or
promotion changes the active pointer.

Therefore:

- finalization requires the promoted pointer to still be active;
- later verification of an already persisted ceremony does not require it to
  still be active.

This preserves honest history.

Example:

```text
generation 0: model A
generation 1: model B  <- COMPLETE L36 ceremony
generation 2: model A  <- later rollback
```

The generation-1 ceremony remains valid evidence that B was successfully
promoted and converged before it was later rolled back.

## Registry audit

The L33 registry audit now also checks every persisted L36 ceremony.

For each receipt it verifies:

- self-digest;
- COMPLETE status;
- filename/generation binding;
- committed transaction;
- committed pointer SHA;
- pointer generation;
- candidate checkpoint;
- L35 authorization;
- L32 and long-horizon evidence digests.

Tampering with a persisted ceremony therefore makes the registry audit fail.

L36 also exposes a dedicated ceremony audit.

## CLI

Finalize after L34 convergence:

```bash
python scripts/finalize_promotion_ceremony.py \
  --registry runtime-data/continual-checkpoint-registry \
  finalize \
  --transaction <transaction-id> \
  --convergence runtime-data/serving-convergence.json
```

Verify one historical ceremony:

```bash
python scripts/finalize_promotion_ceremony.py \
  --registry runtime-data/continual-checkpoint-registry \
  verify --generation 1
```

Audit all ceremonies:

```bash
python scripts/finalize_promotion_ceremony.py \
  --registry runtime-data/continual-checkpoint-registry \
  audit
```

## Courts

The L36 courts exercise the complete authority path using real repository
implementations:

- valid L32 evidence produces an L35 authorization;
- L33 performs an authorized atomic checkpoint transition;
- L34 registers the committed pointer and produces convergence PASS;
- L36 creates and persists COMPLETE evidence;
- registry audit sees the ceremony;
- BLOCKED serving convergence cannot persist final authority;
- a later active-pointer move blocks a new delayed finalization;
- an already completed ceremony remains historically verifiable after rollback;
- convergence timestamp predating pointer swap is blocked;
- persisted ceremony tampering is detected;
- registry pointer and authorization lookup helpers remain evidence verified.

## What L36 does not prove

Synthetic CI proves the release mechanism, not real model quality.

The project still must execute the full chain with:

- approved real user evidence;
- real L31 neural checkpoint updates;
- real L32 multi-window retention evidence;
- real L35 authorization;
- real checkpoint files;
- real multi-process L34 serving reload;
- hard process kills and filesystem behavior on target operating systems.

Until those real courts pass, `v0.36.0` means the **promotion ceremony
mechanism is implemented**, not that autonomous production self-modification is
approved.
