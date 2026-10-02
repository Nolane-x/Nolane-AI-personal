# L35 Evidence-Bound Promotion Authority

## Purpose

L30-L32 produce model-quality and long-horizon evidence.
L33 can atomically switch checkpoint authority.
L34 can safely converge serving processes.

L35 connects those mechanisms with an explicit promotion decision.

A candidate can reach the intended checkpoint-registry CLI only when:

- a complete L32 multicycle chain is supplied;
- every L31/L30 cycle is reverified;
- the fixed long-horizon court is valid and PASS;
- adaptation protocols were unique across cycles;
- the chain contains at least the required number of cycles;
- an explicit local operator request approves exactly this parent, candidate,
  chain and long-horizon court;
- a short-lived authorization remains valid through commit.

## Offline multicycle jump

The final L31 candidate in a two-cycle offline chain has immediate parent
checkpoint 1, while production may still be on checkpoint 0.

L35 deliberately authorizes:

```text
production active: checkpoint 0

offline evidence:
  checkpoint 0 -> checkpoint 1 -> checkpoint 2
                    L30 PASS       L30 PASS
  checkpoint 0 ------------------> checkpoint 2
             fixed-panel PASS

authorized atomic promotion:
  checkpoint 0 ==================> checkpoint 2
```

This avoids the circular requirement that every experimental intermediate model
must first be promoted to production.

The authorization binds checkpoint 0 as the active parent and checkpoint 2 as
the exact final candidate.

## Evidence is recomputed

L35 does not trust a standalone L32 chain digest.

It loads every L31 cycle receipt and the fixed-panel receipt, reconstructs the
multicycle court with the chain's frozen policy, and requires the result to
match the supplied L32 receipt exactly.

A chain created with adaptation-protocol reuse enabled is never eligible for
promotion.

## Explicit operator request

The request contains only cryptographic identities:

- active parent checkpoint SHA-256;
- final candidate checkpoint SHA-256;
- L32 chain SHA-256;
- fixed-panel court SHA-256;
- hash of a one-time nonce;
- explicit approved boolean.

The raw nonce is not stored in the request.

The request has no authority by itself.

## Authorization

When evidence and operator intent agree, L35 issues:

```text
NOLANE-L35-EVIDENCE-BOUND-PROMOTION-AUTHORIZATION-V1
status = AUTHORIZED
```

The authorization is self-digested and contains:

- exact active parent;
- exact final candidate;
- exact L32 chain;
- exact fixed-panel court;
- cycle count;
- operator request digest;
- hashed nonce;
- issue time;
- expiry time.

Default TTL is one hour.

## One-time transaction use

L33 stores the authorization inside the transaction directory.

The same authorization cannot begin a second transaction.

The authorization is checked at:

1. transaction begin;
2. transaction verification;
3. immediately before pointer commit.

If it expires between verification and commit, the pointer is not swapped.

Historical registry audit verifies the authorization digest and binding without
requiring an old authorization to still be unexpired.

## Intended CLI path

Create explicit intent:

```bash
python scripts/authorize_continual_promotion.py request \
  --active-parent <checkpoint-0-sha> \
  --candidate <checkpoint-N-sha> \
  --chain runtime-data/l32/multicycle-chain.json \
  --long-horizon runtime-data/l32/long-horizon-retention.json \
  --approve \
  --output runtime-data/l35/promotion-request.json
```

Issue authorization:

```bash
python scripts/authorize_continual_promotion.py authorize \
  --cycle runtime-data/cycle-1/l31-run-receipt.json \
  --cycle runtime-data/cycle-2/l31-run-receipt.json \
  --chain runtime-data/l32/multicycle-chain.json \
  --long-horizon runtime-data/l32/long-horizon-retention.json \
  --request runtime-data/l35/promotion-request.json \
  --output runtime-data/l35/promotion-authorization.json
```

Begin the L33 transaction:

```bash
python scripts/manage_checkpoint_registry.py \
  --registry runtime-data/continual-checkpoint-registry \
  begin \
  --candidate runtime-data/final-l31-candidate \
  --authorization runtime-data/l35/promotion-authorization.json
```

The operational `begin` command no longer accepts a bare candidate.

## Courts

CI proves:

- valid L32 evidence plus explicit approval authorizes the final artifact;
- denial remains BLOCKED;
- candidate mismatch remains BLOCKED;
- authorization tampering is detected;
- expiration is enforced;
- cycle evidence is recomputed instead of trusting chain SHA alone;
- an authorized 0 -> 2 offline multicycle jump can commit;
- the direct immediate-parent path rejects that same 0 -> 2 jump;
- authorization replay is rejected;
- expiration between verify and commit prevents pointer swap;
- registry audit detects authorization-file tampering.

## Authority boundary

L35 is the first explicit promotion-decision layer in the continual-learning
path, but repository CI still uses synthetic evidence.

Real production authority is not scientifically earned until the same path is
executed with approved real evidence, real checkpoints, real process reloads and
platform crash tests.

The next wave should therefore build a complete release ceremony that binds:

L35 authorization -> L33 commit -> L34 serving convergence -> final immutable
promotion receipt.
