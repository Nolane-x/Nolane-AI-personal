# L40 Unified Promotion Authority

## Purpose

L39 can prove a long-horizon continual-learning chain containing both:

- L31 factorized language-boundary updates;
- L38 recurrent-cortex updates.

L35 promotion authority understands the older L32/L31-only chain and therefore
must not be reused for L39 evidence by renaming schemas.

L40 adds a separate authority path for the whole mixed model state.

It still does **not** modify the active production pointer.

## Evidence recomputation

L40 receives:

- ordered raw L31/L38 run receipts;
- the fixed long-horizon retention receipt;
- the L39 unified chain receipt;
- explicit operator intent.

Before issuing authorization it:

1. verifies the L39 receipt self-digest;
2. verifies the long-horizon receipt self-digest;
3. reconstructs the L39 policy from the receipt;
4. refuses a policy that allowed adaptation-protocol replay;
5. recomputes L39 from every raw cycle receipt;
6. requires the recomputed receipt to equal the supplied L39 receipt exactly.

A valid-looking chain SHA without matching raw evidence cannot authorize
promotion.

## Operator request

The operator request explicitly binds:

- currently expected production parent checkpoint SHA;
- candidate checkpoint SHA;
- L39 unified chain SHA;
- long-horizon retention court SHA;
- approval boolean;
- one-time nonce hash.

The raw nonce is never persisted.

The request has no promotion authority by itself.

## Cortex evidence floor

The default authorization policy requires:

```text
min_cycles = 2
min_cortex_cycles = 1
require_unique_adaptation_protocols = true
require_operator_approval = true
```

Therefore an otherwise valid chain can still be BLOCKED if the authorization
policy demands stronger recurrent-cortex evidence than the chain contains.

## Authorization binding

An AUTHORIZED receipt binds:

- production parent checkpoint;
- final candidate checkpoint;
- L39 chain SHA;
- fixed long-horizon court SHA;
- total cycle count;
- boundary-cycle count;
- cortex-cycle count;
- initial composite model-state SHA;
- final composite model-state SHA;
- operator request SHA;
- operator nonce SHA;
- issuance time;
- expiry time.

This prevents a candidate checkpoint from being separated from the exact mixed
plasticity evidence that justified it.

## TTL

Authorization is short-lived.

The default TTL is one hour.

Verification rejects an authorization after its expiry time.

A later transaction integration must recheck expiry at begin, verify and commit,
matching the L35/L33 safety model.

## CLI

```bash
python scripts/authorize_unified_promotion.py \
  --cycle runtime-data/cycle-1/l31-run-receipt.json \
  --cycle runtime-data/cycle-2/l38-run-receipt.json \
  --long-horizon runtime-data/l39/long-horizon-retention.json \
  --unified-chain runtime-data/l39/unified-chain.json \
  --active-parent-checkpoint-sha256 <production-parent-sha> \
  --candidate-checkpoint-sha256 <final-candidate-sha> \
  --nonce "<one-time-secret-nonce>" \
  --approve \
  --output runtime-data/l40/unified-authorization.json
```

Without `--approve`, the decision is BLOCKED.

The parent and candidate are explicit CLI arguments rather than silently derived
from the chain. A mismatch therefore appears as an authorization failure.

## CI courts

L40 proves:

- a valid two-cycle recurrent-cortex chain can be AUTHORIZED;
- operator denial blocks;
- candidate mismatch blocks;
- raw cycle evidence drift is detected by recomputation;
- stronger cortex-cycle requirements block insufficient evidence;
- TTL expiry is enforced;
- authorization tampering is detected;
- raw operator nonce is absent from persisted request evidence.

## Authority boundary

L40 emits:

```text
EXPLICIT_UNIFIED_EVIDENCE_BOUND_PROMOTION_AUTHORITY
```

but the current L33 transaction registry intentionally does not consume this
schema yet.

That separation is deliberate.

The next wave must extend the transaction/ceremony path to understand L40,
verify a final L31 or L38 candidate bundle according to its native schema, and
preserve the existing hard-kill/recovery courts before a recurrent-cortex
candidate can change production authority.
