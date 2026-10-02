# L38 Recurrent Cortex Continual Plasticity

## Purpose

L31 proved a real continual-learning update path, but deliberately restricted
plasticity to the low-rank factorized language boundary.

That was useful for controlled adaptation, yet it left the deeper Nolane-owned
recurrent cognition frozen.

L38 opens a second neural update path:

```text
factorized checkpoint
        |
        +----------------------+
        |                      |
        v                      v
frozen reference        trainable candidate
        |                      |
boundary frozen          boundary frozen
cortex frozen            recurrent cortex trainable
        |                      |
        +--------- L30 --------+
```

The goal is to let new experience alter the recurrent temporal/depth dynamics
themselves while preserving a strict language-boundary ownership guard.

## Trainable ownership

L38 freezes every candidate language-boundary parameter.

Only parameters in `DeepRecurrentStateSpaceCortex.module` may receive
gradients.

The court requires:

- candidate boundary digest unchanged;
- candidate boundary gradients seen = 0;
- candidate recurrent-cortex digest changed;
- candidate recurrent-cortex finite gradients observed;
- frozen reference boundary unchanged;
- frozen reference cortex unchanged.

Any boundary mutation aborts the update.

## Evidence separation

L38 keeps the same no-contamination structure as L31:

```text
old protocol
  train + dev -> retention rehearsal
  test        -> retention held-out court

new protocol
  train       -> cortex adaptation
  test        -> adaptation held-out court
```

The held-out rows are never optimizer inputs.

Both old and new datasets must independently pass L28 before the real runner
will train.

## Objective

Each new adaptation example is paired round-robin with an old rehearsal
example.

The candidate objective contains:

1. new-evidence task loss;
2. old rehearsal task loss;
3. KL distillation against the frozen reference on old rehearsal;
4. a small recurrent-cortex parameter anchor to the frozen reference.

The anchor is not a substitute for held-out evaluation. It only discourages
unnecessary neural drift while the L30 court decides whether the update
actually preserved old behavior and gained useful adaptation.

## Model-state identity

L31's first continual-learning implementation used the changed boundary digest
as the L30 before/after identity.

For L38, the authority surface is the recurrent cortex, so the court receives a
model-state SHA-256 built from:

```text
SHA256(
  boundary_state_digest,
  cortex_state_digest
)
```

The language boundary stays identical while the recurrent cortex changes, so
the composite model-state digest must change.

## Held-out L30 court

After training, L38 computes NLL on:

- old held-out retention examples;
- new held-out adaptation examples.

The embedded L30 court still gates:

- mean old-group forgetting;
- worst old-group forgetting;
- mean new-group adaptation gain;
- worst new-group regression.

Optimizer completion is not promotion.

A cortex can receive gradients and change weights yet still finish BLOCKED.

## Receipt integrity

The L38 training receipt is self-digested with `update_sha256`.

Its verifier also re-verifies the embedded L30 digest and ownership invariants.

The receipt records:

- model-state before/after SHA;
- boundary before/after digest;
- cortex before/after digest;
- observed gradient ownership;
- train/eval counts;
- NLL before/after;
- training configuration;
- complete L30 court.

Raw source-group hashes are not copied into the L30 receipt.

## Real local-data runner

```bash
python scripts/train_continual_cortex_update.py \
  --factorized runtime-data/l17-factorized-trained/factorized-nolane.pt \
  --retention-dataset /private/old/personalization.jsonl \
  --retention-protocol /private/old/personalization-protocol-v1.json \
  --adaptation-dataset /private/new/personalization.jsonl \
  --adaptation-protocol /private/new/personalization-protocol-v1.json \
  --output-dir runtime-data/l38-continual-cortex-update
```

The output is still a standard standalone factorized Nolane artifact, but its
`training_receipt` is L38 evidence and the runtime still requires no Qwen model
object.

## Important compatibility boundary

L32/L35 currently understand the L31 boundary-update cycle schema.

L38 therefore **does not pretend to be an L31 cycle** and is not yet eligible
for the existing multicycle promotion chain.

Its run authority is:

```text
RECURRENT_CORTEX_UPDATE_EVIDENCE_ONLY_UNPROMOTED
```

A later mixed-cycle ledger must explicitly understand both update types before
an L38 artifact can participate in production promotion evidence.

This prevents a schema shortcut from silently weakening L32/L35.

## CI courts

The PyTorch court proves:

- candidate and reference start from exact same boundary + cortex state;
- only candidate cortex changes;
- only candidate cortex receives gradients;
- candidate boundary remains byte/digest stable;
- frozen reference remains unchanged;
- model-state digest changes;
- held-out L30 can PASS under permissive test policy;
- optimizer success can still be BLOCKED by a strict L30 policy;
- source-group hashes are absent from the serialized receipt;
- boundary/cortex start mismatch fails closed;
- candidate/reference object alias fails closed;
- source-group lineage-length drift fails closed;
- negative anchor weight fails validation;
- receipt tampering is detected.

## Scientific boundary

L38 is stronger evidence of internal plasticity than L31 because the
Nolane-owned recurrent cognition itself changes.

It does not yet prove that this produces better long-horizon companionship.

That needs real multi-window data and a mixed-cycle long-horizon court showing
that recurrent plasticity improves adaptation without destabilizing identity,
memory, language quality or previous behavior.
