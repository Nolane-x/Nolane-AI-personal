# L22 End-to-End Evidence Harness

## Purpose

L21 defines the real 17-stage candidate authority chain, but before L22 ordinary CI only exercised readiness/check-only behavior.

L22 adds a shared evidence-chain executor and a synthetic end-to-end harness that traverses the same canonical stage contract used by the real pipeline.

The harness is explicitly:

```text
SYNTHETIC_NON_AUTHORITY_NEVER_PROMOTABLE
```

Its purpose is to prove orchestration, fail-stop behavior, artifact hashing and receipt privacy. It can never be used as model-quality or promotion evidence.

## Canonical stage contract

The shared engine freezes exactly 17 stages:

```text
01 freeze_l15_spec
02 train_l15_native
03 evaluate_l15_quality
04 benchmark_l15_resources
05 promote_l15
06 export_l16_standalone
07 evaluate_l16_parity
08 benchmark_l16_resources
09 promote_l16
10 search_l18_rank_frontier
11 evaluate_l18_quality
12 benchmark_l18_resources
13 promote_l18
14 export_l19_quantized
15 evaluate_l19_quality
16 benchmark_l19_resources
17 promote_l19
```

The order is SHA-256 bound through a canonical contract digest.

The real L21 pipeline imports this exact contract rather than maintaining its own copy.

## Shared executor

The executor requires, for every stage:

- exact next stage name;
- process exit code zero;
- every declared output file to exist;
- SHA-256 for every declared output.

Any violation blocks immediately.

An out-of-order stage fails before its process is executed.

An extra stage after stage 17 also fails closed.

## Synthetic fixture

Run the full fixture:

```bash
python scripts/run_evidence_chain_fixture.py
```

The fixture invokes an external subprocess for each of the 17 stages and creates deterministic synthetic artifacts.

Its receipt contains only:

- stage index;
- stage name;
- process exit code;
- expected output paths;
- SHA-256 of each generated artifact.

The synthetic input file intentionally contains privacy sentinels. CI verifies those prompt/target sentinels never appear in the receipt.

## Failure injection

Any stage can be forced to fail:

```bash
python scripts/run_evidence_chain_fixture.py \
  --fail-stage evaluate_l16_parity
```

Expected behavior:

- stages 1 through the failed stage are recorded;
- the failed stage records its non-zero exit code;
- no later stage executes;
- the receipt status is `SYNTHETIC_EVIDENCE_CHAIN_BLOCKED`;
- `blocked_stage` names the exact first failure.

## Contract audit

```bash
python scripts/audit_evidence_chain_contract.py
```

CI requires:

- exactly 17 stages;
- unique stage names;
- stable canonical contract digest.

## Real L21 integration

`scripts/run_real_candidate_pipeline.py` now routes all 17 production stages through the same L22 executor.

The real receipt records:

- `stage_contract_sha256`;
- `stage_count_expected=17`;
- per-stage SHA-256 coverage;
- final `stage_count_observed=17` only after complete verification.

A final `REAL_CANDIDATE_EVIDENCE_CHAIN_PASS` is impossible unless the observed stage order exactly equals the frozen contract and every expected output has a digest.

## What L22 proves

L22 proves the orchestration mechanism itself can:

- execute all 17 stages in order;
- preserve exact output lineage;
- stop on the first failure;
- reject order drift;
- reject incomplete digest coverage;
- avoid copying synthetic private prompt/target text into receipts.

## What L22 does not prove

L22 does not prove model quality.

The synthetic fixture never receives promotion authority and never substitutes for the missing user-approved train/dev/test evidence required by L21.

The next real milestone is still an actual L21 execution on sufficient approved personalization data and a valid L14 comparator.
