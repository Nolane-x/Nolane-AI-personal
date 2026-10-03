# L43 Real Longitudinal Learning Execution

## Purpose

L38 proves one recurrent-cortex update can change the Nolane-owned recurrent
cognition while preserving a frozen language boundary.

L39 proves multiple neural updates can form one exact model-state chain.

L43 answers the harder empirical question:

> After Nolane learns several genuinely new windows over time, does the final
> checkpoint still retain the original capabilities **and** the capabilities it
> learned in earlier windows?

This is an execution layer for approved real evidence. It does not invent,
scrape, auto-approve or synthesize the missing user evidence.

## Evidence boundary

Every dataset in an L43 plan must already be a verified L23 approved-evidence
pack.

That means every row has passed the local approval/privacy boundary and the pack
has an L28 evidence-quality PASS.

L43 additionally requires complete source-group lineage for every example.

No fixture or CI-generated pack can be represented as real evidence merely by
placing it in an L43 plan.

## Minimum experiment

A real longitudinal run requires at least **five** recurrent-cortex learning
cycles.

Each cycle has two approved packs:

- retention pack: older behavior rehearsed during the update;
- adaptation pack: genuinely new behavior to learn during that cycle.

The adaptation protocol and adaptation source groups must be fresh across
cycles.

Within one cycle, retention and adaptation source groups must be disjoint.

A separate fixed panel is never trained or rehearsed and must be source-group
disjoint from every training pack in the complete run.

## Why L39 fixed retention alone is not enough

Suppose cycle 1 teaches skill A and cycle 5 teaches skill E.

A fixed panel frozen before cycle 1 can prove that old baseline abilities were
not destroyed, but it does not prove that skill A survived cycles 2-5.

L43 therefore adds **learned-window retention**.

For a five-cycle run:

```text
after cycle 1 -----> final cycle 5   evaluate cycle-1 held-out adaptation test
after cycle 2 -----> final cycle 5   evaluate cycle-2 held-out adaptation test
after cycle 3 -----> final cycle 5   evaluate cycle-3 held-out adaptation test
after cycle 4 -----> final cycle 5   evaluate cycle-4 held-out adaptation test
cycle 5                               protected by its immediate L30 court
```

The final cycle is intentionally not compared with itself because that would
have identical checkpoint identity and would not constitute long-horizon
retention evidence.

## Hard evidence ceilings

The plan may make thresholds stricter, never looser.

Default maximum regressions:

- fixed panel overall: +0.01 NLL
- fixed panel worst group: +0.03 NLL
- learned-window overall: +0.03 NLL
- learned-window worst group: +0.05 NLL

The plan is invalid if it attempts to relax these ceilings or disable evidence
isolation requirements.

## Plan

Example:

```json
{
  "schema": "NOLANE-L43-REAL-LONGITUDINAL-PLAN-V1",
  "initial_factorized": "/private/checkpoints/factorized-nolane.pt",
  "latent": "/private/runtime/latent.json",
  "tokenizer": "/private/models/Qwen3-0.6B",
  "device": "cuda",
  "fixed_panel_manifest": "/private/fixed/approved-evidence-manifest.json",
  "cycles": [
    {
      "retention_manifest": "/private/window-1-old/approved-evidence-manifest.json",
      "adaptation_manifest": "/private/window-1-new/approved-evidence-manifest.json",
      "training": {
        "epochs": 2,
        "learning_rate": 0.0002
      }
    },
    {
      "retention_manifest": "/private/window-2-old/approved-evidence-manifest.json",
      "adaptation_manifest": "/private/window-2-new/approved-evidence-manifest.json"
    },
    {
      "retention_manifest": "/private/window-3-old/approved-evidence-manifest.json",
      "adaptation_manifest": "/private/window-3-new/approved-evidence-manifest.json"
    },
    {
      "retention_manifest": "/private/window-4-old/approved-evidence-manifest.json",
      "adaptation_manifest": "/private/window-4-new/approved-evidence-manifest.json"
    },
    {
      "retention_manifest": "/private/window-5-old/approved-evidence-manifest.json",
      "adaptation_manifest": "/private/window-5-new/approved-evidence-manifest.json"
    }
  ]
}
```

Raw local paths are used only to execute the run. The public plan receipt stores
hash bindings/counts and does not copy paths, prompts, targets or source-group
hash values.

## Validate before training

Run:

```bash
python scripts/run_real_longitudinal_learning.py \
  --plan /private/l43-plan.json \
  --validate-only
```

This performs the full pack verification and leakage/isolation court without
loading or training the neural model.

It should be run before committing compute to the experiment.

## Execute

```bash
python scripts/run_real_longitudinal_learning.py \
  --plan /private/l43-plan.json \
  --output-dir runtime-data/l43-real-longitudinal
```

The executor:

1. validates all approved packs and the plan;
2. trains L38 cycle 1;
3. verifies its saved checkpoint and L38 run receipt;
4. uses that exact checkpoint as the parent of cycle 2;
5. repeats through all configured cycles;
6. evaluates the frozen initial-vs-final fixed panel;
7. evaluates every earlier learned adaptation window against the final model;
8. recomputes L39 over the complete L38 chain;
9. writes an L43 longitudinal report.

## Resume

Each successful cycle is immutable evidence.

After each cycle L43 writes a self-digested journal containing only hashes and
protocol lineage.

A stopped run can continue with:

```bash
python scripts/run_real_longitudinal_learning.py \
  --plan /private/l43-plan.json \
  --output-dir runtime-data/l43-real-longitudinal \
  --resume
```

A resumed cycle is accepted only if:

- its native L38 receipt verifies;
- its checkpoint file SHA matches the receipt;
- its parent checkpoint is exactly the previous cycle output.

A partial or foreign workspace fails closed.

## Output layout

```text
l43-real-longitudinal/
  plan-receipt.json
  execution-journal.json
  cycle-001/
    factorized-nolane.pt
    factorized-nolane-manifest.json
    l38-run-receipt.json
  ...
  cycle-005/
    ...
  fixed-long-horizon.json
  learned-window-retention/
    cycle-001.json
    cycle-002.json
    cycle-003.json
    cycle-004.json
  unified-chain.json
  l43-longitudinal-report.json
```

## PASS meaning

An L43 PASS means, for this measured sequence:

- every L38 cycle passed its immediate adaptation/retention court;
- the exact neural artifact chain remained continuous;
- the initial fixed held-out panel survived the complete run;
- every learned window with future cycles survived to the final checkpoint;
- the complete L39 mixed model-state chain passed.

It does **not** prove unlimited lifelong learning.

Five cycles are the minimum empirical threshold, not a claim that forgetting is
solved forever.

## Promotion boundary

L43 authority is:

```text
REAL_LONGITUDINAL_EVIDENCE_EXECUTION_NO_PROMOTION_AUTHORITY
```

The executor deliberately does not call L40 and cannot promote a model.

If a real L43 run passes, a human/operator must separately review the evidence
and invoke the L40 authorization path.

That separation prevents a training orchestrator from granting itself
production authority.
