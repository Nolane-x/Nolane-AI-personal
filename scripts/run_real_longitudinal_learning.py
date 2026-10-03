from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from nolane_personal.long_horizon_retention import (
    verify_long_horizon_retention_digest,
)
from nolane_personal.longitudinal_execution import (
    build_longitudinal_report,
    validate_longitudinal_plan,
)
from nolane_personal.store import canonical_json, payload_digest
from nolane_personal.unified_continual import (
    verify_l38_run_receipt,
    verify_unified_continual_chain_digest,
)


ROOT = Path(__file__).resolve().parents[1]
JOURNAL_SCHEMA = "NOLANE-L43-REAL-LONGITUDINAL-JOURNAL-V1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def seal(payload: dict[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    result[field] = payload_digest(result)
    return result


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def run_checked(args: list[str]) -> None:
    completed = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "longitudinal stage failed with exit code "
            f"{completed.returncode}: {' '.join(args)}"
        )


def training_args(
    *,
    parent: Path,
    latent: Path,
    tokenizer: Path,
    device: str,
    cycle,
    output_dir: Path,
) -> list[str]:
    args = [
        sys.executable,
        "scripts/train_continual_cortex_update.py",
        "--factorized",
        str(parent),
        "--latent",
        str(latent),
        "--retention-dataset",
        cycle.retention.dataset_path,
        "--retention-protocol",
        cycle.retention.protocol_path,
        "--adaptation-dataset",
        cycle.adaptation.dataset_path,
        "--adaptation-protocol",
        cycle.adaptation.protocol_path,
        "--tokenizer",
        str(tokenizer),
        "--output-dir",
        str(output_dir),
        "--device",
        device,
    ]
    flags = {
        "epochs": "--epochs",
        "learning_rate": "--learning-rate",
        "adaptation_task_weight": "--adaptation-task-weight",
        "retention_task_weight": "--retention-task-weight",
        "retention_distill_weight": "--retention-distill-weight",
        "cortex_anchor_weight": "--cortex-anchor-weight",
        "temperature": "--temperature",
        "max_grad_norm": "--max-grad-norm",
        "max_length": "--max-length",
    }
    for key, flag in flags.items():
        if key in cycle.training:
            args.extend([flag, str(cycle.training[key])])
    return args


def evaluator_args(
    *,
    initial: Path,
    final: Path,
    latent: Path,
    dataset: str,
    protocol: str,
    tokenizer: Path,
    device: str,
    output: Path,
    max_overall_regression: float,
    max_worst_group_regression: float,
) -> list[str]:
    return [
        sys.executable,
        "scripts/evaluate_unified_long_horizon_retention.py",
        "--initial",
        str(initial),
        "--final",
        str(final),
        "--latent",
        str(latent),
        "--dataset",
        dataset,
        "--protocol",
        protocol,
        "--tokenizer",
        str(tokenizer),
        "--device",
        device,
        "--max-overall-regression",
        str(max_overall_regression),
        "--max-worst-group-regression",
        str(max_worst_group_regression),
        "--output",
        str(output),
    ]


def save_journal(
    path: Path,
    *,
    plan_sha256: str,
    status: str,
    cycles: list[dict[str, Any]],
    report_sha256: str | None = None,
) -> dict[str, Any]:
    payload = seal(
        {
            "schema": JOURNAL_SCHEMA,
            "authority": (
                "REAL_LONGITUDINAL_EXECUTION_JOURNAL_NO_PROMOTION_AUTHORITY"
            ),
            "plan_sha256": plan_sha256,
            "status": status,
            "completed_cycles": cycles,
            "report_sha256": report_sha256,
            "privacy": {
                "contains_raw_prompt_target": False,
                "contains_raw_source_group_hash_values": False,
            },
        },
        "journal_sha256",
    )
    write_json(path, payload)
    return payload


def load_completed_cycle(
    cycle_dir: Path,
    *,
    expected_parent_sha256: str,
) -> tuple[dict[str, Any], Path]:
    receipt_path = cycle_dir / "l38-run-receipt.json"
    checkpoint = cycle_dir / "factorized-nolane.pt"
    if not receipt_path.is_file() or not checkpoint.is_file():
        raise ValueError(
            f"incomplete longitudinal cycle workspace: {cycle_dir}"
        )
    receipt = load_json(receipt_path)
    verify_l38_run_receipt(receipt)
    if (
        receipt["parent_factorized_checkpoint_sha256"]
        != expected_parent_sha256
    ):
        raise ValueError("resumed L38 cycle parent checkpoint mismatch")
    if receipt["artifact"]["checkpoint_sha256"] != sha256_file(checkpoint):
        raise ValueError("resumed L38 cycle checkpoint digest mismatch")
    return receipt, checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument(
        "--output-dir",
        default="runtime-data/l43-real-longitudinal",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Verify all approved packs and isolation rules without training.",
    )
    args = parser.parse_args()

    plan = validate_longitudinal_plan(args.plan)
    output = Path(args.output_dir).resolve()
    plan_receipt_path = output / "plan-receipt.json"
    journal_path = output / "execution-journal.json"

    if args.validate_only:
        rendered = json.dumps(
            plan.receipt,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        print(rendered)
        return 0

    if output.exists() and any(output.iterdir()) and not args.resume:
        raise SystemExit(
            f"refusing non-empty L43 workspace without --resume: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)

    if plan_receipt_path.exists():
        existing = load_json(plan_receipt_path)
        if existing != plan.receipt:
            raise SystemExit(
                "existing L43 workspace belongs to a different plan"
            )
    else:
        write_json(plan_receipt_path, plan.receipt)

    parent = plan.initial_factorized
    parent_sha = sha256_file(parent)
    cycle_receipts: list[dict[str, Any]] = []
    cycle_checkpoints: list[Path] = []
    journal_rows: list[dict[str, Any]] = []

    for cycle in plan.cycles:
        cycle_dir = output / f"cycle-{cycle.index:03d}"
        if cycle_dir.exists() and any(cycle_dir.iterdir()):
            if not args.resume:
                raise SystemExit(
                    f"cycle workspace already exists: {cycle_dir}"
                )
            receipt, checkpoint = load_completed_cycle(
                cycle_dir,
                expected_parent_sha256=parent_sha,
            )
        else:
            run_checked(
                training_args(
                    parent=parent,
                    latent=plan.latent,
                    tokenizer=plan.tokenizer,
                    device=plan.device,
                    cycle=cycle,
                    output_dir=cycle_dir,
                )
            )
            receipt, checkpoint = load_completed_cycle(
                cycle_dir,
                expected_parent_sha256=parent_sha,
            )

        cycle_receipts.append(receipt)
        cycle_checkpoints.append(checkpoint)
        parent = checkpoint
        parent_sha = receipt["artifact"]["checkpoint_sha256"]
        journal_rows.append(
            {
                "cycle": cycle.index,
                "checkpoint_sha256": parent_sha,
                "run_receipt_sha256": sha256_file(
                    cycle_dir / "l38-run-receipt.json"
                ),
                "adaptation_protocol_sha256": cycle.adaptation.protocol_sha256,
                "retention_protocol_sha256": cycle.retention.protocol_sha256,
            }
        )
        save_journal(
            journal_path,
            plan_sha256=plan.receipt["plan_sha256"],
            status="IN_PROGRESS",
            cycles=journal_rows,
        )

    final_checkpoint = cycle_checkpoints[-1]

    fixed_path = output / "fixed-long-horizon.json"
    if not fixed_path.exists():
        run_checked(
            evaluator_args(
                initial=plan.initial_factorized,
                final=final_checkpoint,
                latent=plan.latent,
                dataset=plan.fixed_panel.dataset_path,
                protocol=plan.fixed_panel.protocol_path,
                tokenizer=plan.tokenizer,
                device=plan.device,
                output=fixed_path,
                max_overall_regression=(
                    plan.policy.max_fixed_overall_regression
                ),
                max_worst_group_regression=(
                    plan.policy.max_fixed_worst_group_regression
                ),
            )
        )
    fixed_receipt = load_json(fixed_path)
    verify_long_horizon_retention_digest(fixed_receipt)

    learned_dir = output / "learned-window-retention"
    learned_dir.mkdir(parents=True, exist_ok=True)
    learned_receipts: list[dict[str, Any]] = []
    for cycle, learned_checkpoint in zip(
        plan.cycles[:-1],
        cycle_checkpoints[:-1],
        strict=True,
    ):
        retention_path = learned_dir / f"cycle-{cycle.index:03d}.json"
        if not retention_path.exists():
            run_checked(
                evaluator_args(
                    initial=learned_checkpoint,
                    final=final_checkpoint,
                    latent=plan.latent,
                    dataset=cycle.adaptation.dataset_path,
                    protocol=cycle.adaptation.protocol_path,
                    tokenizer=plan.tokenizer,
                    device=plan.device,
                    output=retention_path,
                    max_overall_regression=(
                        plan.policy.max_learned_window_overall_regression
                    ),
                    max_worst_group_regression=(
                        plan.policy.max_learned_window_worst_group_regression
                    ),
                )
            )
        retention = load_json(retention_path)
        verify_long_horizon_retention_digest(retention)
        if (
            retention["initial_checkpoint_sha256"]
            != cycle_receipts[cycle.index - 1]["artifact"][
                "checkpoint_sha256"
            ]
        ):
            raise ValueError("learned-window retention initial checkpoint mismatch")
        if (
            retention["final_checkpoint_sha256"]
            != cycle_receipts[-1]["artifact"]["checkpoint_sha256"]
        ):
            raise ValueError("learned-window retention final checkpoint mismatch")
        learned_receipts.append(retention)

    unified_path = output / "unified-chain.json"
    if not unified_path.exists():
        command = [
            sys.executable,
            "scripts/assess_unified_continual.py",
        ]
        for index in range(1, len(plan.cycles) + 1):
            command.extend(
                [
                    "--cycle",
                    str(output / f"cycle-{index:03d}" / "l38-run-receipt.json"),
                ]
            )
        command.extend(
            [
                "--long-horizon",
                str(fixed_path),
                "--min-cycles",
                str(len(plan.cycles)),
                "--min-cortex-cycles",
                str(len(plan.cycles)),
                "--output",
                str(unified_path),
            ]
        )
        run_checked(command)
    unified = load_json(unified_path)
    verify_unified_continual_chain_digest(unified)

    report = build_longitudinal_report(
        plan_receipt=plan.receipt,
        cycle_receipts=cycle_receipts,
        fixed_panel_receipt=fixed_receipt,
        learned_window_receipts=learned_receipts,
        unified_chain=unified,
    )
    report_path = output / "l43-longitudinal-report.json"
    write_json(report_path, report)
    save_journal(
        journal_path,
        plan_sha256=plan.receipt["plan_sha256"],
        status=report["status"],
        cycles=journal_rows,
        report_sha256=report["report_sha256"],
    )
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
