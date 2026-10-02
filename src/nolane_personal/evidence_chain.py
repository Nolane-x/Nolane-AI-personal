from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .store import canonical_json, payload_digest


REAL_CANDIDATE_STAGE_ORDER: tuple[str, ...] = (
    "freeze_l15_spec",
    "train_l15_native",
    "evaluate_l15_quality",
    "benchmark_l15_resources",
    "promote_l15",
    "export_l16_standalone",
    "evaluate_l16_parity",
    "benchmark_l16_resources",
    "promote_l16",
    "search_l18_rank_frontier",
    "evaluate_l18_quality",
    "benchmark_l18_resources",
    "promote_l18",
    "export_l19_quantized",
    "evaluate_l19_quality",
    "benchmark_l19_resources",
    "promote_l19",
)


def evidence_chain_contract_sha256(
    stage_order: Sequence[str] = REAL_CANDIDATE_STAGE_ORDER,
) -> str:
    return payload_digest({
        "schema": "NOLANE-EVIDENCE-CHAIN-CONTRACT-V1",
        "stages": list(stage_order),
    })


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_output_digest(paths: Sequence[Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in paths:
        if path.exists() and path.is_file():
            result[str(path)] = sha256_file(path)
    return result


def write_receipt(path: Path, receipt: dict) -> None:
    body = dict(receipt)
    body.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = payload_digest(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(receipt) + "\n", encoding="utf-8")


@dataclass(slots=True)
class EvidenceStageError(RuntimeError):
    stage: str
    exit_code: int
    missing_outputs: tuple[str, ...] = ()
    reason: str = "stage_failed"

    def __str__(self) -> str:
        return (
            f"{self.reason}: stage={self.stage} exit_code={self.exit_code} "
            f"missing={list(self.missing_outputs)}"
        )


Runner = Callable[[list[str], Path], int]


def subprocess_runner(command: list[str], cwd: Path) -> int:
    completed = subprocess.run(command, cwd=cwd)
    return int(completed.returncode)


def run_stage(
    *,
    name: str,
    command: list[str],
    expected: list[Path],
    receipt: dict,
    receipt_path: Path,
    root: Path,
    stage_order: Sequence[str] = REAL_CANDIDATE_STAGE_ORDER,
    runner: Runner | None = None,
    blocked_status: str = "REAL_CANDIDATE_PIPELINE_BLOCKED",
) -> None:
    index = len(receipt.setdefault("stages", []))
    if index >= len(stage_order):
        receipt["status"] = blocked_status
        receipt["blocked_stage"] = name
        receipt["contract_error"] = "extra_stage"
        write_receipt(receipt_path, receipt)
        raise EvidenceStageError(name, 4, reason="extra_stage")

    expected_name = str(stage_order[index])
    if name != expected_name:
        receipt["status"] = blocked_status
        receipt["blocked_stage"] = name
        receipt["contract_error"] = (
            f"stage_order_mismatch:expected={expected_name}:actual={name}"
        )
        write_receipt(receipt_path, receipt)
        raise EvidenceStageError(name, 4, reason="stage_order_mismatch")

    print(f"\n[EVIDENCE] {index + 1:02d}/{len(stage_order):02d} {name}")
    print(" ".join(command))

    execute = runner or subprocess_runner
    exit_code = int(execute(command, root))
    output_sha256 = stage_output_digest(expected)
    missing = tuple(str(path) for path in expected if not path.exists())
    row = {
        "stage_index": index + 1,
        "name": name,
        "exit_code": exit_code,
        "expected_outputs": [str(path) for path in expected],
        "output_sha256": output_sha256,
    }
    if missing:
        row["missing_outputs"] = list(missing)
    receipt["stages"].append(row)

    if exit_code != 0 or missing:
        receipt["status"] = blocked_status
        receipt["blocked_stage"] = name
        write_receipt(receipt_path, receipt)
        raise EvidenceStageError(
            name,
            exit_code or 3,
            missing_outputs=missing,
            reason="stage_failed",
        )
    write_receipt(receipt_path, receipt)


def verify_complete_stage_order(
    receipt: dict,
    *,
    stage_order: Sequence[str] = REAL_CANDIDATE_STAGE_ORDER,
) -> None:
    observed = [str(row.get("name")) for row in receipt.get("stages", [])]
    expected = list(stage_order)
    if observed != expected:
        raise ValueError(
            "evidence chain incomplete or out of order: "
            f"observed={observed} expected={expected}"
        )
    for index, row in enumerate(receipt["stages"], start=1):
        if int(row.get("stage_index", -1)) != index:
            raise ValueError("evidence chain stage_index mismatch")
        if int(row.get("exit_code", 1)) != 0:
            raise ValueError("evidence chain contains failed stage")
        expected_outputs = row.get("expected_outputs", [])
        output_sha256 = row.get("output_sha256", {})
        if len(output_sha256) != len(expected_outputs):
            raise ValueError("evidence chain output digest coverage mismatch")
