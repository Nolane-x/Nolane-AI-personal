from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .approved_evidence import resolve_approved_evidence_pack
from .store import payload_digest


PLAN_SCHEMA = "NOLANE-L43-REAL-LONGITUDINAL-PLAN-V1"
PLAN_RECEIPT_SCHEMA = "NOLANE-L43-REAL-LONGITUDINAL-PLAN-RECEIPT-V1"
REPORT_SCHEMA = "NOLANE-L43-REAL-LONGITUDINAL-REPORT-V1"
AUTHORITY = "REAL_LONGITUDINAL_EVIDENCE_EXECUTION_NO_PROMOTION_AUTHORITY"


@dataclass(slots=True)
class LongitudinalExecutionPolicy:
    min_cycles: int = 5
    max_fixed_overall_regression: float = 0.01
    max_fixed_worst_group_regression: float = 0.03
    max_learned_window_overall_regression: float = 0.03
    max_learned_window_worst_group_regression: float = 0.05
    require_full_source_group_lineage: bool = True
    require_unique_adaptation_protocols: bool = True
    require_disjoint_adaptation_groups: bool = True
    require_fixed_panel_isolation: bool = True

    def validate(self) -> None:
        if self.min_cycles < 5:
            raise ValueError("real longitudinal execution requires min_cycles >= 5")
        limits = {
            "max_fixed_overall_regression": 0.01,
            "max_fixed_worst_group_regression": 0.03,
            "max_learned_window_overall_regression": 0.03,
            "max_learned_window_worst_group_regression": 0.05,
        }
        for name, ceiling in limits.items():
            value = float(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            if value > ceiling:
                raise ValueError(
                    f"{name} cannot be looser than the L43 ceiling {ceiling}"
                )
        for name in (
            "require_full_source_group_lineage",
            "require_unique_adaptation_protocols",
            "require_disjoint_adaptation_groups",
            "require_fixed_panel_isolation",
        ):
            if not bool(getattr(self, name)):
                raise ValueError(f"{name} cannot be disabled for real evidence")


@dataclass(slots=True)
class ApprovedPackBinding:
    manifest_path: str
    manifest_sha256: str
    dataset_path: str
    dataset_sha256: str
    protocol_path: str
    protocol_sha256: str
    quality_court_sha256: str
    source_groups: tuple[str, ...]
    output_examples: int

    def public_receipt(self) -> dict[str, Any]:
        return {
            "manifest_sha256": self.manifest_sha256,
            "dataset_sha256": self.dataset_sha256,
            "protocol_sha256": self.protocol_sha256,
            "quality_court_sha256": self.quality_court_sha256,
            "source_group_count": len(set(self.source_groups)),
            "output_examples": self.output_examples,
        }


@dataclass(slots=True)
class LongitudinalCycleBinding:
    index: int
    retention: ApprovedPackBinding
    adaptation: ApprovedPackBinding
    training: dict[str, Any]


@dataclass(slots=True)
class ValidatedLongitudinalPlan:
    path: Path
    initial_factorized: Path
    latent: Path
    tokenizer: Path
    device: str
    fixed_panel: ApprovedPackBinding
    cycles: list[LongitudinalCycleBinding]
    policy: LongitudinalExecutionPolicy
    receipt: dict[str, Any]


def _resolve(base: Path, value: Any, *, field: str) -> Path:
    rendered = str(value or "").strip()
    if not rendered:
        raise ValueError(f"{field} is required")
    path = Path(rendered)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def _pack_binding(path: Path) -> ApprovedPackBinding:
    manifest, dataset, protocol = resolve_approved_evidence_pack(path)
    groups = manifest.get("example_source_group_sha256")
    if not isinstance(groups, list):
        raise ValueError("approved evidence pack source-group lineage missing")
    normalized: list[str] = []
    for value in groups:
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(
                "real longitudinal pack requires complete source-group lineage"
            )
        normalized.append(value)
    stats = manifest.get("stats", {})
    output_examples = int(stats.get("output_examples", 0))
    if output_examples != len(normalized):
        raise ValueError("approved pack source-group/example count mismatch")
    if int(manifest.get("source_group_coverage", -1)) != output_examples:
        raise ValueError("approved pack does not have full source-group coverage")
    return ApprovedPackBinding(
        manifest_path=str(path.resolve()),
        manifest_sha256=str(manifest["manifest_sha256"]),
        dataset_path=str(dataset.resolve()),
        dataset_sha256=str(manifest["dataset_sha256"]),
        protocol_path=str(protocol.resolve()),
        protocol_sha256=str(manifest["protocol_sha256"]),
        quality_court_sha256=str(manifest["quality_court_sha256"]),
        source_groups=tuple(normalized),
        output_examples=output_examples,
    )


def _training_config(payload: Any) -> dict[str, Any]:
    source = dict(payload or {})
    allowed = {
        "epochs": int,
        "learning_rate": float,
        "adaptation_task_weight": float,
        "retention_task_weight": float,
        "retention_distill_weight": float,
        "cortex_anchor_weight": float,
        "temperature": float,
        "max_grad_norm": float,
        "max_length": int,
    }
    result: dict[str, Any] = {}
    for key, caster in allowed.items():
        if key in source:
            result[key] = caster(source[key])
    if int(result.get("epochs", 2)) < 1:
        raise ValueError("cycle epochs must be >=1")
    if float(result.get("learning_rate", 2e-4)) <= 0:
        raise ValueError("cycle learning_rate must be positive")
    if int(result.get("max_length", 384)) < 32:
        raise ValueError("cycle max_length must be >=32")
    for key in (
        "adaptation_task_weight",
        "retention_task_weight",
        "retention_distill_weight",
        "cortex_anchor_weight",
    ):
        if key in result and float(result[key]) < 0:
            raise ValueError(f"cycle {key} must be non-negative")
    if "temperature" in result and float(result["temperature"]) <= 0:
        raise ValueError("cycle temperature must be positive")
    if "max_grad_norm" in result and float(result["max_grad_norm"]) <= 0:
        raise ValueError("cycle max_grad_norm must be positive")
    return result


def validate_longitudinal_plan(
    plan_path: str | Path,
) -> ValidatedLongitudinalPlan:
    path = Path(plan_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != PLAN_SCHEMA:
        raise ValueError("unsupported longitudinal plan schema")

    policy = LongitudinalExecutionPolicy(**dict(payload.get("policy", {})))
    policy.validate()
    cycles_payload = payload.get("cycles")
    if not isinstance(cycles_payload, list) or len(cycles_payload) < policy.min_cycles:
        raise ValueError(
            f"need at least {policy.min_cycles} real longitudinal cycles"
        )

    base = path.parent
    initial_factorized = _resolve(
        base,
        payload.get("initial_factorized"),
        field="initial_factorized",
    )
    latent = _resolve(base, payload.get("latent"), field="latent")
    tokenizer = _resolve(base, payload.get("tokenizer"), field="tokenizer")
    if not initial_factorized.is_file():
        raise FileNotFoundError(initial_factorized)
    if not latent.is_file():
        raise FileNotFoundError(latent)
    if not tokenizer.exists():
        raise FileNotFoundError(tokenizer)

    device = str(payload.get("device", "cpu"))
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")

    fixed_panel = _pack_binding(
        _resolve(
            base,
            payload.get("fixed_panel_manifest"),
            field="fixed_panel_manifest",
        )
    )

    cycles: list[LongitudinalCycleBinding] = []
    adaptation_protocols: set[str] = set()
    adaptation_groups_seen: set[str] = set()
    all_training_groups: set[str] = set()

    for index, raw in enumerate(cycles_payload, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"cycle {index} must be an object")
        retention = _pack_binding(
            _resolve(
                base,
                raw.get("retention_manifest"),
                field=f"cycles[{index}].retention_manifest",
            )
        )
        adaptation = _pack_binding(
            _resolve(
                base,
                raw.get("adaptation_manifest"),
                field=f"cycles[{index}].adaptation_manifest",
            )
        )
        retention_groups = set(retention.source_groups)
        adaptation_groups = set(adaptation.source_groups)
        overlap = retention_groups & adaptation_groups
        if overlap:
            raise ValueError(
                f"cycle {index} retention/adaptation source groups overlap"
            )
        if (
            policy.require_unique_adaptation_protocols
            and adaptation.protocol_sha256 in adaptation_protocols
        ):
            raise ValueError("adaptation protocol reused across longitudinal cycles")
        if (
            policy.require_disjoint_adaptation_groups
            and adaptation_groups & adaptation_groups_seen
        ):
            raise ValueError(
                "adaptation source groups reused across longitudinal cycles"
            )
        adaptation_protocols.add(adaptation.protocol_sha256)
        adaptation_groups_seen.update(adaptation_groups)
        all_training_groups.update(retention_groups)
        all_training_groups.update(adaptation_groups)
        cycles.append(
            LongitudinalCycleBinding(
                index=index,
                retention=retention,
                adaptation=adaptation,
                training=_training_config(raw.get("training")),
            )
        )

    fixed_groups = set(fixed_panel.source_groups)
    if (
        policy.require_fixed_panel_isolation
        and fixed_groups & all_training_groups
    ):
        raise ValueError(
            "fixed long-horizon panel overlaps longitudinal training source groups"
        )

    public_cycles = [
        {
            "cycle": cycle.index,
            "retention": cycle.retention.public_receipt(),
            "adaptation": cycle.adaptation.public_receipt(),
            "training": dict(cycle.training),
        }
        for cycle in cycles
    ]
    receipt = {
        "schema": PLAN_RECEIPT_SCHEMA,
        "authority": AUTHORITY,
        "policy": asdict(policy),
        "device": device,
        "cycles": len(cycles),
        "fixed_panel": fixed_panel.public_receipt(),
        "cycle_bindings": public_cycles,
        "privacy": {
            "contains_raw_prompt_target": False,
            "contains_raw_source_group_hash_values": False,
            "contains_local_paths": False,
        },
    }
    receipt["plan_sha256"] = payload_digest(receipt)

    return ValidatedLongitudinalPlan(
        path=path,
        initial_factorized=initial_factorized,
        latent=latent,
        tokenizer=tokenizer,
        device=device,
        fixed_panel=fixed_panel,
        cycles=cycles,
        policy=policy,
        receipt=receipt,
    )


def build_longitudinal_report(
    *,
    plan_receipt: dict[str, Any],
    cycle_receipts: list[dict[str, Any]],
    fixed_panel_receipt: dict[str, Any],
    learned_window_receipts: list[dict[str, Any]],
    unified_chain: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    if fixed_panel_receipt.get("status") != "PASS":
        reasons.append("fixed_long_horizon_panel_failed")
    if unified_chain.get("status") != "PASS":
        reasons.append("unified_continual_chain_failed")
    if len(cycle_receipts) < 5:
        reasons.append("insufficient_real_cycles")
    expected_learned_courts = max(0, len(cycle_receipts) - 1)
    if len(learned_window_receipts) != expected_learned_courts:
        raise ValueError(
            "learned-window retention court count must equal cycles - 1"
        )
    for index, receipt in enumerate(learned_window_receipts, start=1):
        if receipt.get("status") != "PASS":
            reasons.append(f"learned_window_{index:03d}_forgotten")

    rows: list[dict[str, Any]] = []
    for index, cycle in enumerate(cycle_receipts, start=1):
        training = cycle["training"]
        continual = training["continual_learning"]
        retention = (
            learned_window_receipts[index - 1]
            if index <= len(learned_window_receipts)
            else None
        )
        rows.append(
            {
                "cycle": index,
                "artifact_checkpoint_sha256": cycle["artifact"][
                    "checkpoint_sha256"
                ],
                "adaptation_protocol_sha256": training["lineage"][
                    "adaptation_protocol_sha256"
                ],
                "cycle_adaptation_mean_gain": continual["adaptation"]["summary"][
                    "mean_group_gain"
                ],
                "cycle_retention_worst_group_regression": continual[
                    "retention"
                ]["summary"]["worst_group_regression"],
                "future_cycles_observed": len(cycle_receipts) - index,
                "final_learned_window_overall_regression": (
                    retention["group_robustness"]["summary"][
                        "overall_regression"
                    ]
                    if retention is not None
                    else 0.0
                ),
                "final_learned_window_worst_group_regression": (
                    retention["group_robustness"]["summary"][
                        "worst_group_regression"
                    ]
                    if retention is not None
                    else 0.0
                ),
                "learned_window_court_sha256": (
                    retention["court_sha256"]
                    if retention is not None
                    else None
                ),
            }
        )

    report = {
        "schema": REPORT_SCHEMA,
        "authority": AUTHORITY,
        "status": "PASS" if not reasons else "BLOCKED",
        "reasons": reasons,
        "plan_sha256": plan_receipt["plan_sha256"],
        "cycles": len(cycle_receipts),
        "initial_checkpoint_sha256": unified_chain[
            "first_parent_checkpoint_sha256"
        ],
        "final_checkpoint_sha256": unified_chain[
            "final_artifact_checkpoint_sha256"
        ],
        "fixed_panel_court_sha256": fixed_panel_receipt["court_sha256"],
        "unified_chain_sha256": unified_chain["chain_sha256"],
        "cycle_rows": rows,
        "privacy": {
            "contains_raw_prompt_target": False,
            "contains_raw_source_group_hash_values": False,
            "contains_local_paths": False,
        },
    }
    report["report_sha256"] = payload_digest(report)
    return report


def verify_longitudinal_report_digest(
    report: dict[str, Any],
) -> dict[str, Any]:
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError("unsupported longitudinal report schema")
    supplied = report.get("report_sha256")
    body = dict(report)
    body.pop("report_sha256", None)
    if payload_digest(body) != supplied:
        raise ValueError("longitudinal report digest mismatch")
    if report.get("status") not in {"PASS", "BLOCKED"}:
        raise ValueError("longitudinal report status invalid")
    return report
