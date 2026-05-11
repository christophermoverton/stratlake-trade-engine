from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

from src.research.registry import canonicalize_value

from .models import RobustnessFinding, WalkForwardEfficiencyRow

WFE_CHECK_PREFIX = "walk_forward_efficiency"
WFE_STATUSES: tuple[str, ...] = ("robust", "acceptable", "weak", "broken", "undefined", "missing")


@dataclass(frozen=True)
class WalkForwardEfficiencyThresholds:
    robust_min: float = 0.75
    acceptable_min: float = 0.50
    weak_min: float = 0.00
    near_zero_is_sharpe: float = 1e-12
    severity_by_status: Mapping[str, str] = field(
        default_factory=lambda: {
            "robust": "info",
            "acceptable": "info",
            "weak": "warning",
            "broken": "needs_review",
            "undefined": "needs_review",
            "missing": "needs_review",
        }
    )

    def __post_init__(self) -> None:
        if not (self.weak_min <= self.acceptable_min <= self.robust_min):
            raise ValueError("WFE thresholds must satisfy weak_min <= acceptable_min <= robust_min.")
        if self.near_zero_is_sharpe < 0:
            raise ValueError("near_zero_is_sharpe must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        return canonicalize_value(
            {
                "acceptable_min": self.acceptable_min,
                "near_zero_is_sharpe": self.near_zero_is_sharpe,
                "robust_min": self.robust_min,
                "weak_min": self.weak_min,
            }
        )

    def severity_for(self, status: str) -> str:
        return str(self.severity_by_status.get(status, "needs_review"))


@dataclass(frozen=True)
class WalkForwardEfficiencyInput:
    workflow_type: str
    run_id: str
    split_id: str
    sharpe_is: float | int | str | None = None
    sharpe_oos: float | int | str | None = None
    train_start: str = ""
    train_end: str = ""
    test_start: str = ""
    test_end: str = ""
    n_trades_is: int | None = None
    n_trades_oos: int | None = None
    source_run_id: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WalkForwardEfficiencyResult:
    wfe: float | None
    status: str
    reason: str
    sharpe_is: float | None
    sharpe_oos: float | None


def compute_walk_forward_efficiency(
    sharpe_is: Any,
    sharpe_oos: Any,
    *,
    thresholds: WalkForwardEfficiencyThresholds | None = None,
) -> WalkForwardEfficiencyResult:
    resolved = thresholds or WalkForwardEfficiencyThresholds()
    normalized_is = _finite_float_or_none(sharpe_is)
    normalized_oos = _finite_float_or_none(sharpe_oos)
    if normalized_is is None:
        return WalkForwardEfficiencyResult(None, "missing" if sharpe_is is None else "undefined", _missing_reason("sharpe_is", sharpe_is), normalized_is, normalized_oos)
    if normalized_oos is None:
        return WalkForwardEfficiencyResult(None, "missing" if sharpe_oos is None else "undefined", _missing_reason("sharpe_oos", sharpe_oos), normalized_is, normalized_oos)
    if abs(normalized_is) <= resolved.near_zero_is_sharpe:
        return WalkForwardEfficiencyResult(None, "undefined", "near_zero_in_sample_sharpe", normalized_is, normalized_oos)

    wfe = normalized_oos / normalized_is
    if not math.isfinite(wfe):
        return WalkForwardEfficiencyResult(None, "undefined", "non_finite_walk_forward_efficiency", normalized_is, normalized_oos)
    if normalized_is < 0:
        return WalkForwardEfficiencyResult(_stable_float(wfe), "broken", "negative_in_sample_sharpe", normalized_is, normalized_oos)
    status = classify_walk_forward_efficiency(wfe, thresholds=resolved)
    return WalkForwardEfficiencyResult(_stable_float(wfe), status, f"{status}_threshold", normalized_is, normalized_oos)


def classify_walk_forward_efficiency(
    wfe: Any,
    *,
    thresholds: WalkForwardEfficiencyThresholds | None = None,
) -> str:
    resolved = thresholds or WalkForwardEfficiencyThresholds()
    normalized = _finite_float_or_none(wfe)
    if normalized is None:
        return "undefined"
    if normalized >= resolved.robust_min:
        return "robust"
    if normalized >= resolved.acceptable_min:
        return "acceptable"
    if normalized >= resolved.weak_min:
        return "weak"
    return "broken"


def build_walk_forward_efficiency_rows(
    records: Sequence[WalkForwardEfficiencyInput | Mapping[str, Any]],
    *,
    thresholds: WalkForwardEfficiencyThresholds | None = None,
) -> list[WalkForwardEfficiencyRow]:
    resolved = thresholds or WalkForwardEfficiencyThresholds()
    rows = [_build_row(_coerce_input(record), thresholds=resolved) for record in records]
    return sorted(rows, key=lambda row: (row.workflow_type, row.run_id, row.split_id))


def build_walk_forward_efficiency_findings(
    records: Sequence[WalkForwardEfficiencyInput | Mapping[str, Any]],
    *,
    thresholds: WalkForwardEfficiencyThresholds | None = None,
    include_info: bool = True,
) -> list[RobustnessFinding]:
    resolved = thresholds or WalkForwardEfficiencyThresholds()
    findings: list[RobustnessFinding] = []
    for record in sorted((_coerce_input(item) for item in records), key=lambda item: (item.workflow_type, item.run_id, item.split_id)):
        result = compute_walk_forward_efficiency(record.sharpe_is, record.sharpe_oos, thresholds=resolved)
        if not include_info and result.status in {"robust", "acceptable"}:
            continue
        findings.append(
            RobustnessFinding(
                check_id=f"{WFE_CHECK_PREFIX}.{result.status}",
                severity=resolved.severity_for(result.status),
                workflow_type=record.workflow_type,
                run_id=record.run_id,
                message=_finding_message(result.status, record.split_id),
                details=_details_payload(record, result, thresholds=resolved),
            )
        )
    return findings


def build_walk_forward_efficiency_evidence(
    records: Sequence[WalkForwardEfficiencyInput | Mapping[str, Any]],
    *,
    thresholds: WalkForwardEfficiencyThresholds | None = None,
    include_info_findings: bool = True,
) -> tuple[list[WalkForwardEfficiencyRow], list[RobustnessFinding]]:
    resolved = thresholds or WalkForwardEfficiencyThresholds()
    return (
        build_walk_forward_efficiency_rows(records, thresholds=resolved),
        build_walk_forward_efficiency_findings(records, thresholds=resolved, include_info=include_info_findings),
    )


def _build_row(record: WalkForwardEfficiencyInput, *, thresholds: WalkForwardEfficiencyThresholds) -> WalkForwardEfficiencyRow:
    result = compute_walk_forward_efficiency(record.sharpe_is, record.sharpe_oos, thresholds=thresholds)
    return WalkForwardEfficiencyRow(
        workflow_type=record.workflow_type,
        run_id=record.run_id,
        split_id=record.split_id,
        train_period=_period(record.train_start, record.train_end),
        test_period=_period(record.test_start, record.test_end),
        in_sample_metric=result.sharpe_is,
        out_of_sample_metric=result.sharpe_oos,
        walk_forward_efficiency=result.wfe,
        status=result.status,
        details=_details_payload(record, result, thresholds=thresholds),
    )


def _details_payload(
    record: WalkForwardEfficiencyInput,
    result: WalkForwardEfficiencyResult,
    *,
    thresholds: WalkForwardEfficiencyThresholds,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "metric": "sharpe_ratio",
        "reason": result.reason,
        "sharpe_is": result.sharpe_is,
        "sharpe_oos": result.sharpe_oos,
        "split_id": record.split_id,
        "thresholds": thresholds.to_dict(),
        "wfe": result.wfe,
        "wfe_status": result.status,
    }
    optional = {
        "n_trades_is": record.n_trades_is,
        "n_trades_oos": record.n_trades_oos,
        "source_run_id": record.source_run_id,
        "test_end": record.test_end,
        "test_start": record.test_start,
        "train_end": record.train_end,
        "train_start": record.train_start,
    }
    payload.update({key: value for key, value in optional.items() if value not in {None, ""}})
    payload.update(dict(record.details))
    return canonicalize_value(_drop_none(payload))


def _coerce_input(record: WalkForwardEfficiencyInput | Mapping[str, Any]) -> WalkForwardEfficiencyInput:
    if isinstance(record, WalkForwardEfficiencyInput):
        return record
    return WalkForwardEfficiencyInput(
        workflow_type=str(record.get("workflow_type", "unknown")),
        run_id=str(record.get("run_id") or record.get("source_run_id") or "unknown"),
        split_id=str(record.get("split_id", "")),
        sharpe_is=_first_present(record, "sharpe_is", "in_sample_sharpe", "in_sample_metric"),
        sharpe_oos=_first_present(record, "sharpe_oos", "out_of_sample_sharpe", "out_of_sample_metric"),
        train_start=str(record.get("train_start", "")),
        train_end=str(record.get("train_end", "")),
        test_start=str(record.get("test_start", "")),
        test_end=str(record.get("test_end", "")),
        n_trades_is=_optional_int(_first_present(record, "n_trades_is", "trade_count_is", "in_sample_trade_count")),
        n_trades_oos=_optional_int(_first_present(record, "n_trades_oos", "trade_count_oos", "out_of_sample_trade_count", "trade_count")),
        source_run_id=None if record.get("source_run_id") is None else str(record.get("source_run_id")),
        details=dict(record.get("details", {})) if isinstance(record.get("details", {}), Mapping) else {},
    )


def _first_present(record: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    return None


def _finite_float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(normalized):
        return None
    return _stable_float(normalized)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _stable_float(value: float) -> float:
    return float(round(value, 12))


def _period(start: str, end: str) -> str:
    if start and end:
        return f"{start}/{end}"
    return start or end


def _drop_none(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}


def _missing_reason(metric_name: str, raw_value: Any) -> str:
    if raw_value is None or raw_value == "":
        return f"missing_{metric_name}"
    return f"non_finite_{metric_name}"


def _finding_message(status: str, split_id: str) -> str:
    return {
        "robust": f"Walk-forward efficiency is robust for split '{split_id}'.",
        "acceptable": f"Walk-forward efficiency is acceptable for split '{split_id}'.",
        "weak": f"Walk-forward efficiency is weak for split '{split_id}'.",
        "broken": f"Walk-forward efficiency is broken for split '{split_id}'.",
        "undefined": f"Walk-forward efficiency is undefined for split '{split_id}'.",
        "missing": f"Walk-forward efficiency is missing required metrics for split '{split_id}'.",
    }[status]


__all__ = [
    "WFE_CHECK_PREFIX",
    "WFE_STATUSES",
    "WalkForwardEfficiencyInput",
    "WalkForwardEfficiencyResult",
    "WalkForwardEfficiencyThresholds",
    "build_walk_forward_efficiency_evidence",
    "build_walk_forward_efficiency_findings",
    "build_walk_forward_efficiency_rows",
    "classify_walk_forward_efficiency",
    "compute_walk_forward_efficiency",
]
