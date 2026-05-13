from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

from src.research.registry import canonicalize_value

from .models import RobustnessFinding, SampleSizeValidation

SAMPLE_SIZE_CHECK_PREFIX = "sample_size"
SAMPLE_SIZE_STATUSES: tuple[str, ...] = ("pass", "warning", "needs_review", "missing")


@dataclass(frozen=True)
class SampleSizeThresholds:
    minimum_total_samples: int = 252
    minimum_total_trades: int = 30
    minimum_oos_trades: int = 10
    minimum_trades_per_split: int = 5
    minimum_unique_periods: int = 30
    minimum_regime_count: int = 2
    minimum_trades_per_regime: int = 5
    missing_metadata_severity: str = "needs_review"
    thin_sample_severity: str = "warning"
    thin_trade_count_severity: str = "needs_review"

    def __post_init__(self) -> None:
        for field_name, value in self.to_dict(include_severity=False).items():
            if int(value) < 0:
                raise ValueError(f"{field_name} must be non-negative.")

    def to_dict(self, *, include_severity: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "minimum_oos_trades": self.minimum_oos_trades,
            "minimum_regime_count": self.minimum_regime_count,
            "minimum_total_samples": self.minimum_total_samples,
            "minimum_total_trades": self.minimum_total_trades,
            "minimum_trades_per_regime": self.minimum_trades_per_regime,
            "minimum_trades_per_split": self.minimum_trades_per_split,
            "minimum_unique_periods": self.minimum_unique_periods,
        }
        if include_severity:
            payload.update(
                {
                    "missing_metadata_severity": self.missing_metadata_severity,
                    "thin_sample_severity": self.thin_sample_severity,
                    "thin_trade_count_severity": self.thin_trade_count_severity,
                }
            )
        return canonicalize_value(payload)

    def severity_for(self, check_id: str, status: str) -> str:
        if status == "pass":
            return "info"
        if status == "missing" or ".missing_" in check_id:
            return self.missing_metadata_severity
        if check_id in {
            "sample_size.minimum_total_samples",
            "sample_size.minimum_unique_periods",
            "sample_size.minimum_regime_coverage",
        }:
            return self.thin_sample_severity
        return self.thin_trade_count_severity


@dataclass(frozen=True)
class SampleSizeInput:
    workflow_type: str
    run_id: str
    source_run_id: str | None = None
    sample_count: int | float | str | None = None
    trade_count: int | float | str | None = None
    oos_trade_count: int | float | str | None = None
    split_id: str | None = None
    split_trade_count: int | float | str | None = None
    split_trade_counts: Mapping[str, int | float | str | None] | None = None
    unique_period_count: int | float | str | None = None
    regime_trade_counts: Mapping[str, int | float | str | None] | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


def evaluate_sample_size_guardrails(
    record: SampleSizeInput | Mapping[str, Any],
    *,
    thresholds: SampleSizeThresholds | None = None,
) -> list[SampleSizeValidation]:
    resolved = thresholds or SampleSizeThresholds()
    sample = _coerce_input(record)
    validations: list[SampleSizeValidation] = []

    validations.append(
        _validation(
            sample,
            check_id="sample_size.minimum_total_samples",
            observed=sample.sample_count,
            required=resolved.minimum_total_samples,
            observed_kind="sample_count",
            threshold_kind="minimum_sample_count",
            thresholds=resolved,
        )
    )
    validations.append(
        _validation(
            sample,
            check_id="sample_size.minimum_total_trades",
            observed=sample.trade_count,
            required=resolved.minimum_total_trades,
            observed_kind="trade_count",
            threshold_kind="minimum_trade_count",
            thresholds=resolved,
        )
    )
    validations.append(
        _validation(
            sample,
            check_id="sample_size.minimum_oos_trades",
            observed=sample.oos_trade_count,
            required=resolved.minimum_oos_trades,
            observed_kind="trade_count",
            threshold_kind="minimum_trade_count",
            thresholds=resolved,
            extra_details={"sample_scope": "out_of_sample"},
        )
    )
    validations.append(
        _validation(
            sample,
            check_id="sample_size.minimum_unique_periods",
            observed=sample.unique_period_count,
            required=resolved.minimum_unique_periods,
            observed_kind="sample_count",
            threshold_kind="minimum_sample_count",
            thresholds=resolved,
            extra_details={"sample_scope": "unique_periods"},
        )
    )

    for split_id, split_trade_count in _split_trade_counts(sample).items():
        validations.append(
            _validation(
                sample,
                check_id="sample_size.minimum_trades_per_split",
                observed=split_trade_count,
                required=resolved.minimum_trades_per_split,
                observed_kind="trade_count",
                threshold_kind="minimum_trade_count",
                thresholds=resolved,
                extra_details={"split_id": split_id},
            )
        )

    if sample.regime_trade_counts:
        regime_counts = {str(key): _optional_int(value) for key, value in sample.regime_trade_counts.items()}
        represented = sum(1 for value in regime_counts.values() if value is not None and value > 0)
        validations.append(
            _validation(
                sample,
                check_id="sample_size.minimum_regime_coverage",
                observed=represented,
                required=resolved.minimum_regime_count,
                observed_kind="sample_count",
                threshold_kind="minimum_sample_count",
                thresholds=resolved,
                extra_details={"regime_count": len(regime_counts), "sample_scope": "regime_coverage"},
            )
        )
        for regime_id, trade_count in sorted(regime_counts.items()):
            validations.append(
                _validation(
                    sample,
                    check_id="sample_size.minimum_trades_per_regime",
                    observed=trade_count,
                    required=resolved.minimum_trades_per_regime,
                    observed_kind="trade_count",
                    threshold_kind="minimum_trade_count",
                    thresholds=resolved,
                    extra_details={"regime_id": regime_id},
                )
            )

    return sorted(validations, key=_validation_sort_key)


def build_sample_size_validations(
    records: Sequence[SampleSizeInput | Mapping[str, Any]],
    *,
    thresholds: SampleSizeThresholds | None = None,
) -> list[SampleSizeValidation]:
    validations: list[SampleSizeValidation] = []
    for record in records:
        validations.extend(evaluate_sample_size_guardrails(record, thresholds=thresholds))
    return sorted(validations, key=_validation_sort_key)


def build_sample_size_findings(
    records: Sequence[SampleSizeInput | Mapping[str, Any]],
    *,
    thresholds: SampleSizeThresholds | None = None,
    include_info: bool = False,
) -> list[RobustnessFinding]:
    resolved = thresholds or SampleSizeThresholds()
    findings: list[RobustnessFinding] = []
    for validation in build_sample_size_validations(records, thresholds=resolved):
        if validation.status == "pass" and not include_info:
            continue
        details = dict(validation.details)
        findings.append(
            RobustnessFinding(
                check_id=validation.check_id,
                severity=resolved.severity_for(validation.check_id, validation.status),
                workflow_type=validation.workflow_type,
                run_id=validation.run_id,
                message=_finding_message(validation),
                details=details,
            )
        )
    return sorted(findings, key=lambda finding: (finding.workflow_type, finding.run_id, finding.check_id, str(finding.details.get("split_id", "")), str(finding.details.get("regime_id", ""))))


def build_sample_size_evidence(
    records: Sequence[SampleSizeInput | Mapping[str, Any]],
    *,
    thresholds: SampleSizeThresholds | None = None,
    include_info_findings: bool = False,
) -> tuple[list[SampleSizeValidation], list[RobustnessFinding]]:
    resolved = thresholds or SampleSizeThresholds()
    return (
        build_sample_size_validations(records, thresholds=resolved),
        build_sample_size_findings(records, thresholds=resolved, include_info=include_info_findings),
    )


def _validation(
    sample: SampleSizeInput,
    *,
    check_id: str,
    observed: Any,
    required: int,
    observed_kind: str,
    threshold_kind: str,
    thresholds: SampleSizeThresholds,
    extra_details: Mapping[str, Any] | None = None,
) -> SampleSizeValidation:
    normalized = _optional_int(observed)
    status = _status(normalized, required)
    resolved_check_id = _missing_check_id(check_id) if status == "missing" else check_id
    details = {
        "check_id": resolved_check_id,
        "configured_check_id": check_id,
        "observed_value": normalized,
        "reason": _reason(resolved_check_id, status),
        "required_threshold": required,
        "source_run_id": sample.source_run_id,
        "status": status,
        "thresholds": thresholds.to_dict(),
    }
    details.update(dict(extra_details or {}))
    details.update(dict(sample.details))
    return SampleSizeValidation(
        workflow_type=sample.workflow_type,
        run_id=sample.run_id,
        check_id=resolved_check_id,
        sample_count=normalized if observed_kind == "sample_count" else _optional_int(sample.sample_count),
        trade_count=normalized if observed_kind == "trade_count" else _optional_int(sample.trade_count),
        minimum_sample_count=required if threshold_kind == "minimum_sample_count" else None,
        minimum_trade_count=required if threshold_kind == "minimum_trade_count" else None,
        status=status,
        details=canonicalize_value(_drop_empty(details)),
    )


def _coerce_input(record: SampleSizeInput | Mapping[str, Any]) -> SampleSizeInput:
    if isinstance(record, SampleSizeInput):
        return record
    return SampleSizeInput(
        workflow_type=str(record.get("workflow_type", "unknown")),
        run_id=str(record.get("run_id") or record.get("source_run_id") or "unknown"),
        source_run_id=None if record.get("source_run_id") is None else str(record.get("source_run_id")),
        sample_count=_first_present(record, "sample_count", "total_samples", "observation_count", "row_count"),
        trade_count=_first_present(record, "trade_count", "total_trades"),
        oos_trade_count=_first_present(record, "oos_trade_count", "out_of_sample_trade_count", "test_trade_count"),
        split_id=None if record.get("split_id") is None else str(record.get("split_id")),
        split_trade_count=_first_present(record, "split_trade_count", "trades_per_split"),
        split_trade_counts=dict(record.get("split_trade_counts", {})) if isinstance(record.get("split_trade_counts", {}), Mapping) else None,
        unique_period_count=_first_present(record, "unique_period_count", "period_count", "unique_periods"),
        regime_trade_counts=dict(record.get("regime_trade_counts", {})) if isinstance(record.get("regime_trade_counts", {}), Mapping) else None,
        details=dict(record.get("details", {})) if isinstance(record.get("details", {}), Mapping) else {},
    )


def _split_trade_counts(sample: SampleSizeInput) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if sample.split_trade_counts:
        values.update({str(key): value for key, value in sample.split_trade_counts.items()})
    if sample.split_id is not None or sample.split_trade_count is not None:
        values[str(sample.split_id or "split")] = sample.split_trade_count
    return {key: values[key] for key in sorted(values)}


def _first_present(record: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    return None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(normalized):
        return None
    return int(normalized)


def _status(observed: int | None, required: int) -> str:
    if observed is None:
        return "missing"
    return "pass" if observed >= required else "needs_review"


def _reason(check_id: str, status: str) -> str:
    if status == "pass":
        return "meets_configured_threshold"
    if status == "missing":
        return f"missing_{check_id.rsplit('.', maxsplit=1)[-1]}"
    return f"below_{check_id.rsplit('.', maxsplit=1)[-1]}"


def _missing_check_id(check_id: str) -> str:
    return {
        "sample_size.minimum_total_samples": "sample_size.missing_sample_count",
        "sample_size.minimum_total_trades": "sample_size.missing_trade_count",
        "sample_size.minimum_oos_trades": "sample_size.missing_oos_trade_count",
        "sample_size.minimum_trades_per_split": "sample_size.missing_trade_count",
        "sample_size.minimum_unique_periods": "sample_size.missing_unique_period_count",
        "sample_size.minimum_trades_per_regime": "sample_size.missing_trade_count",
    }.get(check_id, check_id)


def _validation_sort_key(validation: SampleSizeValidation) -> tuple[str, str, str, str, str]:
    details = dict(validation.details)
    return (
        validation.workflow_type,
        validation.run_id,
        validation.check_id,
        str(details.get("split_id", "")),
        str(details.get("regime_id", "")),
    )


def _finding_message(validation: SampleSizeValidation) -> str:
    detail = dict(validation.details)
    suffix = ""
    if detail.get("split_id"):
        suffix = f" for split '{detail['split_id']}'"
    elif detail.get("regime_id"):
        suffix = f" for regime '{detail['regime_id']}'"
    if validation.status == "pass":
        return f"Sample-size guardrail '{validation.check_id}' passed{suffix}."
    if validation.status == "missing":
        return f"Sample-size guardrail '{validation.check_id}' is missing required metadata{suffix}."
    return f"Sample-size guardrail '{validation.check_id}' needs review{suffix}."


def _drop_empty(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None and item != ""}


__all__ = [
    "SAMPLE_SIZE_CHECK_PREFIX",
    "SAMPLE_SIZE_STATUSES",
    "SampleSizeInput",
    "SampleSizeThresholds",
    "build_sample_size_evidence",
    "build_sample_size_findings",
    "build_sample_size_validations",
    "evaluate_sample_size_guardrails",
]
