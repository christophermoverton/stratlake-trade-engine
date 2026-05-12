from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

from src.research.registry import canonicalize_value

from .models import RobustnessFinding, SensitivitySummaryRow

SENSITIVITY_CHECK_PREFIX = "sensitivity"
METRIC_TRANSFORM_ABSOLUTE_MAGNITUDE = "absolute_magnitude_lower_is_better"
SENSITIVITY_STATUSES: tuple[str, ...] = (
    "improved",
    "stable",
    "mildly_sensitive",
    "fragile",
    "undefined",
    "missing",
)


@dataclass(frozen=True)
class SensitivityThresholds:
    stable_relative_delta_max: float = 0.05
    mild_relative_delta_max: float = 0.15
    fragile_relative_delta_min: float = 0.25
    stable_absolute_delta_max: float = 0.02
    mild_absolute_delta_max: float = 0.10
    near_zero_base_metric: float = 1e-12
    default_higher_is_better: bool = True
    missing_metadata_severity: str = "needs_review"
    mild_sensitivity_severity: str = "warning"
    fragile_severity: str = "needs_review"
    undefined_severity: str = "needs_review"

    def __post_init__(self) -> None:
        if self.near_zero_base_metric < 0:
            raise ValueError("near_zero_base_metric must be non-negative.")
        if self.stable_relative_delta_max < 0:
            raise ValueError("stable_relative_delta_max must be non-negative.")
        if self.mild_relative_delta_max < self.stable_relative_delta_max:
            raise ValueError("mild_relative_delta_max must be >= stable_relative_delta_max.")
        if self.fragile_relative_delta_min < self.mild_relative_delta_max:
            raise ValueError("fragile_relative_delta_min must be >= mild_relative_delta_max.")
        if self.stable_absolute_delta_max < 0:
            raise ValueError("stable_absolute_delta_max must be non-negative.")
        if self.mild_absolute_delta_max < self.stable_absolute_delta_max:
            raise ValueError("mild_absolute_delta_max must be >= stable_absolute_delta_max.")

    def to_dict(self) -> dict[str, Any]:
        return canonicalize_value(
            {
                "default_higher_is_better": self.default_higher_is_better,
                "fragile_relative_delta_min": self.fragile_relative_delta_min,
                "mild_absolute_delta_max": self.mild_absolute_delta_max,
                "mild_relative_delta_max": self.mild_relative_delta_max,
                "near_zero_base_metric": self.near_zero_base_metric,
                "stable_absolute_delta_max": self.stable_absolute_delta_max,
                "stable_relative_delta_max": self.stable_relative_delta_max,
            }
        )

    def severity_for(self, status: str) -> str:
        if status in {"improved", "stable"}:
            return "info"
        if status == "mildly_sensitive":
            return self.mild_sensitivity_severity
        if status == "fragile":
            return self.fragile_severity
        if status == "missing":
            return self.missing_metadata_severity
        return self.undefined_severity


@dataclass(frozen=True)
class SensitivityInput:
    workflow_type: str
    run_id: str
    source_run_id: str | None = None
    parameter_name: str = ""
    base_value: Any = None
    perturbed_value: Any = None
    metric_name: str = ""
    base_metric_value: float | int | str | None = None
    perturbed_metric_value: float | int | str | None = None
    higher_is_better: bool | None = None
    scenario_id: str = ""
    perturbation_type: str = ""
    perturbation_size: float | int | str | None = None
    metric_transform: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SensitivityEvaluation:
    status: str
    reason: str
    absolute_delta: float | None
    deterioration: float | None
    relative_deterioration: float | None
    base_metric_value: float | None
    perturbed_metric_value: float | None
    parameter_distance: float | None
    normalized_parameter_distance: float | None
    higher_is_better: bool


def evaluate_parameter_sensitivity(
    record: SensitivityInput | Mapping[str, Any],
    *,
    thresholds: SensitivityThresholds | None = None,
) -> SensitivityEvaluation:
    resolved = thresholds or SensitivityThresholds()
    normalized = _coerce_input(record)
    higher_is_better = (
        resolved.default_higher_is_better
        if normalized.higher_is_better is None
        else bool(normalized.higher_is_better)
    )

    if not normalized.parameter_name:
        return SensitivityEvaluation(
            status="missing",
            reason="missing_parameter_name",
            absolute_delta=None,
            deterioration=None,
            relative_deterioration=None,
            base_metric_value=_finite_float_or_none(normalized.base_metric_value),
            perturbed_metric_value=_finite_float_or_none(normalized.perturbed_metric_value),
            parameter_distance=None,
            normalized_parameter_distance=None,
            higher_is_better=higher_is_better,
        )
    if not normalized.metric_name:
        return SensitivityEvaluation(
            status="missing",
            reason="missing_metric_name",
            absolute_delta=None,
            deterioration=None,
            relative_deterioration=None,
            base_metric_value=_finite_float_or_none(normalized.base_metric_value),
            perturbed_metric_value=_finite_float_or_none(normalized.perturbed_metric_value),
            parameter_distance=None,
            normalized_parameter_distance=None,
            higher_is_better=higher_is_better,
        )

    base_metric = _finite_float_or_none(normalized.base_metric_value)
    perturbed_metric = _finite_float_or_none(normalized.perturbed_metric_value)
    if base_metric is None:
        return SensitivityEvaluation(
            status="missing" if _is_missing(normalized.base_metric_value) else "undefined",
            reason="missing_base_metric_value"
            if _is_missing(normalized.base_metric_value)
            else "non_finite_base_metric_value",
            absolute_delta=None,
            deterioration=None,
            relative_deterioration=None,
            base_metric_value=None,
            perturbed_metric_value=perturbed_metric,
            parameter_distance=_parameter_distance(normalized.base_value, normalized.perturbed_value),
            normalized_parameter_distance=_normalized_parameter_distance(normalized.base_value, normalized.perturbed_value, epsilon=resolved.near_zero_base_metric),
            higher_is_better=higher_is_better,
        )
    if perturbed_metric is None:
        return SensitivityEvaluation(
            status="missing" if _is_missing(normalized.perturbed_metric_value) else "undefined",
            reason="missing_perturbed_metric_value"
            if _is_missing(normalized.perturbed_metric_value)
            else "non_finite_perturbed_metric_value",
            absolute_delta=None,
            deterioration=None,
            relative_deterioration=None,
            base_metric_value=base_metric,
            perturbed_metric_value=None,
            parameter_distance=_parameter_distance(normalized.base_value, normalized.perturbed_value),
            normalized_parameter_distance=_normalized_parameter_distance(normalized.base_value, normalized.perturbed_value, epsilon=resolved.near_zero_base_metric),
            higher_is_better=higher_is_better,
        )

    absolute_delta = _stable_float(perturbed_metric - base_metric)
    parameter_distance = _parameter_distance(normalized.base_value, normalized.perturbed_value)
    normalized_distance = _normalized_parameter_distance(
        normalized.base_value,
        normalized.perturbed_value,
        epsilon=resolved.near_zero_base_metric,
    )

    metric_transform = (normalized.metric_transform or "").strip()
    if metric_transform == METRIC_TRANSFORM_ABSOLUTE_MAGNITUDE:
        base_for_cmp = abs(base_metric)
        perturbed_for_cmp = abs(perturbed_metric)
        deterioration = _stable_float(perturbed_for_cmp - base_for_cmp)
        if base_for_cmp <= resolved.near_zero_base_metric:
            return SensitivityEvaluation(
                status="undefined",
                reason="near_zero_base_metric",
                absolute_delta=absolute_delta,
                deterioration=deterioration,
                relative_deterioration=None,
                base_metric_value=base_metric,
                perturbed_metric_value=perturbed_metric,
                parameter_distance=parameter_distance,
                normalized_parameter_distance=normalized_distance,
                higher_is_better=higher_is_better,
            )
        relative_deterioration = _stable_float(deterioration / base_for_cmp)
        status = classify_fragility(
            deterioration=deterioration,
            relative_deterioration=relative_deterioration,
            thresholds=resolved,
        )
        return SensitivityEvaluation(
            status=status,
            reason=f"{status}_threshold",
            absolute_delta=absolute_delta,
            deterioration=deterioration,
            relative_deterioration=relative_deterioration,
            base_metric_value=base_metric,
            perturbed_metric_value=perturbed_metric,
            parameter_distance=parameter_distance,
            normalized_parameter_distance=normalized_distance,
            higher_is_better=higher_is_better,
        )

    deterioration = (
        _stable_float(base_metric - perturbed_metric)
        if higher_is_better
        else _stable_float(perturbed_metric - base_metric)
    )

    if abs(base_metric) <= resolved.near_zero_base_metric:
        return SensitivityEvaluation(
            status="undefined",
            reason="near_zero_base_metric",
            absolute_delta=absolute_delta,
            deterioration=deterioration,
            relative_deterioration=None,
            base_metric_value=base_metric,
            perturbed_metric_value=perturbed_metric,
            parameter_distance=parameter_distance,
            normalized_parameter_distance=normalized_distance,
            higher_is_better=higher_is_better,
        )

    relative_deterioration = _stable_float(deterioration / abs(base_metric))
    status = classify_fragility(
        deterioration=deterioration,
        relative_deterioration=relative_deterioration,
        thresholds=resolved,
    )
    return SensitivityEvaluation(
        status=status,
        reason=f"{status}_threshold",
        absolute_delta=absolute_delta,
        deterioration=deterioration,
        relative_deterioration=relative_deterioration,
        base_metric_value=base_metric,
        perturbed_metric_value=perturbed_metric,
        parameter_distance=parameter_distance,
        normalized_parameter_distance=normalized_distance,
        higher_is_better=higher_is_better,
    )


def classify_fragility(
    *,
    deterioration: float | None,
    relative_deterioration: float | None,
    thresholds: SensitivityThresholds | None = None,
) -> str:
    resolved = thresholds or SensitivityThresholds()
    if deterioration is None:
        return "undefined"
    if deterioration < 0:
        return "improved"

    absolute_level = _classify_absolute(deterioration, thresholds=resolved)
    relative_level = _classify_relative(relative_deterioration, thresholds=resolved)

    if "fragile" in {absolute_level, relative_level}:
        return "fragile"
    if "mildly_sensitive" in {absolute_level, relative_level}:
        return "mildly_sensitive"
    return "stable"


def build_sensitivity_summary_rows(
    records: Sequence[SensitivityInput | Mapping[str, Any]],
    *,
    thresholds: SensitivityThresholds | None = None,
) -> list[SensitivitySummaryRow]:
    resolved = thresholds or SensitivityThresholds()
    rows: list[SensitivitySummaryRow] = []
    for raw in records:
        record = _coerce_input(raw)
        evaluation = evaluate_parameter_sensitivity(record, thresholds=resolved)
        rows.append(
            SensitivitySummaryRow(
                workflow_type=record.workflow_type,
                run_id=record.run_id,
                scenario_id=record.scenario_id,
                parameter=record.parameter_name,
                baseline_value=record.base_value,
                scenario_value=record.perturbed_value,
                metric=record.metric_name,
                baseline_metric_value=evaluation.base_metric_value,
                scenario_metric_value=evaluation.perturbed_metric_value,
                delta=evaluation.absolute_delta,
                status=evaluation.status,
                details=_details_payload(record, evaluation, thresholds=resolved),
            )
        )
    return sorted(rows, key=_summary_row_sort_key)


def build_sensitivity_findings(
    records: Sequence[SensitivityInput | Mapping[str, Any]],
    *,
    thresholds: SensitivityThresholds | None = None,
    include_info: bool = False,
) -> list[RobustnessFinding]:
    resolved = thresholds or SensitivityThresholds()
    findings: list[RobustnessFinding] = []
    for raw in sorted((_coerce_input(item) for item in records), key=_input_sort_key):
        evaluation = evaluate_parameter_sensitivity(raw, thresholds=resolved)
        if not include_info and evaluation.status in {"improved", "stable"}:
            continue
        findings.append(
            RobustnessFinding(
                check_id=f"{SENSITIVITY_CHECK_PREFIX}.{evaluation.status}",
                severity=resolved.severity_for(evaluation.status),
                workflow_type=raw.workflow_type,
                run_id=raw.run_id,
                message=_finding_message(raw, evaluation.status),
                details=_details_payload(raw, evaluation, thresholds=resolved),
            )
        )
    return sorted(
        findings,
        key=lambda finding: (
            finding.workflow_type,
            finding.run_id,
            finding.check_id,
            str(finding.details.get("parameter_name", "")),
            str(finding.details.get("scenario_id", "")),
            str(finding.details.get("metric_name", "")),
            str(finding.details.get("perturbed_value", "")),
        ),
    )


def build_sensitivity_evidence(
    records: Sequence[SensitivityInput | Mapping[str, Any]],
    *,
    thresholds: SensitivityThresholds | None = None,
    include_info_findings: bool = False,
) -> tuple[list[SensitivitySummaryRow], list[RobustnessFinding]]:
    resolved = thresholds or SensitivityThresholds()
    return (
        build_sensitivity_summary_rows(records, thresholds=resolved),
        build_sensitivity_findings(records, thresholds=resolved, include_info=include_info_findings),
    )


def _coerce_input(record: SensitivityInput | Mapping[str, Any]) -> SensitivityInput:
    if isinstance(record, SensitivityInput):
        return record
    details = dict(record.get("details", {})) if isinstance(record.get("details", {}), Mapping) else {}
    return SensitivityInput(
        workflow_type=str(record.get("workflow_type", "unknown")),
        run_id=str(record.get("run_id") or record.get("source_run_id") or "unknown"),
        source_run_id=None if record.get("source_run_id") is None else str(record.get("source_run_id")),
        parameter_name=str(
            record.get("parameter_name")
            or record.get("parameter")
            or details.get("parameter_name")
            or ""
        ).strip(),
        base_value=_first_present(record, "base_value", "baseline_value", "base_parameter_value"),
        perturbed_value=_first_present(record, "perturbed_value", "scenario_value", "parameter_value"),
        metric_name=str(record.get("metric_name") or record.get("metric") or "").strip(),
        base_metric_value=_first_present(
            record,
            "base_metric_value",
            "baseline_metric_value",
            "metric_base_value",
        ),
        perturbed_metric_value=_first_present(
            record,
            "perturbed_metric_value",
            "scenario_metric_value",
            "metric_perturbed_value",
        ),
        higher_is_better=_optional_bool(record.get("higher_is_better")),
        scenario_id=str(record.get("scenario_id") or ""),
        perturbation_type=str(record.get("perturbation_type") or ""),
        perturbation_size=_first_present(record, "perturbation_size", "distance", "step_size"),
        metric_transform=str(record.get("metric_transform") or "").strip(),
        details=details,
    )


def _details_payload(
    record: SensitivityInput,
    evaluation: SensitivityEvaluation,
    *,
    thresholds: SensitivityThresholds,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "absolute_delta": evaluation.absolute_delta,
        "base_metric_value": evaluation.base_metric_value,
        "base_value": record.base_value,
        "deterioration": evaluation.deterioration,
        "fragility_status": evaluation.status,
        "higher_is_better": evaluation.higher_is_better,
        "metric_name": record.metric_name,
        "normalized_parameter_distance": evaluation.normalized_parameter_distance,
        "parameter_distance": evaluation.parameter_distance,
        "parameter_name": record.parameter_name,
        "perturbation_size": _finite_float_or_none(record.perturbation_size),
        "metric_transform": record.metric_transform or None,
        "perturbation_type": record.perturbation_type,
        "perturbed_metric_value": evaluation.perturbed_metric_value,
        "perturbed_value": record.perturbed_value,
        "reason": evaluation.reason,
        "relative_delta": evaluation.relative_deterioration,
        "relative_deterioration": evaluation.relative_deterioration,
        "scenario_id": record.scenario_id,
        "source_run_id": record.source_run_id,
        "thresholds": thresholds.to_dict(),
    }
    payload.update(dict(record.details))
    return canonicalize_value(_drop_none(payload))


def _summary_row_sort_key(row: SensitivitySummaryRow) -> tuple[str, str, str, str, str, str]:
    details = dict(row.details)
    return (
        row.workflow_type,
        row.run_id,
        row.parameter,
        row.scenario_id,
        row.metric,
        str(details.get("perturbed_value", row.scenario_value)),
    )


def _input_sort_key(record: SensitivityInput) -> tuple[str, str, str, str, str, str]:
    return (
        record.workflow_type,
        record.run_id,
        record.parameter_name,
        record.scenario_id,
        record.metric_name,
        str(record.perturbed_value),
    )


def _classify_relative(value: float | None, *, thresholds: SensitivityThresholds) -> str:
    if value is None:
        return "stable"
    if value >= thresholds.fragile_relative_delta_min:
        return "fragile"
    if value <= thresholds.stable_relative_delta_max:
        return "stable"
    return "mildly_sensitive"


def _classify_absolute(value: float, *, thresholds: SensitivityThresholds) -> str:
    if value <= thresholds.stable_absolute_delta_max:
        return "stable"
    if value <= thresholds.mild_absolute_delta_max:
        return "mildly_sensitive"
    return "fragile"


def _parameter_distance(base_value: Any, perturbed_value: Any) -> float | None:
    base = _finite_float_or_none(base_value)
    perturbed = _finite_float_or_none(perturbed_value)
    if base is None or perturbed is None:
        return None
    return _stable_float(abs(perturbed - base))


def _normalized_parameter_distance(base_value: Any, perturbed_value: Any, *, epsilon: float) -> float | None:
    base = _finite_float_or_none(base_value)
    perturbed = _finite_float_or_none(perturbed_value)
    if base is None or perturbed is None:
        return None
    denominator = max(abs(base), epsilon)
    if denominator <= 0:
        return None
    return _stable_float(abs(perturbed - base) / denominator)


def _finite_float_or_none(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(normalized):
        return None
    return _stable_float(normalized)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return None


def _first_present(record: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    return None


def _is_missing(value: Any) -> bool:
    return value is None or value == ""


def _stable_float(value: float) -> float:
    return float(round(value, 12))


def _drop_none(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None and item != ""}


def _finding_message(record: SensitivityInput, status: str) -> str:
    label = record.parameter_name or "parameter"
    scenario = f" scenario '{record.scenario_id}'" if record.scenario_id else ""
    return {
        "improved": f"Sensitivity check improved for '{label}'{scenario}.",
        "stable": f"Sensitivity check is stable for '{label}'{scenario}.",
        "mildly_sensitive": f"Sensitivity check is mildly sensitive for '{label}'{scenario}.",
        "fragile": f"Sensitivity check is fragile for '{label}'{scenario}.",
        "undefined": f"Sensitivity check is undefined for '{label}'{scenario}.",
        "missing": f"Sensitivity check is missing required metadata for '{label}'{scenario}.",
    }[status]


__all__ = [
    "METRIC_TRANSFORM_ABSOLUTE_MAGNITUDE",
    "SENSITIVITY_CHECK_PREFIX",
    "SENSITIVITY_STATUSES",
    "SensitivityEvaluation",
    "SensitivityInput",
    "SensitivityThresholds",
    "build_sensitivity_evidence",
    "build_sensitivity_findings",
    "build_sensitivity_summary_rows",
    "classify_fragility",
    "evaluate_parameter_sensitivity",
]
