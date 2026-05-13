from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

from src.research.registry import canonicalize_value

from .models import MultipleTestingSummary, RobustnessFinding

MULTIPLE_TESTING_CHECK_PREFIX = "multiple_testing"
MULTIPLE_TESTING_STATUSES: tuple[str, ...] = (
    "low_risk",
    "moderate_risk",
    "high_risk",
    "extreme_risk",
    "missing",
    "undefined",
)

TRIAL_COUNT_FIELDS: tuple[str, ...] = (
    "candidate_count",
    "tested_configuration_count",
    "parameter_combination_count",
    "scenario_count",
    "factor_count",
    "model_count",
    "portfolio_count",
    "campaign_count",
)


@dataclass(frozen=True)
class MultipleTestingThresholds:
    low_risk_trial_count_max: int = 10
    moderate_risk_trial_count_max: int = 100
    high_risk_trial_count_min: int = 101
    extreme_risk_trial_count_min: int = 1000
    missing_metadata_severity: str = "needs_review"
    moderate_risk_severity: str = "warning"
    high_risk_severity: str = "needs_review"
    extreme_risk_severity: str = "needs_review"
    selected_rank_warning_threshold: int = 100

    def __post_init__(self) -> None:
        if self.low_risk_trial_count_max < 0:
            raise ValueError("low_risk_trial_count_max must be non-negative.")
        if self.moderate_risk_trial_count_max < self.low_risk_trial_count_max:
            raise ValueError("moderate_risk_trial_count_max must be >= low_risk_trial_count_max.")
        if self.high_risk_trial_count_min <= self.moderate_risk_trial_count_max:
            raise ValueError("high_risk_trial_count_min must be > moderate_risk_trial_count_max.")
        if self.extreme_risk_trial_count_min < self.high_risk_trial_count_min:
            raise ValueError("extreme_risk_trial_count_min must be >= high_risk_trial_count_min.")
        if self.selected_rank_warning_threshold < 0:
            raise ValueError("selected_rank_warning_threshold must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        return canonicalize_value(
            {
                "extreme_risk_trial_count_min": self.extreme_risk_trial_count_min,
                "high_risk_trial_count_min": self.high_risk_trial_count_min,
                "low_risk_trial_count_max": self.low_risk_trial_count_max,
                "moderate_risk_trial_count_max": self.moderate_risk_trial_count_max,
                "selected_rank_warning_threshold": self.selected_rank_warning_threshold,
            }
        )

    def severity_for(self, status: str) -> str:
        if status == "low_risk":
            return "info"
        if status == "moderate_risk":
            return self.moderate_risk_severity
        if status == "high_risk":
            return self.high_risk_severity
        if status == "extreme_risk":
            return self.extreme_risk_severity
        return self.missing_metadata_severity


@dataclass(frozen=True)
class MultipleTestingInput:
    workflow_type: str
    run_id: str
    source_run_id: str | None = None
    family_id: str = "selection_family"
    candidate_count: int | float | str | None = None
    tested_configuration_count: int | float | str | None = None
    parameter_combination_count: int | float | str | None = None
    scenario_count: int | float | str | None = None
    factor_count: int | float | str | None = None
    model_count: int | float | str | None = None
    portfolio_count: int | float | str | None = None
    campaign_count: int | float | str | None = None
    selected_rank: int | float | str | None = None
    selection_metric: str = ""
    selection_metric_value: int | float | str | None = None
    trial_count_source: str = "unknown"
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MultipleTestingEvaluation:
    status: str
    reason: str
    effective_trial_count: int | None
    trial_counts: Mapping[str, int]
    invalid_fields: Mapping[str, Any]
    selected_rank: int | None
    selected_rank_status: str
    selection_metric_value: float | None


def classify_trial_count_risk(
    effective_trial_count: Any,
    *,
    thresholds: MultipleTestingThresholds | None = None,
) -> str:
    resolved = thresholds or MultipleTestingThresholds()
    normalized = _optional_count(effective_trial_count)
    if normalized is None:
        return "undefined"
    if normalized >= resolved.extreme_risk_trial_count_min:
        return "extreme_risk"
    if normalized >= resolved.high_risk_trial_count_min:
        return "high_risk"
    if normalized <= resolved.low_risk_trial_count_max:
        return "low_risk"
    return "moderate_risk"


def evaluate_multiple_testing_risk(
    record: MultipleTestingInput | Mapping[str, Any],
    *,
    thresholds: MultipleTestingThresholds | None = None,
) -> MultipleTestingEvaluation:
    resolved = thresholds or MultipleTestingThresholds()
    normalized = _coerce_input(record)
    raw_counts = {field_name: getattr(normalized, field_name) for field_name in TRIAL_COUNT_FIELDS}
    trial_counts: dict[str, int] = {}
    invalid_fields: dict[str, Any] = {}
    present_count_fields = 0

    for field_name, value in raw_counts.items():
        if _is_missing(value):
            continue
        present_count_fields += 1
        count = _optional_count(value)
        if count is None:
            invalid_fields[field_name] = _safe_raw_value(value)
        else:
            trial_counts[field_name] = count

    if invalid_fields:
        status = "undefined"
        reason = "invalid_trial_count_metadata"
        effective_trial_count = None
    elif present_count_fields == 0:
        status = "missing"
        reason = "missing_trial_count_metadata"
        effective_trial_count = None
    else:
        effective_trial_count = max(trial_counts.values()) if trial_counts else None
        status = classify_trial_count_risk(effective_trial_count, thresholds=resolved)
        reason = f"{status}_threshold"

    selected_rank, selected_rank_status = _selected_rank_status(
        normalized.selected_rank,
        candidate_count=trial_counts.get("candidate_count"),
    )

    return MultipleTestingEvaluation(
        status=status,
        reason=reason,
        effective_trial_count=effective_trial_count,
        trial_counts=canonicalize_value(trial_counts),
        invalid_fields=canonicalize_value(invalid_fields),
        selected_rank=selected_rank,
        selected_rank_status=selected_rank_status,
        selection_metric_value=_finite_float_or_none(normalized.selection_metric_value),
    )


def build_multiple_testing_summaries(
    records: Sequence[MultipleTestingInput | Mapping[str, Any]],
    *,
    thresholds: MultipleTestingThresholds | None = None,
) -> list[MultipleTestingSummary]:
    resolved = thresholds or MultipleTestingThresholds()
    rows: list[MultipleTestingSummary] = []
    for raw in records:
        record = _coerce_input(raw)
        evaluation = evaluate_multiple_testing_risk(record, thresholds=resolved)
        rows.append(
            MultipleTestingSummary(
                workflow_type=record.workflow_type,
                run_id=record.run_id,
                family_id=record.family_id,
                trial_count=evaluation.effective_trial_count,
                effective_trial_count=evaluation.effective_trial_count,
                adjustment_method="metadata_only",
                status=evaluation.status,
                details=_details_payload(record, evaluation, thresholds=resolved),
            )
        )
    return sorted(rows, key=_summary_sort_key)


def build_multiple_testing_findings(
    records: Sequence[MultipleTestingInput | Mapping[str, Any]],
    *,
    thresholds: MultipleTestingThresholds | None = None,
    include_info: bool = False,
    include_moderate_risk: bool = True,
) -> list[RobustnessFinding]:
    resolved = thresholds or MultipleTestingThresholds()
    findings: list[RobustnessFinding] = []
    for raw in sorted((_coerce_input(item) for item in records), key=_input_sort_key):
        evaluation = evaluate_multiple_testing_risk(raw, thresholds=resolved)
        details = _details_payload(raw, evaluation, thresholds=resolved)
        if _emit_primary_finding(
            evaluation.status,
            include_info=include_info,
            include_moderate_risk=include_moderate_risk,
        ):
            findings.append(
                RobustnessFinding(
                    check_id=_check_id_for_status(evaluation.status),
                    severity=resolved.severity_for(evaluation.status),
                    workflow_type=raw.workflow_type,
                    run_id=raw.run_id,
                    message=_finding_message(raw, evaluation.status),
                    details=details,
                )
            )
        findings.extend(_selected_rank_findings(raw, evaluation, details=details, thresholds=resolved))
    return sorted(findings, key=_finding_sort_key)


def build_multiple_testing_evidence(
    records: Sequence[MultipleTestingInput | Mapping[str, Any]],
    *,
    thresholds: MultipleTestingThresholds | None = None,
    include_info_findings: bool = False,
    include_moderate_risk_findings: bool = True,
) -> tuple[list[MultipleTestingSummary], list[RobustnessFinding]]:
    resolved = thresholds or MultipleTestingThresholds()
    return (
        build_multiple_testing_summaries(records, thresholds=resolved),
        build_multiple_testing_findings(
            records,
            thresholds=resolved,
            include_info=include_info_findings,
            include_moderate_risk=include_moderate_risk_findings,
        ),
    )


def build_multiple_testing_evidence_from_summary(
    summary: Mapping[str, Any],
    *,
    workflow_type: str = "robustness",
    run_id: str | None = None,
    thresholds: MultipleTestingThresholds | None = None,
) -> tuple[list[MultipleTestingSummary], list[RobustnessFinding]]:
    record = _coerce_input(
        {
            "workflow_type": workflow_type,
            "run_id": run_id or summary.get("run_id") or summary.get("source_run_id") or "unknown",
            "candidate_count": _first_present(summary, "candidate_count", "variant_count", "row_count"),
            "tested_configuration_count": _first_present(
                summary,
                "tested_configuration_count",
                "configuration_count",
                "tested_config_count",
            ),
            "parameter_combination_count": _first_present(summary, "parameter_combination_count"),
            "scenario_count": _first_present(summary, "scenario_count", "total_scenario_count"),
            "model_count": _first_present(summary, "model_count"),
            "factor_count": _first_present(summary, "factor_count"),
            "portfolio_count": _first_present(summary, "portfolio_count"),
            "campaign_count": _first_present(summary, "campaign_count"),
            "selected_rank": _first_present(summary, "selected_rank", "rank", "best_rank"),
            "selection_metric": _first_present(summary, "selection_metric", "ranking_metric", "metric"),
            "selection_metric_value": _first_present(summary, "selection_metric_value", "best_metric_value"),
            "trial_count_source": str(summary.get("trial_count_source") or "summary_mapping"),
            "details": {"source_summary_keys": sorted(str(key) for key in summary)},
        }
    )
    return build_multiple_testing_evidence([record], thresholds=thresholds)


def _details_payload(
    record: MultipleTestingInput,
    evaluation: MultipleTestingEvaluation,
    *,
    thresholds: MultipleTestingThresholds,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "candidate_count": evaluation.trial_counts.get("candidate_count"),
        "campaign_count": evaluation.trial_counts.get("campaign_count"),
        "dsr_supported": False,
        "effective_trial_count": evaluation.effective_trial_count,
        "factor_count": evaluation.trial_counts.get("factor_count"),
        "haircut_supported": False,
        "invalid_trial_count_fields": dict(evaluation.invalid_fields),
        "model_count": evaluation.trial_counts.get("model_count"),
        "parameter_combination_count": evaluation.trial_counts.get("parameter_combination_count"),
        "pbo_supported": False,
        "portfolio_count": evaluation.trial_counts.get("portfolio_count"),
        "reason": evaluation.reason,
        "scenario_count": evaluation.trial_counts.get("scenario_count"),
        "selected_rank": evaluation.selected_rank,
        "selected_rank_status": evaluation.selected_rank_status,
        "selection_metric": record.selection_metric or None,
        "selection_metric_value": evaluation.selection_metric_value,
        "source_run_id": record.source_run_id,
        "tested_configuration_count": evaluation.trial_counts.get("tested_configuration_count"),
        "thresholds": thresholds.to_dict(),
        "trial_count_source": record.trial_count_source or "unknown",
        "trial_count_status": evaluation.status,
    }
    payload.update(dict(record.details))
    return canonicalize_value(_drop_empty(payload))


def _selected_rank_findings(
    record: MultipleTestingInput,
    evaluation: MultipleTestingEvaluation,
    *,
    details: Mapping[str, Any],
    thresholds: MultipleTestingThresholds,
) -> list[RobustnessFinding]:
    findings: list[RobustnessFinding] = []
    candidate_count = evaluation.trial_counts.get("candidate_count")
    if (
        evaluation.selected_rank == 1
        and evaluation.effective_trial_count is not None
        and evaluation.effective_trial_count >= thresholds.selected_rank_warning_threshold
    ):
        findings.append(
            _rank_finding(
                record,
                details=details,
                reason="top_rank_selected_from_large_search_space",
                message="Top-ranked result was selected from a large search space.",
            )
        )
    if evaluation.selected_rank_status == "missing" and candidate_count is not None:
        findings.append(
            _rank_finding(
                record,
                details=details,
                reason="missing_selected_rank",
                message="Candidate count is present but selected-rank metadata is missing.",
            )
        )
    if evaluation.selected_rank_status == "undefined":
        findings.append(
            _rank_finding(
                record,
                details=details,
                reason="invalid_selected_rank",
                message="Selected-rank metadata is invalid for the candidate set.",
            )
        )
    if evaluation.selected_rank is not None and not record.selection_metric:
        findings.append(
            _rank_finding(
                record,
                details=details,
                reason="missing_selection_metric",
                message="Selected rank is present but selection metric metadata is missing.",
            )
        )
    return findings


def _rank_finding(
    record: MultipleTestingInput,
    *,
    details: Mapping[str, Any],
    reason: str,
    message: str,
) -> RobustnessFinding:
    payload = dict(details)
    payload["rank_reason"] = reason
    return RobustnessFinding(
        check_id="multiple_testing.selected_rank_warning",
        severity="warning",
        workflow_type=record.workflow_type,
        run_id=record.run_id,
        message=message,
        details=canonicalize_value(payload),
    )


def _coerce_input(record: MultipleTestingInput | Mapping[str, Any]) -> MultipleTestingInput:
    if isinstance(record, MultipleTestingInput):
        return record
    details = dict(record.get("details", {})) if isinstance(record.get("details", {}), Mapping) else {}
    return MultipleTestingInput(
        workflow_type=str(record.get("workflow_type", "unknown")),
        run_id=str(record.get("run_id") or record.get("source_run_id") or "unknown"),
        source_run_id=None if record.get("source_run_id") is None else str(record.get("source_run_id")),
        family_id=str(record.get("family_id") or record.get("selection_family_id") or "selection_family"),
        candidate_count=_first_present(record, "candidate_count", "candidate_config_count", "variant_count"),
        tested_configuration_count=_first_present(
            record,
            "tested_configuration_count",
            "tested_config_count",
            "configuration_count",
        ),
        parameter_combination_count=_first_present(
            record,
            "parameter_combination_count",
            "parameter_grid_count",
            "parameter_count",
        ),
        scenario_count=_first_present(record, "scenario_count", "total_scenario_count"),
        factor_count=_first_present(record, "factor_count"),
        model_count=_first_present(record, "model_count"),
        portfolio_count=_first_present(record, "portfolio_count"),
        campaign_count=_first_present(record, "campaign_count"),
        selected_rank=_first_present(record, "selected_rank", "rank", "best_rank"),
        selection_metric=str(record.get("selection_metric") or record.get("ranking_metric") or record.get("metric") or "").strip(),
        selection_metric_value=_first_present(record, "selection_metric_value", "best_metric_value", "metric_value"),
        trial_count_source=str(record.get("trial_count_source") or details.get("trial_count_source") or "unknown"),
        details=details,
    )


def _selected_rank_status(value: Any, *, candidate_count: int | None) -> tuple[int | None, str]:
    if _is_missing(value):
        return None, "missing"
    selected_rank = _optional_count(value)
    if selected_rank is None or selected_rank < 1:
        return None, "undefined"
    if candidate_count is not None and selected_rank > candidate_count:
        return selected_rank, "undefined"
    return selected_rank, "recorded"


def _optional_count(value: Any) -> int | None:
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        return None
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(normalized) or normalized < 0:
        return None
    if not normalized.is_integer():
        return None
    return int(normalized)


def _finite_float_or_none(value: Any) -> float | None:
    if _is_missing(value) or isinstance(value, bool):
        return None
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(normalized):
        return None
    return float(round(normalized, 12))


def _safe_raw_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "non_finite_float"
    return value


def _first_present(record: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    return None


def _is_missing(value: Any) -> bool:
    return value is None or value == ""


def _drop_empty(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None and item != "" and item != {}}


def _summary_sort_key(row: MultipleTestingSummary) -> tuple[str, str, str, str, str]:
    details = dict(row.details)
    return (
        row.workflow_type,
        row.run_id,
        str(details.get("source_run_id", "")),
        str(details.get("trial_count_source", "")),
        str(details.get("selection_metric", "")),
    )


def _input_sort_key(record: MultipleTestingInput) -> tuple[str, str, str, str, str]:
    return (
        record.workflow_type,
        record.run_id,
        str(record.source_run_id or ""),
        record.trial_count_source,
        record.selection_metric,
    )


def _finding_sort_key(finding: RobustnessFinding) -> tuple[str, str, str, str, str]:
    return (
        finding.workflow_type,
        finding.run_id,
        str(finding.details.get("source_run_id", "")),
        str(finding.details.get("trial_count_source", "")),
        finding.check_id,
    )


def _emit_primary_finding(
    status: str,
    *,
    include_info: bool,
    include_moderate_risk: bool,
) -> bool:
    if status == "low_risk":
        return include_info
    if status == "moderate_risk":
        return include_moderate_risk
    return status in {"high_risk", "extreme_risk", "missing", "undefined"}


def _check_id_for_status(status: str) -> str:
    if status == "missing":
        return "multiple_testing.missing_trial_count_metadata"
    if status == "undefined":
        return "multiple_testing.undefined_trial_count_metadata"
    return f"multiple_testing.{status}"


def _finding_message(record: MultipleTestingInput, status: str) -> str:
    family = record.family_id or "selection_family"
    return {
        "low_risk": f"Multiple-testing metadata indicates a controlled search space for '{family}'.",
        "moderate_risk": f"Multiple-testing metadata indicates a non-trivial search space for '{family}'.",
        "high_risk": f"Multiple-testing metadata indicates a large search space for '{family}'.",
        "extreme_risk": f"Multiple-testing metadata indicates a very large search space for '{family}'.",
        "missing": f"Multiple-testing metadata is missing trial-count evidence for '{family}'.",
        "undefined": f"Multiple-testing metadata has invalid trial-count evidence for '{family}'.",
    }[status]


__all__ = [
    "MULTIPLE_TESTING_CHECK_PREFIX",
    "MULTIPLE_TESTING_STATUSES",
    "MultipleTestingEvaluation",
    "MultipleTestingInput",
    "MultipleTestingThresholds",
    "build_multiple_testing_evidence",
    "build_multiple_testing_evidence_from_summary",
    "build_multiple_testing_findings",
    "build_multiple_testing_summaries",
    "classify_trial_count_risk",
    "evaluate_multiple_testing_risk",
]
