from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import yaml

from src.research.registry import canonicalize_value, serialize_canonical_json

DEFAULT_PROMOTION_ARTIFACT_FILENAME = "promotion_gates.json"
DEFAULT_STATUS_ON_PASS = "eligible"
DEFAULT_STATUS_ON_FAIL = "blocked"
_SUPPORTED_SEVERITIES = frozenset({"warn", "review", "reject", "block"})
_SEVERITY_RANK = {
    "warn": 0,
    "review": 1,
    "reject": 2,
    "block": 3,
}
_SEVERITY_PROMOTION_STATUS = {
    "warn": "warn",
    "review": "needs_review",
    "reject": "rejected",
    "block": "blocked",
}
_SEVERITY_REASON_CODES = {
    "warn": "severity_warn",
    "review": "severity_review",
    "reject": "severity_reject",
    "block": "severity_block",
}
_SUPPORTED_SOURCES = frozenset(
    {
        "metrics",
        "qa_summary",
        "manifest",
        "config",
        "metadata",
        "split_metrics",
        "timeseries",
        "aggregate_metrics",
    }
)
_SUPPORTED_STATISTICS = frozenset(
    {
        "value",
        "count",
        "non_null_count",
        "min",
        "max",
        "mean",
        "median",
        "std",
        "range",
        "abs_max",
    }
)
_SUPPORTED_COMPARATORS = frozenset({"gt", "gte", "lt", "lte", "eq", "ne"})


class PromotionGateError(ValueError):
    """Raised when promotion gate configuration is malformed."""


@dataclass(frozen=True)
class PromotionGateDefinition:
    gate_id: str
    source: str
    metric_path: str
    comparator: str
    threshold: float
    statistic: str = "value"
    missing_behavior: str = "fail"
    description: str | None = None
    severity: str | None = None


@dataclass(frozen=True)
class PromotionGateResult:
    gate_id: str
    status: str
    source: str
    metric_path: str
    statistic: str
    comparator: str
    threshold: float
    actual_value: float | None
    missing_behavior: str
    description: str | None
    severity: str | None
    reason_codes: list[str]
    reason: str


@dataclass(frozen=True)
class PromotionGateEvaluation:
    configured: bool
    run_type: str
    evaluation_status: str
    promotion_status: str | None
    status_on_pass: str | None
    status_on_fail: str | None
    gate_count: int
    passed_gate_count: int
    failed_gate_count: int
    missing_gate_count: int
    highest_severity: str | None
    severity_counts: dict[str, int]
    warning_gate_count: int
    review_gate_count: int
    rejected_gate_count: int
    blocked_gate_count: int
    decision_reason_codes: list[str]
    artifact_filename: str
    definitions: list[PromotionGateDefinition]
    results: list[PromotionGateResult]

    def to_payload(self) -> dict[str, Any]:
        return canonicalize_value(
            {
                "configured": self.configured,
                "run_type": self.run_type,
                "evaluation_status": self.evaluation_status,
                "promotion_status": self.promotion_status,
                "status_on_pass": self.status_on_pass,
                "status_on_fail": self.status_on_fail,
                "gate_count": self.gate_count,
                "passed_gate_count": self.passed_gate_count,
                "failed_gate_count": self.failed_gate_count,
                "missing_gate_count": self.missing_gate_count,
                "highest_severity": self.highest_severity,
                "severity_counts": self.severity_counts,
                "warning_gate_count": self.warning_gate_count,
                "review_gate_count": self.review_gate_count,
                "rejected_gate_count": self.rejected_gate_count,
                "blocked_gate_count": self.blocked_gate_count,
                "decision_reason_codes": self.decision_reason_codes,
                "artifact_filename": self.artifact_filename,
                "definitions": [asdict(definition) for definition in self.definitions],
                "results": [asdict(result) for result in self.results],
            }
        )

    def summary(self) -> dict[str, Any]:
        return canonicalize_value(
            {
                "configured": self.configured,
                "evaluation_status": self.evaluation_status,
                "promotion_status": self.promotion_status,
                "status_on_pass": self.status_on_pass,
                "status_on_fail": self.status_on_fail,
                "gate_count": self.gate_count,
                "passed_gate_count": self.passed_gate_count,
                "failed_gate_count": self.failed_gate_count,
                "missing_gate_count": self.missing_gate_count,
                "highest_severity": self.highest_severity,
                "severity_counts": self.severity_counts,
                "warning_gate_count": self.warning_gate_count,
                "review_gate_count": self.review_gate_count,
                "rejected_gate_count": self.rejected_gate_count,
                "blocked_gate_count": self.blocked_gate_count,
                "decision_reason_codes": self.decision_reason_codes,
                "artifact_filename": self.artifact_filename,
            }
        )


def load_promotion_gate_config(path: str | Path) -> dict[str, Any]:
    """Load one JSON or YAML promotion gate config."""

    resolved_path = Path(path)
    if not resolved_path.exists():
        raise PromotionGateError(f"Promotion gate config file does not exist: {resolved_path}")

    suffix = resolved_path.suffix.lower()
    with resolved_path.open("r", encoding="utf-8") as handle:
        if suffix == ".json":
            payload = json.load(handle)
        elif suffix in {".yaml", ".yml"}:
            payload = yaml.safe_load(handle) or {}
        else:
            raise PromotionGateError(
                f"Unsupported promotion gate config format {resolved_path.suffix!r}. Use JSON, YAML, or YML."
            )
    if not isinstance(payload, dict):
        raise PromotionGateError("Promotion gate config must deserialize to a mapping.")
    nested = payload.get("promotion_gates")
    if nested is None:
        return payload
    if not isinstance(nested, dict):
        raise PromotionGateError("promotion_gates config section must be a mapping.")
    return nested


def evaluate_promotion_gates(
    *,
    run_type: str,
    config: Mapping[str, Any] | None,
    sources: Mapping[str, Any],
    artifact_filename: str = DEFAULT_PROMOTION_ARTIFACT_FILENAME,
) -> PromotionGateEvaluation | None:
    """Evaluate one normalized promotion gate config against deterministic sources."""

    if config is None:
        return None

    normalized_config = _normalize_promotion_gate_config(config)
    definitions = normalized_config["definitions"]
    results = [
        _evaluate_definition(definition, sources=sources)
        for definition in definitions
    ]
    failed_gate_count = sum(result.status == "fail" for result in results)
    missing_gate_count = sum(result.status == "missing" for result in results)
    passed_gate_count = sum(result.status == "pass" for result in results)
    evaluation_status = "pass" if failed_gate_count == 0 and missing_gate_count == 0 else "fail"
    highest_severity = _highest_failing_severity(results)
    severity_counts = _severity_counts(results)
    return PromotionGateEvaluation(
        configured=True,
        run_type=run_type,
        evaluation_status=evaluation_status,
        promotion_status=_resolve_promotion_status(
            evaluation_status=evaluation_status,
            highest_severity=highest_severity,
            status_on_pass=normalized_config["status_on_pass"],
            status_on_fail=normalized_config["status_on_fail"],
        ),
        status_on_pass=normalized_config["status_on_pass"],
        status_on_fail=normalized_config["status_on_fail"],
        gate_count=len(results),
        passed_gate_count=passed_gate_count,
        failed_gate_count=failed_gate_count,
        missing_gate_count=missing_gate_count,
        highest_severity=highest_severity,
        severity_counts=severity_counts,
        warning_gate_count=severity_counts["warn"],
        review_gate_count=severity_counts["review"],
        rejected_gate_count=severity_counts["reject"],
        blocked_gate_count=severity_counts["block"],
        decision_reason_codes=_decision_reason_codes(results),
        artifact_filename=artifact_filename,
        definitions=definitions,
        results=results,
    )


def write_promotion_gate_artifact(
    output_dir: str | Path,
    evaluation: PromotionGateEvaluation | None,
    *,
    artifact_filename: str = DEFAULT_PROMOTION_ARTIFACT_FILENAME,
) -> Path | None:
    """Persist a promotion gate evaluation as stable JSON when configured."""

    if evaluation is None:
        return None
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = resolved_output_dir / artifact_filename
    with artifact_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(evaluation.to_payload(), handle, indent=2, sort_keys=True)
    return artifact_path


def promotion_gate_config_digest(config: Mapping[str, Any] | None) -> str | None:
    if config is None:
        return None
    normalized_config = _normalize_promotion_gate_config(config)
    payload = {
        "status_on_pass": normalized_config["status_on_pass"],
        "status_on_fail": normalized_config["status_on_fail"],
        "definitions": [asdict(definition) for definition in normalized_config["definitions"]],
    }
    return serialize_canonical_json(payload)


def _normalize_promotion_gate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise PromotionGateError("promotion_gates must be a mapping when provided.")

    status_on_pass = _normalize_required_string(
        config.get("status_on_pass", DEFAULT_STATUS_ON_PASS),
        field_name="promotion_gates.status_on_pass",
    )
    status_on_fail = _normalize_required_string(
        config.get("status_on_fail", DEFAULT_STATUS_ON_FAIL),
        field_name="promotion_gates.status_on_fail",
    )
    raw_definitions = config.get("gates")
    if not isinstance(raw_definitions, list) or not raw_definitions:
        raise PromotionGateError("promotion_gates.gates must be a non-empty list.")
    definitions = [_normalize_definition(item, index=index) for index, item in enumerate(raw_definitions)]
    return {
        "status_on_pass": status_on_pass,
        "status_on_fail": status_on_fail,
        "definitions": definitions,
    }


def _normalize_definition(raw_definition: Any, *, index: int) -> PromotionGateDefinition:
    if not isinstance(raw_definition, Mapping):
        raise PromotionGateError(f"promotion_gates.gates[{index}] must be a mapping.")

    gate_id = _normalize_required_string(
        raw_definition.get("gate_id"),
        field_name=f"promotion_gates.gates[{index}].gate_id",
    )
    source = _normalize_required_string(
        raw_definition.get("source", "metrics"),
        field_name=f"promotion_gates.gates[{index}].source",
    ).lower()
    if source not in _SUPPORTED_SOURCES:
        formatted = ", ".join(sorted(_SUPPORTED_SOURCES))
        raise PromotionGateError(
            f"promotion_gates.gates[{index}].source must be one of: {formatted}."
        )

    metric_path = raw_definition.get("metric_path", raw_definition.get("metric"))
    metric_path = _normalize_required_string(
        metric_path,
        field_name=f"promotion_gates.gates[{index}].metric_path",
    )
    comparator = _normalize_required_string(
        raw_definition.get("comparator"),
        field_name=f"promotion_gates.gates[{index}].comparator",
    ).lower()
    if comparator not in _SUPPORTED_COMPARATORS:
        formatted = ", ".join(sorted(_SUPPORTED_COMPARATORS))
        raise PromotionGateError(
            f"promotion_gates.gates[{index}].comparator must be one of: {formatted}."
        )
    threshold = _coerce_finite_float(
        raw_definition.get("threshold"),
        field_name=f"promotion_gates.gates[{index}].threshold",
    )
    statistic = _normalize_required_string(
        raw_definition.get("statistic", "value"),
        field_name=f"promotion_gates.gates[{index}].statistic",
    ).lower()
    if statistic not in _SUPPORTED_STATISTICS:
        formatted = ", ".join(sorted(_SUPPORTED_STATISTICS))
        raise PromotionGateError(
            f"promotion_gates.gates[{index}].statistic must be one of: {formatted}."
        )
    missing_behavior = _normalize_required_string(
        raw_definition.get("missing_behavior", "fail"),
        field_name=f"promotion_gates.gates[{index}].missing_behavior",
    ).lower()
    if missing_behavior not in {"fail", "skip"}:
        raise PromotionGateError(
            f"promotion_gates.gates[{index}].missing_behavior must be one of: fail, skip."
        )
    description = raw_definition.get("description")
    if description is not None:
        description = _normalize_required_string(
            description,
            field_name=f"promotion_gates.gates[{index}].description",
        )
    severity = _normalize_optional_severity(
        raw_definition.get("severity"),
        field_name=f"promotion_gates.gates[{index}].severity",
    )
    return PromotionGateDefinition(
        gate_id=gate_id,
        source=source,
        metric_path=metric_path,
        comparator=comparator,
        threshold=threshold,
        statistic=statistic,
        missing_behavior=missing_behavior,
        description=description,
        severity=severity,
    )


def _evaluate_definition(
    definition: PromotionGateDefinition,
    *,
    sources: Mapping[str, Any],
) -> PromotionGateResult:
    raw_source = sources.get(definition.source)
    actual_value = _extract_actual_value(
        raw_source,
        metric_path=definition.metric_path,
        statistic=definition.statistic,
    )
    if actual_value is None:
        status = "pass" if definition.missing_behavior == "skip" else "missing"
        reason = (
            "metric missing and gate skipped"
            if definition.missing_behavior == "skip"
            else "metric missing"
        )
        reason_codes = _result_reason_codes(
            status=status,
            severity=definition.severity,
            base_reason_code=(
                "gate_missing_skipped"
                if definition.missing_behavior == "skip"
                else "gate_missing"
            ),
        )
        return PromotionGateResult(
            gate_id=definition.gate_id,
            status=status,
            source=definition.source,
            metric_path=definition.metric_path,
            statistic=definition.statistic,
            comparator=definition.comparator,
            threshold=definition.threshold,
            actual_value=None,
            missing_behavior=definition.missing_behavior,
            description=definition.description,
            severity=definition.severity,
            reason_codes=reason_codes,
            reason=reason,
        )

    passed = _compare(
        actual_value=actual_value,
        comparator=definition.comparator,
        threshold=definition.threshold,
    )
    return PromotionGateResult(
        gate_id=definition.gate_id,
        status="pass" if passed else "fail",
        source=definition.source,
        metric_path=definition.metric_path,
        statistic=definition.statistic,
        comparator=definition.comparator,
        threshold=definition.threshold,
        actual_value=actual_value,
        missing_behavior=definition.missing_behavior,
        description=definition.description,
        severity=definition.severity,
        reason_codes=_result_reason_codes(
            status="pass" if passed else "fail",
            severity=definition.severity,
            base_reason_code="gate_passed" if passed else "gate_failed_threshold",
        ),
        reason=(
            f"{actual_value} {definition.comparator} {definition.threshold}"
            if passed
            else f"{actual_value} not {definition.comparator} {definition.threshold}"
        ),
    )


def _normalize_optional_severity(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    severity = _normalize_required_string(value, field_name=field_name).lower()
    if severity not in _SUPPORTED_SEVERITIES:
        formatted = ", ".join(sorted(_SUPPORTED_SEVERITIES))
        raise PromotionGateError(f"{field_name} must be one of: {formatted}.")
    return severity


def _result_reason_codes(*, status: str, severity: str | None, base_reason_code: str) -> list[str]:
    codes = [base_reason_code]
    if status in {"fail", "missing"} and severity is not None:
        codes.append(_SEVERITY_REASON_CODES[severity])
    return codes


def _nonpassing_results(results: Iterable[PromotionGateResult]) -> list[PromotionGateResult]:
    return [result for result in results if result.status in {"fail", "missing"}]


def _highest_failing_severity(results: Iterable[PromotionGateResult]) -> str | None:
    severities = [
        result.severity
        for result in _nonpassing_results(results)
        if result.severity is not None
    ]
    if not severities:
        return None
    return max(severities, key=lambda severity: _SEVERITY_RANK[severity])


def _severity_counts(results: Iterable[PromotionGateResult]) -> dict[str, int]:
    counts = {severity: 0 for severity in sorted(_SUPPORTED_SEVERITIES)}
    for result in _nonpassing_results(results):
        if result.severity is not None:
            counts[result.severity] += 1
    return counts


def _decision_reason_codes(results: Iterable[PromotionGateResult]) -> list[str]:
    codes: set[str] = set()
    for result in _nonpassing_results(results):
        codes.update(result.reason_codes)
    return sorted(codes)


def _resolve_promotion_status(
    *,
    evaluation_status: str,
    highest_severity: str | None,
    status_on_pass: str,
    status_on_fail: str,
) -> str:
    if evaluation_status == "pass":
        return status_on_pass
    if highest_severity is None:
        return status_on_fail
    return _SEVERITY_PROMOTION_STATUS[highest_severity]


def _extract_actual_value(
    raw_source: Any,
    *,
    metric_path: str,
    statistic: str,
) -> float | None:
    values = _extract_values(raw_source, metric_path=metric_path)
    if not values:
        return None
    if statistic == "value":
        return values[0] if len(values) == 1 else None
    if statistic == "count":
        return float(len(values))
    if statistic == "non_null_count":
        return float(len(values))

    series = pd.Series(values, dtype="float64")
    if series.empty:
        return None
    if statistic == "min":
        return float(series.min())
    if statistic == "max":
        return float(series.max())
    if statistic == "mean":
        return float(series.mean())
    if statistic == "median":
        return float(series.median())
    if statistic == "std":
        return float(series.std(ddof=0))
    if statistic == "range":
        return float(series.max() - series.min())
    if statistic == "abs_max":
        return float(series.abs().max())
    raise PromotionGateError(f"Unsupported statistic {statistic!r}.")


def _extract_values(raw_source: Any, *, metric_path: str) -> list[float]:
    if raw_source is None:
        return []
    if isinstance(raw_source, pd.DataFrame):
        if metric_path not in raw_source.columns:
            return []
        return _coerce_numeric_iterable(raw_source[metric_path].tolist())
    if isinstance(raw_source, Mapping):
        resolved = _lookup_mapping_value(raw_source, metric_path)
        if resolved is None:
            return []
        numeric = _coerce_numeric_value(resolved)
        return [] if numeric is None else [numeric]
    if isinstance(raw_source, list | tuple):
        collected: list[float] = []
        for item in raw_source:
            if isinstance(item, Mapping):
                resolved = _lookup_mapping_value(item, metric_path)
                numeric = _coerce_numeric_value(resolved)
                if numeric is not None:
                    collected.append(numeric)
        return collected
    return []


def _lookup_mapping_value(mapping: Mapping[str, Any], metric_path: str) -> Any:
    current: Any = mapping
    for part in metric_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _coerce_numeric_iterable(values: Iterable[Any]) -> list[float]:
    normalized: list[float] = []
    for value in values:
        numeric = _coerce_numeric_value(value)
        if numeric is not None:
            normalized.append(numeric)
    return normalized


def _coerce_numeric_value(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        if isinstance(value, bool):
            return float(value)
        return None
    if isinstance(value, int | float):
        numeric = float(value)
    else:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _compare(*, actual_value: float, comparator: str, threshold: float) -> bool:
    if comparator == "gt":
        return actual_value > threshold
    if comparator == "gte":
        return actual_value >= threshold
    if comparator == "lt":
        return actual_value < threshold
    if comparator == "lte":
        return actual_value <= threshold
    if comparator == "eq":
        return actual_value == threshold
    if comparator == "ne":
        return actual_value != threshold
    raise PromotionGateError(f"Unsupported comparator {comparator!r}.")


def _normalize_required_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise PromotionGateError(f"{field_name} must be a non-empty string.")
    normalized = value.strip()
    if not normalized:
        raise PromotionGateError(f"{field_name} must be a non-empty string.")
    return normalized


def _coerce_finite_float(value: object, *, field_name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise PromotionGateError(f"{field_name} must be a finite float.") from exc
    if math.isnan(numeric) or math.isinf(numeric):
        raise PromotionGateError(f"{field_name} must be a finite float.")
    return numeric


__all__ = [
    "DEFAULT_PROMOTION_ARTIFACT_FILENAME",
    "PromotionGateDefinition",
    "PromotionGateEvaluation",
    "PromotionGateError",
    "PromotionGateResult",
    "evaluate_promotion_gates",
    "load_promotion_gate_config",
    "promotion_gate_config_digest",
    "write_promotion_gate_artifact",
]
