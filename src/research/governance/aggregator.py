from __future__ import annotations

from collections import Counter
import hashlib
import math
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping

from src.artifacts.safety import portable_path
from src.research.registry import canonicalize_value, serialize_canonical_json

from .models import GovernanceSourceRecord
from .normalization import normalize_promotion_status, normalize_review_status

OUTCOME_MATRIX_COLUMNS = [
    "run_id",
    "workflow_type",
    "promotion_status",
    "highest_severity",
    "review_status",
    "decision_reason_codes",
    "triggered_gate_names",
    "registry_path",
    "manifest_path",
    "campaign_id",
    "scenario_id",
    "campaign_status",
    "scenario_status",
    "candidate_id",
    "candidate_selection_run_id",
    "selected_candidate_id",
    "selected_run_id",
    "upstream_run_ids",
    "strategy_name",
    "portfolio_name",
    "alpha_model_name",
    "effective_n",
    "p_value",
    "hit_rate_p_value",
    "sharpe_stability_ratio",
]
SEVERITIES = ("warn", "review", "reject", "block")


def build_governance_outcome_rows(
    records: list[GovernanceSourceRecord],
    *,
    base_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    resolved_base_dir = Path.cwd() if base_dir is None else Path(base_dir)
    rows = [_record_to_row(record, base_dir=resolved_base_dir) for record in records]
    return sorted(rows, key=lambda row: (str(row["workflow_type"]), str(row["run_id"])))


def build_governance_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    row_count = len(rows)
    status_counts = _counts(row["promotion_status"] for row in rows)
    highest_severity_counts = _counts(row["highest_severity"] for row in rows)
    workflow_type_counts = _counts(row["workflow_type"] for row in rows)
    reason_code_counts = _reason_code_counts(rows)
    severity_counts = {severity: highest_severity_counts.get(severity, 0) for severity in SEVERITIES}
    eligible = status_counts.get("eligible", 0)
    blocked = status_counts.get("blocked", 0)
    review = status_counts.get("needs_review", 0) + status_counts.get("warn", 0)
    summary = {
        "row_count": row_count,
        "campaign_count": sum(row["workflow_type"] == "campaign" for row in rows),
        "campaign_scenario_count": sum(row["workflow_type"] == "campaign_scenario" for row in rows),
        "candidate_review_count": sum(row["workflow_type"] == "candidate_review" for row in rows),
        "candidate_selection_count": len(_non_empty_values(row.get("candidate_selection_run_id") for row in rows)),
        "selected_candidate_count": len(_non_empty_values(row.get("selected_candidate_id") for row in rows)),
        "promotion_status_counts": status_counts,
        "highest_severity_counts": highest_severity_counts,
        "severity_counts": severity_counts,
        "reason_code_counts": reason_code_counts,
        "workflow_type_counts": workflow_type_counts,
        "campaign_status_counts": _counts(row.get("campaign_status", "") for row in rows if row["workflow_type"] == "campaign"),
        "scenario_status_counts": _counts(row.get("scenario_status", "") for row in rows if row["workflow_type"] == "campaign_scenario"),
        "candidate_status_counts": _counts(
            row.get("promotion_status", "")
            for row in rows
            if row["workflow_type"] == "candidate_review" or row.get("candidate_id")
        ),
        "warning_total": severity_counts["warn"],
        "review_total": severity_counts["review"],
        "reject_total": severity_counts["reject"],
        "block_total": severity_counts["block"],
        "top_blocking_reason_codes": _top_reason_codes(reason_code_counts, prefix_filter={"severity_block", "gate_missing", "gate_failed_threshold"}),
        "top_review_reason_codes": _top_reason_codes(reason_code_counts, prefix_filter={"severity_review", "severity_warn"}),
        "eligible_fraction": _fraction(eligible, row_count),
        "blocked_fraction": _fraction(blocked, row_count),
        "review_fraction": _fraction(review, row_count),
    }
    return canonicalize_value(summary)


def build_reason_code_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = _reason_code_counts(rows)
    return [
        {"reason_code": code, "count": count}
        for code, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def build_severity_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    highest_counts = _counts(row["highest_severity"] for row in rows)
    triggered_counts = Counter()
    for row in rows:
        for severity in _split_codes(row["decision_reason_codes"]):
            if severity.startswith("severity_"):
                triggered_counts[severity.removeprefix("severity_")] += 1
    return [
        {
            "severity": severity,
            "highest_severity_count": highest_counts.get(severity, 0),
            "triggered_reason_count": int(triggered_counts.get(severity, 0)),
        }
        for severity in SEVERITIES
    ]


def build_workflow_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["workflow_type"] or "unknown"), []).append(row)
    return [
        {
            "workflow_type": workflow_type,
            "row_count": len(group_rows),
            "eligible_count": sum(row["promotion_status"] == "eligible" for row in group_rows),
            "blocked_count": sum(row["promotion_status"] == "blocked" for row in group_rows),
            "needs_review_count": sum(row["promotion_status"] in {"needs_review", "warn"} for row in group_rows),
            "rejected_count": sum(row["promotion_status"] == "rejected" for row in group_rows),
        }
        for workflow_type, group_rows in sorted(grouped.items())
    ]


def build_governance_report_id(rows: list[dict[str, Any]]) -> str:
    identity_rows = [
        {
            "run_id": row["run_id"],
            "workflow_type": row["workflow_type"],
            "promotion_status": row["promotion_status"],
            "highest_severity": row["highest_severity"],
        }
        for row in rows
    ]
    digest = hashlib.sha256(serialize_canonical_json(identity_rows).encode("utf-8")).hexdigest()[:12]
    return f"promotion_governance_{digest}"


def _record_to_row(record: GovernanceSourceRecord, *, base_dir: Path) -> dict[str, Any]:
    entry = record.registry_entry
    manifest = record.manifest or {}
    summary = record.promotion_gate_summary or {}
    metadata = record.governance_metadata
    metrics = _mapping(entry.get("metrics_summary")) or _mapping(entry.get("metrics")) or _mapping(manifest.get("metric_summary"))
    review_metadata = _mapping(entry.get("review_metadata"))
    return {
        "run_id": record.run_id,
        "workflow_type": record.workflow_type,
        "promotion_status": _text(normalize_promotion_status(summary.get("promotion_status") or entry.get("promotion_status"))),
        "highest_severity": _text(summary.get("highest_severity")),
        "review_status": _text(normalize_review_status(entry.get("review_status") or review_metadata.get("status"))),
        "decision_reason_codes": "|".join(_string_list(summary.get("decision_reason_codes"))),
        "triggered_gate_names": "|".join(_triggered_gate_names(summary, record.promotion_gates)),
        "registry_path": _relative_path(
            record.registry_path,
            base_dir=base_dir,
            roots=_record_path_roots(record, path=record.registry_path),
        ),
        "manifest_path": _relative_path(
            record.manifest_path,
            base_dir=base_dir,
            roots=_record_path_roots(record, path=record.manifest_path),
        ),
        "campaign_id": _text(metadata.get("campaign_id") or entry.get("campaign_id") or manifest.get("campaign_run_id")),
        "scenario_id": _text(metadata.get("scenario_id") or entry.get("scenario_id") or manifest.get("scenario_id")),
        "candidate_id": _text(metadata.get("selected_candidate_id") or metadata.get("candidate_id") or entry.get("candidate_id")),
        "candidate_selection_run_id": _text(metadata.get("candidate_selection_run_id") or entry.get("candidate_selection_run_id")),
        "selected_candidate_id": _text(metadata.get("selected_candidate_id") or entry.get("selected_candidate_id")),
        "selected_run_id": _text(metadata.get("selected_run_id") or entry.get("selected_run_id")),
        "upstream_run_ids": "|".join(_string_list(metadata.get("upstream_run_ids") or entry.get("upstream_run_ids"))),
        "strategy_name": _text(entry.get("strategy_name") or manifest.get("strategy_name")),
        "portfolio_name": _text(entry.get("portfolio_name") or manifest.get("portfolio_name")),
        "alpha_model_name": _text(entry.get("alpha_name") or entry.get("alpha_model_name") or manifest.get("alpha_name")),
        "effective_n": _number_or_empty(metrics.get("effective_n")),
        "p_value": _number_or_empty(metrics.get("p_value")),
        "hit_rate_p_value": _number_or_empty(metrics.get("hit_rate_p_value")),
        "sharpe_stability_ratio": _number_or_empty(metrics.get("sharpe_stability_ratio")),
        "campaign_status": _text(metadata.get("campaign_status")),
        "scenario_status": _text(metadata.get("scenario_status")),
    }


def _triggered_gate_names(*payloads: Mapping[str, Any] | None) -> list[str]:
    results = None
    for payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        candidate_results = payload.get("results")
        if isinstance(candidate_results, list):
            results = candidate_results
            break
    if results is None:
        return []
    names = []
    for result in results:
        if not isinstance(result, Mapping):
            continue
        if result.get("status") not in {"fail", "missing"}:
            continue
        gate_id = result.get("gate_id")
        if isinstance(gate_id, str) and gate_id.strip():
            names.append(gate_id.strip())
    return sorted(set(names))


def _reason_code_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(_split_codes(row["decision_reason_codes"]))
    return dict(sorted(counts.items()))


def _split_codes(value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    return sorted({part.strip() for part in value.split("|") if part.strip()})


def _counts(values: Any) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for value in values:
        text = _text(value)
        if text:
            counts[text] += 1
    return dict(sorted(counts.items()))


def _non_empty_values(values: Any) -> set[str]:
    return {_text(value) for value in values if _text(value)}


def _top_reason_codes(counts: Mapping[str, int], *, prefix_filter: set[str]) -> list[dict[str, Any]]:
    rows = [
        {"reason_code": code, "count": count}
        for code, count in counts.items()
        if code in prefix_filter
    ]
    return sorted(rows, key=lambda row: (-int(row["count"]), str(row["reason_code"])))[:10]


def _fraction(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _relative_path(path: Path | None, *, base_dir: Path, roots: tuple[Path, ...] = ()) -> str:
    if path is None:
        return ""
    value = str(path)
    if PureWindowsPath(value).is_absolute():
        return portable_path(value)
    return portable_path(path, roots=(base_dir, *roots))


def _record_path_roots(record: GovernanceSourceRecord, *, path: Path | None) -> tuple[Path, ...]:
    roots: list[Path] = []
    if record.artifact_dir is not None:
        roots.append(record.artifact_dir.parent)
    if record.registry_path is not None:
        roots.append(record.registry_path.parent)
    if path is not None and path.is_absolute():
        roots.append(path.parent.parent if path.parent.parent != path.parent else path.parent)
    return tuple(roots)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted({str(item).strip() for item in value if str(item).strip()})


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _number_or_empty(value: Any) -> float | str:
    if value is None or isinstance(value, str) and not value.strip():
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(numeric):
        return ""
    return numeric


__all__ = [
    "OUTCOME_MATRIX_COLUMNS",
    "build_governance_outcome_rows",
    "build_governance_report_id",
    "build_governance_summary",
    "build_reason_code_summary",
    "build_severity_summary",
    "build_workflow_summary",
]
