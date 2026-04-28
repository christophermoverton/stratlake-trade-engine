from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from src.config.regime_promotion_gates import (
    GateCategoryConfig,
    RegimePromotionGateConfig,
)
from src.research.registry import canonicalize_value


GATE_RESULT_COLUMNS = (
    "variant_name",
    "variant_type",
    "regime_source",
    "calibration_profile",
    "classifier_model",
    "policy_name",
    "gate_name",
    "gate_category",
    "metric_name",
    "metric_value",
    "threshold_operator",
    "threshold_value",
    "severity",
    "status",
    "decision_impact",
    "message",
)
DECISION_ORDER = ("accepted", "accepted_with_warnings", "needs_review", "rejected")


@dataclass(frozen=True)
class RegimePromotionGateRunResult:
    benchmark_run_id: str
    gate_config_name: str
    decision_policy: str
    source_benchmark_path: Path
    output_dir: Path
    gate_config_path: Path
    gate_results_csv_path: Path
    gate_results_json_path: Path
    decision_summary_path: Path
    failed_gates_csv_path: Path
    warning_gates_csv_path: Path
    manifest_path: Path
    decision_summary: dict[str, Any]
    manifest: dict[str, Any]


def run_regime_promotion_gates(
    benchmark_path: Path,
    config: RegimePromotionGateConfig,
    *,
    output_dir: Path | None = None,
) -> RegimePromotionGateRunResult:
    benchmark_root = benchmark_path.resolve()
    if not benchmark_root.exists():
        raise FileNotFoundError(
            f"Regime benchmark path does not exist: {benchmark_root.as_posix()}"
        )
    matrix_path = _resolve_matrix_path(benchmark_root)
    matrix_rows = _load_benchmark_matrix(matrix_path)
    if not matrix_rows:
        raise ValueError(
            f"Regime benchmark matrix contains no variant rows: {matrix_path.as_posix()}"
        )

    summary = _load_optional_json(benchmark_root / "benchmark_summary.json")
    provenance = _load_optional_json(benchmark_root / "artifact_provenance.json")
    transition_metrics = _load_transition_metrics(benchmark_root / "transition_summary.json")
    policy_metrics = _load_policy_metrics(benchmark_root / "policy_comparison.csv")
    benchmark_run_id = _resolve_benchmark_run_id(matrix_rows, summary)
    resolved_output_dir = (output_dir or (benchmark_root / "promotion_gates")).resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    rows = _evaluate_rows(
        matrix_rows,
        config=config,
        transition_metrics=transition_metrics,
        policy_metrics=policy_metrics,
    )
    decisions = _build_variant_decisions(rows, config=config)
    decision_summary = _build_decision_summary(
        benchmark_run_id=benchmark_run_id,
        benchmark_root=benchmark_root,
        config=config,
        decisions=decisions,
        rows=rows,
    )

    gate_config_path = resolved_output_dir / "gate_config.json"
    gate_results_csv_path = resolved_output_dir / "gate_results.csv"
    gate_results_json_path = resolved_output_dir / "gate_results.json"
    decision_summary_path = resolved_output_dir / "decision_summary.json"
    failed_gates_csv_path = resolved_output_dir / "failed_gates.csv"
    warning_gates_csv_path = resolved_output_dir / "warning_gates.csv"
    manifest_path = resolved_output_dir / "manifest.json"

    _write_json(gate_config_path, config.to_dict())
    _write_csv(gate_results_csv_path, rows, fieldnames=GATE_RESULT_COLUMNS)
    _write_json(
        gate_results_json_path,
        {
            "benchmark_run_id": benchmark_run_id,
            "gate_config_name": config.gate_name,
            "row_count": len(rows),
            "columns": list(GATE_RESULT_COLUMNS),
            "rows": rows,
        },
    )
    _write_json(decision_summary_path, decision_summary)
    failed_rows = [row for row in rows if row["decision_impact"] == "fail"]
    warning_rows = [row for row in rows if row["decision_impact"] == "warn"]
    _write_csv(failed_gates_csv_path, failed_rows, fieldnames=GATE_RESULT_COLUMNS)
    _write_csv(warning_gates_csv_path, warning_rows, fieldnames=GATE_RESULT_COLUMNS)

    manifest = _build_manifest(
        benchmark_run_id=benchmark_run_id,
        benchmark_root=benchmark_root,
        output_dir=resolved_output_dir,
        config=config,
        rows=rows,
        failed_rows=failed_rows,
        warning_rows=warning_rows,
        decision_summary=decision_summary,
        source_paths={
            "benchmark_matrix": matrix_path,
            "benchmark_summary": benchmark_root / "benchmark_summary.json"
            if isinstance(summary, dict)
            else None,
            "artifact_provenance": benchmark_root / "artifact_provenance.json"
            if isinstance(provenance, dict)
            else None,
        },
    )
    _write_json(manifest_path, manifest)

    return RegimePromotionGateRunResult(
        benchmark_run_id=benchmark_run_id,
        gate_config_name=config.gate_name,
        decision_policy=config.decision_policy,
        source_benchmark_path=benchmark_root,
        output_dir=resolved_output_dir,
        gate_config_path=gate_config_path,
        gate_results_csv_path=gate_results_csv_path,
        gate_results_json_path=gate_results_json_path,
        decision_summary_path=decision_summary_path,
        failed_gates_csv_path=failed_gates_csv_path,
        warning_gates_csv_path=warning_gates_csv_path,
        manifest_path=manifest_path,
        decision_summary=decision_summary,
        manifest=manifest,
    )


def _evaluate_rows(
    matrix_rows: list[dict[str, Any]],
    *,
    config: RegimePromotionGateConfig,
    transition_metrics: Mapping[str, Mapping[str, Any]],
    policy_metrics: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in matrix_rows:
        enriched = dict(row)
        variant_name = str(enriched.get("variant_name"))
        enriched.update(transition_metrics.get(variant_name, {}))
        enriched.update(policy_metrics.get(variant_name, {}))
        for spec in _gate_specs():
            category = config.category(spec["category"])
            if not category.enabled:
                continue
            results.append(_evaluate_metric(enriched, category=category, config=config, spec=spec))
    return [canonicalize_value(row) for row in results]


def _gate_specs() -> list[dict[str, Any]]:
    return [
        _spec("confidence", "min_mean_confidence", "mean_confidence", ">=", "min", _gmm_applicable),
        _spec("confidence", "min_median_confidence", "median_confidence", ">=", "min", _gmm_applicable),
        _spec("confidence", "max_low_confidence_pct", "low_confidence_pct", "<=", "max", _gmm_applicable),
        _spec("entropy", "max_mean_entropy", "mean_entropy", "<=", "max", _gmm_applicable),
        _spec("entropy", "max_high_entropy_pct", "high_entropy_pct", "<=", "max", _gmm_applicable),
        _spec("stability", "min_avg_regime_duration", "avg_regime_duration", ">=", "min", _regime_applicable),
        _spec("stability", "max_dominant_regime_share", "dominant_regime_share", "<=", "max", _regime_applicable),
        _spec("stability", "max_transition_rate", "transition_rate", "<=", "max", _regime_applicable),
        _spec("transition_behavior", "max_transition_rate", "transition_rate", "<=", "max", _regime_applicable),
        _spec(
            "transition_behavior",
            "max_transition_instability_score",
            "transition_instability_score",
            "<=",
            "max",
            _regime_applicable,
        ),
        _spec(
            "transition_behavior",
            "max_transition_concentration",
            "transition_concentration",
            "<=",
            "max",
            _regime_applicable,
        ),
        _spec("adaptive_uplift", "min_return_delta", "adaptive_vs_static_return_delta", ">=", "min", _policy_applicable),
        _spec("adaptive_uplift", "min_sharpe_delta", "adaptive_vs_static_sharpe_delta", ">=", "min", _policy_applicable),
        _spec(
            "adaptive_uplift",
            "min_drawdown_delta",
            "adaptive_vs_static_max_drawdown_delta",
            ">=",
            "min",
            _policy_applicable,
        ),
        _spec("drawdown", "max_allowed_drawdown", "max_drawdown", ">=", "min", _drawdown_applicable),
        _spec(
            "drawdown",
            "max_high_vol_drawdown",
            "high_volatility_regime_drawdown",
            ">=",
            "min",
            _policy_applicable,
        ),
        _spec("policy_turnover", "max_policy_turnover", "policy_turnover", "<=", "max", _policy_applicable),
        _spec(
            "policy_turnover",
            "max_policy_state_changes",
            "policy_state_change_count",
            "<=",
            "max",
            _policy_applicable,
        ),
    ]


def _spec(
    category: str,
    gate_name: str,
    metric_name: str,
    operator: str,
    threshold_kind: str,
    applicable: Callable[[Mapping[str, Any]], bool],
) -> dict[str, Any]:
    return {
        "category": category,
        "gate_name": gate_name,
        "metric_name": metric_name,
        "operator": operator,
        "threshold_kind": threshold_kind,
        "applicable": applicable,
    }


def _evaluate_metric(
    row: Mapping[str, Any],
    *,
    category: GateCategoryConfig,
    config: RegimePromotionGateConfig,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    variant_type = str(row.get("variant_type") or "")
    metric_name = str(spec["metric_name"])
    threshold_name = str(spec["gate_name"])
    threshold = category.thresholds.get(threshold_name)
    applies = bool(spec["applicable"](row))
    required = variant_type in set(category.required_for_variants)
    base = {
        "variant_name": row.get("variant_name"),
        "variant_type": row.get("variant_type"),
        "regime_source": row.get("regime_source"),
        "calibration_profile": row.get("calibration_profile"),
        "classifier_model": row.get("classifier_model"),
        "policy_name": row.get("policy_name"),
        "gate_name": threshold_name,
        "gate_category": spec["category"],
        "metric_name": metric_name,
        "threshold_operator": spec["operator"],
        "threshold_value": threshold,
    }
    if threshold is None:
        return {
            **base,
            "metric_value": None,
            "severity": "info",
            "status": "not_applicable",
            "decision_impact": "ignore",
            "message": f"Gate {threshold_name} has no configured threshold and was skipped.",
        }
    if not applies:
        impact = config.missing_metric_policy.not_applicable_metric
        return {
            **base,
            "metric_value": None,
            "severity": _severity_for_impact(impact),
            "status": "not_applicable",
            "decision_impact": impact,
            "message": (
                f"Metric {metric_name} missing for {variant_type} and treated as "
                "not_applicable."
            ),
        }

    value = _metric_value(row, metric_name)
    if value is None:
        impact = (
            config.missing_metric_policy.required_metric_missing
            if required
            else config.missing_metric_policy.optional_metric_missing
        )
        required_label = "required" if required else "optional"
        return {
            **base,
            "metric_value": None,
            "severity": _severity_for_impact(impact),
            "status": "metric_missing",
            "decision_impact": impact,
            "message": (
                f"Metric {metric_name} missing for {variant_type} and treated as "
                f"{impact} because it is {required_label}."
            ),
        }

    passed = _compare(value, float(threshold), operator=str(spec["operator"]))
    if passed:
        return {
            **base,
            "metric_value": value,
            "severity": "info",
            "status": "passed",
            "decision_impact": "pass",
            "message": (
                f"Metric {metric_name} passed: {_format_value(value)} "
                f"{spec['operator']} {_format_value(threshold)}."
            ),
        }
    impact = category.failure_impact
    return {
        **base,
        "metric_value": value,
        "severity": _severity_for_impact(impact),
        "status": "needs_review" if impact == "review" else "failed",
        "decision_impact": impact,
        "message": (
            f"Metric {metric_name} failed: {_format_value(value)} "
            f"{spec['operator']} {_format_value(threshold)}."
        ),
    }


def _metric_value(row: Mapping[str, Any], metric_name: str) -> float | None:
    if metric_name == "low_confidence_pct":
        return _ratio(row.get("low_confidence_observation_count"), row.get("observation_count"))
    if metric_name == "high_entropy_pct":
        return _ratio(row.get("high_entropy_observation_count"), row.get("observation_count"))
    return _finite_float(row.get(metric_name))


def _build_variant_decisions(
    rows: list[dict[str, Any]],
    *,
    config: RegimePromotionGateConfig,
) -> list[dict[str, Any]]:
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant_name"]), []).append(row)

    decisions: list[dict[str, Any]] = []
    for variant_name, gate_rows in by_variant.items():
        impacts = [str(row["decision_impact"]) for row in gate_rows]
        if "fail" in impacts:
            decision_key = "fail"
        elif "review" in impacts:
            decision_key = "review"
        elif "warn" in impacts:
            decision_key = "warn"
        else:
            decision_key = "pass"
        decision = config.outcomes[decision_key]
        reasons = [
            str(row["message"])
            for row in gate_rows
            if row["decision_impact"] in {"fail", "review", "warn"}
        ]
        if not reasons:
            reasons = [
                str(row["message"])
                for row in gate_rows
                if row["status"] == "passed"
            ][:3]
        decisions.append(
            canonicalize_value(
                {
                    "variant_name": variant_name,
                    "variant_type": gate_rows[0].get("variant_type"),
                    "decision": decision,
                    "passed_gates": _count_status(gate_rows, "passed"),
                    "warning_gates": impacts.count("warn"),
                    "review_gates": impacts.count("review"),
                    "failed_gates": impacts.count("fail"),
                    "missing_metric_gates": _count_status(gate_rows, "metric_missing"),
                    "not_applicable_gates": _count_status(gate_rows, "not_applicable"),
                    "primary_reasons": reasons[:5],
                }
            )
        )
    return decisions


def _build_decision_summary(
    *,
    benchmark_run_id: str,
    benchmark_root: Path,
    config: RegimePromotionGateConfig,
    decisions: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    decision_counts = {decision: 0 for decision in DECISION_ORDER}
    for decision in decisions:
        decision_counts[str(decision["decision"])] = decision_counts.get(str(decision["decision"]), 0) + 1
    return canonicalize_value(
        {
            "benchmark_run_id": benchmark_run_id,
            "gate_config_name": config.gate_name,
            "decision_policy": config.decision_policy,
            "source_benchmark_path": _relative_path(benchmark_root, base_dir=Path.cwd()),
            "decision_counts": decision_counts,
            "variant_decisions": decisions,
            "passed_gate_count": _count_status(rows, "passed"),
            "warning_gate_count": _count_impact(rows, "warn"),
            "review_gate_count": _count_impact(rows, "review"),
            "failed_gate_count": _count_impact(rows, "fail"),
            "missing_metric_gate_count": _count_status(rows, "metric_missing"),
            "not_applicable_gate_count": _count_status(rows, "not_applicable"),
        }
    )


def _build_manifest(
    *,
    benchmark_run_id: str,
    benchmark_root: Path,
    output_dir: Path,
    config: RegimePromotionGateConfig,
    rows: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
    warning_rows: list[dict[str, Any]],
    decision_summary: Mapping[str, Any],
    source_paths: Mapping[str, Path | None],
) -> dict[str, Any]:
    generated_files = [
        "gate_config.json",
        "gate_results.csv",
        "gate_results.json",
        "decision_summary.json",
        "failed_gates.csv",
        "warning_gates.csv",
        "manifest.json",
    ]
    return canonicalize_value(
        {
            "artifact_type": "regime_promotion_gates",
            "schema_version": 1,
            "source_benchmark_path": _relative_path(benchmark_root, base_dir=Path.cwd()),
            "source_benchmark_run_id": benchmark_run_id,
            "gate_config_name": config.gate_name,
            "generated_files": generated_files,
            "row_counts": {
                "gate_results": len(rows),
                "failed_gates": len(failed_rows),
                "warning_gates": len(warning_rows),
                "variant_decisions": len(decision_summary.get("variant_decisions", [])),
            },
            "decision_counts": decision_summary.get("decision_counts", {}),
            "warning_count": len(warning_rows),
            "failure_count": len(failed_rows),
            "source_paths": {
                key: _relative_path(path, base_dir=benchmark_root)
                for key, path in source_paths.items()
                if path is not None
            },
            "file_inventory": {
                relative: {
                    "path": relative,
                    "rows": _row_count(output_dir / relative),
                }
                for relative in generated_files
            },
        }
    )


def _resolve_matrix_path(benchmark_root: Path) -> Path:
    csv_path = benchmark_root / "benchmark_matrix.csv"
    json_path = benchmark_root / "benchmark_matrix.json"
    if csv_path.exists():
        return csv_path
    if json_path.exists():
        return json_path
    raise FileNotFoundError(
        "Regime benchmark path must contain benchmark_matrix.csv or "
        f"benchmark_matrix.json: {benchmark_root.as_posix()}"
    )


def _load_benchmark_matrix(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise ValueError(f"Benchmark matrix JSON must contain a rows array: {path.as_posix()}")
        return [dict(row) for row in rows if isinstance(row, Mapping)]
    frame = pd.read_csv(path, keep_default_na=True)
    return [_normalize_csv_row(row) for row in frame.to_dict(orient="records")]


def _load_transition_metrics(path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_optional_json(path)
    if not isinstance(payload, dict):
        return {}
    variants = payload.get("variants")
    if not isinstance(variants, list):
        return {}
    metrics: dict[str, dict[str, Any]] = {}
    for item in variants:
        if not isinstance(item, Mapping) or item.get("variant_name") is None:
            continue
        metrics[str(item["variant_name"])] = {
            key: item.get(key)
            for key in ("transition_instability_score", "transition_concentration")
        }
    return metrics


def _load_policy_metrics(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path, keep_default_na=True)
    metrics: dict[str, dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        variant_name = row.get("variant_name")
        if variant_name is None or _is_missing(variant_name):
            continue
        metrics[str(variant_name)] = {
            key: row.get(key)
            for key in (
                "high_volatility_regime_drawdown",
                "transition_window_performance_delta",
            )
        }
    return metrics


def _resolve_benchmark_run_id(
    matrix_rows: list[dict[str, Any]],
    summary: Mapping[str, Any] | None,
) -> str:
    if isinstance(summary, Mapping) and summary.get("benchmark_run_id") is not None:
        return str(summary["benchmark_run_id"])
    value = matrix_rows[0].get("benchmark_run_id")
    if value is None or _is_missing(value):
        return "unknown"
    return str(value)


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _normalize_csv_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: (None if _is_missing(value) else value) for key, value in row.items()}


def _gmm_applicable(row: Mapping[str, Any]) -> bool:
    return str(row.get("variant_type")) in {"gmm_classifier", "gmm_calibrated_overlay", "policy_optimized"}


def _regime_applicable(row: Mapping[str, Any]) -> bool:
    return str(row.get("variant_type")) != "static_baseline"


def _policy_applicable(row: Mapping[str, Any]) -> bool:
    return str(row.get("variant_type")) == "policy_optimized"


def _drawdown_applicable(row: Mapping[str, Any]) -> bool:
    return _policy_applicable(row) or _metric_value(row, "max_drawdown") is not None


def _compare(value: float, threshold: float, *, operator: str) -> bool:
    if operator == ">=":
        return value >= threshold
    if operator == "<=":
        return value <= threshold
    raise ValueError(f"Unsupported gate threshold operator: {operator!r}")


def _severity_for_impact(impact: str) -> str:
    if impact == "fail":
        return "failure"
    if impact == "review":
        return "review"
    if impact == "warn":
        return "warning"
    return "info"


def _ratio(numerator: Any, denominator: Any) -> float | None:
    top = _finite_float(numerator)
    bottom = _finite_float(denominator)
    if top is None or bottom is None or bottom == 0.0:
        return None
    return _finite_float(top / bottom)


def _finite_float(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _count_status(rows: list[dict[str, Any]], status: str) -> int:
    return int(sum(1 for row in rows if row.get("status") == status))


def _count_impact(rows: list[dict[str, Any]], impact: str) -> int:
    return int(sum(1 for row in rows if row.get("decision_impact") == impact))


def _format_value(value: Any) -> str:
    numeric = _finite_float(value)
    if numeric is None:
        return str(value)
    return f"{numeric:.6g}"


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(canonicalize_value(payload), indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )


def _relative_path(path: Path, *, base_dir: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _row_count(path: Path) -> int | None:
    if not path.exists():
        return None
    suffix = path.suffix.lower()
    if suffix == ".csv":
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return 0
        count = text.count("\n")
        if text.endswith("\n"):
            count -= 1
        return max(count, 0)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            return int(len(payload["rows"]))
    return None


__all__ = [
    "GATE_RESULT_COLUMNS",
    "RegimePromotionGateRunResult",
    "run_regime_promotion_gates",
]
