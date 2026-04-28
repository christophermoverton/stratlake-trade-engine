from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.config.regime_review_pack import RegimeReviewPackConfig
from src.research.registry import canonicalize_value


DECISION_ORDER = ("accepted", "accepted_with_warnings", "needs_review", "rejected")
LEADERBOARD_COLUMNS = (
    "rank",
    "variant_name",
    "variant_type",
    "regime_source",
    "calibration_profile",
    "classifier_model",
    "policy_name",
    "decision",
    "decision_rank",
    "accepted",
    "warning_count",
    "failed_gate_count",
    "passed_gate_count",
    "mean_confidence",
    "mean_entropy",
    "transition_rate",
    "avg_regime_duration",
    "dominant_regime_share",
    "adaptive_vs_static_return_delta",
    "adaptive_vs_static_sharpe_delta",
    "adaptive_vs_static_max_drawdown_delta",
    "policy_turnover",
    "primary_reason",
    "source_benchmark_run_id",
    "source_gate_config_name",
)
REJECTED_COLUMNS = (
    "variant_name",
    "variant_type",
    "decision",
    "primary_failure_category",
    "failed_gate_count",
    "warning_gate_count",
    "primary_reason",
    "failed_metrics",
    "source_metric_values",
    "source_thresholds",
    "review_notes",
)
_HIGHER_IS_BETTER = {
    "adaptive_vs_static_sharpe_delta",
    "adaptive_vs_static_return_delta",
    "adaptive_vs_static_max_drawdown_delta",
    "mean_confidence",
    "avg_regime_duration",
}
_LOWER_IS_BETTER = {"transition_rate", "policy_turnover", "mean_entropy", "dominant_regime_share"}
_KEY_METRICS = (
    "mean_confidence",
    "mean_entropy",
    "transition_rate",
    "avg_regime_duration",
    "dominant_regime_share",
    "adaptive_vs_static_return_delta",
    "adaptive_vs_static_sharpe_delta",
    "adaptive_vs_static_max_drawdown_delta",
    "policy_turnover",
)


@dataclass(frozen=True)
class RegimeReviewPackRunResult:
    review_run_id: str
    review_name: str
    source_benchmark_run_id: str
    source_gate_config_name: str
    output_dir: Path
    leaderboard_csv_path: Path
    leaderboard_json_path: Path
    decision_log_path: Path
    review_summary_path: Path
    accepted_variants_path: Path
    warning_variants_path: Path
    needs_review_variants_path: Path
    rejected_candidates_path: Path
    evidence_index_path: Path
    config_path: Path
    manifest_path: Path
    report_path: Path | None
    review_inputs_path: Path
    review_summary: dict[str, Any]
    manifest: dict[str, Any]


def run_regime_review_pack(config: RegimeReviewPackConfig) -> RegimeReviewPackRunResult:
    benchmark_root = Path(config.benchmark_path).resolve()
    gates_root = Path(config.promotion_gates_path).resolve()
    _require_dir(benchmark_root, label="Regime benchmark path")
    _require_dir(gates_root, label="Regime promotion-gates path")
    required = _required_paths(benchmark_root, gates_root)
    for label, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"Regime review pack requires {label}: {path.as_posix()}")

    optional = {
        "benchmark_summary": benchmark_root / "benchmark_summary.json",
        "artifact_provenance": benchmark_root / "artifact_provenance.json",
        "gate_results_json": gates_root / "gate_results.json",
        "failed_gates": gates_root / "failed_gates.csv",
        "warning_gates": gates_root / "warning_gates.csv",
        "promotion_manifest": gates_root / "manifest.json",
    }
    missing_optional = [
        f"Optional artifact {label} was not found at {_rel(path)}."
        for label, path in optional.items()
        if not path.exists()
    ]

    matrix_rows = _load_rows(required["benchmark_matrix"])
    gate_rows = _load_rows(required["gate_results"])
    decision_summary = _load_json(required["decision_summary"])
    benchmark_summary = _load_optional_json(optional["benchmark_summary"])
    source_benchmark_run_id = _source_benchmark_run_id(matrix_rows, benchmark_summary, decision_summary)
    source_gate_config_name = str(decision_summary.get("gate_config_name") or "unknown")
    review_run_id = _build_review_run_id(config, source_benchmark_run_id, source_gate_config_name)
    output_dir = Path(config.output_root).resolve() / review_run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    decisions = {
        str(item["variant_name"]): dict(item)
        for item in decision_summary.get("variant_decisions", [])
        if isinstance(item, Mapping) and item.get("variant_name") is not None
    }
    if len(decisions) != len(matrix_rows):
        missing = sorted(str(row.get("variant_name")) for row in matrix_rows if str(row.get("variant_name")) not in decisions)
        if missing:
            raise ValueError(f"Promotion-gate decision_summary is missing variant decisions for: {missing}.")

    gate_rows_by_variant = _group_gate_rows(gate_rows)
    leaderboard = _build_leaderboard(
        matrix_rows,
        decisions=decisions,
        gate_rows_by_variant=gate_rows_by_variant,
        config=config,
        source_benchmark_run_id=source_benchmark_run_id,
        source_gate_config_name=source_gate_config_name,
    )
    generated_files = [
        "leaderboard.csv",
        "leaderboard.json",
        "decision_log.json",
        "review_summary.json",
        "accepted_variants.csv",
        "warning_variants.csv",
        "needs_review_variants.csv",
        "rejected_candidates.csv",
        "evidence_index.json",
        "config.json",
        "manifest.json",
        "review_inputs.json",
    ]
    if config.report.write_markdown:
        generated_files.append("report.md")
    evidence = _build_evidence_index(
        benchmark_matrix_path=required["benchmark_matrix"],
        gates_root=gates_root,
        output_dir=output_dir,
        leaderboard=leaderboard,
        generated_files=generated_files,
        missing_optional=missing_optional,
    )
    decision_log = _build_decision_log(
        leaderboard,
        decisions=decisions,
        gate_rows_by_variant=gate_rows_by_variant,
        evidence=evidence,
    )

    rejected = _build_rejected_rows(leaderboard, gate_rows_by_variant)
    summary = _build_review_summary(
        review_run_id=review_run_id,
        config=config,
        leaderboard=leaderboard,
        source_benchmark_run_id=source_benchmark_run_id,
        source_gate_config_name=source_gate_config_name,
        generated_files=generated_files,
        benchmark_root=benchmark_root,
        gates_root=gates_root,
    )

    paths = {
        "leaderboard_csv": output_dir / "leaderboard.csv",
        "leaderboard_json": output_dir / "leaderboard.json",
        "decision_log": output_dir / "decision_log.json",
        "review_summary": output_dir / "review_summary.json",
        "accepted": output_dir / "accepted_variants.csv",
        "warnings": output_dir / "warning_variants.csv",
        "needs_review": output_dir / "needs_review_variants.csv",
        "rejected": output_dir / "rejected_candidates.csv",
        "evidence": output_dir / "evidence_index.json",
        "config": output_dir / "config.json",
        "manifest": output_dir / "manifest.json",
        "review_inputs": output_dir / "review_inputs.json",
    }
    _write_csv(paths["leaderboard_csv"], leaderboard, fieldnames=LEADERBOARD_COLUMNS)
    _write_json(paths["leaderboard_json"], {"rows": leaderboard, "columns": list(LEADERBOARD_COLUMNS), "row_count": len(leaderboard)})
    _write_json(paths["decision_log"], {"rows": decision_log, "row_count": len(decision_log)})
    _write_json(paths["review_summary"], summary)
    _write_csv(paths["accepted"], [r for r in leaderboard if r["decision"] == "accepted"], fieldnames=LEADERBOARD_COLUMNS)
    _write_csv(paths["warnings"], [r for r in leaderboard if r["decision"] == "accepted_with_warnings"], fieldnames=LEADERBOARD_COLUMNS)
    _write_csv(paths["needs_review"], [r for r in leaderboard if r["decision"] == "needs_review"], fieldnames=LEADERBOARD_COLUMNS)
    _write_csv(paths["rejected"], rejected, fieldnames=REJECTED_COLUMNS)
    _write_json(paths["evidence"], evidence)
    _write_json(paths["config"], config.to_dict())
    _write_json(paths["review_inputs"], _review_inputs(benchmark_root, gates_root, required, optional, missing_optional))
    report_path = output_dir / "report.md" if config.report.write_markdown else None
    if report_path is not None:
        _write_report(report_path, summary=summary, leaderboard=leaderboard, evidence=evidence, config=config)

    manifest = _build_manifest(
        review_run_id=review_run_id,
        config=config,
        leaderboard=leaderboard,
        summary=summary,
        generated_files=generated_files,
        output_dir=output_dir,
        missing_optional=missing_optional,
        source_benchmark_run_id=source_benchmark_run_id,
        source_gate_config_name=source_gate_config_name,
        benchmark_root=benchmark_root,
        gates_root=gates_root,
    )
    _write_json(paths["manifest"], manifest)
    return RegimeReviewPackRunResult(
        review_run_id=review_run_id,
        review_name=config.review_name,
        source_benchmark_run_id=source_benchmark_run_id,
        source_gate_config_name=source_gate_config_name,
        output_dir=output_dir,
        leaderboard_csv_path=paths["leaderboard_csv"],
        leaderboard_json_path=paths["leaderboard_json"],
        decision_log_path=paths["decision_log"],
        review_summary_path=paths["review_summary"],
        accepted_variants_path=paths["accepted"],
        warning_variants_path=paths["warnings"],
        needs_review_variants_path=paths["needs_review"],
        rejected_candidates_path=paths["rejected"],
        evidence_index_path=paths["evidence"],
        config_path=paths["config"],
        manifest_path=paths["manifest"],
        report_path=report_path,
        review_inputs_path=paths["review_inputs"],
        review_summary=summary,
        manifest=manifest,
    )


def _build_leaderboard(
    matrix_rows: list[dict[str, Any]],
    *,
    decisions: Mapping[str, dict[str, Any]],
    gate_rows_by_variant: Mapping[str, list[dict[str, Any]]],
    config: RegimeReviewPackConfig,
    source_benchmark_run_id: str,
    source_gate_config_name: str,
) -> list[dict[str, Any]]:
    decision_rank = {decision: index + 1 for index, decision in enumerate(config.ranking.decision_order)}
    rows = []
    for matrix in matrix_rows:
        variant_name = str(matrix.get("variant_name"))
        decision = decisions[variant_name]
        gate_rows = gate_rows_by_variant.get(variant_name, [])
        row = {
            "rank": None,
            "variant_name": variant_name,
            "variant_type": matrix.get("variant_type"),
            "regime_source": matrix.get("regime_source"),
            "calibration_profile": matrix.get("calibration_profile"),
            "classifier_model": matrix.get("classifier_model"),
            "policy_name": matrix.get("policy_name"),
            "decision": decision.get("decision"),
            "decision_rank": decision_rank.get(str(decision.get("decision")), len(decision_rank) + 1),
            "accepted": str(decision.get("decision")) in {"accepted", "accepted_with_warnings"},
            "warning_count": _count_impact(gate_rows, "warn"),
            "failed_gate_count": _count_impact(gate_rows, "fail"),
            "passed_gate_count": _count_status(gate_rows, "passed"),
            "primary_reason": _primary_reason(decision),
            "source_benchmark_run_id": source_benchmark_run_id,
            "source_gate_config_name": source_gate_config_name,
        }
        for metric in _KEY_METRICS:
            row[metric] = _finite_float(matrix.get(metric))
        rows.append(canonicalize_value(row))
    rows.sort(key=lambda row: _ranking_key(row, config=config))
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows


def _ranking_key(row: Mapping[str, Any], *, config: RegimeReviewPackConfig) -> tuple[Any, ...]:
    key: list[Any] = [int(row.get("decision_rank") or 999)]
    for metric in config.ranking.metric_priority:
        value = _finite_float(row.get(metric))
        if value is None:
            key.append((1, 0.0))
        elif metric in _LOWER_IS_BETTER:
            key.append((0, value))
        else:
            key.append((0, -value))
    for field in config.ranking.tie_breakers:
        value = row.get(field)
        key.append("" if value is None else str(value))
    key.append(str(row.get("variant_name") or ""))
    return tuple(key)


def _build_decision_log(
    leaderboard: list[dict[str, Any]],
    *,
    decisions: Mapping[str, dict[str, Any]],
    gate_rows_by_variant: Mapping[str, list[dict[str, Any]]],
    evidence: Mapping[str, Any],
) -> list[dict[str, Any]]:
    entries = []
    for row in leaderboard:
        variant_name = str(row["variant_name"])
        gate_rows = gate_rows_by_variant.get(variant_name, [])
        entry = {
            "variant_name": variant_name,
            "variant_type": row.get("variant_type"),
            "decision": row.get("decision"),
            "rank": row.get("rank"),
            "summary": _decision_summary_sentence(row, decisions[variant_name]),
            "passed_gates": _gate_names(gate_rows, status="passed"),
            "warning_gates": _gate_names(gate_rows, impact="warn"),
            "review_gates": _gate_names(gate_rows, impact="review"),
            "failed_gates": _gate_names(gate_rows, impact="fail"),
            "missing_metric_gates": _gate_names(gate_rows, status="metric_missing"),
            "not_applicable_gates": _gate_names(gate_rows, status="not_applicable"),
            "primary_reasons": list(decisions[variant_name].get("primary_reasons") or []),
            "key_metrics": {metric: row.get(metric) for metric in _KEY_METRICS if row.get(metric) is not None},
            "evidence": evidence["variant_evidence"].get(variant_name, {}),
        }
        entries.append(canonicalize_value(entry))
    return entries


def _build_evidence_index(
    *,
    benchmark_matrix_path: Path,
    gates_root: Path,
    output_dir: Path,
    leaderboard: list[dict[str, Any]],
    generated_files: list[str],
    missing_optional: list[str],
) -> dict[str, Any]:
    source_artifacts = {
        "benchmark_matrix": _rel(benchmark_matrix_path),
        "gate_results": _rel(gates_root / "gate_results.csv"),
        "decision_summary": _rel(gates_root / "decision_summary.json"),
    }
    variant_evidence = {
        str(row["variant_name"]): {
            "benchmark_matrix": source_artifacts["benchmark_matrix"],
            "gate_results": source_artifacts["gate_results"],
            "decision_summary": source_artifacts["decision_summary"],
        }
        for row in leaderboard
    }
    return canonicalize_value(
        {
            "source_artifacts": source_artifacts,
            "variant_evidence": variant_evidence,
            "generated_review_artifacts": {name: _rel(output_dir / name) for name in generated_files},
            "missing_optional_artifacts": missing_optional,
        }
    )


def _build_review_summary(
    *,
    review_run_id: str,
    config: RegimeReviewPackConfig,
    leaderboard: list[dict[str, Any]],
    source_benchmark_run_id: str,
    source_gate_config_name: str,
    generated_files: list[str],
    benchmark_root: Path,
    gates_root: Path,
) -> dict[str, Any]:
    counts = {decision: 0 for decision in DECISION_ORDER}
    for row in leaderboard:
        counts[str(row["decision"])] = counts.get(str(row["decision"]), 0) + 1
    top = leaderboard[0] if leaderboard else None
    return canonicalize_value(
        {
            "review_run_id": review_run_id,
            "review_name": config.review_name,
            "source_benchmark_run_id": source_benchmark_run_id,
            "source_gate_config_name": source_gate_config_name,
            "variant_count": len(leaderboard),
            "decision_counts": counts,
            "top_variant": None
            if top is None
            else {"variant_name": top["variant_name"], "decision": top["decision"], "rank": top["rank"]},
            "rejected_count": counts.get("rejected", 0),
            "warning_count": counts.get("accepted_with_warnings", 0),
            "needs_review_count": counts.get("needs_review", 0),
            "accepted_count": counts.get("accepted", 0),
            "rejection_summary": _reason_summary(leaderboard, "rejected"),
            "warning_summary": _reason_summary(leaderboard, "accepted_with_warnings"),
            "needs_review_summary": _reason_summary(leaderboard, "needs_review"),
            "generated_artifacts": generated_files,
            "source_paths": {"benchmark_path": _rel(benchmark_root), "promotion_gates_path": _rel(gates_root)},
        }
    )


def _build_rejected_rows(
    leaderboard: list[dict[str, Any]],
    gate_rows_by_variant: Mapping[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for row in leaderboard:
        if row["decision"] != "rejected":
            continue
        gates = [gate for gate in gate_rows_by_variant.get(str(row["variant_name"]), []) if gate.get("decision_impact") == "fail"]
        rows.append(
            canonicalize_value(
                {
                    "variant_name": row["variant_name"],
                    "variant_type": row["variant_type"],
                    "decision": row["decision"],
                    "primary_failure_category": gates[0].get("gate_category") if gates else None,
                    "failed_gate_count": row["failed_gate_count"],
                    "warning_gate_count": row["warning_count"],
                    "primary_reason": row["primary_reason"],
                    "failed_metrics": ";".join(str(gate.get("metric_name")) for gate in gates),
                    "source_metric_values": json.dumps({str(gate.get("metric_name")): gate.get("metric_value") for gate in gates}, sort_keys=True),
                    "source_thresholds": json.dumps({str(gate.get("metric_name")): gate.get("threshold_value") for gate in gates}, sort_keys=True),
                    "review_notes": "Review failed gate rows and source benchmark metrics before reconsideration.",
                }
            )
        )
    return rows


def _build_manifest(
    *,
    review_run_id: str,
    config: RegimeReviewPackConfig,
    leaderboard: list[dict[str, Any]],
    summary: Mapping[str, Any],
    generated_files: list[str],
    output_dir: Path,
    missing_optional: list[str],
    source_benchmark_run_id: str,
    source_gate_config_name: str,
    benchmark_root: Path,
    gates_root: Path,
) -> dict[str, Any]:
    return canonicalize_value(
        {
            "artifact_type": "regime_review_pack",
            "schema_version": 1,
            "review_run_id": review_run_id,
            "review_name": config.review_name,
            "source_benchmark_run_id": source_benchmark_run_id,
            "source_gate_config_name": source_gate_config_name,
            "source_paths": {"benchmark_path": _rel(benchmark_root), "promotion_gates_path": _rel(gates_root)},
            "generated_files": generated_files,
            "row_counts": {
                "leaderboard": len(leaderboard),
                "accepted_variants": _decision_count(leaderboard, "accepted"),
                "warning_variants": _decision_count(leaderboard, "accepted_with_warnings"),
                "needs_review_variants": _decision_count(leaderboard, "needs_review"),
                "rejected_candidates": _decision_count(leaderboard, "rejected"),
            },
            "decision_counts": summary.get("decision_counts", {}),
            "warning_count": summary.get("warning_count", 0),
            "rejection_count": summary.get("rejected_count", 0),
            "missing_optional_artifact_warnings": missing_optional,
            "file_inventory": {
                name: {"path": name, "rows": _row_count(output_dir / name)}
                for name in generated_files
            },
        }
    )


def _write_report(
    path: Path,
    *,
    summary: Mapping[str, Any],
    leaderboard: list[dict[str, Any]],
    evidence: Mapping[str, Any],
    config: RegimeReviewPackConfig,
) -> None:
    lines = [
        f"# {config.review_name}",
        "",
        f"Source benchmark run id: `{summary['source_benchmark_run_id']}`",
        f"Source gate config name: `{summary['source_gate_config_name']}`",
        "",
        "## Decision Counts",
    ]
    counts = summary.get("decision_counts", {})
    for decision in DECISION_ORDER:
        lines.append(f"- {decision}: {counts.get(decision, 0)}")
    top = summary.get("top_variant")
    if top:
        lines.extend(["", "## Top Variant", f"- {top['rank']}. {top['variant_name']} ({top['decision']})"])
    for title, decision in (
        ("Accepted Variants", "accepted"),
        ("Warning Variants", "accepted_with_warnings"),
        ("Needs-Review Variants", "needs_review"),
        ("Rejected Variants", "rejected"),
    ):
        if decision == "accepted_with_warnings" and not config.report.include_warning_summary:
            continue
        if decision == "rejected" and not config.report.include_rejected_candidates:
            continue
        lines.extend(["", f"## {title}"])
        rows = [row for row in leaderboard if row["decision"] == decision]
        lines.extend([f"- #{row['rank']} {row['variant_name']}: {row['primary_reason']}" for row in rows] or ["- None"])
    warnings = evidence.get("missing_optional_artifacts", [])
    lines.extend(["", "## Key Limitations / Warnings"])
    lines.extend([f"- {warning}" for warning in warnings] or ["- No optional artifact warnings."])
    if config.report.include_evidence_index:
        lines.extend(["", "## Artifact Index"])
        for label, artifact_path in evidence.get("source_artifacts", {}).items():
            lines.append(f"- {label}: `{artifact_path}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _review_inputs(
    benchmark_root: Path,
    gates_root: Path,
    required: Mapping[str, Path],
    optional: Mapping[str, Path],
    missing_optional: list[str],
) -> dict[str, Any]:
    return canonicalize_value(
        {
            "benchmark_path": _rel(benchmark_root),
            "promotion_gates_path": _rel(gates_root),
            "required_artifacts": {key: _rel(path) for key, path in required.items()},
            "optional_artifacts": {key: _rel(path) for key, path in optional.items()},
            "missing_optional_artifacts": missing_optional,
        }
    )


def _required_paths(benchmark_root: Path, gates_root: Path) -> dict[str, Path]:
    matrix = benchmark_root / "benchmark_matrix.csv"
    if not matrix.exists() and (benchmark_root / "benchmark_matrix.json").exists():
        matrix = benchmark_root / "benchmark_matrix.json"
    return {"benchmark_matrix": matrix, "gate_results": gates_root / "gate_results.csv", "decision_summary": gates_root / "decision_summary.json"}


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = _load_json(path)
        rows = payload.get("rows") if isinstance(payload, Mapping) else None
        if not isinstance(rows, list):
            raise ValueError(f"JSON artifact must contain a rows array: {path.as_posix()}")
        return [_normalize_row(row) for row in rows if isinstance(row, Mapping)]
    frame = pd.read_csv(path, keep_default_na=True)
    return [_normalize_row(row) for row in frame.to_dict(orient="records")]


def _normalize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): (None if _is_missing(value) else value) for key, value in row.items()}


def _group_gate_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("variant_name")), []).append(row)
    return grouped


def _gate_names(rows: list[dict[str, Any]], *, status: str | None = None, impact: str | None = None) -> list[str]:
    selected = []
    for row in rows:
        if status is not None and row.get("status") != status:
            continue
        if impact is not None and row.get("decision_impact") != impact:
            continue
        selected.append(f"{row.get('gate_category')}.{row.get('gate_name')}")
    return selected


def _decision_summary_sentence(row: Mapping[str, Any], decision: Mapping[str, Any]) -> str:
    reasons = decision.get("primary_reasons") or []
    reason = str(reasons[0]) if reasons else str(row.get("primary_reason") or "No gate issues were recorded.")
    return f"{row['decision']} because {reason[0].lower() + reason[1:] if reason else 'the gate evidence supported this outcome.'}"


def _primary_reason(decision: Mapping[str, Any]) -> str:
    reasons = decision.get("primary_reasons") or []
    if reasons:
        return str(reasons[0])
    return "No gate issues were recorded."


def _reason_summary(leaderboard: list[dict[str, Any]], decision: str) -> list[dict[str, Any]]:
    return [
        {"variant_name": row["variant_name"], "rank": row["rank"], "primary_reason": row["primary_reason"]}
        for row in leaderboard
        if row["decision"] == decision
    ]


def _source_benchmark_run_id(
    rows: list[dict[str, Any]],
    benchmark_summary: Mapping[str, Any] | None,
    decision_summary: Mapping[str, Any],
) -> str:
    if benchmark_summary and benchmark_summary.get("benchmark_run_id") is not None:
        return str(benchmark_summary["benchmark_run_id"])
    if decision_summary.get("benchmark_run_id") is not None:
        return str(decision_summary["benchmark_run_id"])
    if rows and rows[0].get("benchmark_run_id") is not None:
        return str(rows[0]["benchmark_run_id"])
    return "unknown"


def _build_review_run_id(config: RegimeReviewPackConfig, benchmark_run_id: str, gate_config_name: str) -> str:
    payload = canonicalize_value({"config": config.to_dict(), "benchmark_run_id": benchmark_run_id, "gate_config_name": gate_config_name})
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:12]
    return f"{config.review_name}_{digest}"


def _require_dir(path: Path, *, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path.as_posix()}")
    if not path.is_dir():
        raise ValueError(f"{label} must be a directory: {path.as_posix()}")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path.as_posix()}")
    return payload


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    return _load_json(path) if path.exists() else None


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(canonicalize_value(payload), indent=2, sort_keys=True), encoding="utf-8", newline="\n")


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


def _decision_count(rows: list[dict[str, Any]], decision: str) -> int:
    return int(sum(1 for row in rows if row.get("decision") == decision))


def _row_count(path: Path) -> int | None:
    if not path.exists():
        return None
    if path.suffix.lower() == ".csv":
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return 0
        count = text.count("\n")
        return max(count - 1 if text.endswith("\n") else count, 0)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            return len(payload["rows"])
    return None


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


__all__ = ["DECISION_ORDER", "LEADERBOARD_COLUMNS", "RegimeReviewPackRunResult", "run_regime_review_pack"]
