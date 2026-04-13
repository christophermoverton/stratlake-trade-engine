from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.research.reporting.campaign_milestone_report import (
    build_campaign_milestone_report_payloads,
    generate_campaign_milestone_report,
    load_campaign_reporting_payloads,
    resolve_campaign_artifact_dir,
)
from src.research.reporting.milestone_artifacts import (
    MILESTONE_DECISION_LOG_FILENAME,
    MILESTONE_MANIFEST_FILENAME,
    MILESTONE_SUMMARY_FILENAME,
    MilestoneArtifactValidationError,
    MilestoneDecisionEntry,
    MilestoneReport,
    build_milestone_decision_log_payload,
    build_milestone_report_id,
    build_milestone_report_manifest_payload,
    build_milestone_report_summary_payload,
    resolve_milestone_artifact_dir,
    validate_milestone_decision_log_payload,
    validate_milestone_report,
    validate_milestone_report_payload,
    write_milestone_report_artifacts,
)
from src.research.reporting.report_generator import generate_strategy_plots, generate_strategy_report


def load_run_artifacts(run_dir: Path | str) -> dict[str, Any]:
    """Load the standard artifact files for one strategy run."""

    root = Path(run_dir)
    payload: dict[str, Any] = {
        "run_dir": root,
        "manifest": _load_json_if_exists(root / "manifest.json"),
        "config": _load_json_if_exists(root / "config.json"),
        "metrics": _load_json_if_exists(root / "metrics.json"),
        "metrics_by_split": _load_csv_if_exists(root / "metrics_by_split.csv"),
        "equity_curve": _load_csv_if_exists(root / "equity_curve.csv"),
        "signals": _load_parquet_if_exists(root / "signals.parquet"),
        "trades": _load_parquet_if_exists(root / "trades.parquet"),
    }
    payload["splits"] = _load_split_artifacts(root / "splits")
    return payload


def summarize_run(run_dir: Path | str) -> dict[str, Any]:
    """Return a small summary payload for quick inspection and reporting."""

    artifacts = load_run_artifacts(run_dir)
    manifest = artifacts["manifest"] or {}
    metrics = artifacts["metrics"] or {}
    metrics_by_split = artifacts["metrics_by_split"]
    trades = artifacts["trades"]

    return {
        "run_id": manifest.get("run_id") or Path(run_dir).name,
        "strategy_name": manifest.get("strategy_name"),
        "evaluation_mode": manifest.get("evaluation_mode"),
        "split_count": manifest.get("split_count"),
        "primary_metric": manifest.get("primary_metric", "sharpe_ratio"),
        "primary_metric_value": metrics.get(manifest.get("primary_metric", "sharpe_ratio")),
        "cumulative_return": metrics.get("cumulative_return"),
        "sharpe_ratio": metrics.get("sharpe_ratio"),
        "max_drawdown": metrics.get("max_drawdown"),
        "trade_count": 0 if trades is None else len(trades),
        "artifact_count": len(manifest.get("artifact_files") or []),
        "split_metrics_rows": 0 if metrics_by_split is None else len(metrics_by_split),
    }


def print_quick_report(run_dir: Path | str) -> None:
    """Print a concise console summary for one saved run."""

    summary = summarize_run(run_dir)
    print(f"run_id: {summary['run_id']}")
    print(f"strategy: {summary['strategy_name']}")
    print(f"mode: {summary['evaluation_mode']}")
    if summary["split_count"] is not None:
        print(f"split_count: {summary['split_count']}")
    print(f"{summary['primary_metric']}: {_format_metric(summary['primary_metric_value'])}")
    print(f"cumulative_return: {_format_metric(summary['cumulative_return'])}")
    print(f"sharpe_ratio: {_format_metric(summary['sharpe_ratio'])}")
    print(f"max_drawdown: {_format_metric(summary['max_drawdown'])}")
    print(f"trade_count: {summary['trade_count']}")
    print(f"artifact_count: {summary['artifact_count']}")


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _load_parquet_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _load_split_artifacts(splits_dir: Path) -> dict[str, dict[str, Any]]:
    if not splits_dir.exists():
        return {}

    payload: dict[str, dict[str, Any]] = {}
    for split_dir in sorted(path for path in splits_dir.iterdir() if path.is_dir()):
        payload[split_dir.name] = {
            "split": _load_json_if_exists(split_dir / "split.json"),
            "metrics": _load_json_if_exists(split_dir / "metrics.json"),
            "equity_curve": _load_csv_if_exists(split_dir / "equity_curve.csv"),
            "signals": _load_parquet_if_exists(split_dir / "signals.parquet"),
            "trades": _load_parquet_if_exists(split_dir / "trades.parquet"),
        }
    return payload


def _format_metric(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.6f}"


__all__ = [
    "MILESTONE_DECISION_LOG_FILENAME",
    "MILESTONE_MANIFEST_FILENAME",
    "MILESTONE_SUMMARY_FILENAME",
    "MilestoneArtifactValidationError",
    "MilestoneDecisionEntry",
    "MilestoneReport",
    "build_milestone_decision_log_payload",
    "build_campaign_milestone_report_payloads",
    "build_milestone_report_id",
    "build_milestone_report_manifest_payload",
    "build_milestone_report_summary_payload",
    "generate_campaign_milestone_report",
    "generate_strategy_plots",
    "generate_strategy_report",
    "load_campaign_reporting_payloads",
    "load_run_artifacts",
    "print_quick_report",
    "resolve_campaign_artifact_dir",
    "resolve_milestone_artifact_dir",
    "summarize_run",
    "validate_milestone_decision_log_payload",
    "validate_milestone_report",
    "validate_milestone_report_payload",
    "write_milestone_report_artifacts",
]
