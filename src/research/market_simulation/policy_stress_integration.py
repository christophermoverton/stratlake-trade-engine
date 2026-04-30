from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.config.regime_policy_stress_tests import RegimePolicyMarketSimulationStressConfig
from src.research.market_simulation.metrics import (
    LEADERBOARD_COLUMNS,
    PATH_METRIC_COLUMNS,
    SUMMARY_COLUMNS,
)
from src.research.registry import canonicalize_value


MARKET_SIMULATION_STRESS_SUMMARY_FILENAME = "market_simulation_stress_summary.json"
MARKET_SIMULATION_STRESS_LEADERBOARD_FILENAME = "market_simulation_stress_leaderboard.csv"
REGIME_ONLY_MONTE_CARLO_NOTE = (
    "Monte Carlo paths are regime-only and do not fabricate return or policy metrics."
)

REQUIRED_METRIC_FILES = {
    "simulation_path_metrics_csv": "simulation_path_metrics.csv",
    "simulation_summary_csv": "simulation_summary.csv",
    "simulation_leaderboard_csv": "simulation_leaderboard.csv",
    "policy_failure_summary_json": "policy_failure_summary.json",
    "simulation_metric_config_json": "simulation_metric_config.json",
    "manifest_json": "manifest.json",
}


@dataclass(frozen=True)
class MarketSimulationPolicyStressIntegrationResult:
    summary: dict[str, Any]
    summary_path: Path
    leaderboard_path: Path
    source_metrics_dir: Path


def run_market_simulation_policy_stress_integration(
    config: RegimePolicyMarketSimulationStressConfig,
    *,
    output_dir: Path,
) -> MarketSimulationPolicyStressIntegrationResult | None:
    if not config.enabled:
        return None
    metrics_dir = _metrics_dir_from_config(config)
    summary = load_market_simulation_stress_summary(
        metrics_dir,
        mode=config.mode,
        include_in_policy_stress_summary=config.include_in_policy_stress_summary,
        include_in_case_study_report=config.include_in_case_study_report,
    )
    summary_path = output_dir / MARKET_SIMULATION_STRESS_SUMMARY_FILENAME
    leaderboard_path = output_dir / MARKET_SIMULATION_STRESS_LEADERBOARD_FILENAME
    _write_json(summary_path, summary)
    shutil.copyfile(
        metrics_dir / REQUIRED_METRIC_FILES["simulation_leaderboard_csv"],
        leaderboard_path,
    )
    return MarketSimulationPolicyStressIntegrationResult(
        summary=summary,
        summary_path=summary_path,
        leaderboard_path=leaderboard_path,
        source_metrics_dir=metrics_dir,
    )


def load_market_simulation_stress_summary(
    metrics_dir: str | Path,
    *,
    mode: str = "existing_artifacts",
    include_in_policy_stress_summary: bool = True,
    include_in_case_study_report: bool = True,
) -> dict[str, Any]:
    root = Path(metrics_dir).resolve()
    _validate_metrics_dir(root)

    path_metrics = pd.read_csv(root / "simulation_path_metrics.csv")
    summary = pd.read_csv(root / "simulation_summary.csv")
    leaderboard = pd.read_csv(root / "simulation_leaderboard.csv")
    failure_summary = _read_json(root / "policy_failure_summary.json")
    manifest = _read_json(root / "manifest.json")

    _require_columns(path_metrics, PATH_METRIC_COLUMNS, "simulation_path_metrics.csv")
    _require_columns(summary, SUMMARY_COLUMNS, "simulation_summary.csv")
    _require_columns(leaderboard, LEADERBOARD_COLUMNS, "simulation_leaderboard.csv")

    simulation_types = sorted(
        {
            str(value)
            for value in summary.get("simulation_type", pd.Series(dtype="string")).dropna().tolist()
            if str(value)
        }
    )
    best = _ranked_scenario(leaderboard.iloc[0].to_dict()) if not leaderboard.empty else {}
    worst = _ranked_scenario(leaderboard.iloc[-1].to_dict()) if not leaderboard.empty else {}
    return canonicalize_value(
        {
            "market_simulation_enabled": True,
            "market_simulation_available": True,
            "market_simulation_mode": mode,
            "source_market_simulation_run_id": _source_run_id(manifest, summary, path_metrics),
            "simulation_type_count": len(simulation_types),
            "simulation_types": simulation_types,
            "path_metric_row_count": int(len(path_metrics)),
            "summary_row_count": int(len(summary)),
            "leaderboard_row_count": int(len(leaderboard)),
            "policy_failure_rate": _float(
                failure_summary.get("policy_failure_rate"),
                default=_policy_failure_rate(path_metrics),
            ),
            "best_ranked_market_simulation_scenario": best,
            "worst_ranked_market_simulation_scenario": worst,
            "regime_only_monte_carlo_note": REGIME_ONLY_MONTE_CARLO_NOTE,
            "source_artifact_paths": {
                key: _relative_path(root / filename)
                for key, filename in sorted(REQUIRED_METRIC_FILES.items())
            },
            "include_in_policy_stress_summary": bool(include_in_policy_stress_summary),
            "include_in_case_study_report": bool(include_in_case_study_report),
            "limitations": [
                "M27 market simulation stress evidence is optional and complements deterministic regime shock stress tests.",
                "Simulation outputs are not forecasts or trading recommendations.",
                REGIME_ONLY_MONTE_CARLO_NOTE,
            ],
        }
    )


def _metrics_dir_from_config(config: RegimePolicyMarketSimulationStressConfig) -> Path:
    if config.mode == "existing_artifacts":
        if config.simulation_metrics_dir is None:
            raise ValueError("market_simulation_stress.simulation_metrics_dir is required.")
        return Path(config.simulation_metrics_dir).resolve()
    if config.mode == "run_config":
        if config.config_path is None:
            raise ValueError("market_simulation_stress.config_path is required.")
        from src.execution.market_simulation import run_market_simulation_scenarios

        result = run_market_simulation_scenarios(config_path=Path(config.config_path)).raw_result
        metrics = result.simulation_stress_metrics_result
        if metrics is None:
            raise ValueError(
                "market_simulation_stress run_config mode requires stress_metrics.enabled=true."
            )
        return metrics.output_dir.resolve()
    raise ValueError(f"Unsupported market_simulation_stress mode: {config.mode!r}.")


def _validate_metrics_dir(root: Path) -> None:
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Market simulation metrics directory does not exist: {_relative_path(root)}")
    missing = sorted(filename for filename in REQUIRED_METRIC_FILES.values() if not (root / filename).exists())
    if missing:
        raise FileNotFoundError(
            "Market simulation metrics directory is missing required file(s): "
            f"{missing} in {_relative_path(root)}."
        )


def _require_columns(frame: pd.DataFrame, expected: tuple[str, ...], filename: str) -> None:
    missing = sorted(set(expected) - set(frame.columns))
    if missing:
        raise ValueError(f"{filename} is missing required column(s): {missing}.")


def _source_run_id(
    manifest: Mapping[str, Any],
    summary: pd.DataFrame,
    path_metrics: pd.DataFrame,
) -> str | None:
    value = manifest.get("simulation_run_id")
    if value:
        return str(value)
    for frame in (summary, path_metrics):
        if "simulation_run_id" in frame.columns and not frame.empty:
            candidate = frame["simulation_run_id"].dropna()
            if not candidate.empty:
                return str(candidate.iloc[0])
    return None


def _ranked_scenario(row: Mapping[str, Any]) -> dict[str, Any]:
    return canonicalize_value(
        {
            "rank": _int(row.get("rank")),
            "scenario_id": row.get("scenario_id"),
            "scenario_name": row.get("scenario_name"),
            "simulation_type": row.get("simulation_type"),
            "policy_name": row.get("policy_name"),
            "ranking_metric": row.get("ranking_metric"),
            "ranking_value": _float(row.get("ranking_value"), default=None),
            "decision_label": row.get("decision_label"),
            "primary_reason": row.get("primary_reason"),
        }
    )


def _policy_failure_rate(path_metrics: pd.DataFrame) -> float:
    if path_metrics.empty or "policy_failure" not in path_metrics:
        return 0.0
    values = path_metrics["policy_failure"].dropna().astype(str).str.lower()
    if values.empty:
        return 0.0
    return float(values.isin({"true", "1", "yes"}).mean())


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {_relative_path(path)}.")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(canonicalize_value(dict(payload)), indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )


def _float(value: Any, *, default: float | None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any) -> int | None:
    number = _float(value, default=None)
    return None if number is None else int(number)


def _relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_leaderboard_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
