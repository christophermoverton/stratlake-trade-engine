from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config.simulation import SimulationConfig, resolve_simulation_config
from src.portfolio.risk import PortfolioRiskError, historical_cvar, historical_var, validate_return_series
from src.research.metrics import (
    DEFAULT_PERIODS_PER_YEAR,
    annualized_return,
    annualized_volatility,
    max_drawdown,
    sharpe_ratio,
    total_return,
    win_rate,
)

DEFAULT_SIMULATION_METRICS = (
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "win_rate",
    "value_at_risk",
    "conditional_value_at_risk",
    "final_equity",
)
_SUMMARY_PERCENTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
_MAX_PATHS = 100_000
_MAX_PATH_LENGTH = 100_000


class SimulationError(ValueError):
    """Raised when simulation inputs or deterministic outputs are invalid."""


@dataclass(frozen=True)
class SimulationRunResult:
    """Deterministic structured simulation result."""

    config: dict[str, Any]
    assumptions: dict[str, Any]
    path_metrics: pd.DataFrame
    paths: pd.DataFrame
    summary: dict[str, Any]


def run_return_simulation(
    returns: pd.Series,
    *,
    config: SimulationConfig | dict[str, Any],
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    owner: str = "returns",
    var_confidence_level: float = 0.95,
    cvar_confidence_level: float = 0.95,
) -> SimulationRunResult:
    resolved = resolve_simulation_config(config)
    if resolved is None:
        raise SimulationError("Simulation config is required.")
    normalized_returns = _validate_simulation_returns(returns, owner=owner)
    _validate_config_bounds(resolved)
    path_length = resolved.path_length or len(normalized_returns)
    if path_length <= 0:
        raise SimulationError("Simulation path_length must resolve to a positive integer.")

    if resolved.method == "bootstrap":
        assumptions = {
            "method": "bootstrap",
            "sampling_scheme": "iid_with_replacement",
            "input_observation_count": int(len(normalized_returns)),
        }
        paths = generate_bootstrap_paths(normalized_returns, resolved, path_length=path_length)
    elif resolved.method == "monte_carlo":
        assumptions, paths = generate_monte_carlo_paths(normalized_returns, resolved, path_length=path_length)
    else:  # pragma: no cover - guarded by config resolution
        raise SimulationError(f"Unsupported simulation method: {resolved.method!r}.")

    path_metrics = evaluate_simulation_paths(
        paths,
        periods_per_year=periods_per_year,
        metrics=resolved.metrics,
        var_confidence_level=var_confidence_level,
        cvar_confidence_level=cvar_confidence_level,
    )
    summary = summarize_simulation_metrics(
        path_metrics,
        config=resolved,
        assumptions=assumptions,
    )
    _validate_simulation_outputs(path_metrics, summary)
    return SimulationRunResult(
        config={**resolved.to_dict(), "path_length": int(path_length)},
        assumptions=assumptions,
        path_metrics=path_metrics,
        paths=paths,
        summary=summary,
    )


def generate_bootstrap_paths(
    returns: pd.Series,
    config: SimulationConfig,
    *,
    path_length: int,
) -> pd.DataFrame:
    normalized = _validate_simulation_returns(returns, owner="bootstrap returns")
    rng = np.random.default_rng(config.seed)
    values = normalized.to_numpy(copy=True, dtype="float64")
    timestamp_labels = [str(index) for index in normalized.index.tolist()]
    rows: list[dict[str, Any]] = []
    for path_index, path_id in enumerate(_path_ids(config.num_paths)):
        sampled_positions = rng.integers(0, len(values), size=path_length)
        sampled_returns = values[sampled_positions]
        for step, (sampled_position, sampled_return) in enumerate(zip(sampled_positions.tolist(), sampled_returns.tolist(), strict=False)):
            rows.append(
                {
                    "path_id": path_id,
                    "path_index": path_index,
                    "step": step,
                    "simulated_return": float(sampled_return),
                    "source_observation_index": int(sampled_position),
                    "source_index_label": timestamp_labels[int(sampled_position)],
                }
            )
    return pd.DataFrame(rows)


def generate_monte_carlo_paths(
    returns: pd.Series,
    config: SimulationConfig,
    *,
    path_length: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    normalized = _validate_simulation_returns(returns, owner="monte_carlo returns")
    if config.monte_carlo_mean is None or config.monte_carlo_volatility is None:
        if len(normalized) < config.min_samples:
            raise SimulationError(
                "Monte Carlo simulation requires sufficient return observations to estimate parameters "
                f"(required={config.min_samples}, observed={len(normalized)})."
            )

    mean = float(normalized.mean()) if config.monte_carlo_mean is None else float(config.monte_carlo_mean)
    volatility = (
        float(normalized.std(ddof=1))
        if config.monte_carlo_volatility is None
        else float(config.monte_carlo_volatility)
    )
    if volatility < 0.0 or not math.isfinite(volatility):
        raise SimulationError("Monte Carlo volatility must be finite and non-negative.")

    rng = np.random.default_rng(config.seed)
    paths_matrix = rng.normal(loc=mean, scale=volatility, size=(config.num_paths, path_length)).astype("float64")
    rows: list[dict[str, Any]] = []
    for path_index, path_id in enumerate(_path_ids(config.num_paths)):
        for step, value in enumerate(paths_matrix[path_index].tolist()):
            rows.append(
                {
                    "path_id": path_id,
                    "path_index": path_index,
                    "step": step,
                    "simulated_return": float(value),
                    "source_observation_index": None,
                    "source_index_label": None,
                }
            )
    assumptions = {
        "method": "monte_carlo",
        "distribution": "normal",
        "parameter_source": "historical_estimate"
        if config.monte_carlo_mean is None or config.monte_carlo_volatility is None
        else "explicit",
        "estimated_mean": mean,
        "estimated_volatility": volatility,
        "input_observation_count": int(len(normalized)),
    }
    return assumptions, pd.DataFrame(rows)


def evaluate_simulation_paths(
    paths: pd.DataFrame,
    *,
    periods_per_year: int,
    metrics: tuple[str, ...] | None,
    var_confidence_level: float,
    cvar_confidence_level: float,
) -> pd.DataFrame:
    if not isinstance(paths, pd.DataFrame) or paths.empty:
        raise SimulationError("Simulation paths must be provided as a non-empty DataFrame.")
    if "path_id" not in paths.columns or "simulated_return" not in paths.columns:
        raise SimulationError("Simulation paths must include 'path_id' and 'simulated_return' columns.")

    selected_metrics = tuple(metrics or DEFAULT_SIMULATION_METRICS)
    rows: list[dict[str, Any]] = []
    ordered = paths.sort_values(["path_index", "path_id", "step"], kind="stable").reset_index(drop=True)
    for path_id, frame in ordered.groupby("path_id", sort=False):
        path_returns = pd.Series(frame["simulated_return"].to_numpy(copy=True), dtype="float64")
        metric_row = evaluate_return_path(
            path_returns,
            periods_per_year=periods_per_year,
            var_confidence_level=var_confidence_level,
            cvar_confidence_level=cvar_confidence_level,
        )
        rows.append({"path_id": str(path_id), **{key: metric_row[key] for key in selected_metrics if key in metric_row}})
    return pd.DataFrame(rows).sort_values("path_id", kind="stable").reset_index(drop=True)


def evaluate_return_path(
    returns: pd.Series,
    *,
    periods_per_year: int,
    var_confidence_level: float,
    cvar_confidence_level: float,
) -> dict[str, float]:
    normalized = validate_return_series(returns, owner="simulated path returns", min_samples=1)
    equity = (1.0 + normalized).cumprod()
    return {
        "cumulative_return": float(total_return(normalized)),
        "annualized_return": float(annualized_return(normalized, periods_per_year=periods_per_year)),
        "annualized_volatility": float(annualized_volatility(normalized, periods_per_year=periods_per_year)),
        "sharpe_ratio": float(sharpe_ratio(normalized, periods_per_year=periods_per_year)),
        "max_drawdown": float(max_drawdown(normalized)),
        "win_rate": float(win_rate(normalized)),
        "value_at_risk": float(historical_var(normalized, confidence_level=var_confidence_level)),
        "conditional_value_at_risk": float(historical_cvar(normalized, confidence_level=cvar_confidence_level)),
        "final_equity": float(equity.iloc[-1]) if not equity.empty else 1.0,
    }


def summarize_simulation_metrics(
    path_metrics: pd.DataFrame,
    *,
    config: SimulationConfig,
    assumptions: dict[str, Any],
) -> dict[str, Any]:
    if path_metrics.empty:
        raise SimulationError("Simulation path metrics must be non-empty.")

    metric_summaries: dict[str, dict[str, float]] = {}
    for metric_name in [column for column in path_metrics.columns if column != "path_id"]:
        series = pd.to_numeric(path_metrics[metric_name], errors="coerce")
        if series.isna().any():
            raise SimulationError(f"Simulation metric {metric_name!r} contains non-numeric values.")
        metric_summaries[metric_name] = _distribution_summary(series)

    cumulative_returns = pd.to_numeric(path_metrics["cumulative_return"], errors="coerce")
    max_drawdowns = pd.to_numeric(path_metrics["max_drawdown"], errors="coerce")
    summary: dict[str, Any] = {
        "method": config.method,
        "num_paths": int(config.num_paths),
        "path_length": int(config.path_length) if config.path_length is not None else None,
        "seed": int(config.seed),
        "distribution": config.distribution,
        "assumptions": assumptions,
        "metric_statistics": metric_summaries,
        "probability_of_loss": float((cumulative_returns < 0.0).mean()),
        "probability_of_non_positive_return": float((cumulative_returns <= 0.0).mean()),
    }
    if config.drawdown_threshold is not None:
        summary["drawdown_exceedance_probability"] = float((max_drawdowns >= config.drawdown_threshold).mean())
        summary["drawdown_threshold"] = float(config.drawdown_threshold)
    return summary


def write_simulation_artifacts(
    output_dir: str | Path,
    result: SimulationRunResult,
    *,
    parent_manifest_dir: str | Path | None = None,
) -> dict[str, Any]:
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    _write_json(resolved_output_dir / "config.json", result.config)
    _write_json(resolved_output_dir / "assumptions.json", result.assumptions)
    _write_csv(resolved_output_dir / "path_metrics.csv", result.path_metrics)
    _write_csv(resolved_output_dir / "simulated_paths.csv", result.paths)
    _write_json(resolved_output_dir / "summary.json", result.summary)
    manifest = _build_simulation_manifest(resolved_output_dir, result)
    _write_json(resolved_output_dir / "manifest.json", manifest)

    if parent_manifest_dir is not None:
        _augment_parent_manifest(Path(parent_manifest_dir), resolved_output_dir.name, result.summary, result.config)

    return manifest


def _build_simulation_manifest(output_dir: Path, result: SimulationRunResult) -> dict[str, Any]:
    artifact_files = sorted(
        set([*(path.name for path in output_dir.iterdir() if path.is_file()), "manifest.json"])
    )
    return {
        "artifact_files": artifact_files,
        "files_written": int(len(artifact_files)),
        "method": result.config["method"],
        "num_paths": int(result.config["num_paths"]),
        "path_length": int(result.config["path_length"]),
        "seed": int(result.config["seed"]),
        "metric_summary": result.summary.get("metric_statistics", {}),
        "probability_of_loss": result.summary.get("probability_of_loss"),
        "summary_path": "summary.json",
    }


def _augment_parent_manifest(
    experiment_dir: Path,
    simulation_dir_name: str,
    summary: dict[str, Any],
    config: dict[str, Any],
) -> None:
    manifest_path = experiment_dir / "manifest.json"
    if not manifest_path.exists():
        return
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_files = payload.get("artifact_files", [])
    if not isinstance(artifact_files, list):
        artifact_files = []
    simulation_files = sorted(
        str(Path(simulation_dir_name, filename).as_posix())
        for filename in (
            "assumptions.json",
            "config.json",
            "manifest.json",
            "path_metrics.csv",
            "simulated_paths.csv",
            "summary.json",
        )
    )
    payload["artifact_files"] = sorted(set([*artifact_files, *simulation_files]))
    payload["simulation"] = {
        "enabled": True,
        "method": config.get("method"),
        "num_paths": config.get("num_paths"),
        "path_length": config.get("path_length"),
        "seed": config.get("seed"),
        "config_path": Path(simulation_dir_name, "config.json").as_posix(),
        "manifest_path": Path(simulation_dir_name, "manifest.json").as_posix(),
        "summary_path": Path(simulation_dir_name, "summary.json").as_posix(),
        "artifact_path": simulation_dir_name,
        "artifact_files": simulation_files,
        "probability_of_loss": summary.get("probability_of_loss"),
        "summary": {
            "cumulative_return": (
                summary.get("metric_statistics", {})
                .get("cumulative_return", {})
            ),
            "max_drawdown": (
                summary.get("metric_statistics", {})
                .get("max_drawdown", {})
            ),
        },
    }
    artifact_groups = payload.get("artifact_groups")
    if not isinstance(artifact_groups, dict):
        artifact_groups = {}
    artifact_groups["simulation"] = simulation_files
    payload["artifact_groups"] = {
        key: sorted(value) if isinstance(value, list) else value
        for key, value in sorted(artifact_groups.items())
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _distribution_summary(values: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce").astype("float64")
    if numeric.isna().any():
        raise SimulationError("Simulation summary values must be numeric and finite.")
    summary = {
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
    }
    for percentile in _SUMMARY_PERCENTILES:
        summary[_percentile_key(percentile)] = float(numeric.quantile(percentile, interpolation="linear"))
    return summary


def _percentile_key(percentile: float) -> str:
    return f"p{int(round(percentile * 100)):02d}"


def _validate_simulation_returns(returns: pd.Series, *, owner: str) -> pd.Series:
    try:
        normalized = validate_return_series(returns, owner=owner, min_samples=1)
    except PortfolioRiskError as exc:
        raise SimulationError(str(exc)) from exc
    if normalized.empty:
        raise SimulationError(f"{owner} must contain at least one observation.")
    return normalized


def _validate_config_bounds(config: SimulationConfig) -> None:
    if config.num_paths > _MAX_PATHS:
        raise SimulationError(f"Simulation num_paths must be <= {_MAX_PATHS}.")
    if config.path_length is not None and config.path_length > _MAX_PATH_LENGTH:
        raise SimulationError(f"Simulation path_length must be <= {_MAX_PATH_LENGTH}.")


def _validate_simulation_outputs(path_metrics: pd.DataFrame, summary: dict[str, Any]) -> None:
    if path_metrics["path_id"].duplicated().any():
        raise SimulationError("Simulation path metrics must contain unique path_id values.")
    for column in [column for column in path_metrics.columns if column != "path_id"]:
        series = pd.to_numeric(path_metrics[column], errors="coerce")
        if series.isna().any() or not np.isfinite(series.to_numpy(copy=True)).all():
            raise SimulationError(f"Simulation metric column {column!r} must be finite.")
    probability_of_loss = summary.get("probability_of_loss")
    if probability_of_loss is None or not (0.0 <= float(probability_of_loss) <= 1.0):
        raise SimulationError("Simulation probability_of_loss must be within [0, 1].")


def _path_ids(num_paths: int) -> list[str]:
    width = max(4, len(str(max(num_paths - 1, 0))))
    return [f"path_{index:0{width}d}" for index in range(num_paths)]


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


__all__ = [
    "DEFAULT_SIMULATION_METRICS",
    "SimulationError",
    "SimulationRunResult",
    "evaluate_return_path",
    "evaluate_simulation_paths",
    "generate_bootstrap_paths",
    "generate_monte_carlo_paths",
    "run_return_simulation",
    "summarize_simulation_metrics",
    "write_simulation_artifacts",
]
