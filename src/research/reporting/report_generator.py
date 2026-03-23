from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.research.visualization.diagnostics import (
    compute_trade_statistics,
    plot_drawdown,
    plot_rolling_sharpe,
    plot_trade_return_distribution,
    plot_win_loss_distribution,
)
from src.research.visualization.equity import plot_equity_curve

_KEY_METRICS: tuple[tuple[str, str], ...] = (
    ("sharpe_ratio", "Sharpe"),
    ("total_return", "Total Return"),
    ("cumulative_return", "Cumulative Return"),
    ("max_drawdown", "Max Drawdown"),
    ("volatility", "Volatility"),
    ("annualized_return", "Annualized Return"),
    ("annualized_volatility", "Annualized Volatility"),
    ("win_rate", "Win Rate"),
    ("profit_factor", "Profit Factor"),
)
_METADATA_KEYS: tuple[tuple[str, str], ...] = (
    ("run_id", "Run ID"),
    ("strategy_name", "Strategy"),
    ("evaluation_mode", "Evaluation Mode"),
    ("split_count", "Split Count"),
    ("primary_metric", "Primary Metric"),
    ("evaluation_config_path", "Evaluation Config Path"),
)
_TRADE_METRICS: tuple[tuple[str, str], ...] = (
    ("count", "Trade Count"),
    ("win_count", "Winning Trades"),
    ("loss_count", "Losing Trades"),
    ("win_rate", "Win Rate"),
    ("loss_rate", "Loss Rate"),
    ("mean_return", "Mean Return"),
    ("median_return", "Median Return"),
    ("std_return", "Std Return"),
)
_ROLLING_SHARPE_WINDOW = 20


def generate_strategy_report(run_dir: Path, output_path: Path | None = None) -> Path:
    """Generate a deterministic Markdown report for one strategy run directory."""

    resolved_run_dir = Path(run_dir)
    _validate_run_dir(resolved_run_dir)

    resolved_output_path = Path(output_path) if output_path is not None else resolved_run_dir / "report.md"
    plots_dir = resolved_run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(resolved_run_dir / "metrics.json")
    manifest = _load_json_if_exists(resolved_run_dir / "manifest.json") or {}
    config = _load_json_if_exists(resolved_run_dir / "config.json") or {}
    equity_curve = _load_equity_curve(resolved_run_dir)
    trades = _load_optional_parquet(resolved_run_dir / "trades.parquet")
    signals = _load_optional_parquet(resolved_run_dir / "signals.parquet")

    plot_paths = generate_report_plots(
        run_dir=resolved_run_dir,
        plots_dir=plots_dir,
        equity_curve=equity_curve,
        trades=trades,
    )
    markdown = build_markdown_report(
        run_dir=resolved_run_dir,
        output_path=resolved_output_path,
        metrics=metrics,
        manifest=manifest,
        config=config,
        equity_curve=equity_curve,
        trades=trades,
        signals=signals,
        plot_paths=plot_paths,
    )
    resolved_output_path.write_text(markdown, encoding="utf-8")
    return resolved_output_path


def load_metrics(metrics_path: Path) -> dict[str, Any]:
    """Load metrics.json from a strategy run."""

    if not metrics_path.exists():
        raise FileNotFoundError(f"Required metrics artifact not found: {metrics_path}")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def format_metrics_table(metrics: dict[str, Any]) -> str:
    """Render a deterministic Markdown table for key performance metrics."""

    rows = ["| Metric | Value |", "| --- | --- |"]
    for key, label in _KEY_METRICS:
        if key not in metrics:
            continue
        rows.append(f"| {label} | {_format_value(metrics.get(key))} |")
    if len(rows) == 2:
        rows.append("| No metrics available | NA |")
    return "\n".join(rows)


def generate_plot_if_needed(
    *,
    output_path: Path,
    plotter: Any,
    overwrite: bool = False,
    **plot_kwargs: Any,
) -> Path:
    """Generate a plot artifact unless an existing artifact should be reused."""

    if output_path.exists() and not overwrite:
        return output_path
    result = plotter(output_path=output_path, **plot_kwargs)
    return Path(result)


def build_markdown_report(
    *,
    run_dir: Path,
    output_path: Path,
    metrics: dict[str, Any],
    manifest: dict[str, Any],
    config: dict[str, Any],
    equity_curve: pd.DataFrame | None,
    trades: pd.DataFrame | None,
    signals: pd.DataFrame | None,
    plot_paths: dict[str, Path],
) -> str:
    """Assemble the final Markdown report contents."""

    title = f"# Strategy Report: {_resolve_report_title(run_dir, manifest, config)}"
    sections = [
        title,
        "",
        "## Run Metadata",
        _build_metadata_section(run_dir, manifest, config),
        "",
        "## Performance Summary",
        format_metrics_table(metrics),
        "",
    ]

    if "equity_curve" in plot_paths:
        sections.extend(
            [
                "## Equity Curve",
                "![Equity Curve](%s)" % _relative_markdown_path(output_path, plot_paths["equity_curve"]),
                "",
            ]
        )

    if "drawdown" in plot_paths:
        sections.extend(
            [
                "## Drawdown",
                "![Drawdown](%s)" % _relative_markdown_path(output_path, plot_paths["drawdown"]),
                "",
            ]
        )

    sections.extend(_build_rolling_metrics_section(output_path=output_path, plot_paths=plot_paths))
    sections.extend(
        _build_trade_analysis_section(
            output_path=output_path,
            trades=trades,
            plot_paths=plot_paths,
        )
    )
    sections.extend(_build_optional_artifact_notes(equity_curve=equity_curve, signals=signals))
    sections.extend(
        [
            "## Observations",
            "_Add analysis notes here._",
            "",
        ]
    )
    return "\n".join(sections).rstrip() + "\n"


def generate_report_plots(
    *,
    run_dir: Path,
    plots_dir: Path,
    equity_curve: pd.DataFrame | None,
    trades: pd.DataFrame | None,
) -> dict[str, Path]:
    """Generate or reuse report plots derived from run artifacts."""

    del run_dir
    plot_paths: dict[str, Path] = {}

    if equity_curve is not None:
        equity_series = _select_equity_series(equity_curve)
        plot_paths["equity_curve"] = generate_plot_if_needed(
            output_path=plots_dir / "equity_curve.png",
            plotter=plot_equity_curve,
            equity_data=equity_series,
            input_type="equity",
            title="Equity Curve",
        )
        plot_paths["drawdown"] = generate_plot_if_needed(
            output_path=plots_dir / "drawdown.png",
            plotter=plot_drawdown,
            equity_data=equity_series,
            input_type="equity",
            title="Drawdown",
        )

        returns_series = _select_returns_series(equity_curve)
        if returns_series is not None and len(returns_series) >= _ROLLING_SHARPE_WINDOW:
            plot_paths["rolling_sharpe"] = generate_plot_if_needed(
                output_path=plots_dir / "rolling_sharpe.png",
                plotter=plot_rolling_sharpe,
                returns=returns_series,
                window=_ROLLING_SHARPE_WINDOW,
                title=f"Rolling Sharpe ({_ROLLING_SHARPE_WINDOW}-period)",
            )

    trade_returns = _select_trade_returns(trades)
    if trade_returns is not None:
        plot_paths["trade_return_distribution"] = generate_plot_if_needed(
            output_path=plots_dir / "trade_return_distribution.png",
            plotter=plot_trade_return_distribution,
            trade_returns=trade_returns,
            title="Trade Return Distribution",
        )
        plot_paths["win_loss_distribution"] = generate_plot_if_needed(
            output_path=plots_dir / "win_loss_distribution.png",
            plotter=plot_win_loss_distribution,
            trade_returns=trade_returns,
            title="Win/Loss Distribution",
        )

    return plot_paths


def _validate_run_dir(run_dir: Path) -> None:
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    if not run_dir.is_dir():
        raise ValueError(f"Run directory must be a directory: {run_dir}")
    if not (run_dir / "metrics.json").exists():
        raise FileNotFoundError(f"Required metrics artifact not found: {run_dir / 'metrics.json'}")


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_equity_curve(run_dir: Path) -> pd.DataFrame | None:
    csv_path = run_dir / "equity_curve.csv"
    if not csv_path.exists():
        return None
    frame = pd.read_csv(csv_path)
    if frame.empty:
        return None
    return frame


def _load_optional_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    if frame.empty:
        return None
    return frame


def _resolve_report_title(run_dir: Path, manifest: dict[str, Any], config: dict[str, Any]) -> str:
    return str(
        manifest.get("strategy_name")
        or config.get("strategy_name")
        or manifest.get("run_id")
        or run_dir.name
    )


def _build_metadata_section(run_dir: Path, manifest: dict[str, Any], config: dict[str, Any]) -> str:
    metadata = {**manifest}
    if "strategy_name" not in metadata and "strategy_name" in config:
        metadata["strategy_name"] = config["strategy_name"]

    lines: list[str] = []
    for key, label in _METADATA_KEYS:
        value = metadata.get(key)
        if value is None:
            continue
        lines.append(f"- {label}: `{value}`")

    if "parameters" in config and isinstance(config["parameters"], dict) and config["parameters"]:
        serialized_parameters = ", ".join(
            f"{name}={config['parameters'][name]!r}"
            for name in sorted(config["parameters"])
        )
        lines.append(f"- Parameters: `{serialized_parameters}`")

    if not lines:
        lines.append(f"- Run Directory: `{run_dir.name}`")
    return "\n".join(lines)


def _build_rolling_metrics_section(*, output_path: Path, plot_paths: dict[str, Path]) -> list[str]:
    section = ["## Rolling Metrics"]
    if "rolling_sharpe" in plot_paths:
        section.append(
            "![Rolling Sharpe](%s)" % _relative_markdown_path(output_path, plot_paths["rolling_sharpe"])
        )
    else:
        section.append("_Rolling Sharpe unavailable for this run._")
    section.append("")
    return section


def _build_trade_analysis_section(
    *,
    output_path: Path,
    trades: pd.DataFrame | None,
    plot_paths: dict[str, Path],
) -> list[str]:
    section = ["## Trade Analysis"]
    trade_returns = _select_trade_returns(trades)
    if trade_returns is None:
        section.extend(["_Trade data unavailable for this run._", ""])
        return section

    statistics = compute_trade_statistics(trade_returns)
    section.extend(_format_statistics_table(statistics, _TRADE_METRICS))

    if "trade_return_distribution" in plot_paths:
        section.append(
            "![Trade Return Distribution](%s)"
            % _relative_markdown_path(output_path, plot_paths["trade_return_distribution"])
        )
    if "win_loss_distribution" in plot_paths:
        section.append(
            "![Win Loss Distribution](%s)"
            % _relative_markdown_path(output_path, plot_paths["win_loss_distribution"])
        )
    section.append("")
    return section


def _build_optional_artifact_notes(*, equity_curve: pd.DataFrame | None, signals: pd.DataFrame | None) -> list[str]:
    notes: list[str] = []
    if equity_curve is not None and "signal" in equity_curve.columns:
        notes.extend(
            [
                "## Signal Summary",
                f"- Signal observations: `{len(equity_curve)}` rows available in equity artifacts.",
                "",
            ]
        )
    elif signals is not None:
        notes.extend(
            [
                "## Signal Summary",
                f"- Signal observations: `{len(signals)}` rows available.",
                "",
            ]
        )
    return notes


def _format_statistics_table(statistics: dict[str, float], ordered_keys: tuple[tuple[str, str], ...]) -> list[str]:
    lines = ["| Metric | Value |", "| --- | --- |"]
    for key, label in ordered_keys:
        if key not in statistics:
            continue
        lines.append(f"| {label} | {_format_value(statistics[key])} |")
    lines.append("")
    return lines


def _select_equity_series(equity_curve: pd.DataFrame) -> pd.Series:
    if "equity" in equity_curve.columns:
        return _series_with_time_index(equity_curve, "equity")
    if "equity_curve" in equity_curve.columns:
        return _series_with_time_index(equity_curve, "equity_curve")
    raise ValueError("Equity artifact must contain an 'equity' or 'equity_curve' column.")


def _select_returns_series(equity_curve: pd.DataFrame) -> pd.Series | None:
    if "strategy_return" not in equity_curve.columns:
        return None
    returns = _series_with_time_index(equity_curve, "strategy_return").dropna()
    if returns.empty:
        return None
    return returns


def _select_trade_returns(trades: pd.DataFrame | None) -> pd.Series | None:
    if trades is None or "return" not in trades.columns:
        return None
    trade_returns = pd.to_numeric(trades["return"], errors="coerce").dropna()
    if trade_returns.empty:
        return None
    trade_returns.name = "return"
    return trade_returns


def _series_with_time_index(frame: pd.DataFrame, value_column: str) -> pd.Series:
    series = pd.to_numeric(frame[value_column], errors="coerce")
    index = _resolve_index(frame)
    resolved = pd.Series(series.to_numpy(dtype="float64"), index=index, name=value_column)
    return resolved.dropna()


def _resolve_index(frame: pd.DataFrame) -> pd.Index:
    for column in ("ts_utc", "date"):
        if column in frame.columns:
            parsed = pd.to_datetime(frame[column], utc=True, errors="coerce")
            if parsed.notna().any():
                return pd.DatetimeIndex(parsed)
    return pd.RangeIndex(start=0, stop=len(frame), step=1)


def _relative_markdown_path(report_path: Path, target_path: Path) -> str:
    return target_path.relative_to(report_path.parent).as_posix()


def _format_value(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if pd.isna(value):
            return "NA"
        return f"{value:.6f}"
    return str(value)
