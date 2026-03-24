from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from src.research.visualization.artifacts import (
    get_canonical_plot_name,
    get_plot_dir,
    get_plot_intent,
    get_plot_path,
    is_standard_plot_dir,
)
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
    ("turnover", "Turnover"),
    ("exposure_pct", "Exposure"),
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
_PERCENT_METRIC_KEYS = {
    "annualized_return",
    "annualized_volatility",
    "cumulative_return",
    "loss_rate",
    "max_drawdown",
    "mean_return",
    "median_return",
    "std_return",
    "total_return",
    "volatility",
    "win_rate",
}
_DECIMAL_METRIC_KEYS = {"profit_factor", "sharpe_ratio", "turnover"}
_COUNT_METRIC_KEYS = {"count", "win_count", "loss_count"}
_REPORT_PLOT_ORDER: tuple[tuple[str, str | None, str], ...] = (
    ("equity_curve", "### Performance Overview", "Equity Curve"),
    ("drawdown", None, "Drawdown"),
)


def generate_strategy_report(run_dir: Path, output_path: Path | None = None) -> Path:
    """Generate a deterministic Markdown report for one strategy run directory.

    Plot artifacts are always resolved under the standardized ``<run_dir>/plots``
    directory so report generation and direct plot generation share one layout.
    """

    resolved_run_dir = Path(run_dir)
    _validate_run_dir(resolved_run_dir)

    resolved_output_path = Path(output_path) if output_path is not None else resolved_run_dir / "report.md"
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    plots_dir = get_plot_dir(resolved_run_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(resolved_run_dir / "metrics.json")
    manifest = _load_json_if_exists(resolved_run_dir / "manifest.json") or {}
    config = _load_json_if_exists(resolved_run_dir / "config.json") or {}
    equity_curve = _load_equity_curve(resolved_run_dir)
    trades = _load_optional_parquet(resolved_run_dir / "trades.parquet")
    signals = _load_optional_parquet(resolved_run_dir / "signals.parquet")

    plot_paths = generate_plot_artifacts(
        run_dir=resolved_run_dir,
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
        plot_paths=_select_plot_paths_by_intent(plot_paths, intent="report"),
    )
    resolved_output_path.write_text(markdown, encoding="utf-8")
    return resolved_output_path


def generate_strategy_plots(run_dir: Path, plots_dir: Path | None = None) -> dict[str, Path]:
    """Generate deterministic plot artifacts for one saved strategy run.

    The plotting contract mirrors the plot-generation step used by
    ``generate_strategy_report()``. Only artifacts supported by the saved run
    inputs are emitted, and artifacts are anchored under the standardized
    run-scoped plots directory.
    """

    resolved_run_dir = Path(run_dir)
    _validate_run_dir(resolved_run_dir)

    resolved_plots_dir = get_plot_dir(resolved_run_dir)
    if plots_dir is not None and not is_standard_plot_dir(resolved_run_dir, Path(plots_dir)):
        raise ValueError(
            "plots_dir must match the standardized run plot directory "
            f"'{resolved_plots_dir}'."
        )
    resolved_plots_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = generate_plot_artifacts(
        run_dir=resolved_run_dir,
        equity_curve=_load_equity_curve(resolved_run_dir),
        trades=_load_optional_parquet(resolved_run_dir / "trades.parquet"),
    )
    if not plot_paths:
        raise ValueError(
            "No plot artifacts can be generated from this run directory. "
            "Expected an equity_curve.csv artifact and/or trades.parquet with a numeric 'return' column."
        )
    return plot_paths


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
        rows.append(f"| {label} | {_format_metric_value(key, metrics.get(key))} |")
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

    context = _build_report_context(
        run_dir=run_dir,
        manifest=manifest,
        config=config,
        equity_curve=equity_curve,
        signals=signals,
    )
    sections = [
        render_header(context),
        "",
        "## Run Configuration Summary",
        render_configuration_summary(context),
        "",
        "## Key Metrics",
        format_metrics_table(metrics),
        "",
    ]
    sections.extend(render_visualizations(output_path=output_path, trades=trades, plot_paths=plot_paths))
    sections.extend(render_interpretation(context=context, metrics=metrics, trades=trades, signals=signals))
    sections.extend(render_artifact_links(run_dir=run_dir, output_path=output_path, plot_paths=plot_paths))
    return "\n".join(sections).rstrip() + "\n"


def generate_plot_artifacts(
    *,
    run_dir: Path,
    equity_curve: pd.DataFrame | None,
    trades: pd.DataFrame | None,
) -> dict[str, Path]:
    """Generate or reuse deterministic report and debug plots derived from run artifacts."""

    plot_paths: dict[str, Path] = {}

    if equity_curve is not None:
        equity_series = _select_equity_series(equity_curve)
        plot_paths["equity_curve"] = generate_plot_if_needed(
            output_path=get_plot_path(run_dir, "equity_curve"),
            plotter=plot_equity_curve,
            equity_data=equity_series,
            input_type="equity",
            title="Equity Curve",
        )
        plot_paths["drawdown"] = generate_plot_if_needed(
            output_path=get_plot_path(run_dir, "drawdown"),
            plotter=plot_drawdown,
            equity_data=equity_series,
            input_type="equity",
            title="Drawdown",
        )

        returns_series = _select_returns_series(equity_curve)
        if returns_series is not None and len(returns_series) >= _ROLLING_SHARPE_WINDOW:
            plot_paths["rolling_sharpe_debug"] = generate_plot_if_needed(
                output_path=get_plot_path(run_dir, "rolling_sharpe_debug"),
                plotter=plot_rolling_sharpe,
                returns=returns_series,
                window=_ROLLING_SHARPE_WINDOW,
                title=f"Rolling Sharpe ({_ROLLING_SHARPE_WINDOW}-period, Debug)",
            )

    trade_returns = _select_trade_returns(trades)
    if trade_returns is not None:
        plot_paths["trade_return_distribution_debug"] = generate_plot_if_needed(
            output_path=get_plot_path(run_dir, "trade_return_distribution_debug"),
            plotter=plot_trade_return_distribution,
            trade_returns=trade_returns,
            title="Trade Return Distribution (Debug)",
        )
        plot_paths["win_loss_distribution_debug"] = generate_plot_if_needed(
            output_path=get_plot_path(run_dir, "win_loss_distribution_debug"),
            plotter=plot_win_loss_distribution,
            trade_returns=trade_returns,
            title="Win/Loss Distribution (Debug)",
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


def render_header(context: dict[str, Any]) -> str:
    lines = [f"# Strategy Report: {context['title']}", ""]
    lines.extend(_render_key_value_table(context["header_rows"]))
    return "\n".join(lines)


def render_configuration_summary(context: dict[str, Any]) -> str:
    return "\n".join(_render_key_value_table(context["config_rows"]))


def render_visualizations(
    *,
    output_path: Path,
    trades: pd.DataFrame | None,
    plot_paths: dict[str, Path],
) -> list[str]:
    sections = ["## Visualizations"]

    performance_blocks = _render_plot_blocks(output_path=output_path, plot_paths=plot_paths, plot_order=_REPORT_PLOT_ORDER)

    if performance_blocks:
        sections.extend(performance_blocks)
    else:
        sections.extend(["_No visualization artifacts were available for this run._", ""])

    sections.extend(_render_trade_metrics_subsection(trades=trades))

    return sections


def render_interpretation(
    *,
    context: dict[str, Any],
    metrics: dict[str, Any],
    trades: pd.DataFrame | None,
    signals: pd.DataFrame | None,
) -> list[str]:
    notes = ["## Interpretation"]
    summary_points = _build_interpretation_points(
        context=context,
        metrics=metrics,
        trades=trades,
        signals=signals,
    )
    if not summary_points:
        notes.extend(["- Interpretation unavailable from the saved artifacts.", ""])
        return notes

    notes.extend(f"- {point}" for point in summary_points)
    notes.append("")
    return notes


def render_artifact_links(
    *,
    run_dir: Path,
    output_path: Path,
    plot_paths: dict[str, Path],
) -> list[str]:
    rows: list[str] = ["## Artifact References"]

    for label, artifact_path in _ordered_artifact_links(run_dir=run_dir, plot_paths=plot_paths):
        rows.append(f"- [{label}]({_relative_markdown_path(output_path, artifact_path)})")

    rows.append("")
    return rows


def _format_statistics_table(statistics: dict[str, float], ordered_keys: tuple[tuple[str, str], ...]) -> list[str]:
    lines = ["| Metric | Value |", "| --- | --- |"]
    for key, label in ordered_keys:
        if key not in statistics:
            continue
        lines.append(f"| {label} | {_format_metric_value(key, statistics[key])} |")
    lines.append("")
    return lines


def _build_report_context(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    config: dict[str, Any],
    equity_curve: pd.DataFrame | None,
    signals: pd.DataFrame | None,
) -> dict[str, Any]:
    strategy_name = str(
        manifest.get("strategy_name")
        or config.get("strategy_name")
        or manifest.get("run_id")
        or run_dir.name
    )
    run_id = str(manifest.get("run_id") or run_dir.name)
    evaluation_mode = str(manifest.get("evaluation_mode") or "single")
    timeframe = _resolve_timeframe(config=config, equity_curve=equity_curve, signals=signals)
    date_range = _resolve_date_range(config=config, equity_curve=equity_curve, signals=signals)

    header_rows = [
        ("Strategy", strategy_name),
        ("Run ID", run_id),
        ("Evaluation Mode", _humanize_token(evaluation_mode)),
    ]
    if timeframe is not None:
        header_rows.append(("Timeframe", timeframe))
    if date_range is not None:
        header_rows.append(("Date Range", date_range))
    if manifest.get("split_count") is not None:
        header_rows.append(("Split Count", str(manifest["split_count"])))

    config_rows: list[tuple[str, str]] = []
    dataset = config.get("dataset")
    if dataset is not None:
        config_rows.append(("Dataset", str(dataset)))

    parameters = config.get("parameters")
    if isinstance(parameters, dict) and parameters:
        config_rows.append(
            (
                "Parameters",
                ", ".join(f"{key}={parameters[key]!r}" for key in sorted(parameters)),
            )
        )

    evaluation = config.get("evaluation")
    if isinstance(evaluation, dict) and evaluation:
        config_rows.append(("Evaluation Config", _serialize_mapping(evaluation)))

    evaluation_config_path = manifest.get("evaluation_config_path") or config.get("evaluation_config_path")
    if evaluation_config_path is not None:
        config_rows.append(("Evaluation Config Path", str(evaluation_config_path)))

    primary_metric = manifest.get("primary_metric")
    if primary_metric is not None:
        config_rows.append(("Primary Metric", _humanize_metric_name(str(primary_metric))))

    if not config_rows:
        config_rows.append(("Run Directory", run_dir.name))

    return {
        "title": strategy_name,
        "strategy_name": strategy_name,
        "run_id": run_id,
        "evaluation_mode": evaluation_mode,
        "header_rows": header_rows,
        "config_rows": config_rows,
        "timeframe": timeframe,
        "date_range": date_range,
        "split_count": manifest.get("split_count"),
    }


def _render_plot_blocks(
    *,
    output_path: Path,
    plot_paths: dict[str, Path],
    plot_order: tuple[tuple[str, str | None, str], ...],
) -> list[str]:
    rows: list[str] = []
    current_heading: str | None = None
    for key, heading, alt_text in plot_order:
        if key not in plot_paths:
            continue
        if heading is not None and heading != current_heading:
            rows.extend([heading, ""])
            current_heading = heading
        rows.append(f"![{alt_text}]({_relative_markdown_path(output_path, plot_paths[key])})")
        rows.append("")
    return rows


def _render_trade_metrics_subsection(*, trades: pd.DataFrame | None) -> list[str]:
    section = ["### Trade Summary", ""]
    trade_returns = _select_trade_returns(trades)
    if trade_returns is None:
        section.extend(["_Trade data unavailable for this run._", ""])
        return section

    statistics = compute_trade_statistics(trade_returns)
    section.extend(_format_statistics_table(statistics, _TRADE_METRICS))
    return section


def _build_interpretation_points(
    *,
    context: dict[str, Any],
    metrics: dict[str, Any],
    trades: pd.DataFrame | None,
    signals: pd.DataFrame | None,
) -> list[str]:
    points: list[str] = []

    total_return = _coerce_float(metrics.get("total_return"))
    sharpe_ratio = _coerce_float(metrics.get("sharpe_ratio"))
    max_drawdown = _coerce_float(metrics.get("max_drawdown"))
    win_rate = _coerce_float(metrics.get("win_rate"))

    if total_return is not None:
        direction = "positive" if total_return >= 0 else "negative"
        message = f"The run finished with a {direction} total return of {_format_metric_value('total_return', total_return)}."
        if sharpe_ratio is not None:
            message += f" Sharpe was {_format_metric_value('sharpe_ratio', sharpe_ratio)}."
        points.append(message)

    if max_drawdown is not None:
        points.append(
            f"Peak-to-trough drawdown reached {_format_metric_value('max_drawdown', max_drawdown)}, which frames the downside seen in the equity and drawdown plots."
        )

    trade_returns = _select_trade_returns(trades)
    if trade_returns is not None:
        trade_count = len(trade_returns)
        trade_message = f"Trade diagnostics cover `{trade_count}` closed trades."
        if win_rate is not None:
            trade_message += f" Win rate was {_format_metric_value('win_rate', win_rate)}."
        points.append(trade_message)
    elif signals is not None:
        points.append(f"Signal artifacts are available with `{len(signals)}` rows, but no trade summary artifact was present.")

    if context.get("evaluation_mode") == "walk_forward" and context.get("split_count") is not None:
        points.append(f"Walk-forward artifacts summarize `{context['split_count']}` saved split(s); use `metrics_by_split.csv` for fold-level detail.")

    return points


def _ordered_artifact_links(*, run_dir: Path, plot_paths: dict[str, Path]) -> list[tuple[str, Path]]:
    ordered: list[tuple[str, Path]] = []
    for name in (
        "manifest.json",
        "config.json",
        "metrics.json",
        "equity_curve.csv",
        "signals.parquet",
        "trades.parquet",
        "metrics_by_split.csv",
    ):
        artifact_path = run_dir / name
        if artifact_path.exists():
            ordered.append((name, artifact_path))

    if plot_paths:
        plots_dir = get_plot_dir(run_dir)
        if plots_dir.exists():
            ordered.append(("plots/", plots_dir))
        for plot_name in sorted(plot_paths):
            if get_plot_intent(plot_name) != "report":
                continue
            ordered.append((plot_paths[plot_name].name, plot_paths[plot_name]))

    return ordered


def _render_key_value_table(rows: list[tuple[str, str]]) -> list[str]:
    table = ["| Field | Value |", "| --- | --- |"]
    for label, value in rows:
        table.append(f"| {label} | {value} |")
    return table


def _serialize_mapping(mapping: dict[str, Any]) -> str:
    return ", ".join(f"{key}={mapping[key]!r}" for key in sorted(mapping))


def _resolve_timeframe(
    *,
    config: dict[str, Any],
    equity_curve: pd.DataFrame | None,
    signals: pd.DataFrame | None,
) -> str | None:
    for frame in (equity_curve, signals):
        if frame is None or "timeframe" not in frame.columns:
            continue
        series = frame["timeframe"].dropna()
        if not series.empty:
            return str(series.iloc[0])

    evaluation = config.get("evaluation")
    if isinstance(evaluation, dict) and evaluation.get("timeframe") is not None:
        return str(evaluation["timeframe"])

    dataset = config.get("dataset")
    if isinstance(dataset, str):
        normalized = dataset.lower()
        if normalized.endswith("daily"):
            return "1D"
        if normalized.endswith("1m"):
            return "1Min"
    return None


def _resolve_date_range(
    *,
    config: dict[str, Any],
    equity_curve: pd.DataFrame | None,
    signals: pd.DataFrame | None,
) -> str | None:
    for frame in (equity_curve, signals):
        resolved = _date_range_from_frame(frame)
        if resolved is not None:
            return resolved

    evaluation = config.get("evaluation")
    if isinstance(evaluation, dict):
        start = evaluation.get("start") or evaluation.get("train_start")
        end = evaluation.get("end") or evaluation.get("test_end")
        if start is not None or end is not None:
            return _format_date_range(start, end)

    if config.get("start") is not None or config.get("end") is not None:
        return _format_date_range(config.get("start"), config.get("end"))
    return None


def _date_range_from_frame(frame: pd.DataFrame | None) -> str | None:
    if frame is None or frame.empty:
        return None

    for column in ("ts_utc", "date"):
        if column not in frame.columns:
            continue
        parsed = pd.to_datetime(frame[column], utc=True, errors="coerce").dropna()
        if parsed.empty:
            continue
        start = parsed.min()
        end = parsed.max()
        if column == "date":
            return _format_date_range(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        return _format_date_range(
            start.isoformat().replace("+00:00", "Z"),
            end.isoformat().replace("+00:00", "Z"),
        )
    return None


def _format_date_range(start: Any, end: Any) -> str:
    start_text = "NA" if start is None else str(start)
    end_text = "NA" if end is None else str(end)
    return f"{start_text} to {end_text}"


def _humanize_token(value: str) -> str:
    return value.replace("_", " ").strip().title()


def _humanize_metric_name(value: str) -> str:
    return value.replace("_", " ").strip()


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float) and not pd.isna(value):
        return float(value)
    return None


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
    return Path(os.path.relpath(target_path, start=report_path.parent)).as_posix()


def _select_plot_paths_by_intent(plot_paths: dict[str, Path], *, intent: str) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    for plot_name, plot_path in plot_paths.items():
        canonical_name = get_canonical_plot_name(plot_name)
        if get_plot_intent(canonical_name) == intent:
            selected[canonical_name] = plot_path
    return selected


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


def _format_metric_value(metric_name: str, value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if pd.isna(value):
            return "NA"
        if metric_name in _COUNT_METRIC_KEYS and float(value).is_integer():
            return str(int(value))
        if metric_name == "exposure_pct":
            return f"{value:.2f}%"
        if metric_name in _PERCENT_METRIC_KEYS:
            return f"{value * 100:.2f}%"
        if metric_name in _DECIMAL_METRIC_KEYS:
            return f"{value:.3f}"
        return f"{value:.6f}"
    return str(value)
