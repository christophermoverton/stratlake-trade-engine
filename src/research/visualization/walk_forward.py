"""Plotting interfaces for walk-forward splits and fold-level evaluation results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

from src.research.visualization.equity import DEFAULT_FIGSIZE

if TYPE_CHECKING:
    from matplotlib.figure import Figure
else:
    Figure = Any

PlotResult = Path | Figure
RecordInput = pd.DataFrame | Sequence[Mapping[str, object]]

_TRAIN_COLOR = "tab:blue"
_TEST_COLOR = "tab:orange"


def plot_walk_forward_splits(
    splits_frame: RecordInput,
    *,
    title: str = "Walk-Forward Splits",
    output_path: Path | None = None,
) -> PlotResult:
    """Plot train and test windows for each walk-forward split.

    Args:
        splits_frame: A DataFrame or sequence of records with one row per fold.
            Required columns are ``fold``, ``train_start``, ``train_end``,
            ``test_start``, and ``test_end``.
        title: Chart title.
        output_path: Optional output location. When provided, the figure is
            saved as a PNG and the saved path is returned.
    """

    normalized = normalize_walk_forward_folds(splits_frame)

    figure, axis = plt.subplots(figsize=DEFAULT_FIGSIZE)
    y_positions = list(range(len(normalized)))
    fold_labels = normalized["fold"].astype(str).tolist()

    for y_position, fold in zip(y_positions, normalized.itertuples(index=False), strict=False):
        axis.barh(
            y_position,
            width=mdates.date2num(fold.train_end) - mdates.date2num(fold.train_start),
            left=mdates.date2num(fold.train_start),
            height=0.6,
            color=_TRAIN_COLOR,
            alpha=0.8,
        )
        axis.barh(
            y_position,
            width=mdates.date2num(fold.test_end) - mdates.date2num(fold.test_start),
            left=mdates.date2num(fold.test_start),
            height=0.6,
            color=_TEST_COLOR,
            alpha=0.9,
            hatch="//",
        )

    axis.set_title(title)
    axis.set_xlabel("Date")
    axis.set_ylabel("Fold")
    axis.set_yticks(y_positions)
    axis.set_yticklabels(fold_labels)
    axis.invert_yaxis()
    axis.xaxis_date()
    locator = mdates.AutoDateLocator()
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axis.grid(True, axis="x", linestyle=":", linewidth=0.75, alpha=0.7)
    axis.legend(
        handles=[
            Patch(facecolor=_TRAIN_COLOR, edgecolor=_TRAIN_COLOR, alpha=0.8, label="Train"),
            Patch(facecolor=_TEST_COLOR, edgecolor=_TEST_COLOR, alpha=0.9, hatch="//", label="Test"),
        ]
    )
    figure.autofmt_xdate()
    figure.tight_layout()

    return _save_or_return_figure(figure, output_path)


def plot_fold_level_metrics(
    fold_metrics: RecordInput,
    *,
    metric_name: str,
    title: str | None = None,
    output_path: Path | None = None,
) -> PlotResult:
    """Plot one metric across walk-forward folds.

    Args:
        fold_metrics: A DataFrame or sequence of records with one row per fold.
            Required columns are ``fold`` and the requested ``metric_name``.
        metric_name: Numeric metric column to plot, such as ``"sharpe"`` or
            ``"max_drawdown"``.
        title: Optional chart title. When omitted, a deterministic title is
            generated from ``metric_name``.
        output_path: Optional output location. When provided, the figure is
            saved as a PNG and the saved path is returned.
    """

    normalized = normalize_fold_metrics(fold_metrics, metric_name=metric_name)
    resolved_title = title or f"Fold-Level {metric_name}"
    values = normalized[metric_name].astype("float64")
    labels = normalized["fold"].astype(str)
    colors = ["tab:green" if value >= 0.0 else "tab:red" for value in values]

    figure, axis = plt.subplots(figsize=DEFAULT_FIGSIZE)
    axis.bar(labels, values, color=colors, edgecolor="black", linewidth=0.8, alpha=0.85)
    axis.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.8)
    axis.set_title(resolved_title)
    axis.set_xlabel("Fold")
    axis.set_ylabel(metric_name)
    axis.grid(True, axis="y", linestyle=":", linewidth=0.75, alpha=0.7)
    figure.tight_layout()

    return _save_or_return_figure(figure, output_path)


def plot_walk_forward_results(
    fold_results: Sequence[pd.DataFrame],
    *,
    labels: Sequence[str] | None = None,
    title: str = "Walk-Forward Results",
    output_path: Path | None = None,
) -> PlotResult:
    """Plot a per-fold return summary from walk-forward result frames.

    Args:
        fold_results: Sequence of per-fold result DataFrames. Each frame must be
            non-empty and contain either an ``equity`` column, a
            ``strategy_return`` column, or another numeric column.
        labels: Optional fold labels. Defaults to ``fold_0``, ``fold_1``, and
            so on when omitted.
        title: Chart title.
        output_path: Optional output location. When provided, the figure is
            saved as a PNG and the saved path is returned.
    """

    normalized = _normalize_fold_result_summaries(fold_results, labels=labels)

    figure, axis = plt.subplots(figsize=DEFAULT_FIGSIZE)
    axis.plot(
        normalized["fold"].astype(str),
        normalized["value"].astype("float64"),
        marker="o",
        linewidth=2.0,
        color="tab:blue",
        label=normalized["metric_label"].iat[0],
    )
    axis.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.8)
    axis.set_title(title)
    axis.set_xlabel("Fold")
    axis.set_ylabel(normalized["metric_label"].iat[0])
    axis.legend()
    axis.grid(True, axis="y", linestyle=":", linewidth=0.75, alpha=0.7)
    figure.tight_layout()

    return _save_or_return_figure(figure, output_path)


def normalize_walk_forward_folds(splits_frame: RecordInput) -> pd.DataFrame:
    """Return a copied fold table with validated date windows."""

    normalized = _coerce_records_to_frame(splits_frame, input_name="Walk-forward folds")
    _validate_required_columns(
        normalized,
        required_columns=["fold", "train_start", "train_end", "test_start", "test_end"],
        input_name="Walk-forward folds",
    )

    result = normalized.loc[:, ["fold", "train_start", "train_end", "test_start", "test_end"]].copy(deep=True)
    for column in ("train_start", "train_end", "test_start", "test_end"):
        result[column] = _coerce_datetime_column(result[column], column_name=column)

    _validate_interval_order(result["train_start"], result["train_end"], label="train")
    _validate_interval_order(result["test_start"], result["test_end"], label="test")
    return result.reset_index(drop=True)


def normalize_fold_metrics(fold_metrics: RecordInput, *, metric_name: str) -> pd.DataFrame:
    """Return a copied fold metric table with validated numeric values."""

    normalized = _coerce_records_to_frame(fold_metrics, input_name="Fold metrics")
    _validate_required_columns(
        normalized,
        required_columns=["fold", metric_name],
        input_name="Fold metrics",
    )

    result = normalized.loc[:, ["fold", metric_name]].copy(deep=True)
    result[metric_name] = pd.to_numeric(result[metric_name], errors="coerce")
    if result[metric_name].isna().any():
        raise ValueError(f"Fold metrics column '{metric_name}' must contain only numeric values.")

    return result.reset_index(drop=True)


def _coerce_records_to_frame(data: RecordInput, *, input_name: str) -> pd.DataFrame:
    """Copy supported record inputs into a DataFrame without mutating the caller."""

    if isinstance(data, pd.DataFrame):
        frame = data.copy(deep=True)
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        if len(data) == 0:
            raise ValueError(f"{input_name} must be non-empty.")
        frame = pd.DataFrame(list(data))
    else:
        raise ValueError(f"{input_name} must be a pandas DataFrame or a sequence of records.")

    if frame.empty:
        raise ValueError(f"{input_name} must be non-empty.")
    return frame


def _validate_required_columns(
    frame: pd.DataFrame,
    *,
    required_columns: Sequence[str],
    input_name: str,
) -> None:
    """Validate that a frame contains all required columns."""

    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        missing_list = ", ".join(missing_columns)
        raise ValueError(f"{input_name} is missing required columns: {missing_list}.")


def _coerce_datetime_column(values: pd.Series, *, column_name: str) -> pd.Series:
    """Convert a column to datetimes or raise a clear validation error."""

    try:
        return pd.to_datetime(values, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Column '{column_name}' must contain valid datetime values.") from exc


def _validate_interval_order(start_values: pd.Series, end_values: pd.Series, *, label: str) -> None:
    """Ensure each interval has an end timestamp greater than or equal to its start."""

    if (end_values < start_values).any():
        raise ValueError(f"Each {label} window must end on or after its start.")


def _normalize_fold_result_summaries(
    fold_results: Sequence[pd.DataFrame],
    *,
    labels: Sequence[str] | None,
) -> pd.DataFrame:
    """Summarize each fold result frame into one comparable numeric value."""

    if len(fold_results) == 0:
        raise ValueError("fold_results must be non-empty.")
    if labels is not None and len(labels) != len(fold_results):
        raise ValueError("labels must match the number of fold result frames.")

    rows: list[dict[str, object]] = []
    for index, frame in enumerate(fold_results):
        if not isinstance(frame, pd.DataFrame):
            raise ValueError("Each fold result must be a pandas DataFrame.")
        if frame.empty:
            raise ValueError("Each fold result must be non-empty.")

        label = labels[index] if labels is not None else f"fold_{index}"
        value, metric_label = _extract_fold_summary_value(frame.copy(deep=True))
        rows.append({"fold": label, "value": value, "metric_label": metric_label})

    return pd.DataFrame(rows)


def _extract_fold_summary_value(frame: pd.DataFrame) -> tuple[float, str]:
    """Derive a deterministic summary value from one fold result DataFrame."""

    if "equity" in frame.columns:
        equity = pd.to_numeric(frame["equity"], errors="coerce").dropna()
        if equity.empty:
            raise ValueError("Fold result column 'equity' must contain numeric values.")
        return float(equity.iloc[-1] - 1.0), "Fold Return"

    if "strategy_return" in frame.columns:
        returns = pd.to_numeric(frame["strategy_return"], errors="coerce").dropna()
        if returns.empty:
            raise ValueError("Fold result column 'strategy_return' must contain numeric values.")
        return float((1.0 + returns).prod() - 1.0), "Fold Return"

    numeric_columns = frame.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    if numeric_columns.empty:
        raise ValueError(
            "Each fold result must contain an 'equity' column, a 'strategy_return' column, or another numeric column."
        )

    column_name = str(numeric_columns.columns[0])
    series = numeric_columns[column_name].dropna()
    if series.empty:
        raise ValueError("Fold result numeric columns must contain at least one numeric value.")
    return float(series.iloc[-1]), column_name


def _save_or_return_figure(figure: Figure, output_path: Path | None) -> PlotResult:
    """Save a figure as a PNG artifact or return it directly."""

    if output_path is None:
        return figure

    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(resolved_output_path, format="png", dpi=100)
    plt.close(figure)
    return resolved_output_path
