from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from src.research.metrics import (
    TRADING_DAYS_PER_YEAR,
    annualized_return,
    annualized_volatility,
    max_drawdown,
    sharpe_ratio,
    total_return,
    volatility,
    win_rate,
)
from src.research.regimes.taxonomy import (
    REGIME_DIMENSIONS,
    REGIME_STATE_COLUMNS,
    TAXONOMY_VERSION,
    UNDEFINED_REGIME_LABEL,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default output prefix assumed from M24.2 alignment helpers.
_DEFAULT_REGIME_PREFIX = "regime_"

#: Minimum observation count required to compute regime-conditioned metrics.
#: Subsets below this threshold receive null/undefined metric values.
MIN_REGIME_OBSERVATIONS = 5

#: Column name for the composite regime label produced by alignment helpers.
REGIME_LABEL_COLUMN = "regime_label"

#: Alignment status values from M24.2.
_STATUS_DEFINED = "matched_defined"
_STATUS_UNDEFINED = "matched_undefined"
_STATUS_UNMATCHED = "unmatched_timestamp"
_STATUS_UNAVAILABLE = "regime_labels_unavailable"

#: Surface identifiers that govern which metric set is applied.
ConditionalSurface = Literal["strategy", "alpha", "portfolio"]

#: Stable column ordering for metrics_by_regime CSV output.
_STRATEGY_METRIC_COLUMNS: tuple[str, ...] = (
    "regime_label",
    "dimension",
    "observation_count",
    "coverage_status",
    "total_return",
    "annualized_return",
    "volatility",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "win_rate",
)

_ALPHA_METRIC_COLUMNS: tuple[str, ...] = (
    "regime_label",
    "dimension",
    "observation_count",
    "coverage_status",
    "mean_ic",
    "mean_rank_ic",
    "ic_std",
    "rank_ic_std",
    "ic_ir",
    "rank_ic_ir",
)

_PORTFOLIO_METRIC_COLUMNS: tuple[str, ...] = (
    "regime_label",
    "dimension",
    "observation_count",
    "coverage_status",
    "total_return",
    "annualized_return",
    "volatility",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "win_rate",
)

#: Coverage status labels for regime subsets.
_COVERAGE_SUFFICIENT = "sufficient"
_COVERAGE_SPARSE = "sparse"
_COVERAGE_EMPTY = "empty"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeConditionalConfig:
    """Configuration for regime-conditional evaluation.

    Parameters
    ----------
    min_observations:
        Minimum number of ``matched_defined`` rows required within a regime
        subset to compute meaningful metrics. Subsets below this threshold
        receive null metric values and a ``sparse`` coverage status.
    regime_prefix:
        Column prefix used by M24.2 alignment helpers. Must match the prefix
        that was configured when ``align_regime_labels`` was called.
    periods_per_year:
        Annualization factor for return-based metrics. Defaults to 252 for
        daily data.
    taxonomy_version:
        Regime taxonomy version tag included in artifact metadata for
        traceability.
    metadata:
        Arbitrary additional metadata propagated to artifact payloads.
    """

    min_observations: int = MIN_REGIME_OBSERVATIONS
    regime_prefix: str = _DEFAULT_REGIME_PREFIX
    periods_per_year: int = TRADING_DAYS_PER_YEAR
    taxonomy_version: str = TAXONOMY_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.min_observations < 1:
            raise ValueError("min_observations must be at least 1.")
        if not self.regime_prefix:
            raise ValueError("regime_prefix must be a non-empty string.")
        if self.periods_per_year < 1:
            raise ValueError("periods_per_year must be at least 1.")


def resolve_regime_conditional_config(
    config: RegimeConditionalConfig | dict[str, Any] | None = None,
) -> RegimeConditionalConfig:
    """Return a resolved ``RegimeConditionalConfig`` from any accepted input form."""

    if config is None:
        return RegimeConditionalConfig()
    if isinstance(config, RegimeConditionalConfig):
        return config
    if not isinstance(config, dict):
        raise TypeError("RegimeConditionalConfig must be a dataclass instance or a dict.")

    allowed = set(RegimeConditionalConfig.__dataclass_fields__)
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(f"Unsupported RegimeConditionalConfig fields: {unknown}.")
    return RegimeConditionalConfig(**config)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeConditionalResult:
    """Container for regime-conditional evaluation outputs.

    Parameters
    ----------
    surface:
        Workflow surface (``"strategy"``, ``"alpha"``, or ``"portfolio"``).
    dimension:
        Regime dimension name used as the grouping key (e.g.
        ``"composite"`` for the combined label, or a taxonomy dimension such as
        ``"volatility"``).
    metrics_by_regime:
        One row per regime label with stable metric columns and observation
        counts. Never empty — regimes with insufficient observations carry
        null metric values and a ``sparse`` coverage status.
    alignment_summary:
        Observation-count breakdown by alignment status.
    config:
        The resolved configuration used for this evaluation.
    metadata:
        Propagated metadata from the configuration and evaluation context.
    """

    surface: str
    dimension: str
    metrics_by_regime: pd.DataFrame
    alignment_summary: dict[str, int]
    config: RegimeConditionalConfig
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public evaluation functions
# ---------------------------------------------------------------------------


def evaluate_strategy_metrics_by_regime(
    aligned_frame: pd.DataFrame,
    *,
    return_column: str = "strategy_return",
    config: RegimeConditionalConfig | dict[str, Any] | None = None,
    dimension: str = "composite",
) -> RegimeConditionalResult:
    """Compute regime-conditioned strategy metrics from an M24.2-aligned frame.

    The function expects ``aligned_frame`` to have already had regime context
    attached via ``align_regimes_to_strategy_timeseries`` or
    ``align_regime_labels``. Only ``matched_defined`` rows contribute to
    metric computations.

    Parameters
    ----------
    aligned_frame:
        Strategy timeseries with regime columns attached by M24.2 alignment.
    return_column:
        Name of the period-level strategy return column.
    config:
        Evaluation configuration. Defaults to ``RegimeConditionalConfig()``.
    dimension:
        Regime grouping dimension. Use ``"composite"`` for the full
        ``regime_label`` column, or pass a taxonomy dimension name such as
        ``"volatility"`` to group by a single dimension's state column.

    Returns
    -------
    RegimeConditionalResult
        Regime-conditioned strategy metrics with stable column ordering.
    """

    resolved_config = resolve_regime_conditional_config(config)
    _validate_aligned_frame(aligned_frame, config=resolved_config)
    _require_column(aligned_frame, return_column, context="strategy evaluation")

    group_column = _resolve_group_column(dimension, config=resolved_config)
    defined_rows = _filter_defined_rows(aligned_frame, config=resolved_config)
    alignment_summary = _build_alignment_summary(aligned_frame, config=resolved_config)
    regime_groups = _stable_regime_groups(defined_rows, group_column=group_column)

    rows: list[dict[str, Any]] = []
    for regime_label, group in regime_groups:
        returns = group[return_column].dropna()
        n = int(len(returns))
        coverage = _coverage_status(n, min_obs=resolved_config.min_observations)

        row: dict[str, Any] = {
            "regime_label": str(regime_label),
            "dimension": dimension,
            "observation_count": n,
            "coverage_status": coverage,
        }
        if coverage == _COVERAGE_SUFFICIENT:
            row.update(
                {
                    "total_return": _safe_float(total_return(returns)),
                    "annualized_return": _safe_float(
                        annualized_return(returns, periods_per_year=resolved_config.periods_per_year)
                    ),
                    "volatility": _safe_float(volatility(returns)),
                    "annualized_volatility": _safe_float(
                        annualized_volatility(returns, periods_per_year=resolved_config.periods_per_year)
                    ),
                    "sharpe_ratio": _safe_float(
                        sharpe_ratio(returns, periods_per_year=resolved_config.periods_per_year)
                    ),
                    "max_drawdown": _safe_float(max_drawdown(returns)),
                    "win_rate": _safe_float(win_rate(returns)),
                }
            )
        else:
            row.update(
                {
                    "total_return": None,
                    "annualized_return": None,
                    "volatility": None,
                    "annualized_volatility": None,
                    "sharpe_ratio": None,
                    "max_drawdown": None,
                    "win_rate": None,
                }
            )
        rows.append(row)

    metrics_frame = _build_metrics_frame(rows, columns=_STRATEGY_METRIC_COLUMNS)
    metadata = _build_result_metadata(resolved_config, surface="strategy", dimension=dimension)

    return RegimeConditionalResult(
        surface="strategy",
        dimension=dimension,
        metrics_by_regime=metrics_frame,
        alignment_summary=alignment_summary,
        config=resolved_config,
        metadata=metadata,
    )


def evaluate_alpha_metrics_by_regime(
    aligned_ic_frame: pd.DataFrame,
    *,
    ic_column: str = "ic",
    rank_ic_column: str = "rank_ic",
    config: RegimeConditionalConfig | dict[str, Any] | None = None,
    dimension: str = "composite",
) -> RegimeConditionalResult:
    """Compute regime-conditioned IC / Rank IC metrics from an M24.2-aligned IC frame.

    The function expects ``aligned_ic_frame`` to be a timestamp-level IC
    timeseries (as produced by ``evaluate_information_coefficient``) with
    regime context attached via ``align_regimes_to_alpha_windows``.

    Only ``matched_defined`` rows contribute to metric computations. Subsets
    with fewer than ``min_observations`` IC values receive null outputs with a
    ``sparse`` coverage status rather than potentially misleading statistics.

    Parameters
    ----------
    aligned_ic_frame:
        IC timeseries with regime columns attached by M24.2 alignment.
    ic_column:
        Name of the Pearson IC column.
    rank_ic_column:
        Name of the Spearman Rank IC column.
    config:
        Evaluation configuration. Defaults to ``RegimeConditionalConfig()``.
    dimension:
        Regime grouping dimension.

    Returns
    -------
    RegimeConditionalResult
        Regime-conditioned IC / Rank IC metrics with stable column ordering.
    """

    resolved_config = resolve_regime_conditional_config(config)
    _validate_aligned_frame(aligned_ic_frame, config=resolved_config)
    _require_column(aligned_ic_frame, ic_column, context="alpha IC evaluation")
    _require_column(aligned_ic_frame, rank_ic_column, context="alpha rank IC evaluation")

    group_column = _resolve_group_column(dimension, config=resolved_config)
    defined_rows = _filter_defined_rows(aligned_ic_frame, config=resolved_config)
    alignment_summary = _build_alignment_summary(aligned_ic_frame, config=resolved_config)
    regime_groups = _stable_regime_groups(defined_rows, group_column=group_column)

    rows: list[dict[str, Any]] = []
    for regime_label, group in regime_groups:
        ic_values = group[ic_column].dropna()
        rank_ic_values = group[rank_ic_column].dropna()
        n = int(len(group))
        coverage = _coverage_status(n, min_obs=resolved_config.min_observations)

        row: dict[str, Any] = {
            "regime_label": str(regime_label),
            "dimension": dimension,
            "observation_count": n,
            "coverage_status": coverage,
        }
        if coverage == _COVERAGE_SUFFICIENT:
            ic_std = float(ic_values.std()) if len(ic_values) >= 2 else float("nan")
            rank_ic_std = float(rank_ic_values.std()) if len(rank_ic_values) >= 2 else float("nan")
            mean_ic = float(ic_values.mean()) if not ic_values.empty else float("nan")
            mean_rank_ic = float(rank_ic_values.mean()) if not rank_ic_values.empty else float("nan")
            row.update(
                {
                    "mean_ic": _safe_float(mean_ic),
                    "mean_rank_ic": _safe_float(mean_rank_ic),
                    "ic_std": _safe_float(ic_std),
                    "rank_ic_std": _safe_float(rank_ic_std),
                    "ic_ir": _safe_float(_information_ratio(mean_ic, ic_std)),
                    "rank_ic_ir": _safe_float(_information_ratio(mean_rank_ic, rank_ic_std)),
                }
            )
        else:
            row.update(
                {
                    "mean_ic": None,
                    "mean_rank_ic": None,
                    "ic_std": None,
                    "rank_ic_std": None,
                    "ic_ir": None,
                    "rank_ic_ir": None,
                }
            )
        rows.append(row)

    metrics_frame = _build_metrics_frame(rows, columns=_ALPHA_METRIC_COLUMNS)
    metadata = _build_result_metadata(resolved_config, surface="alpha", dimension=dimension)

    return RegimeConditionalResult(
        surface="alpha",
        dimension=dimension,
        metrics_by_regime=metrics_frame,
        alignment_summary=alignment_summary,
        config=resolved_config,
        metadata=metadata,
    )


def evaluate_portfolio_metrics_by_regime(
    aligned_frame: pd.DataFrame,
    *,
    return_column: str = "portfolio_return",
    config: RegimeConditionalConfig | dict[str, Any] | None = None,
    dimension: str = "composite",
) -> RegimeConditionalResult:
    """Compute regime-conditioned portfolio metrics from an M24.2-aligned frame.

    The function expects ``aligned_frame`` to have already had regime context
    attached via ``align_regimes_to_portfolio_windows`` or
    ``align_regime_labels``. Only ``matched_defined`` rows contribute to
    metric computations.

    Parameters
    ----------
    aligned_frame:
        Portfolio return timeseries with regime columns attached by M24.2 alignment.
    return_column:
        Name of the period-level portfolio return column.
    config:
        Evaluation configuration. Defaults to ``RegimeConditionalConfig()``.
    dimension:
        Regime grouping dimension.

    Returns
    -------
    RegimeConditionalResult
        Regime-conditioned portfolio metrics with stable column ordering.
    """

    resolved_config = resolve_regime_conditional_config(config)
    _validate_aligned_frame(aligned_frame, config=resolved_config)
    _require_column(aligned_frame, return_column, context="portfolio evaluation")

    group_column = _resolve_group_column(dimension, config=resolved_config)
    defined_rows = _filter_defined_rows(aligned_frame, config=resolved_config)
    alignment_summary = _build_alignment_summary(aligned_frame, config=resolved_config)
    regime_groups = _stable_regime_groups(defined_rows, group_column=group_column)

    rows: list[dict[str, Any]] = []
    for regime_label, group in regime_groups:
        returns = group[return_column].dropna()
        n = int(len(returns))
        coverage = _coverage_status(n, min_obs=resolved_config.min_observations)

        row: dict[str, Any] = {
            "regime_label": str(regime_label),
            "dimension": dimension,
            "observation_count": n,
            "coverage_status": coverage,
        }
        if coverage == _COVERAGE_SUFFICIENT:
            row.update(
                {
                    "total_return": _safe_float(total_return(returns)),
                    "annualized_return": _safe_float(
                        annualized_return(returns, periods_per_year=resolved_config.periods_per_year)
                    ),
                    "volatility": _safe_float(volatility(returns)),
                    "annualized_volatility": _safe_float(
                        annualized_volatility(returns, periods_per_year=resolved_config.periods_per_year)
                    ),
                    "sharpe_ratio": _safe_float(
                        sharpe_ratio(returns, periods_per_year=resolved_config.periods_per_year)
                    ),
                    "max_drawdown": _safe_float(max_drawdown(returns)),
                    "win_rate": _safe_float(win_rate(returns)),
                }
            )
        else:
            row.update(
                {
                    "total_return": None,
                    "annualized_return": None,
                    "volatility": None,
                    "annualized_volatility": None,
                    "sharpe_ratio": None,
                    "max_drawdown": None,
                    "win_rate": None,
                }
            )
        rows.append(row)

    metrics_frame = _build_metrics_frame(rows, columns=_PORTFOLIO_METRIC_COLUMNS)
    metadata = _build_result_metadata(resolved_config, surface="portfolio", dimension=dimension)

    return RegimeConditionalResult(
        surface="portfolio",
        dimension=dimension,
        metrics_by_regime=metrics_frame,
        alignment_summary=alignment_summary,
        config=resolved_config,
        metadata=metadata,
    )


def evaluate_all_dimensions(
    aligned_frame: pd.DataFrame,
    *,
    surface: ConditionalSurface,
    return_column: str | None = None,
    ic_column: str = "ic",
    rank_ic_column: str = "rank_ic",
    config: RegimeConditionalConfig | dict[str, Any] | None = None,
) -> dict[str, RegimeConditionalResult]:
    """Evaluate regime-conditioned metrics across the composite label and all dimensions.

    This is a convenience wrapper that calls the appropriate surface-specific
    function once for the composite label and once for each taxonomy dimension,
    returning results keyed by dimension name.

    Parameters
    ----------
    aligned_frame:
        M24.2-aligned frame for the target surface.
    surface:
        Workflow surface: ``"strategy"``, ``"alpha"``, or ``"portfolio"``.
    return_column:
        Return column name for strategy/portfolio surfaces. When ``None``,
        defaults to ``"strategy_return"`` for strategy and
        ``"portfolio_return"`` for portfolio.
    ic_column:
        IC column name for alpha surface.
    rank_ic_column:
        Rank IC column name for alpha surface.
    config:
        Evaluation configuration.

    Returns
    -------
    dict[str, RegimeConditionalResult]
        Results keyed by ``"composite"`` plus each taxonomy dimension name.
    """

    resolved_config = resolve_regime_conditional_config(config)
    dimensions = ["composite", *REGIME_DIMENSIONS]
    results: dict[str, RegimeConditionalResult] = {}

    for dimension in dimensions:
        if surface == "strategy":
        	rc_col = return_column if return_column is not None else "strategy_return"
        	results[dimension] = evaluate_strategy_metrics_by_regime(
        		aligned_frame,
        		return_column=rc_col,
        		config=resolved_config,
        		dimension=dimension,
        	)
        elif surface == "alpha":
        	results[dimension] = evaluate_alpha_metrics_by_regime(
        		aligned_frame,
        		ic_column=ic_column,
        		rank_ic_column=rank_ic_column,
        		config=resolved_config,
        		dimension=dimension,
        	)
        elif surface == "portfolio":
        	pc_col = return_column if return_column is not None else "portfolio_return"
        	results[dimension] = evaluate_portfolio_metrics_by_regime(
        		aligned_frame,
        		return_column=pc_col,
        		config=resolved_config,
        		dimension=dimension,
        	)
        else:
        	raise ValueError(f"Unsupported surface: {surface!r}. Expected 'strategy', 'alpha', or 'portfolio'.")

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_aligned_frame(
    frame: pd.DataFrame,
    *,
    config: RegimeConditionalConfig,
) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("aligned_frame must be a pandas DataFrame.")
    if frame.empty:
        raise ValueError("aligned_frame must not be empty.")

    status_col = f"{config.regime_prefix}alignment_status"
    if status_col not in frame.columns:
        raise ValueError(
            f"aligned_frame is missing the regime alignment status column {status_col!r}. "
            "Ensure the frame was produced by an M24.2 alignment helper before calling "
            "regime-conditional evaluation."
        )


def _require_column(frame: pd.DataFrame, column: str, *, context: str) -> None:
    if column not in frame.columns:
        raise ValueError(
            f"Column {column!r} required for {context} is missing from the aligned frame."
        )


def _resolve_group_column(dimension: str, *, config: RegimeConditionalConfig) -> str:
    """Return the DataFrame column name to group by for the requested dimension."""

    if dimension == "composite":
        return f"{config.regime_prefix}label"

    if dimension not in REGIME_DIMENSIONS:
        raise ValueError(
            f"Unknown regime dimension {dimension!r}. "
            f"Expected 'composite' or one of: {sorted(REGIME_DIMENSIONS)}."
        )
    state_column = REGIME_STATE_COLUMNS[dimension]
    return f"{config.regime_prefix}{state_column}"


def _filter_defined_rows(
    frame: pd.DataFrame,
    *,
    config: RegimeConditionalConfig,
) -> pd.DataFrame:
    """Return only rows with ``matched_defined`` alignment status."""

    status_col = f"{config.regime_prefix}alignment_status"
    mask = frame[status_col] == _STATUS_DEFINED
    return frame.loc[mask].copy()


def _build_alignment_summary(
    frame: pd.DataFrame,
    *,
    config: RegimeConditionalConfig,
) -> dict[str, int]:
    """Return observation counts for each alignment status category."""

    status_col = f"{config.regime_prefix}alignment_status"
    counts = frame[status_col].value_counts()
    return {
        "total_rows": int(len(frame)),
        "matched_defined": int(counts.get(_STATUS_DEFINED, 0)),
        "matched_undefined": int(counts.get(_STATUS_UNDEFINED, 0)),
        "unmatched_timestamp": int(counts.get(_STATUS_UNMATCHED, 0)),
        "regime_labels_unavailable": int(counts.get(_STATUS_UNAVAILABLE, 0)),
    }


def _stable_regime_groups(
    defined_frame: pd.DataFrame,
    *,
    group_column: str,
) -> list[tuple[str, pd.DataFrame]]:
    """Return deterministically ordered regime-label groups."""

    if group_column not in defined_frame.columns:
        return []

    groups: list[tuple[str, pd.DataFrame]] = []
    seen_labels = sorted(defined_frame[group_column].dropna().unique().tolist())
    for label in seen_labels:
        subset = defined_frame.loc[defined_frame[group_column] == label]
        groups.append((str(label), subset))
    return groups


def _coverage_status(n: int, *, min_obs: int) -> str:
    if n == 0:
        return _COVERAGE_EMPTY
    if n < min_obs:
        return _COVERAGE_SPARSE
    return _COVERAGE_SUFFICIENT


def _safe_float(value: float) -> float | None:
    """Return ``None`` for NaN/Inf values, else the float."""

    try:
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _information_ratio(mean_ic: float, ic_std: float) -> float:
    """Compute IC information ratio (mean / std). Returns NaN when undefined."""

    if math.isnan(mean_ic) or math.isnan(ic_std):
        return float("nan")
    if ic_std == 0.0:
        return float("nan")
    return mean_ic / ic_std


def _build_metrics_frame(
    rows: list[dict[str, Any]],
    *,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    """Build a stable-column-ordered metrics DataFrame from row dicts."""

    if not rows:
        return pd.DataFrame(columns=list(columns))

    frame = pd.DataFrame(rows)
    # Ensure all expected columns are present, adding missing ones as None.
    for column in columns:
        if column not in frame.columns:
            frame[column] = None

    frame = frame[list(columns)].copy()
    frame = frame.sort_values("regime_label", kind="stable").reset_index(drop=True)
    frame.attrs = {}
    return frame


def _build_result_metadata(
    config: RegimeConditionalConfig,
    *,
    surface: str,
    dimension: str,
) -> dict[str, Any]:
    return {
        "surface": surface,
        "dimension": dimension,
        "taxonomy_version": config.taxonomy_version,
        "min_observations": config.min_observations,
        "periods_per_year": config.periods_per_year,
        "regime_prefix": config.regime_prefix,
        **config.metadata,
    }


__all__ = [
    "MIN_REGIME_OBSERVATIONS",
    "REGIME_LABEL_COLUMN",
    "ConditionalSurface",
    "RegimeConditionalConfig",
    "RegimeConditionalResult",
    "evaluate_all_dimensions",
    "evaluate_alpha_metrics_by_regime",
    "evaluate_portfolio_metrics_by_regime",
    "evaluate_strategy_metrics_by_regime",
    "resolve_regime_conditional_config",
]
