from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Literal

import pandas as pd

from src.research.metrics import TRADING_DAYS_PER_YEAR, max_drawdown, total_return, volatility, win_rate
from src.research.regimes.conditional import MIN_REGIME_OBSERVATIONS
from src.research.regimes.taxonomy import (
    DRAWDOWN_RECOVERY_DIMENSION,
    REGIME_DIMENSIONS,
    REGIME_STATE_COLUMNS,
    STRESS_DIMENSION,
    TAXONOMY_VERSION,
)
from src.research.regimes.validation import validate_regime_labels

TransitionSurface = Literal["strategy", "alpha", "portfolio"]

TRANSITION_EVENT_COLUMNS: tuple[str, ...] = (
    "event_id",
    "event_order",
    "dimension_event_order",
    "ts_utc",
    "transition_dimension",
    "previous_state",
    "current_state",
    "transition_label",
    "transition_category",
    "transition_direction",
    "is_stress_transition",
    "taxonomy_version",
)

TRANSITION_WINDOW_TAG_COLUMNS: tuple[str, ...] = (
    "transition_event_id",
    "transition_event_order",
    "transition_dimension_event_order",
    "transition_ts_utc",
    "transition_dimension",
    "transition_previous_state",
    "transition_current_state",
    "transition_label",
    "transition_category",
    "transition_direction",
    "transition_is_stress_transition",
    "transition_window_role",
    "transition_row_offset",
    "transition_timestamp_offset_seconds",
    "transition_overlap_count",
    "transition_has_window_overlap",
    "transition_is_valid_evidence",
)

_RETURN_SUMMARY_COLUMNS: tuple[str, ...] = (
    *TRANSITION_EVENT_COLUMNS,
    "surface",
    "pre_observation_count",
    "event_observation_count",
    "post_observation_count",
    "window_observation_count",
    "valid_observation_count",
    "coverage_status",
    "pre_transition_return",
    "event_window_return",
    "post_transition_return",
    "window_cumulative_return",
    "window_volatility",
    "window_max_drawdown",
    "window_win_rate",
)

_ALPHA_SUMMARY_COLUMNS: tuple[str, ...] = (
    *TRANSITION_EVENT_COLUMNS,
    "surface",
    "pre_observation_count",
    "event_observation_count",
    "post_observation_count",
    "window_observation_count",
    "valid_observation_count",
    "coverage_status",
    "pre_mean_ic",
    "event_mean_ic",
    "post_mean_ic",
    "window_mean_ic",
    "pre_mean_rank_ic",
    "event_mean_rank_ic",
    "post_mean_rank_ic",
    "window_mean_rank_ic",
)

_DIMENSION_ORDER = ("composite", *REGIME_DIMENSIONS)
_DEFINED_ALIGNMENT_STATUS = "matched_defined"


class RegimeTransitionError(ValueError):
    """Raised when deterministic transition analysis cannot be completed."""


@dataclass(frozen=True)
class RegimeTransitionConfig:
    pre_event_rows: int = 2
    post_event_rows: int = 2
    allow_window_overlap: bool = True
    timestamp_column: str = "ts_utc"
    regime_prefix: str = "regime_"
    min_observations: int = MIN_REGIME_OBSERVATIONS
    periods_per_year: int = TRADING_DAYS_PER_YEAR
    taxonomy_version: str = TAXONOMY_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.pre_event_rows < 0:
            raise RegimeTransitionError("pre_event_rows must be >= 0.")
        if self.post_event_rows < 0:
            raise RegimeTransitionError("post_event_rows must be >= 0.")
        if self.min_observations < 1:
            raise RegimeTransitionError("min_observations must be at least 1.")
        if self.periods_per_year < 1:
            raise RegimeTransitionError("periods_per_year must be at least 1.")
        if not self.timestamp_column:
            raise RegimeTransitionError("timestamp_column must be a non-empty string.")
        if not self.regime_prefix:
            raise RegimeTransitionError("regime_prefix must be a non-empty string.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "allow_window_overlap": self.allow_window_overlap,
            "metadata": dict(sorted(self.metadata.items())),
            "min_observations": self.min_observations,
            "periods_per_year": self.periods_per_year,
            "post_event_rows": self.post_event_rows,
            "pre_event_rows": self.pre_event_rows,
            "regime_prefix": self.regime_prefix,
            "taxonomy_version": self.taxonomy_version,
            "timestamp_column": self.timestamp_column,
        }


@dataclass(frozen=True)
class RegimeTransitionAnalysisResult:
    surface: TransitionSurface
    events: pd.DataFrame
    windows: pd.DataFrame
    event_summaries: pd.DataFrame
    config: RegimeTransitionConfig
    metadata: dict[str, Any] = field(default_factory=dict)


def resolve_regime_transition_config(
    config: RegimeTransitionConfig | dict[str, Any] | None = None,
) -> RegimeTransitionConfig:
    if config is None:
        return RegimeTransitionConfig()
    if isinstance(config, RegimeTransitionConfig):
        return config
    if not isinstance(config, dict):
        raise TypeError("RegimeTransitionConfig must be a dataclass instance or a dict.")
    allowed = set(RegimeTransitionConfig.__dataclass_fields__)
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise RegimeTransitionError(f"Unsupported RegimeTransitionConfig fields: {unknown}.")
    return RegimeTransitionConfig(**config)


def detect_regime_transitions(
    regime_labels: pd.DataFrame,
    *,
    dimensions: str | list[str] | tuple[str, ...] = "all",
) -> pd.DataFrame:
    labels = validate_regime_labels(regime_labels)
    resolved_dimensions = _resolve_transition_dimensions(dimensions)
    rows: list[dict[str, Any]] = []
    dimension_event_counts = {dimension: 0 for dimension in resolved_dimensions}

    for position in range(1, len(labels)):
        previous = labels.iloc[position - 1]
        current = labels.iloc[position]
        if not bool(previous["is_defined"]) or not bool(current["is_defined"]):
            continue
        for dimension in resolved_dimensions:
            previous_state = _state_for_dimension(previous, dimension)
            current_state = _state_for_dimension(current, dimension)
            if previous_state == current_state:
                continue
            dimension_event_counts[dimension] += 1
            rows.append(
                {
                    "event_id": f"{dimension}:{dimension_event_counts[dimension]:04d}",
                    "event_order": 0,
                    "dimension_event_order": dimension_event_counts[dimension],
                    "ts_utc": current["ts_utc"],
                    "transition_dimension": dimension,
                    "previous_state": previous_state,
                    "current_state": current_state,
                    "transition_label": f"{dimension}:{previous_state}->{current_state}",
                    "transition_category": classify_transition_category(
                        dimension=dimension,
                        previous_state=previous_state,
                        current_state=current_state,
                    ),
                    "transition_direction": classify_transition_direction(
                        dimension=dimension,
                        previous_state=previous_state,
                        current_state=current_state,
                    ),
                    "is_stress_transition": is_stress_transition(
                        dimension=dimension,
                        previous_state=previous_state,
                        current_state=current_state,
                    ),
                    "taxonomy_version": TAXONOMY_VERSION,
                }
            )

    events = pd.DataFrame(rows, columns=list(TRANSITION_EVENT_COLUMNS))
    if events.empty:
        return events
    events = events.sort_values(
        ["ts_utc", "transition_dimension", "dimension_event_order"],
        key=lambda series: _sort_key_for_events(series),
        kind="stable",
    ).reset_index(drop=True)
    events["event_order"] = range(1, len(events) + 1)
    return events.loc[:, list(TRANSITION_EVENT_COLUMNS)]


def tag_transition_windows(
    frame: pd.DataFrame,
    transition_events: pd.DataFrame,
    *,
    config: RegimeTransitionConfig | dict[str, Any] | None = None,
) -> pd.DataFrame:
    resolved_config = resolve_regime_transition_config(config)
    normalized = _normalize_timestamp_frame(frame, timestamp_column=resolved_config.timestamp_column)
    events = _normalize_transition_events(transition_events)
    if events.empty:
        return pd.DataFrame(columns=[*TRANSITION_WINDOW_TAG_COLUMNS, *normalized.columns])

    indexed = normalized.reset_index(drop=True).copy()
    indexed["_transition_row_position"] = range(len(indexed))
    rows: list[dict[str, Any]] = []

    for event in events.itertuples(index=False):
        event_positions = indexed.index[indexed[resolved_config.timestamp_column].eq(event.ts_utc)].tolist()
        if not event_positions:
            continue
        start = max(0, min(event_positions) - resolved_config.pre_event_rows)
        end = min(len(indexed) - 1, max(event_positions) + resolved_config.post_event_rows)
        for position in range(start, end + 1):
            row = indexed.iloc[position]
            row_offset = _row_offset(position, event_positions=event_positions)
            rows.append(
                {
                    "transition_event_id": event.event_id,
                    "transition_event_order": int(event.event_order),
                    "transition_dimension_event_order": int(event.dimension_event_order),
                    "transition_ts_utc": event.ts_utc,
                    "transition_dimension": event.transition_dimension,
                    "transition_previous_state": event.previous_state,
                    "transition_current_state": event.current_state,
                    "transition_label": event.transition_label,
                    "transition_category": event.transition_category,
                    "transition_direction": event.transition_direction,
                    "transition_is_stress_transition": bool(event.is_stress_transition),
                    "transition_window_role": _window_role_for_offset(row_offset),
                    "transition_row_offset": row_offset,
                    "transition_timestamp_offset_seconds": _timestamp_offset_seconds(
                        row[resolved_config.timestamp_column],
                        event.ts_utc,
                    ),
                    "transition_overlap_count": 0,
                    "transition_has_window_overlap": False,
                    "transition_is_valid_evidence": _row_is_valid_evidence(row, config=resolved_config),
                    "_transition_row_position": int(row["_transition_row_position"]),
                    **{column: row[column] for column in normalized.columns},
                }
            )

    windows = pd.DataFrame(rows)
    if windows.empty:
        return pd.DataFrame(columns=[*TRANSITION_WINDOW_TAG_COLUMNS, *normalized.columns])

    overlap_counts = windows.groupby("_transition_row_position", sort=False)["transition_event_id"].size().astype("int64")
    windows["transition_overlap_count"] = windows["_transition_row_position"].map(overlap_counts).fillna(0).astype("int64")
    windows["transition_has_window_overlap"] = windows["transition_overlap_count"].gt(1)

    if not resolved_config.allow_window_overlap and windows["transition_has_window_overlap"].any():
        raise RegimeTransitionError("Transition window overlap is not allowed for this configuration.")

    windows = windows.sort_values(["transition_event_order", "_transition_row_position"], kind="stable").reset_index(drop=True)
    windows = windows.drop(columns="_transition_row_position")
    return windows.loc[:, [*TRANSITION_WINDOW_TAG_COLUMNS, *normalized.columns]]


def analyze_strategy_regime_transitions(
    aligned_frame: pd.DataFrame,
    regime_labels: pd.DataFrame,
    *,
    return_column: str = "strategy_return",
    dimensions: str | list[str] | tuple[str, ...] = "all",
    config: RegimeTransitionConfig | dict[str, Any] | None = None,
) -> RegimeTransitionAnalysisResult:
    return _analyze_return_surface(
        aligned_frame,
        regime_labels,
        surface="strategy",
        value_column=return_column,
        dimensions=dimensions,
        config=config,
    )


def analyze_portfolio_regime_transitions(
    aligned_frame: pd.DataFrame,
    regime_labels: pd.DataFrame,
    *,
    return_column: str = "portfolio_return",
    dimensions: str | list[str] | tuple[str, ...] = "all",
    config: RegimeTransitionConfig | dict[str, Any] | None = None,
) -> RegimeTransitionAnalysisResult:
    return _analyze_return_surface(
        aligned_frame,
        regime_labels,
        surface="portfolio",
        value_column=return_column,
        dimensions=dimensions,
        config=config,
    )


def analyze_alpha_regime_transitions(
    aligned_frame: pd.DataFrame,
    regime_labels: pd.DataFrame,
    *,
    ic_column: str = "ic",
    rank_ic_column: str = "rank_ic",
    dimensions: str | list[str] | tuple[str, ...] = "all",
    config: RegimeTransitionConfig | dict[str, Any] | None = None,
) -> RegimeTransitionAnalysisResult:
    resolved_config = resolve_regime_transition_config(config)
    _require_column(aligned_frame, ic_column, context="alpha transition analysis")
    _require_column(aligned_frame, rank_ic_column, context="alpha transition analysis")
    events = detect_regime_transitions(regime_labels, dimensions=dimensions)
    windows = tag_transition_windows(aligned_frame, events, config=resolved_config)
    summaries = _summarize_alpha_windows(
        windows,
        ic_column=ic_column,
        rank_ic_column=rank_ic_column,
        config=resolved_config,
    )
    return RegimeTransitionAnalysisResult(
        surface="alpha",
        events=events,
        windows=windows,
        event_summaries=summaries,
        config=resolved_config,
        metadata=_build_analysis_metadata(surface="alpha", config=resolved_config),
    )


def classify_transition_category(
    *,
    dimension: str,
    previous_state: str,
    current_state: str,
) -> str:
    if dimension == "composite":
        previous_components = _parse_composite_label(previous_state)
        current_components = _parse_composite_label(current_state)
        for component_dimension in REGIME_DIMENSIONS:
            if previous_components.get(component_dimension) != current_components.get(component_dimension):
                category = classify_transition_category(
                    dimension=component_dimension,
                    previous_state=str(previous_components.get(component_dimension)),
                    current_state=str(current_components.get(component_dimension)),
                )
                if category != "generic_transition":
                    return category
        return "generic_transition"

    if dimension == "volatility":
        previous_rank = _ordered_rank(previous_state, ("low_volatility", "normal_volatility", "high_volatility"))
        current_rank = _ordered_rank(current_state, ("low_volatility", "normal_volatility", "high_volatility"))
        if current_rank > previous_rank:
            return "volatility_upshift"
        if current_rank < previous_rank:
            return "volatility_downshift"
        return "generic_transition"

    if dimension == "trend":
        if current_state == "downtrend" and previous_state != "downtrend":
            return "trend_breakdown"
        return "generic_transition"

    if dimension == DRAWDOWN_RECOVERY_DIMENSION:
        if current_state == "drawdown" and previous_state != "drawdown":
            return "drawdown_onset"
        if current_state == "recovery" and previous_state != "recovery":
            return "recovery_onset"
        return "generic_transition"

    if dimension == STRESS_DIMENSION:
        if previous_state == "normal_stress" and current_state != "normal_stress":
            return "stress_onset"
        if previous_state != "normal_stress" and current_state == "normal_stress":
            return "stress_relief"
        return "generic_transition"

    return "generic_transition"


def classify_transition_direction(
    *,
    dimension: str,
    previous_state: str,
    current_state: str,
) -> str:
    if dimension == "composite":
        return "state_change"
    if dimension == "volatility":
        previous_rank = _ordered_rank(previous_state, ("low_volatility", "normal_volatility", "high_volatility"))
        current_rank = _ordered_rank(current_state, ("low_volatility", "normal_volatility", "high_volatility"))
        if current_rank > previous_rank:
            return "up"
        if current_rank < previous_rank:
            return "down"
        return "flat"
    if dimension == STRESS_DIMENSION:
        if previous_state == "normal_stress" and current_state != "normal_stress":
            return "enter"
        if previous_state != "normal_stress" and current_state == "normal_stress":
            return "exit"
        return "state_change"
    if dimension == DRAWDOWN_RECOVERY_DIMENSION and current_state in {"drawdown", "recovery"}:
        if previous_state != current_state:
            return "enter"
    return "state_change"


def is_stress_transition(
    *,
    dimension: str,
    previous_state: str,
    current_state: str,
) -> bool:
    if dimension == STRESS_DIMENSION:
        return True
    if dimension != "composite":
        return False
    previous_components = _parse_composite_label(previous_state)
    current_components = _parse_composite_label(current_state)
    return previous_components.get(STRESS_DIMENSION) != current_components.get(STRESS_DIMENSION)


def _analyze_return_surface(
    aligned_frame: pd.DataFrame,
    regime_labels: pd.DataFrame,
    *,
    surface: Literal["strategy", "portfolio"],
    value_column: str,
    dimensions: str | list[str] | tuple[str, ...],
    config: RegimeTransitionConfig | dict[str, Any] | None,
) -> RegimeTransitionAnalysisResult:
    resolved_config = resolve_regime_transition_config(config)
    _require_column(aligned_frame, value_column, context=f"{surface} transition analysis")
    events = detect_regime_transitions(regime_labels, dimensions=dimensions)
    windows = tag_transition_windows(aligned_frame, events, config=resolved_config)
    summaries = _summarize_return_windows(
        windows,
        surface=surface,
        value_column=value_column,
        config=resolved_config,
    )
    return RegimeTransitionAnalysisResult(
        surface=surface,
        events=events,
        windows=windows,
        event_summaries=summaries,
        config=resolved_config,
        metadata=_build_analysis_metadata(surface=surface, config=resolved_config),
    )


def _normalize_timestamp_frame(frame: pd.DataFrame, *, timestamp_column: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Transition analysis frame must be a pandas DataFrame.")
    if timestamp_column not in frame.columns:
        raise RegimeTransitionError(
            f"Transition analysis frame must include timestamp column {timestamp_column!r}."
        )
    normalized = frame.copy(deep=True)
    normalized.attrs = {}
    normalized[timestamp_column] = pd.to_datetime(normalized[timestamp_column], utc=True, errors="coerce")
    if normalized[timestamp_column].isna().any():
        raise RegimeTransitionError(
            f"Transition analysis frame contains unparsable timestamps in {timestamp_column!r}."
        )
    normalized = normalized.sort_values(timestamp_column, kind="stable").reset_index(drop=True)
    return normalized


def _normalize_transition_events(transition_events: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(transition_events, pd.DataFrame):
        raise TypeError("transition_events must be a pandas DataFrame.")
    missing = [column for column in TRANSITION_EVENT_COLUMNS if column not in transition_events.columns]
    if missing:
        raise RegimeTransitionError(f"transition_events is missing required columns: {missing}.")
    normalized = transition_events.copy(deep=True)
    normalized.attrs = {}
    normalized["ts_utc"] = pd.to_datetime(normalized["ts_utc"], utc=True, errors="coerce")
    if normalized["ts_utc"].isna().any():
        raise RegimeTransitionError("transition_events contains unparsable ts_utc values.")
    return normalized.loc[:, list(TRANSITION_EVENT_COLUMNS)]


def _resolve_transition_dimensions(dimensions: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if dimensions == "all":
        return _DIMENSION_ORDER
    if isinstance(dimensions, str):
        resolved = (dimensions,)
    elif isinstance(dimensions, list | tuple):
        resolved = tuple(dimensions)
    else:
        raise TypeError("dimensions must be 'all', a string, or a list/tuple of strings.")
    unknown = sorted(set(resolved) - set(_DIMENSION_ORDER))
    if unknown:
        raise RegimeTransitionError(
            f"Unknown transition dimensions: {unknown}. Expected members of {_DIMENSION_ORDER}."
        )
    return tuple(resolved)


def _state_for_dimension(row: pd.Series, dimension: str) -> str:
    if dimension == "composite":
        return str(row["regime_label"])
    return str(row[REGIME_STATE_COLUMNS[dimension]])


def _ordered_rank(value: str, ordered_states: tuple[str, ...]) -> int:
    try:
        return ordered_states.index(value)
    except ValueError:
        return -1


def _parse_composite_label(label: str) -> dict[str, str]:
    parts: dict[str, str] = {}
    for component in str(label).split("|"):
        if "=" not in component:
            continue
        key, value = component.split("=", 1)
        parts[key] = value
    return parts


def _sort_key_for_events(series: pd.Series) -> pd.Series:
    if series.name == "transition_dimension":
        mapping = {dimension: order for order, dimension in enumerate(_DIMENSION_ORDER)}
        return series.map(mapping).fillna(len(_DIMENSION_ORDER))
    return series


def _row_offset(position: int, *, event_positions: list[int]) -> int:
    first_position = min(event_positions)
    last_position = max(event_positions)
    if first_position <= position <= last_position:
        return 0
    if position < first_position:
        return position - first_position
    return position - last_position


def _window_role_for_offset(offset: int) -> str:
    if offset < 0:
        return "pre_window"
    if offset > 0:
        return "post_window"
    return "event_timestamp"


def _timestamp_offset_seconds(ts_utc: pd.Timestamp, event_ts_utc: pd.Timestamp) -> float:
    return float((ts_utc - event_ts_utc).total_seconds())


def _row_is_valid_evidence(row: pd.Series, *, config: RegimeTransitionConfig) -> bool:
    alignment_column = f"{config.regime_prefix}alignment_status"
    if alignment_column not in row.index:
        return True
    return str(row[alignment_column]) == _DEFINED_ALIGNMENT_STATUS


def _require_column(frame: pd.DataFrame, column: str, *, context: str) -> None:
    if column not in frame.columns:
        raise RegimeTransitionError(
            f"Column {column!r} required for {context} is missing from the frame."
        )


def _summarize_return_windows(
    windows: pd.DataFrame,
    *,
    surface: Literal["strategy", "portfolio"],
    value_column: str,
    config: RegimeTransitionConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _event_id, group in _iter_event_windows(windows):
        valid = group.loc[group["transition_is_valid_evidence"]].copy()
        returns = pd.to_numeric(valid[value_column], errors="coerce").dropna().astype("float64")
        pre_returns = pd.to_numeric(valid.loc[valid["transition_window_role"] == "pre_window", value_column], errors="coerce").dropna()
        event_returns = pd.to_numeric(valid.loc[valid["transition_window_role"] == "event_timestamp", value_column], errors="coerce").dropna()
        post_returns = pd.to_numeric(valid.loc[valid["transition_window_role"] == "post_window", value_column], errors="coerce").dropna()
        coverage = _coverage_status(
            valid_count=int(len(returns)),
            min_observations=config.min_observations,
            required_sections=(pre_returns, event_returns, post_returns),
        )
        rows.append(
            {
                **_event_metadata_from_window_group(group),
                "surface": surface,
                "pre_observation_count": int(len(pre_returns)),
                "event_observation_count": int(len(event_returns)),
                "post_observation_count": int(len(post_returns)),
                "window_observation_count": int(len(group)),
                "valid_observation_count": int(len(returns)),
                "coverage_status": coverage,
                "pre_transition_return": _safe_metric(total_return(pre_returns)) if len(pre_returns) else None,
                "event_window_return": _safe_metric(total_return(event_returns)) if len(event_returns) else None,
                "post_transition_return": _safe_metric(total_return(post_returns)) if len(post_returns) else None,
                "window_cumulative_return": _safe_metric(total_return(returns)) if len(returns) else None,
                "window_volatility": _safe_metric(volatility(returns)) if len(returns) else None,
                "window_max_drawdown": _safe_metric(max_drawdown(returns)) if len(returns) else None,
                "window_win_rate": _safe_metric(win_rate(returns)) if len(returns) else None,
            }
        )
    return _build_summary_frame(rows, columns=_RETURN_SUMMARY_COLUMNS)


def _summarize_alpha_windows(
    windows: pd.DataFrame,
    *,
    ic_column: str,
    rank_ic_column: str,
    config: RegimeTransitionConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _event_id, group in _iter_event_windows(windows):
        valid = group.loc[group["transition_is_valid_evidence"]].copy()
        pre_ic = pd.to_numeric(valid.loc[valid["transition_window_role"] == "pre_window", ic_column], errors="coerce").dropna()
        event_ic = pd.to_numeric(valid.loc[valid["transition_window_role"] == "event_timestamp", ic_column], errors="coerce").dropna()
        post_ic = pd.to_numeric(valid.loc[valid["transition_window_role"] == "post_window", ic_column], errors="coerce").dropna()
        all_ic = pd.to_numeric(valid[ic_column], errors="coerce").dropna()
        pre_rank_ic = pd.to_numeric(valid.loc[valid["transition_window_role"] == "pre_window", rank_ic_column], errors="coerce").dropna()
        event_rank_ic = pd.to_numeric(valid.loc[valid["transition_window_role"] == "event_timestamp", rank_ic_column], errors="coerce").dropna()
        post_rank_ic = pd.to_numeric(valid.loc[valid["transition_window_role"] == "post_window", rank_ic_column], errors="coerce").dropna()
        all_rank_ic = pd.to_numeric(valid[rank_ic_column], errors="coerce").dropna()
        coverage = _coverage_status(
            valid_count=int(len(all_ic)),
            min_observations=config.min_observations,
            required_sections=(pre_ic, event_ic, post_ic),
        )
        rows.append(
            {
                **_event_metadata_from_window_group(group),
                "surface": "alpha",
                "pre_observation_count": int(len(pre_ic)),
                "event_observation_count": int(len(event_ic)),
                "post_observation_count": int(len(post_ic)),
                "window_observation_count": int(len(group)),
                "valid_observation_count": int(len(all_ic)),
                "coverage_status": coverage,
                "pre_mean_ic": _safe_metric(pre_ic.mean()) if len(pre_ic) else None,
                "event_mean_ic": _safe_metric(event_ic.mean()) if len(event_ic) else None,
                "post_mean_ic": _safe_metric(post_ic.mean()) if len(post_ic) else None,
                "window_mean_ic": _safe_metric(all_ic.mean()) if len(all_ic) else None,
                "pre_mean_rank_ic": _safe_metric(pre_rank_ic.mean()) if len(pre_rank_ic) else None,
                "event_mean_rank_ic": _safe_metric(event_rank_ic.mean()) if len(event_rank_ic) else None,
                "post_mean_rank_ic": _safe_metric(post_rank_ic.mean()) if len(post_rank_ic) else None,
                "window_mean_rank_ic": _safe_metric(all_rank_ic.mean()) if len(all_rank_ic) else None,
            }
        )
    return _build_summary_frame(rows, columns=_ALPHA_SUMMARY_COLUMNS)


def _iter_event_windows(windows: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    if windows.empty:
        return []
    groups: list[tuple[str, pd.DataFrame]] = []
    for event_id in windows["transition_event_id"].drop_duplicates().tolist():
        groups.append((str(event_id), windows.loc[windows["transition_event_id"] == event_id].copy()))
    return groups


def _event_metadata_from_window_group(group: pd.DataFrame) -> dict[str, Any]:
    row = group.iloc[0]
    return {
        "event_id": row["transition_event_id"],
        "event_order": int(row["transition_event_order"]),
        "dimension_event_order": int(row["transition_dimension_event_order"]),
        "ts_utc": row["transition_ts_utc"],
        "transition_dimension": row["transition_dimension"],
        "previous_state": row["transition_previous_state"],
        "current_state": row["transition_current_state"],
        "transition_label": row["transition_label"],
        "transition_category": row["transition_category"],
        "transition_direction": row["transition_direction"],
        "is_stress_transition": bool(row["transition_is_stress_transition"]),
        "taxonomy_version": TAXONOMY_VERSION,
    }


def _coverage_status(
    *,
    valid_count: int,
    min_observations: int,
    required_sections: tuple[pd.Series, pd.Series, pd.Series],
) -> str:
    if valid_count == 0:
        return "empty"
    if any(len(section) == 0 for section in required_sections):
        return "sparse"
    if valid_count < min_observations:
        return "sparse"
    return "sufficient"


def _safe_metric(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _build_summary_frame(rows: list[dict[str, Any]], *, columns: tuple[str, ...]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=list(columns))
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
    frame = frame.loc[:, list(columns)].copy()
    frame = frame.sort_values(["event_order", "transition_dimension"], kind="stable").reset_index(drop=True)
    frame.attrs = {}
    return frame


def _build_analysis_metadata(*, surface: str, config: RegimeTransitionConfig) -> dict[str, Any]:
    return {
        "surface": surface,
        "taxonomy_version": config.taxonomy_version,
        "window": {
            "pre_event_rows": config.pre_event_rows,
            "post_event_rows": config.post_event_rows,
            "allow_window_overlap": config.allow_window_overlap,
        },
        "min_observations": config.min_observations,
        "regime_prefix": config.regime_prefix,
        **config.metadata,
    }


__all__ = [
    "TRANSITION_EVENT_COLUMNS",
    "TRANSITION_WINDOW_TAG_COLUMNS",
    "RegimeTransitionAnalysisResult",
    "RegimeTransitionConfig",
    "RegimeTransitionError",
    "TransitionSurface",
    "analyze_alpha_regime_transitions",
    "analyze_portfolio_regime_transitions",
    "analyze_strategy_regime_transitions",
    "classify_transition_category",
    "classify_transition_direction",
    "detect_regime_transitions",
    "is_stress_transition",
    "resolve_regime_transition_config",
    "tag_transition_windows",
]
