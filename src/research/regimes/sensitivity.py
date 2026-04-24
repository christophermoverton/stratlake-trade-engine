from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import pandas as pd

from src.research.metrics import total_return, volatility
from src.research.regimes.alignment import (
    align_regimes_to_alpha_windows,
    align_regimes_to_portfolio_windows,
    align_regimes_to_strategy_timeseries,
)
from src.research.regimes.attribution import summarize_regime_attribution
from src.research.regimes.calibration import (
    DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    RegimeCalibrationProfile,
    RegimeCalibrationResult,
    apply_regime_calibration,
    resolve_regime_calibration_profile,
)
from src.research.regimes.conditional import (
    RegimeConditionalResult,
    evaluate_alpha_metrics_by_regime,
    evaluate_portfolio_metrics_by_regime,
    evaluate_strategy_metrics_by_regime,
)
from src.research.regimes.taxonomy import TAXONOMY_VERSION
from src.research.regimes.validation import validate_regime_labels
from src.research.registry import canonicalize_value

SensitivitySurface = Literal["strategy", "alpha", "portfolio"]

_MATRIX_COLUMNS: tuple[str, ...] = (
    "profile_name",
    "total_observations",
    "regime_count",
    "transition_count",
    "flip_rate",
    "single_day_flip_count",
    "single_day_flip_share",
    "average_regime_duration",
    "median_regime_duration",
    "minimum_regime_duration",
    "maximum_regime_duration",
    "unstable_regime_share",
    "low_confidence_share",
    "attribution_eligible_regime_count",
    "attribution_ineligible_regime_count",
    "is_unstable_profile",
    "exceeds_max_flip_rate",
    "exceeds_max_single_day_flip_share",
    "exceeds_low_confidence_share",
    "has_unstable_runs",
    "warning_count",
    "fallback_rows_total",
    "unknown_fallback_rows",
    "low_confidence_fallback_rows",
    "unstable_profile_fallback_rows",
    "dominant_regime_label",
    "dominant_regime_share",
    "defined_observation_share",
    "stable_profile_rank",
    "stability_score",
    "eligible_for_downstream_decisioning",
    "is_recommended_profile",
)


class RegimeSensitivityError(ValueError):
    """Raised when regime calibration sensitivity analysis cannot be completed."""


@dataclass(frozen=True)
class RegimeCalibrationSensitivityResult:
    """Structured multi-profile calibration sensitivity result."""

    matrix: pd.DataFrame
    profile_results: dict[str, RegimeCalibrationResult]
    profile_order: tuple[str, ...]
    summary: dict[str, Any]
    performance_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    profile_performance: dict[str, pd.DataFrame] = field(default_factory=dict)
    profile_attribution: dict[str, RegimeConditionalResult] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    taxonomy_version: str = TAXONOMY_VERSION


def run_regime_calibration_sensitivity(
    regime_labels: pd.DataFrame,
    *,
    profiles: Sequence[RegimeCalibrationProfile | Mapping[str, Any] | str] | None = None,
    confidence_column: str | None = None,
    low_confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    performance_frame: pd.DataFrame | None = None,
    performance_surface: SensitivitySurface | None = None,
    performance_timestamp_column: str = "ts_utc",
    performance_value_column: str | None = None,
    rank_ic_column: str = "rank_ic",
    source_artifact_references: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RegimeCalibrationSensitivityResult:
    if not isinstance(regime_labels, pd.DataFrame):
        raise TypeError("regime_labels must be a pandas DataFrame.")
    input_labels = regime_labels.copy(deep=True)
    input_labels.attrs = {}
    validate_regime_labels(input_labels)
    resolved_profiles = _resolve_profiles(profiles)

    profile_results: dict[str, RegimeCalibrationResult] = {}
    matrix_rows: list[dict[str, Any]] = []
    profile_order: list[str] = []

    for position, profile in enumerate(resolved_profiles):
        calibration = apply_regime_calibration(
            input_labels,
            profile=profile,
            confidence_column=confidence_column,
            low_confidence_threshold=low_confidence_threshold,
            metadata={"sensitivity_input_order": position},
        )
        profile_results[profile.name] = calibration
        profile_order.append(profile.name)
        matrix_rows.append(_matrix_row_for_result(calibration))

    matrix = build_regime_sensitivity_matrix(matrix_rows, profile_order=profile_order)
    matrix = rank_regime_calibration_profiles(matrix, profile_order=profile_order)

    profile_performance: dict[str, pd.DataFrame] = {}
    profile_attribution: dict[str, RegimeConditionalResult] = {}
    performance_summary = pd.DataFrame()
    if performance_frame is not None:
        (
            performance_summary,
            profile_performance,
            profile_attribution,
        ) = _build_performance_outputs(
            performance_frame=performance_frame,
            calibrated_results=profile_results,
            profile_order=profile_order,
            surface=performance_surface,
            timestamp_column=performance_timestamp_column,
            value_column=performance_value_column,
            rank_ic_column=rank_ic_column,
        )

    summary = summarize_regime_sensitivity(
        matrix,
        profile_results=profile_results,
        performance_summary=performance_summary,
        profile_order=profile_order,
        source_artifact_references=source_artifact_references,
        metadata=metadata,
    )
    return RegimeCalibrationSensitivityResult(
        matrix=matrix,
        profile_results=profile_results,
        profile_order=tuple(profile_order),
        summary=summary,
        performance_summary=performance_summary,
        profile_performance=profile_performance,
        profile_attribution=profile_attribution,
        metadata=canonicalize_value(dict(metadata or {})),
    )


def build_regime_sensitivity_matrix(
    rows: Sequence[Mapping[str, Any]] | pd.DataFrame,
    *,
    profile_order: Sequence[str] | None = None,
) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        frame = rows.copy(deep=True)
    else:
        frame = pd.DataFrame([dict(row) for row in rows])
    if frame.empty:
        return pd.DataFrame(columns=list(_MATRIX_COLUMNS))

    for column in _MATRIX_COLUMNS:
        if column not in frame.columns:
            frame[column] = None

    frame = frame.loc[:, list(_MATRIX_COLUMNS)].copy()
    frame.attrs = {}
    if profile_order is not None:
        order_lookup = {name: index for index, name in enumerate(profile_order)}
        frame["_profile_order"] = frame["profile_name"].map(order_lookup).fillna(len(order_lookup)).astype("int64")
        frame = frame.sort_values(["_profile_order", "profile_name"], kind="stable").drop(columns="_profile_order")
        frame = frame.reset_index(drop=True)
    return frame


def rank_regime_calibration_profiles(
    matrix: pd.DataFrame,
    *,
    profile_order: Sequence[str] | None = None,
) -> pd.DataFrame:
    ranked = build_regime_sensitivity_matrix(matrix)
    if ranked.empty:
        return ranked

    order_lookup = {name: index for index, name in enumerate(profile_order or ranked["profile_name"].tolist())}
    ranked["defined_observation_share"] = pd.to_numeric(ranked["defined_observation_share"], errors="coerce").fillna(0.0)
    ranked["attribution_eligible_regime_count"] = (
        pd.to_numeric(ranked["attribution_eligible_regime_count"], errors="coerce").fillna(0).astype("int64")
    )
    ranked["regime_count"] = pd.to_numeric(ranked["regime_count"], errors="coerce").fillna(0).astype("int64")
    ranked["warning_count"] = pd.to_numeric(ranked["warning_count"], errors="coerce").fillna(0).astype("int64")
    ranked["flip_rate"] = pd.to_numeric(ranked["flip_rate"], errors="coerce").fillna(0.0)
    ranked["single_day_flip_share"] = pd.to_numeric(ranked["single_day_flip_share"], errors="coerce").fillna(0.0)
    ranked["unstable_regime_share"] = pd.to_numeric(ranked["unstable_regime_share"], errors="coerce").fillna(0.0)
    ranked["is_unstable_profile"] = ranked["is_unstable_profile"].astype("bool")

    ranked["eligible_for_downstream_decisioning"] = (
        ~ranked["is_unstable_profile"]
        & ranked["defined_observation_share"].gt(0.0)
        & ranked["attribution_eligible_regime_count"].gt(0)
    )
    ranked["stability_score"] = ranked.apply(_stability_score, axis=1).round(6)

    sortable = ranked.copy(deep=True)
    sortable["_profile_order"] = sortable["profile_name"].map(order_lookup).fillna(len(order_lookup)).astype("int64")
    sortable = sortable.sort_values(
        [
            "is_unstable_profile",
            "flip_rate",
            "single_day_flip_share",
            "unstable_regime_share",
            "attribution_eligible_regime_count",
            "defined_observation_share",
            "warning_count",
            "_profile_order",
            "profile_name",
        ],
        ascending=[True, True, True, True, False, False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)

    rank_lookup = {str(row["profile_name"]): index + 1 for index, row in sortable.iterrows()}
    recommended_profile = None
    eligible = sortable.loc[sortable["eligible_for_downstream_decisioning"]]
    if not eligible.empty:
        recommended_profile = str(eligible.iloc[0]["profile_name"])

    ranked["stable_profile_rank"] = ranked["profile_name"].map(rank_lookup).astype("int64")
    ranked["is_recommended_profile"] = ranked["profile_name"].eq(recommended_profile)
    return build_regime_sensitivity_matrix(ranked, profile_order=profile_order)


def summarize_regime_sensitivity(
    matrix: pd.DataFrame,
    *,
    profile_results: Mapping[str, RegimeCalibrationResult] | None = None,
    performance_summary: pd.DataFrame | None = None,
    profile_order: Sequence[str] | None = None,
    source_artifact_references: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = build_regime_sensitivity_matrix(matrix, profile_order=profile_order)
    recommended_rows = normalized.loc[normalized["is_recommended_profile"].astype("bool")]
    recommended_profile = None if recommended_rows.empty else str(recommended_rows.iloc[0]["profile_name"])

    unstable_profiles = (
        normalized.loc[normalized["is_unstable_profile"].astype("bool"), "profile_name"].astype("string").tolist()
    )
    rankings = (
        normalized.sort_values("stable_profile_rank", kind="stable")
        .loc[
            :,
            [
                "profile_name",
                "stable_profile_rank",
                "stability_score",
                "is_unstable_profile",
                "flip_rate",
                "single_day_flip_share",
                "unstable_regime_share",
                "defined_observation_share",
                "warning_count",
                "eligible_for_downstream_decisioning",
            ],
        ]
        .astype("object")
        .where(pd.notna(normalized.sort_values("stable_profile_rank", kind="stable").loc[
            :,
            [
                "profile_name",
                "stable_profile_rank",
                "stability_score",
                "is_unstable_profile",
                "flip_rate",
                "single_day_flip_share",
                "unstable_regime_share",
                "defined_observation_share",
                "warning_count",
                "eligible_for_downstream_decisioning",
            ],
        ]), None)
        .to_dict(orient="records")
    )

    payload: dict[str, Any] = {
        "artifact_type": "regime_sensitivity_summary",
        "schema_version": 1,
        "taxonomy_version": TAXONOMY_VERSION,
        "profile_count": int(len(normalized)),
        "profile_names": list(normalized["profile_name"].astype("string")),
        "profile_order": list(profile_order or normalized["profile_name"].astype("string")),
        "recommended_profile": recommended_profile,
        "unstable_profiles": unstable_profiles,
        "rankings": canonicalize_value(rankings),
        "matrix_columns": list(normalized.columns),
        "source_regime_artifact_references": canonicalize_value(dict(source_artifact_references or {})),
        "metadata": canonicalize_value(dict(metadata or {})),
    }
    if profile_results is not None:
        payload["warning_counts_by_profile"] = {
            name: int(len(result.warnings))
            for name, result in sorted(profile_results.items())
        }
    if performance_summary is not None and not performance_summary.empty:
        payload["performance_summary"] = canonicalize_value(
            performance_summary.astype("object").where(pd.notna(performance_summary), None).to_dict(orient="records")
        )
    return canonicalize_value(payload)


def _resolve_profiles(
    profiles: Sequence[RegimeCalibrationProfile | Mapping[str, Any] | str] | None,
) -> list[RegimeCalibrationProfile]:
    requested = profiles or ("baseline", "conservative", "reactive", "crisis_sensitive")
    resolved = [resolve_regime_calibration_profile(profile) for profile in requested]
    names = [profile.name for profile in resolved]
    duplicates = sorted(name for name in set(names) if names.count(name) > 1)
    if duplicates:
        raise RegimeSensitivityError(f"Profile names must be unique for sensitivity analysis. Duplicates: {duplicates}.")
    return resolved


def _matrix_row_for_result(result: RegimeCalibrationResult) -> dict[str, Any]:
    labels = result.labels.copy(deep=True)
    counts = labels["regime_label"].astype("string").value_counts(sort=False)
    if counts.empty:
        dominant_regime_label = None
        dominant_regime_share = 0.0
    else:
        dominant_regime_label = sorted(counts.index.astype(str).tolist(), key=lambda name: (-int(counts.loc[name]), name))[0]
        dominant_regime_share = float(int(counts.loc[dominant_regime_label]) / max(len(labels), 1))

    fallback_rows_total = int(
        result.fallback_summary.get("unknown_fallback_rows", 0)
        + result.fallback_summary.get("low_confidence_fallback_rows", 0)
        + result.fallback_summary.get("unstable_profile_fallback_rows", 0)
    )
    metrics = dict(result.stability_metrics)
    flags = dict(result.profile_flags)
    return canonicalize_value(
        {
            "profile_name": result.profile.name,
            "total_observations": int(metrics.get("total_observations", len(labels))),
            "regime_count": int(metrics.get("regime_count", 0)),
            "transition_count": int(metrics.get("transition_count", 0)),
            "flip_rate": float(metrics.get("flip_rate", 0.0)),
            "single_day_flip_count": int(metrics.get("single_day_flip_count", 0)),
            "single_day_flip_share": float(metrics.get("single_day_flip_share", 0.0)),
            "average_regime_duration": float(metrics.get("average_regime_duration", 0.0)),
            "median_regime_duration": float(metrics.get("median_regime_duration", 0.0)),
            "minimum_regime_duration": int(metrics.get("minimum_regime_duration", 0)),
            "maximum_regime_duration": int(metrics.get("maximum_regime_duration", 0)),
            "unstable_regime_share": float(metrics.get("unstable_regime_share", 0.0)),
            "low_confidence_share": float(metrics.get("low_confidence_share", 0.0)),
            "attribution_eligible_regime_count": int(metrics.get("attribution_eligible_regime_count", 0)),
            "attribution_ineligible_regime_count": int(metrics.get("attribution_ineligible_regime_count", 0)),
            "is_unstable_profile": bool(flags.get("is_unstable_profile", False)),
            "exceeds_max_flip_rate": bool(flags.get("exceeds_max_flip_rate", False)),
            "exceeds_max_single_day_flip_share": bool(flags.get("exceeds_max_single_day_flip_share", False)),
            "exceeds_low_confidence_share": bool(flags.get("exceeds_low_confidence_share", False)),
            "has_unstable_runs": bool(flags.get("has_unstable_runs", False)),
            "warning_count": int(len(result.warnings)),
            "fallback_rows_total": fallback_rows_total,
            "unknown_fallback_rows": int(result.fallback_summary.get("unknown_fallback_rows", 0)),
            "low_confidence_fallback_rows": int(result.fallback_summary.get("low_confidence_fallback_rows", 0)),
            "unstable_profile_fallback_rows": int(result.fallback_summary.get("unstable_profile_fallback_rows", 0)),
            "dominant_regime_label": dominant_regime_label,
            "dominant_regime_share": dominant_regime_share,
            "defined_observation_share": float(labels["is_defined"].astype("float64").mean()) if len(labels) else 0.0,
            "stable_profile_rank": None,
            "stability_score": None,
            "eligible_for_downstream_decisioning": None,
            "is_recommended_profile": False,
        }
    )


def _stability_score(row: pd.Series) -> float:
    regime_count = max(int(row.get("regime_count", 0) or 0), 1)
    eligible_ratio = float(row.get("attribution_eligible_regime_count", 0) or 0) / regime_count
    score = 100.0
    score -= 45.0 * float(row.get("flip_rate", 0.0) or 0.0)
    score -= 25.0 * float(row.get("single_day_flip_share", 0.0) or 0.0)
    score -= 20.0 * float(row.get("unstable_regime_share", 0.0) or 0.0)
    score -= 10.0 * (1.0 - float(row.get("defined_observation_share", 0.0) or 0.0))
    score -= 3.0 * float(row.get("warning_count", 0.0) or 0.0)
    score += 8.0 * eligible_ratio
    if bool(row.get("is_unstable_profile", False)):
        score -= 15.0
    return max(score, 0.0)


def _build_performance_outputs(
    *,
    performance_frame: pd.DataFrame,
    calibrated_results: Mapping[str, RegimeCalibrationResult],
    profile_order: Sequence[str],
    surface: SensitivitySurface | None,
    timestamp_column: str,
    value_column: str | None,
    rank_ic_column: str,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, RegimeConditionalResult]]:
    normalized_frame = _normalize_performance_frame(performance_frame, timestamp_column=timestamp_column)
    resolved_surface, resolved_value_column = _resolve_performance_surface_and_column(
        normalized_frame,
        surface=surface,
        value_column=value_column,
        rank_ic_column=rank_ic_column,
    )

    profile_tables: dict[str, pd.DataFrame] = {}
    profile_attribution: dict[str, RegimeConditionalResult] = {}
    summary_rows: list[dict[str, Any]] = []

    for profile_name in profile_order:
        calibration = calibrated_results[profile_name]
        aligned = _align_performance_frame(
            normalized_frame,
            calibration.labels,
            surface=resolved_surface,
            timestamp_column=timestamp_column,
        )
        conditional = _evaluate_profile_performance(
            aligned,
            surface=resolved_surface,
            value_column=resolved_value_column,
            rank_ic_column=rank_ic_column,
        )
        profile_attribution[profile_name] = conditional
        detail, summary = _summarize_profile_performance(
            conditional,
            aligned,
            surface=resolved_surface,
            value_column=resolved_value_column,
            rank_ic_column=rank_ic_column,
            profile_name=profile_name,
        )
        profile_tables[profile_name] = detail
        summary_rows.append(summary)

    summary_frame = pd.DataFrame(summary_rows)
    if not summary_frame.empty:
        summary_frame = summary_frame.sort_values(["profile_name"], kind="stable").reset_index(drop=True)
        summary_frame.attrs = {}
    return summary_frame, profile_tables, profile_attribution


def _normalize_performance_frame(frame: pd.DataFrame, *, timestamp_column: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("performance_frame must be a pandas DataFrame.")
    if timestamp_column not in frame.columns:
        raise RegimeSensitivityError(
            f"performance_frame must include timestamp column {timestamp_column!r}."
        )
    normalized = frame.copy(deep=True)
    normalized.attrs = {}
    normalized[timestamp_column] = pd.to_datetime(normalized[timestamp_column], utc=True, errors="coerce")
    if normalized[timestamp_column].isna().any():
        raise RegimeSensitivityError(
            f"performance_frame contains unparsable timestamps in {timestamp_column!r}."
        )
    return normalized


def _resolve_performance_surface_and_column(
    frame: pd.DataFrame,
    *,
    surface: SensitivitySurface | None,
    value_column: str | None,
    rank_ic_column: str,
) -> tuple[SensitivitySurface, str]:
    if surface is None:
        if value_column is not None:
            if value_column == "portfolio_return":
                surface = "portfolio"
            elif value_column == "ic":
                surface = "alpha"
            else:
                surface = "strategy"
        elif "strategy_return" in frame.columns:
            surface = "strategy"
            value_column = "strategy_return"
        elif "portfolio_return" in frame.columns:
            surface = "portfolio"
            value_column = "portfolio_return"
        elif "ic" in frame.columns and rank_ic_column in frame.columns:
            surface = "alpha"
            value_column = "ic"
        else:
            raise RegimeSensitivityError(
                "Could not infer performance_surface/performance_value_column from performance_frame."
            )
    if value_column is None:
        value_column = {
            "strategy": "strategy_return",
            "portfolio": "portfolio_return",
            "alpha": "ic",
        }[surface]
    if value_column not in frame.columns:
        raise RegimeSensitivityError(
            f"performance_frame is missing value column {value_column!r} for surface {surface!r}."
        )
    if surface == "alpha" and rank_ic_column not in frame.columns:
        raise RegimeSensitivityError(
            f"performance_frame is missing rank IC column {rank_ic_column!r} for alpha sensitivity analysis."
        )
    return surface, value_column


def _align_performance_frame(
    frame: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    surface: SensitivitySurface,
    timestamp_column: str,
) -> pd.DataFrame:
    if surface == "strategy":
        return align_regimes_to_strategy_timeseries(frame, labels, timestamp_column=timestamp_column)
    if surface == "portfolio":
        return align_regimes_to_portfolio_windows(frame, labels, timestamp_column=timestamp_column)
    return align_regimes_to_alpha_windows(frame, labels, timestamp_column=timestamp_column)


def _evaluate_profile_performance(
    aligned: pd.DataFrame,
    *,
    surface: SensitivitySurface,
    value_column: str,
    rank_ic_column: str,
) -> RegimeConditionalResult:
    if surface == "strategy":
        return evaluate_strategy_metrics_by_regime(aligned, return_column=value_column)
    if surface == "portfolio":
        return evaluate_portfolio_metrics_by_regime(aligned, return_column=value_column)
    return evaluate_alpha_metrics_by_regime(aligned, ic_column=value_column, rank_ic_column=rank_ic_column)


def _summarize_profile_performance(
    conditional: RegimeConditionalResult,
    aligned: pd.DataFrame,
    *,
    surface: SensitivitySurface,
    value_column: str,
    rank_ic_column: str,
    profile_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    label_column = "regime_label"
    if surface == "alpha":
        detail = conditional.metrics_by_regime.copy(deep=True)
        detail.attrs = {}
        attribution = summarize_regime_attribution(conditional)
        summary = {
            "profile_name": profile_name,
            "surface": surface,
            "best_regime_by_mean_return": attribution.summary["best_regime"]["regime_label"],
            "worst_regime_by_mean_return": attribution.summary["worst_regime"]["regime_label"],
            "profile_mean_return": None,
            "profile_volatility": None,
            "profile_cumulative_return": None,
            "best_regime_primary_metric": attribution.summary["best_regime"]["metric_value"],
            "worst_regime_primary_metric": attribution.summary["worst_regime"]["metric_value"],
        }
        return detail, canonicalize_value(summary)

    defined = aligned.loc[aligned["regime_alignment_status"] == "matched_defined", [label_column, value_column]].copy()
    rows: list[dict[str, Any]] = []
    for regime_label in sorted(defined[label_column].dropna().astype("string").unique().tolist()):
        series = pd.to_numeric(
            defined.loc[defined[label_column].eq(regime_label), value_column],
            errors="coerce",
        ).dropna()
        rows.append(
            {
                "profile_name": profile_name,
                "regime_label": str(regime_label),
                "observation_count": int(len(series)),
                "mean_return": float(series.mean()) if len(series) else None,
                "volatility": float(volatility(series)) if len(series) else None,
                "cumulative_return": float(total_return(series)) if len(series) else None,
                "min_return": float(series.min()) if len(series) else None,
                "max_return": float(series.max()) if len(series) else None,
            }
        )
    detail = pd.DataFrame(rows)
    if detail.empty:
        detail = pd.DataFrame(
            columns=[
                "profile_name",
                "regime_label",
                "observation_count",
                "mean_return",
                "volatility",
                "cumulative_return",
                "min_return",
                "max_return",
            ]
        )
    else:
        detail = detail.sort_values(["regime_label"], kind="stable").reset_index(drop=True)
    detail.attrs = {}

    profile_series = pd.to_numeric(defined[value_column], errors="coerce").dropna()
    if detail.empty:
        best_regime = None
        worst_regime = None
    else:
        ordered = detail.sort_values(["mean_return", "regime_label"], ascending=[False, True], kind="stable")
        best_regime = str(ordered.iloc[0]["regime_label"])
        worst_regime = str(ordered.iloc[-1]["regime_label"])

    summary = {
        "profile_name": profile_name,
        "surface": surface,
        "best_regime_by_mean_return": best_regime,
        "worst_regime_by_mean_return": worst_regime,
        "profile_mean_return": float(profile_series.mean()) if len(profile_series) else None,
        "profile_volatility": float(volatility(profile_series)) if len(profile_series) else None,
        "profile_cumulative_return": float(total_return(profile_series)) if len(profile_series) else None,
        "best_regime_primary_metric": None if best_regime is None else float(detail.loc[detail["regime_label"].eq(best_regime), "mean_return"].iloc[0]),
        "worst_regime_primary_metric": None if worst_regime is None else float(detail.loc[detail["regime_label"].eq(worst_regime), "mean_return"].iloc[0]),
    }
    return detail, canonicalize_value(summary)


__all__ = [
    "RegimeCalibrationSensitivityResult",
    "RegimeSensitivityError",
    "SensitivitySurface",
    "build_regime_sensitivity_matrix",
    "rank_regime_calibration_profiles",
    "run_regime_calibration_sensitivity",
    "summarize_regime_sensitivity",
]
