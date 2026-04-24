from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

import pandas as pd

from src.research.regimes.conditional import RegimeConditionalResult
from src.research.regimes.transition import RegimeTransitionAnalysisResult
from src.research.registry import canonicalize_value

_SCHEMA_VERSION = 1

_COVERAGE_ORDER = {"sufficient": 0, "sparse": 1, "empty": 2}
_SURFACE_PRIMARY_METRICS: dict[str, str] = {
    "strategy": "total_return",
    "portfolio": "total_return",
    "alpha": "mean_ic",
}
_SURFACE_PRIMARY_DIRECTIONS: dict[str, str] = {
    "strategy": "desc",
    "portfolio": "desc",
    "alpha": "desc",
}
_RETURN_TRANSITION_PRIMARY_METRIC = "post_transition_return"
_ALPHA_TRANSITION_PRIMARY_METRIC = "post_mean_ic"
_FRAGILITY_DOMINANT_SHARE_THRESHOLD = 0.6
_FRAGILITY_CONCENTRATION_THRESHOLD = 0.7

_ATTRIBUTION_TABLE_COLUMNS: tuple[str, ...] = (
    "regime_label",
    "dimension",
    "surface",
    "observation_count",
    "coverage_status",
    "primary_metric",
    "primary_metric_value",
    "metric_direction",
    "positive_contribution_share",
    "absolute_contribution_share",
    "contribution_sign",
    "metric_rank",
    "is_best_regime",
    "is_worst_regime",
    "evidence_warning",
    "coverage_caveat",
)

_TRANSITION_ATTRIBUTION_BASE_COLUMNS: tuple[str, ...] = (
    "transition_category",
    "transition_dimension",
    "surface",
    "event_count",
    "sufficient_event_count",
    "sparse_event_count",
    "empty_event_count",
    "primary_metric",
    "primary_metric_value",
    "metric_rank",
    "adverse_event_count",
    "degradation_event_count",
    "sparse_warning",
)

_COMPARISON_BASE_COLUMNS: tuple[str, ...] = (
    "run_id",
    "surface",
    "dimension",
    "regime_label",
    "primary_metric",
    "primary_metric_value",
    "metric_direction",
    "coverage_status",
    "observation_count",
    "rank_within_regime",
    "warning_flag",
)


@dataclass(frozen=True)
class RegimeAttributionResult:
    surface: str
    dimension: str
    attribution_table: pd.DataFrame
    summary: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RegimeTransitionAttributionResult:
    surface: str
    attribution_table: pd.DataFrame
    summary: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RegimeComparisonResult:
    surface: str
    dimension: str
    comparison_table: pd.DataFrame
    summary: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


class RegimeAttributionError(ValueError):
    """Raised when regime attribution surfaces cannot be computed safely."""


def summarize_regime_attribution(
    result: RegimeConditionalResult,
    *,
    run_id: str | None = None,
    primary_metric: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RegimeAttributionResult:
    frame = result.metrics_by_regime.copy()
    metric_name = _resolve_primary_metric(result.surface, frame, primary_metric=primary_metric)
    metric_direction = _metric_direction(metric_name, surface=result.surface)
    working = _normalize_metric_frame(frame, metric_name=metric_name, surface=result.surface)
    ranked = _apply_metric_ranks(working, metric_name=metric_name, direction=metric_direction)
    summary = _build_regime_attribution_summary(
        ranked,
        result=result,
        run_id=run_id,
        metric_name=metric_name,
        metric_direction=metric_direction,
        metadata=metadata,
    )
    return RegimeAttributionResult(
        surface=result.surface,
        dimension=result.dimension,
        attribution_table=ranked,
        summary=summary,
        metadata=canonicalize_value(dict(metadata or {})),
    )


def summarize_transition_attribution(
    result: RegimeTransitionAnalysisResult,
    *,
    run_id: str | None = None,
    primary_metric: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RegimeTransitionAttributionResult:
    summaries = result.event_summaries.copy()
    metric_name = _resolve_transition_primary_metric(result.surface, summaries, primary_metric=primary_metric)
    aggregated = _aggregate_transition_categories(summaries, surface=result.surface, metric_name=metric_name)
    aggregated = _apply_transition_ranks(
        aggregated,
        metric_name=metric_name,
        direction=_metric_direction(metric_name, surface=result.surface),
    )
    summary = _build_transition_summary(
        aggregated,
        result=result,
        run_id=run_id,
        metric_name=metric_name,
        metadata=metadata,
    )
    return RegimeTransitionAttributionResult(
        surface=result.surface,
        attribution_table=aggregated,
        summary=summary,
        metadata=canonicalize_value(dict(metadata or {})),
    )


def compare_regime_results(
    results: Mapping[str, RegimeConditionalResult | pd.DataFrame],
    *,
    surface: str | None = None,
    dimension: str = "composite",
    primary_metric: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RegimeComparisonResult:
    if not results:
        raise RegimeAttributionError("results must not be empty.")

    normalized_rows: list[dict[str, Any]] = []
    inferred_surface: str | None = surface
    for run_id, value in sorted(results.items()):
        frame, row_surface, row_dimension = _coerce_comparison_input(
            value,
            requested_dimension=dimension,
            surface=surface,
        )
        if inferred_surface is None:
            inferred_surface = row_surface
        if row_surface != inferred_surface:
            raise RegimeAttributionError("All comparison inputs must share the same surface.")
        metric_name = _resolve_primary_metric(row_surface, frame, primary_metric=primary_metric)
        metric_direction = _metric_direction(metric_name, surface=row_surface)
        for row in frame.sort_values(["regime_label"], kind="stable").to_dict(orient="records"):
            normalized_rows.append(
                {
                    "run_id": str(run_id),
                    "surface": row_surface,
                    "dimension": row_dimension,
                    "regime_label": str(row["regime_label"]),
                    "primary_metric": metric_name,
                    "primary_metric_value": _safe_float(row.get(metric_name)),
                    "metric_direction": metric_direction,
                    "coverage_status": str(row.get("coverage_status", "empty")),
                    "observation_count": int(row.get("observation_count", 0) or 0),
                    "warning_flag": _comparison_warning_flag(row),
                }
            )

    if inferred_surface is None:
        raise RegimeAttributionError("Could not infer comparison surface.")

    comparison = _rank_comparison_rows(normalized_rows, surface=inferred_surface)
    summary = _build_comparison_summary(
        comparison,
        surface=inferred_surface,
        dimension=dimension,
        metadata=metadata,
    )
    return RegimeComparisonResult(
        surface=inferred_surface,
        dimension=dimension,
        comparison_table=comparison,
        summary=summary,
        metadata=canonicalize_value(dict(metadata or {})),
    )


def render_regime_attribution_report(
    attribution: RegimeAttributionResult,
    *,
    transition: RegimeTransitionAttributionResult | None = None,
    comparison: RegimeComparisonResult | None = None,
) -> str:
    lines: list[str] = []
    surface_title = attribution.surface.title()
    summary = attribution.summary
    lines.append(f"# {surface_title} Regime Attribution Report")
    lines.append("")
    lines.append("## Executive Summary")
    for bullet in summary.get("interpretation_highlights", []):
        lines.append(f"- {bullet}")
    lines.append("")
    lines.append("## Regime Attribution Highlights")
    lines.append(f"- Primary metric: `{summary['primary_metric']}` ({summary['metric_direction']}).")
    lines.append(
        f"- Best regime: `{summary['best_regime']['regime_label']}` "
        f"({_format_number(summary['best_regime']['metric_value'])})."
    )
    lines.append(
        f"- Worst regime: `{summary['worst_regime']['regime_label']}` "
        f"({_format_number(summary['worst_regime']['metric_value'])})."
    )
    lines.append("")
    lines.append("### Best / Worst Regimes")
    lines.extend(_render_markdown_table(attribution.attribution_table, max_rows=8))
    lines.append("")
    lines.append("## Concentration and Fragility Warnings")
    lines.append(
        f"- Dominant regime: `{summary['dominant_regime_label']}` "
        f"with share {_format_number(summary['dominant_regime_share'])}."
    )
    lines.append(
        f"- Concentration score: {_format_number(summary['regime_concentration_score'])}."
    )
    lines.append(f"- Fragility flag: `{str(summary['fragility_flag']).lower()}`.")
    lines.append(f"- Fragility reason: {summary['fragility_reason']}.")
    if summary.get("coverage_caveats"):
        for caveat in summary["coverage_caveats"]:
            lines.append(f"- Caveat: {caveat}")
    lines.append("")
    if transition is not None:
        transition_summary = transition.summary
        lines.append("## Transition Attribution Highlights")
        for bullet in transition_summary.get("interpretation_highlights", []):
            lines.append(f"- {bullet}")
        lines.append("")
        lines.append("### Transition Categories")
        lines.extend(_render_markdown_table(transition.attribution_table, max_rows=8))
        lines.append("")
    if comparison is not None:
        lines.append("## Comparison Tables")
        for bullet in comparison.summary.get("interpretation_highlights", []):
            lines.append(f"- {bullet}")
        lines.append("")
        lines.extend(_render_markdown_table(comparison.comparison_table, max_rows=12))
        lines.append("")
    lines.append("## Interpretation Notes")
    lines.append("- Results are descriptive regime slices, not causal claims.")
    lines.append("- Sparse and empty regimes remain visible and should limit confidence.")
    lines.append("- Undefined or unmatched rows are excluded from metric attribution.")
    lines.append("")
    lines.append("## Limitations and Cautions")
    lines.append("- Concentration and fragility flags are deterministic heuristics.")
    lines.append("- Transition summaries describe behavior around regime changes only.")
    return "\n".join(lines).strip() + "\n"


def _normalize_metric_frame(frame: pd.DataFrame, *, metric_name: str, surface: str) -> pd.DataFrame:
    working = frame.copy()
    if metric_name not in working.columns:
        raise RegimeAttributionError(f"Primary metric '{metric_name}' is not present in metrics_by_regime.")

    working["surface"] = surface
    working["primary_metric"] = metric_name
    working["metric_direction"] = _metric_direction(metric_name, surface=surface)
    metric_values = [_safe_float(value) for value in working[metric_name].tolist()]
    working["primary_metric_value"] = pd.Series(metric_values, dtype="object")
    positive_total = sum(max(value, 0.0) for value in metric_values if value is not None)
    absolute_total = sum(abs(value) for value in metric_values if value is not None)
    positive_shares = [
        _share(max(value, 0.0), positive_total) if value is not None else None
        for value in metric_values
    ]
    absolute_shares = [
        _share(abs(value), absolute_total) if value is not None else None
        for value in metric_values
    ]
    working["positive_contribution_share"] = pd.Series(positive_shares, dtype="object")
    working["absolute_contribution_share"] = pd.Series(absolute_shares, dtype="object")
    working["contribution_sign"] = working["primary_metric_value"].apply(_contribution_sign)
    working["evidence_warning"] = working["coverage_status"].map(_evidence_warning)
    working["coverage_caveat"] = working["coverage_status"].map(_coverage_caveat)
    for column in _ATTRIBUTION_TABLE_COLUMNS:
        if column not in working.columns:
            working[column] = None
    ordered_columns = list(_ATTRIBUTION_TABLE_COLUMNS) + [
        column for column in working.columns if column not in _ATTRIBUTION_TABLE_COLUMNS
    ]
    working = working.loc[:, ordered_columns].copy()
    working.attrs = {}
    return working.sort_values(["regime_label"], kind="stable").reset_index(drop=True)


def _apply_metric_ranks(frame: pd.DataFrame, *, metric_name: str, direction: str) -> pd.DataFrame:
    ranked = frame.copy()
    sufficient = ranked.loc[ranked["coverage_status"] == "sufficient"].copy()
    ordered_indices = _sorted_metric_indices(sufficient, metric_name=metric_name, direction=direction)
    rank_lookup = {index: position + 1 for position, index in enumerate(ordered_indices)}
    ranked["metric_rank"] = ranked.index.map(rank_lookup)
    ranked["is_best_regime"] = ranked["metric_rank"].eq(1)
    worst_rank = max(rank_lookup.values(), default=None)
    ranked["is_worst_regime"] = ranked["metric_rank"].eq(worst_rank) if worst_rank is not None else False
    return ranked


def _build_regime_attribution_summary(
    frame: pd.DataFrame,
    *,
    result: RegimeConditionalResult,
    run_id: str | None,
    metric_name: str,
    metric_direction: str,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    sufficient = frame.loc[frame["coverage_status"] == "sufficient"].copy()
    best_row = _pick_best_row(sufficient, metric_name=metric_name, direction=metric_direction)
    worst_row = _pick_worst_row(sufficient, metric_name=metric_name, direction=metric_direction)
    dominant_row = _pick_dominant_row(frame)
    concentration_score = _concentration_score(frame["positive_contribution_share"].tolist())
    coverage_counts = frame["coverage_status"].value_counts().sort_index().to_dict()
    sufficient_count = int(coverage_counts.get("sufficient", 0))
    positive_count = int(
        sum(1 for value in frame["primary_metric_value"].tolist() if value is not None and value > 0)
    )
    negative_count = int(
        sum(1 for value in frame["primary_metric_value"].tolist() if value is not None and value < 0)
    )
    sparse_count = int(coverage_counts.get("sparse", 0))
    empty_count = int(coverage_counts.get("empty", 0))
    dominant_share = _safe_float(dominant_row.get("positive_contribution_share")) if dominant_row is not None else None
    fragility_reason = _fragility_reason(
        sufficient_count=sufficient_count,
        positive_count=positive_count,
        negative_count=negative_count,
        dominant_share=dominant_share,
        concentration_score=concentration_score,
    )
    fragility_flag = fragility_reason != "no deterministic fragility heuristic triggered"
    summary: dict[str, Any] = {
        "artifact_type": "regime_attribution_summary",
        "schema_version": _SCHEMA_VERSION,
        "surface": result.surface,
        "dimension": result.dimension,
        "taxonomy_version": result.config.taxonomy_version,
        "primary_metric": metric_name,
        "metric_direction": metric_direction,
        "regime_count": int(len(frame)),
        "coverage_counts": {str(key): int(value) for key, value in coverage_counts.items()},
        "positive_regime_count": positive_count,
        "negative_regime_count": negative_count,
        "sparse_regime_count": sparse_count,
        "empty_regime_count": empty_count,
        "sufficient_regime_count": sufficient_count,
        "dominant_regime_label": dominant_row.get("regime_label") if dominant_row is not None else None,
        "dominant_regime_metric": _safe_float(dominant_row.get("primary_metric_value")) if dominant_row is not None else None,
        "dominant_regime_share": dominant_share,
        "regime_concentration_score": concentration_score,
        "fragility_flag": fragility_flag,
        "fragility_reason": fragility_reason,
        "best_regime": _summary_regime_row(best_row),
        "worst_regime": _summary_regime_row(worst_row),
        "coverage_caveats": _coverage_caveats(frame, result.alignment_summary),
        "alignment_summary": canonicalize_value(result.alignment_summary),
        "metric_columns": list(result.metrics_by_regime.columns),
        "interpretation_highlights": _regime_interpretation_highlights(
            frame,
            metric_name=metric_name,
            dominant_label=dominant_row.get("regime_label") if dominant_row is not None else None,
            dominant_share=dominant_share,
            fragility_reason=fragility_reason,
        ),
        "regimes": canonicalize_value(_summary_rows(frame)),
        "metadata": canonicalize_value({**result.metadata, **dict(metadata or {})}),
    }
    if run_id is not None:
        summary["run_id"] = str(run_id)
    return canonicalize_value(summary)


def _aggregate_transition_categories(
    frame: pd.DataFrame,
    *,
    surface: str,
    metric_name: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=list(_TRANSITION_ATTRIBUTION_BASE_COLUMNS))

    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(["transition_category", "transition_dimension"], sort=True, dropna=False)
    for (category, dimension), group in grouped:
        sufficient = int(group["coverage_status"].eq("sufficient").sum())
        sparse = int(group["coverage_status"].eq("sparse").sum())
        empty = int(group["coverage_status"].eq("empty").sum())
        row: dict[str, Any] = {
            "transition_category": str(category),
            "transition_dimension": str(dimension),
            "surface": surface,
            "event_count": int(len(group)),
            "sufficient_event_count": sufficient,
            "sparse_event_count": sparse,
            "empty_event_count": empty,
            "primary_metric": metric_name,
            "adverse_event_count": _adverse_event_count(group, surface=surface),
            "degradation_event_count": _degradation_event_count(group, surface=surface),
            "sparse_warning": sparse > 0 or empty > 0,
        }
        for source_column, output_column in _transition_average_column_map(surface):
            row[output_column] = _mean_metric(group[source_column]) if source_column in group.columns else None
        row["primary_metric_value"] = row.get(_transition_output_metric_name(metric_name))
        rows.append(row)

    aggregated = pd.DataFrame(rows)
    base_columns = list(_TRANSITION_ATTRIBUTION_BASE_COLUMNS)
    for _source_column, output_column in _transition_average_column_map(surface):
        if output_column not in aggregated.columns:
            aggregated[output_column] = None
    for column in base_columns:
        if column not in aggregated.columns:
            aggregated[column] = None
    ordered_columns = base_columns + [
        output_column for _source_column, output_column in _transition_average_column_map(surface)
        if output_column not in base_columns
    ]
    aggregated = aggregated.loc[:, ordered_columns].copy()
    aggregated.attrs = {}
    return aggregated.sort_values(["transition_category", "transition_dimension"], kind="stable").reset_index(drop=True)


def _apply_transition_ranks(frame: pd.DataFrame, *, metric_name: str, direction: str) -> pd.DataFrame:
    ranked = frame.copy()
    sufficient = ranked.loc[ranked["sufficient_event_count"] > 0].copy()
    ordered_indices = _sorted_metric_indices(sufficient, metric_name="primary_metric_value", direction=direction)
    rank_lookup = {index: position + 1 for position, index in enumerate(ordered_indices)}
    ranked["metric_rank"] = ranked.index.map(rank_lookup)
    return ranked


def _build_transition_summary(
    frame: pd.DataFrame,
    *,
    result: RegimeTransitionAnalysisResult,
    run_id: str | None,
    metric_name: str,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    direction = _metric_direction(metric_name, surface=result.surface)
    best_row = _pick_best_row(frame.loc[frame["sufficient_event_count"] > 0], metric_name="primary_metric_value", direction=direction)
    worst_row = _pick_worst_row(frame.loc[frame["sufficient_event_count"] > 0], metric_name="primary_metric_value", direction=direction)
    sparse_categories = frame.loc[frame["sparse_warning"], "transition_category"].tolist()
    summary: dict[str, Any] = {
        "artifact_type": "regime_transition_attribution_summary",
        "schema_version": _SCHEMA_VERSION,
        "surface": result.surface,
        "taxonomy_version": result.config.taxonomy_version,
        "primary_metric": metric_name,
        "transition_category_count": int(len(frame)),
        "event_count": int(len(result.event_summaries)),
        "coverage_counts": canonicalize_value(
            result.event_summaries["coverage_status"].value_counts().sort_index().to_dict()
            if not result.event_summaries.empty
            else {}
        ),
        "best_transition_category": _summary_transition_row(best_row),
        "worst_transition_category": _summary_transition_row(worst_row),
        "stress_onset_summary": _stress_onset_summary(frame),
        "sparse_transition_categories": [str(value) for value in sparse_categories],
        "degradation_categories": [
            str(value)
            for value in frame.loc[frame["degradation_event_count"] > 0, "transition_category"].tolist()
        ],
        "interpretation_highlights": _transition_interpretation_highlights(
            frame,
            surface=result.surface,
            metric_name=metric_name,
        ),
        "transition_categories": canonicalize_value(_summary_rows(frame)),
        "metadata": canonicalize_value({**result.metadata, **dict(metadata or {})}),
    }
    if run_id is not None:
        summary["run_id"] = str(run_id)
    return canonicalize_value(summary)


def _coerce_comparison_input(
    value: RegimeConditionalResult | pd.DataFrame,
    *,
    requested_dimension: str,
    surface: str | None,
) -> tuple[pd.DataFrame, str, str]:
    if isinstance(value, RegimeConditionalResult):
        return value.metrics_by_regime.copy(), value.surface, value.dimension
    if not isinstance(value, pd.DataFrame):
        raise RegimeAttributionError("Comparison inputs must be RegimeConditionalResult objects or DataFrames.")
    frame = value.copy()
    resolved_dimension = requested_dimension
    if "dimension" in frame.columns:
        frame = frame.loc[frame["dimension"] == requested_dimension].copy()
        if frame.empty:
            raise RegimeAttributionError(f"Comparison DataFrame does not contain dimension '{requested_dimension}'.")
    if surface is None:
        raise RegimeAttributionError("surface is required when comparing raw DataFrames.")
    return frame, surface, resolved_dimension


def _rank_comparison_rows(rows: list[dict[str, Any]], *, surface: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=list(_COMPARISON_BASE_COLUMNS))
    frame = pd.DataFrame(rows)
    metric_direction = _metric_direction(frame["primary_metric"].iloc[0], surface=surface)
    ranked_groups: list[pd.DataFrame] = []
    for _regime_label, group in frame.groupby("regime_label", sort=True):
        group = group.copy()
        group["_coverage_rank"] = group["coverage_status"].map(_coverage_rank)
        group["_metric_sort"] = group["primary_metric_value"].apply(
            lambda value: _metric_sort_value(value, direction=metric_direction)
        )
        group = group.sort_values(
            ["_coverage_rank", "_metric_sort", "run_id"],
            kind="stable",
        ).reset_index(drop=True)
        group["rank_within_regime"] = range(1, len(group) + 1)
        ranked_groups.append(group.drop(columns=["_coverage_rank", "_metric_sort"]))
    ranked = pd.concat(ranked_groups, ignore_index=True)
    for column in _COMPARISON_BASE_COLUMNS:
        if column not in ranked.columns:
            ranked[column] = None
    ranked = ranked.loc[:, list(_COMPARISON_BASE_COLUMNS)]
    ranked.attrs = {}
    return ranked.sort_values(["regime_label", "rank_within_regime", "run_id"], kind="stable").reset_index(drop=True)


def _build_comparison_summary(
    frame: pd.DataFrame,
    *,
    surface: str,
    dimension: str,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    winners_frame = frame.loc[
        frame["rank_within_regime"] == 1, ["regime_label", "run_id", "primary_metric_value"]
    ].sort_values(["regime_label", "run_id"], kind="stable")
    winners = [
        {
            "regime_label": str(row["regime_label"]),
            "run_id": str(row["run_id"]),
            "primary_metric_value": _safe_float(row["primary_metric_value"]),
        }
        for row in winners_frame.to_dict(orient="records")
    ]
    win_counts = (
        pd.Series([row["run_id"] for row in winners]).value_counts().sort_index().to_dict()
        if winners
        else {}
    )
    robust_runs = sorted(run_id for run_id, count in win_counts.items() if count > 1)
    summary = {
        "artifact_type": "regime_comparison_summary",
        "schema_version": _SCHEMA_VERSION,
        "surface": surface,
        "dimension": dimension,
        "run_count": int(frame["run_id"].nunique()) if not frame.empty else 0,
        "regime_count": int(frame["regime_label"].nunique()) if not frame.empty else 0,
        "regime_winners": canonicalize_value(winners),
        "winner_counts_by_run": {str(key): int(value) for key, value in sorted(win_counts.items())},
        "robust_run_ids": robust_runs,
        "interpretation_highlights": _comparison_interpretation_highlights(
            robust_runs=robust_runs,
            winners=winners,
            dimension=dimension,
        ),
        "metadata": canonicalize_value(dict(metadata or {})),
    }
    return canonicalize_value(summary)


def _resolve_primary_metric(surface: str, frame: pd.DataFrame, *, primary_metric: str | None) -> str:
    metric = primary_metric or _SURFACE_PRIMARY_METRICS.get(surface)
    if metric is None:
        raise RegimeAttributionError(f"No default primary metric is defined for surface '{surface}'.")
    if metric not in frame.columns:
        raise RegimeAttributionError(f"Primary metric '{metric}' is not present for surface '{surface}'.")
    return metric


def _resolve_transition_primary_metric(
    surface: str,
    frame: pd.DataFrame,
    *,
    primary_metric: str | None,
) -> str:
    metric = primary_metric or (
        _ALPHA_TRANSITION_PRIMARY_METRIC if surface == "alpha" else _RETURN_TRANSITION_PRIMARY_METRIC
    )
    if not frame.empty and metric not in frame.columns:
        raise RegimeAttributionError(f"Transition primary metric '{metric}' is not present for surface '{surface}'.")
    return metric


def _metric_direction(metric_name: str, *, surface: str) -> str:
    if metric_name in {"max_drawdown", "window_max_drawdown", "average_window_max_drawdown"}:
        return "asc"
    return _SURFACE_PRIMARY_DIRECTIONS.get(surface, "desc")


def _pick_best_row(frame: pd.DataFrame, *, metric_name: str, direction: str) -> pd.Series | None:
    if frame.empty:
        return None
    ordered = _sorted_metric_indices(frame, metric_name=metric_name, direction=direction)
    return frame.loc[ordered[0]] if ordered else None


def _pick_worst_row(frame: pd.DataFrame, *, metric_name: str, direction: str) -> pd.Series | None:
    if frame.empty:
        return None
    ordered = _sorted_metric_indices(frame, metric_name=metric_name, direction=direction)
    return frame.loc[ordered[-1]] if ordered else None


def _pick_dominant_row(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None
    working = frame.copy()
    working["_dominant_share"] = working["positive_contribution_share"].apply(
        lambda value: _safe_float(value) if _safe_float(value) is not None else 0.0
    )
    positive = working.loc[working["_dominant_share"] > 0].copy()
    if not positive.empty:
        positive = positive.sort_values(
            ["_dominant_share", "regime_label"],
            ascending=[False, True],
            kind="stable",
        )
        return positive.iloc[0]

    working["_absolute_share"] = working["absolute_contribution_share"].apply(
        lambda value: _safe_float(value) if _safe_float(value) is not None else -1.0
    )
    working = working.sort_values(
        ["_absolute_share", "regime_label"],
        ascending=[False, True],
        kind="stable",
    )
    return working.iloc[0]


def _sorted_metric_indices(frame: pd.DataFrame, *, metric_name: str, direction: str) -> list[int]:
    if frame.empty:
        return []
    working = frame.copy()
    working["_coverage_rank"] = working["coverage_status"].map(_coverage_rank) if "coverage_status" in working.columns else 0
    working["_metric_sort"] = working[metric_name].apply(lambda value: _metric_sort_value(value, direction=direction))
    label_column = "regime_label" if "regime_label" in working.columns else "transition_category"
    extra_column = "transition_dimension" if "transition_dimension" in working.columns else label_column
    working = working.sort_values(
        ["_coverage_rank", "_metric_sort", label_column, extra_column],
        kind="stable",
    )
    return working.index.tolist()


def _metric_sort_value(value: Any, *, direction: str) -> float:
    numeric = _safe_float(value)
    if numeric is None:
        return math.inf
    return -numeric if direction == "desc" else numeric


def _coverage_rank(value: str) -> int:
    return _COVERAGE_ORDER.get(str(value), 99)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _share(value: float, total: float) -> float | None:
    if total <= 0.0:
        return None
    return _safe_float(value / total)


def _contribution_sign(value: float | None) -> str:
    if value is None:
        return "undefined"
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "flat"


def _evidence_warning(status: Any) -> str | None:
    if status == "sparse":
        return "sparse_evidence"
    if status == "empty":
        return "empty_regime"
    return None


def _coverage_caveat(status: Any) -> str | None:
    if status == "sparse":
        return "regime has fewer than min_observations rows"
    if status == "empty":
        return "regime has zero matched_defined rows"
    return None


def _summary_regime_row(row: pd.Series | None) -> dict[str, Any]:
    if row is None:
        return {"regime_label": None, "metric_value": None, "coverage_status": None, "observation_count": 0}
    return canonicalize_value(
        {
            "regime_label": row.get("regime_label"),
            "metric_value": _safe_float(row.get("primary_metric_value")),
            "coverage_status": row.get("coverage_status"),
            "observation_count": int(row.get("observation_count", 0) or 0),
        }
    )


def _summary_transition_row(row: pd.Series | None) -> dict[str, Any]:
    if row is None:
        return {
            "transition_category": None,
            "transition_dimension": None,
            "metric_value": None,
            "event_count": 0,
        }
    return canonicalize_value(
        {
            "transition_category": row.get("transition_category"),
            "transition_dimension": row.get("transition_dimension"),
            "metric_value": _safe_float(row.get("primary_metric_value")),
            "event_count": int(row.get("event_count", 0) or 0),
        }
    )


def _coverage_caveats(frame: pd.DataFrame, alignment_summary: Mapping[str, Any]) -> list[str]:
    caveats: list[str] = []
    if int(frame["coverage_status"].eq("sparse").sum()) > 0:
        caveats.append("one or more regimes are sparse and carry null metrics")
    if int(frame["coverage_status"].eq("empty").sum()) > 0:
        caveats.append("one or more regimes are empty and should not support conclusions")
    if int(alignment_summary.get("matched_undefined", 0)) > 0:
        caveats.append("matched_undefined rows were excluded from attribution")
    if int(alignment_summary.get("unmatched_timestamp", 0)) > 0:
        caveats.append("unmatched_timestamp rows were excluded from attribution")
    return caveats


def _concentration_score(shares: list[Any]) -> float | None:
    normalized = [_safe_float(value) for value in shares]
    clean = [value for value in normalized if value is not None and value > 0]
    if not clean:
        return None
    return _safe_float(sum(value * value for value in clean))


def _fragility_reason(
    *,
    sufficient_count: int,
    positive_count: int,
    negative_count: int,
    dominant_share: float | None,
    concentration_score: float | None,
) -> str:
    if sufficient_count <= 1:
        return "only one regime has sufficient evidence"
    if positive_count <= 1:
        return "positive performance is isolated to one regime"
    if dominant_share is not None and dominant_share >= _FRAGILITY_DOMINANT_SHARE_THRESHOLD:
        return "most positive contribution comes from one dominant regime"
    if concentration_score is not None and concentration_score >= _FRAGILITY_CONCENTRATION_THRESHOLD:
        return "positive contribution is highly concentrated across a small set of regimes"
    if positive_count > 0 and negative_count > positive_count:
        return "aggregate success is paired with broader regime weakness"
    return "no deterministic fragility heuristic triggered"


def _regime_interpretation_highlights(
    frame: pd.DataFrame,
    *,
    metric_name: str,
    dominant_label: str | None,
    dominant_share: float | None,
    fragility_reason: str,
) -> list[str]:
    best = frame.loc[frame["is_best_regime"]].head(1)
    worst = frame.loc[frame["is_worst_regime"]].head(1)
    highlights: list[str] = []
    if not best.empty:
        highlights.append(
            f"Best {metric_name} came from `{best.iloc[0]['regime_label']}` at {_format_number(best.iloc[0]['primary_metric_value'])}."
        )
    if not worst.empty:
        highlights.append(
            f"Weakest regime was `{worst.iloc[0]['regime_label']}` at {_format_number(worst.iloc[0]['primary_metric_value'])}."
        )
    if dominant_label is not None and dominant_share is not None:
        highlights.append(
            f"Positive contribution is led by `{dominant_label}` with share {_format_number(dominant_share)}."
        )
    highlights.append(f"Fragility heuristic: {fragility_reason}.")
    return highlights


def _transition_average_column_map(surface: str) -> tuple[tuple[str, str], ...]:
    if surface == "alpha":
        return (
            ("pre_mean_ic", "average_pre_mean_ic"),
            ("event_mean_ic", "average_event_mean_ic"),
            ("post_mean_ic", "average_post_mean_ic"),
            ("window_mean_ic", "average_window_mean_ic"),
            ("pre_mean_rank_ic", "average_pre_mean_rank_ic"),
            ("event_mean_rank_ic", "average_event_mean_rank_ic"),
            ("post_mean_rank_ic", "average_post_mean_rank_ic"),
            ("window_mean_rank_ic", "average_window_mean_rank_ic"),
        )
    return (
        ("pre_transition_return", "average_pre_transition_return"),
        ("event_window_return", "average_event_window_return"),
        ("post_transition_return", "average_post_transition_return"),
        ("window_cumulative_return", "average_window_cumulative_return"),
        ("window_max_drawdown", "average_window_max_drawdown"),
        ("window_win_rate", "average_window_win_rate"),
    )


def _transition_output_metric_name(metric_name: str) -> str:
    if metric_name.startswith("average_"):
        return metric_name
    return f"average_{metric_name}"


def _mean_metric(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return _safe_float(float(values.mean()))


def _adverse_event_count(group: pd.DataFrame, *, surface: str) -> int:
    if surface == "alpha":
        values = pd.to_numeric(group.get("post_mean_ic"), errors="coerce")
        return int(values.lt(0).fillna(False).sum())
    values = pd.to_numeric(group.get("post_transition_return"), errors="coerce")
    return int(values.lt(0).fillna(False).sum())


def _degradation_event_count(group: pd.DataFrame, *, surface: str) -> int:
    if surface == "alpha":
        pre = pd.to_numeric(group.get("pre_mean_ic"), errors="coerce")
        post = pd.to_numeric(group.get("post_mean_ic"), errors="coerce")
    else:
        pre = pd.to_numeric(group.get("pre_transition_return"), errors="coerce")
        post = pd.to_numeric(group.get("post_transition_return"), errors="coerce")
    return int(post.lt(pre).fillna(False).sum())


def _transition_interpretation_highlights(
    frame: pd.DataFrame,
    *,
    surface: str,
    metric_name: str,
) -> list[str]:
    highlights: list[str] = []
    direction = _metric_direction(metric_name, surface=surface)
    best = _pick_best_row(frame.loc[frame["sufficient_event_count"] > 0], metric_name="primary_metric_value", direction=direction)
    worst = _pick_worst_row(frame.loc[frame["sufficient_event_count"] > 0], metric_name="primary_metric_value", direction=direction)
    if best is not None:
        highlights.append(
            f"Best transition category was `{best['transition_category']}` at {_format_number(best['primary_metric_value'])}."
        )
    if worst is not None:
        highlights.append(
            f"Worst transition category was `{worst['transition_category']}` at {_format_number(worst['primary_metric_value'])}."
        )
    stress = frame.loc[frame["transition_category"] == "stress_onset"].head(1)
    if not stress.empty:
        highlights.append(
            f"`stress_onset` shows {metric_name} at {_format_number(stress.iloc[0]['primary_metric_value'])}."
        )
    sparse = sorted(frame.loc[frame["sparse_warning"], "transition_category"].tolist())
    if sparse:
        highlights.append(f"Sparse transition categories: {', '.join(f'`{value}`' for value in sparse)}.")
    return highlights


def _stress_onset_summary(frame: pd.DataFrame) -> dict[str, Any]:
    stress = frame.loc[frame["transition_category"] == "stress_onset"].head(1)
    if stress.empty:
        return {"transition_category": "stress_onset", "present": False}
    row = stress.iloc[0]
    return canonicalize_value(
        {
            "transition_category": "stress_onset",
            "present": True,
            "primary_metric_value": _safe_float(row.get("primary_metric_value")),
            "event_count": int(row.get("event_count", 0) or 0),
            "adverse_event_count": int(row.get("adverse_event_count", 0) or 0),
            "degradation_event_count": int(row.get("degradation_event_count", 0) or 0),
        }
    )


def _comparison_warning_flag(row: Mapping[str, Any]) -> str | None:
    status = str(row.get("coverage_status", "empty"))
    if status == "sufficient":
        return None
    if status == "sparse":
        return "sparse_evidence"
    return "empty_regime"


def _comparison_interpretation_highlights(
    *,
    robust_runs: list[str],
    winners: list[dict[str, Any]],
    dimension: str,
) -> list[str]:
    if not winners:
        return [f"No comparable regime winners were available for `{dimension}`."]
    highlights = [f"Comparison is aligned on shared regime dimension `{dimension}`."]
    if robust_runs:
        highlights.append(
            f"Runs leading in more than one regime: {', '.join(f'`{run_id}`' for run_id in robust_runs)}."
        )
    else:
        highlights.append("No run led across multiple regimes, suggesting mixed regime leadership.")
    return highlights


def _summary_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    normalized = frame.astype("object").where(pd.notna(frame), None)
    return normalized.to_dict(orient="records")


def _render_markdown_table(frame: pd.DataFrame, *, max_rows: int) -> list[str]:
    if frame.empty:
        return ["No rows available."]
    display = frame.head(max_rows).copy()
    headers = list(display.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in display.itertuples(index=False):
        rendered = []
        for value in row:
            if isinstance(value, bool):
                rendered.append(str(value).lower())
            elif isinstance(value, float):
                rendered.append(_format_number(value))
            elif value is None:
                rendered.append("NA")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return lines


def _format_number(value: Any) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return "NA"
    return f"{numeric:.4f}"


__all__ = [
    "RegimeAttributionError",
    "RegimeAttributionResult",
    "RegimeTransitionAttributionResult",
    "RegimeComparisonResult",
    "compare_regime_results",
    "render_regime_attribution_report",
    "summarize_regime_attribution",
    "summarize_transition_attribution",
]
