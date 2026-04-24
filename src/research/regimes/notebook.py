from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.execution.result import ExecutionResult
from src.research.regimes.artifacts import (
    REGIME_LABELS_FILENAME,
    REGIME_MANIFEST_FILENAME,
    REGIME_SUMMARY_FILENAME,
    load_regime_labels,
    load_regime_manifest,
    load_regime_summary,
)
from src.research.regimes.attribution import (
    RegimeAttributionResult,
    RegimeComparisonResult,
    RegimeTransitionAttributionResult,
    compare_regime_results,
    summarize_transition_attribution,
)
from src.research.regimes.attribution_artifacts import (
    REGIME_ATTRIBUTION_MANIFEST_FILENAME,
    REGIME_ATTRIBUTION_REPORT_FILENAME,
    REGIME_ATTRIBUTION_SUMMARY_FILENAME,
    REGIME_ATTRIBUTION_TABLE_FILENAME,
    REGIME_COMPARISON_SUMMARY_FILENAME,
    REGIME_COMPARISON_TABLE_FILENAME,
    load_regime_attribution_manifest,
    load_regime_attribution_summary,
    load_regime_attribution_table,
    load_regime_comparison_summary,
    load_regime_comparison_table,
)
from src.research.regimes.conditional import RegimeConditionalResult
from src.research.regimes.conditional_artifacts import (
    METRICS_BY_REGIME_FILENAME,
    REGIME_CONDITIONAL_MANIFEST_FILENAME,
    REGIME_CONDITIONAL_SUMMARY_FILENAME,
    load_regime_conditional_manifest,
    load_regime_conditional_metrics,
    load_regime_conditional_summary,
)
from src.research.regimes.taxonomy import REGIME_DIMENSIONS, REGIME_STATE_COLUMNS
from src.research.regimes.transition import (
    RegimeTransitionAnalysisResult,
    RegimeTransitionConfig,
)
from src.research.regimes.transition_artifacts import (
    REGIME_TRANSITION_EVENTS_FILENAME,
    REGIME_TRANSITION_MANIFEST_FILENAME,
    REGIME_TRANSITION_SUMMARY_FILENAME,
    REGIME_TRANSITION_WINDOWS_FILENAME,
    load_regime_transition_events,
    load_regime_transition_manifest,
    load_regime_transition_summary,
    load_regime_transition_windows,
)

_SECTION_ORDER = ("regime", "conditional", "transition", "attribution", "comparison")
_ALIGNMENT_STATUS_ORDER = (
    "matched_defined",
    "matched_undefined",
    "unmatched_timestamp",
    "regime_labels_unavailable",
)
_COVERAGE_STATUS_ORDER = ("sufficient", "sparse", "empty")

_ARTIFACT_SPECS: tuple[tuple[str, str, str, Any], ...] = (
    ("regime", "regime_labels", REGIME_LABELS_FILENAME, load_regime_labels),
    ("regime", "regime_summary", REGIME_SUMMARY_FILENAME, load_regime_summary),
    ("regime", "regime_manifest", REGIME_MANIFEST_FILENAME, load_regime_manifest),
    ("conditional", "conditional_metrics", METRICS_BY_REGIME_FILENAME, load_regime_conditional_metrics),
    ("conditional", "conditional_summary", REGIME_CONDITIONAL_SUMMARY_FILENAME, load_regime_conditional_summary),
    ("conditional", "conditional_manifest", REGIME_CONDITIONAL_MANIFEST_FILENAME, load_regime_conditional_manifest),
    ("transition", "transition_events", REGIME_TRANSITION_EVENTS_FILENAME, load_regime_transition_events),
    ("transition", "transition_windows", REGIME_TRANSITION_WINDOWS_FILENAME, load_regime_transition_windows),
    ("transition", "transition_summary", REGIME_TRANSITION_SUMMARY_FILENAME, load_regime_transition_summary),
    ("transition", "transition_manifest", REGIME_TRANSITION_MANIFEST_FILENAME, load_regime_transition_manifest),
    ("attribution", "attribution_summary", REGIME_ATTRIBUTION_SUMMARY_FILENAME, load_regime_attribution_summary),
    ("attribution", "attribution_table", REGIME_ATTRIBUTION_TABLE_FILENAME, load_regime_attribution_table),
    ("attribution", "attribution_manifest", REGIME_ATTRIBUTION_MANIFEST_FILENAME, load_regime_attribution_manifest),
    ("comparison", "comparison_summary", REGIME_COMPARISON_SUMMARY_FILENAME, load_regime_comparison_summary),
    ("comparison", "comparison_table", REGIME_COMPARISON_TABLE_FILENAME, load_regime_comparison_table),
)


@dataclass(frozen=True)
class RegimeReviewBundle:
    """Notebook-facing view of one regime-aware artifact bundle."""

    root: Path | None = None
    regime_labels: pd.DataFrame | None = None
    regime_summary: dict[str, Any] | None = None
    regime_manifest: dict[str, Any] | None = None
    conditional_metrics: pd.DataFrame | None = None
    conditional_summary: dict[str, Any] | None = None
    conditional_manifest: dict[str, Any] | None = None
    transition_events: pd.DataFrame | None = None
    transition_windows: pd.DataFrame | None = None
    transition_summary: dict[str, Any] | None = None
    transition_manifest: dict[str, Any] | None = None
    attribution_summary: dict[str, Any] | None = None
    attribution_table: pd.DataFrame | None = None
    comparison_summary: dict[str, Any] | None = None
    comparison_table: pd.DataFrame | None = None
    attribution_manifest: dict[str, Any] | None = None
    attribution_report: str | None = None
    source_paths: dict[str, Path] = field(default_factory=dict)

    def available_sections(self) -> tuple[str, ...]:
        available: list[str] = []
        for section in _SECTION_ORDER:
            if any(name.startswith(f"{section}_") or name == "attribution_report" for name in self.source_paths):
                available.append(section)
        return tuple(available)

    def inventory(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for section, name, _, _loader in _ARTIFACT_SPECS:
            path = self.source_paths.get(name)
            rows.append(
                {
                    "section": section,
                    "artifact_name": name,
                    "loaded": path is not None,
                    "path": None if path is None else path.as_posix(),
                }
            )
        rows.append(
            {
                "section": "attribution",
                "artifact_name": "attribution_report",
                "loaded": "attribution_report" in self.source_paths,
                "path": None
                if "attribution_report" not in self.source_paths
                else self.source_paths["attribution_report"].as_posix(),
            }
        )
        frame = pd.DataFrame(rows, columns=["section", "artifact_name", "loaded", "path"])
        frame.attrs = {}
        return frame.sort_values(["section", "artifact_name"], kind="stable").reset_index(drop=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": None if self.root is None else self.root.as_posix(),
            "available_sections": list(self.available_sections()),
            "source_paths": {key: value.as_posix() for key, value in sorted(self.source_paths.items())},
        }


def load_regime_review_bundle(
    source: str | Path | ExecutionResult | None = None,
    *,
    root: str | Path | ExecutionResult | None = None,
    regime: str | Path | None = None,
    conditional: str | Path | None = None,
    transition: str | Path | None = None,
    attribution: str | Path | None = None,
    comparison: str | Path | None = None,
    paths: Mapping[str, str | Path] | None = None,
) -> RegimeReviewBundle:
    """Load regime-aware artifacts from one shared root, family roots, or explicit paths."""

    resolved_root = _coerce_source_root(root if root is not None else source)
    section_roots = {
        "regime": _coerce_optional_path(regime) or resolved_root,
        "conditional": _coerce_optional_path(conditional) or resolved_root,
        "transition": _coerce_optional_path(transition) or resolved_root,
        "attribution": _coerce_optional_path(attribution) or resolved_root,
        "comparison": _coerce_optional_path(comparison) or _coerce_optional_path(attribution) or resolved_root,
    }
    explicit_paths = {key: Path(value) for key, value in dict(paths or {}).items()}
    supported_paths = {name for _section, name, _filename, _loader in _ARTIFACT_SPECS} | {"attribution_report"}
    unknown = sorted(set(explicit_paths) - supported_paths)
    if unknown:
        raise ValueError(f"Unsupported explicit regime review paths: {unknown}.")

    payload: dict[str, Any] = {"root": resolved_root, "source_paths": {}}
    for section, name, filename, loader in _ARTIFACT_SPECS:
        explicit = explicit_paths.get(name)
        if explicit is not None:
            loaded_path = _require_existing_path(explicit, name=name)
            payload[name] = loader(loaded_path)
            payload["source_paths"][name] = loaded_path
            continue

        section_root = section_roots.get(section)
        if section_root is None:
            continue
        candidate = section_root / filename
        if candidate.exists():
            payload[name] = loader(section_root)
            payload["source_paths"][name] = candidate

    report_path = explicit_paths.get("attribution_report")
    if report_path is not None:
        resolved_report_path = _require_existing_path(report_path, name="attribution_report")
        payload["attribution_report"] = resolved_report_path.read_text(encoding="utf-8")
        payload["source_paths"]["attribution_report"] = resolved_report_path
    elif section_roots["attribution"] is not None:
        candidate = section_roots["attribution"] / REGIME_ATTRIBUTION_REPORT_FILENAME
        if candidate.exists():
            payload["attribution_report"] = candidate.read_text(encoding="utf-8")
            payload["source_paths"]["attribution_report"] = candidate

    return RegimeReviewBundle(**payload)


def inspect_regime_artifacts(source: RegimeReviewBundle | str | Path | ExecutionResult) -> dict[str, Any]:
    """Return notebook-friendly inspection tables and summaries for one bundle."""

    bundle = _coerce_bundle(source)
    return {
        "bundle": bundle.to_dict(),
        "artifact_inventory": bundle.inventory(),
        "available_regimes": available_regime_labels(bundle),
        "coverage_summary": summarize_regime_coverage(bundle),
        "transition_categories": list_transition_categories(bundle),
        "attribution_warnings": list_attribution_warnings(bundle),
        "alignment_statuses": _alignment_status_frame(bundle),
    }


def available_regime_labels(source: RegimeReviewBundle | pd.DataFrame) -> pd.DataFrame:
    """List regime dimensions and labels visible in a labels frame or loaded bundle."""

    if isinstance(source, pd.DataFrame):
        frame = source.copy(deep=True)
        rows = _rows_from_metrics_like_frame(frame)
    else:
        bundle = _coerce_bundle(source)
        rows: list[dict[str, Any]] = []
        if bundle.regime_labels is not None:
            labels = bundle.regime_labels.copy(deep=True)
            for dimension in REGIME_DIMENSIONS:
                column = REGIME_STATE_COLUMNS[dimension]
                counts = labels[column].astype("string").value_counts(dropna=False).sort_index()
                for label, row_count in counts.items():
                    rows.append(
                        {
                            "dimension": dimension,
                            "regime_label": str(label),
                            "row_count": int(row_count),
                            "source": "regime_labels",
                        }
                    )
        if bundle.conditional_metrics is not None:
            rows.extend(_rows_from_metrics_like_frame(bundle.conditional_metrics.copy(deep=True)))

    frame = pd.DataFrame(rows, columns=["dimension", "regime_label", "row_count", "source"])
    if frame.empty:
        frame.attrs = {}
        return frame
    frame.attrs = {}
    return frame.sort_values(["dimension", "regime_label", "source"], kind="stable").reset_index(drop=True)


def summarize_regime_coverage(source: RegimeReviewBundle | pd.DataFrame) -> pd.DataFrame:
    """Summarize defined coverage and conditional evidence status for notebook display."""

    if isinstance(source, pd.DataFrame):
        metrics = _require_columns(source, ("dimension", "regime_label", "coverage_status", "observation_count"))
        grouped = metrics.groupby(["dimension", "coverage_status"], sort=True)["observation_count"].agg(
            regime_count="size",
            observation_count="sum",
        )
        frame = grouped.reset_index()
        frame.attrs = {}
        return _sort_coverage_frame(frame)

    bundle = _coerce_bundle(source)
    rows: list[dict[str, Any]] = []
    if bundle.regime_summary is not None:
        defined_count = int(bundle.regime_summary.get("defined_row_count", 0))
        undefined_count = int(bundle.regime_summary.get("undefined_row_count", 0))
        rows.extend(
            [
                {
                    "dimension": "all",
                    "coverage_status": "defined_rows",
                    "regime_count": defined_count,
                    "observation_count": defined_count,
                },
                {
                    "dimension": "all",
                    "coverage_status": "undefined_rows",
                    "regime_count": undefined_count,
                    "observation_count": undefined_count,
                },
            ]
        )
    if bundle.conditional_metrics is not None:
        rows.extend(summarize_regime_coverage(bundle.conditional_metrics).to_dict(orient="records"))

    frame = pd.DataFrame(rows, columns=["dimension", "coverage_status", "regime_count", "observation_count"])
    frame.attrs = {}
    if frame.empty:
        return frame
    return _sort_coverage_frame(frame)


def list_transition_categories(
    source: RegimeReviewBundle | RegimeTransitionAnalysisResult | RegimeTransitionAttributionResult,
) -> pd.DataFrame:
    """List transition categories with deterministic event and warning counts."""

    if isinstance(source, RegimeTransitionAttributionResult):
        frame = source.attribution_table.copy(deep=True)
        frame.attrs = {}
        return frame.sort_values(["transition_category", "transition_dimension"], kind="stable").reset_index(drop=True)

    if isinstance(source, RegimeTransitionAnalysisResult):
        return list_transition_categories(summarize_transition_attribution(source))

    bundle = _coerce_bundle(source)
    if bundle.transition_summary is None:
        frame = pd.DataFrame(
            columns=[
                "transition_category",
                "transition_dimension",
                "event_count",
                "sufficient_event_count",
                "sparse_event_count",
                "empty_event_count",
            ]
        )
        frame.attrs = {}
        return frame

    if bundle.transition_events is not None:
        grouped = (
            bundle.transition_events.groupby(["transition_category", "transition_dimension"], sort=True)
            .size()
            .rename("event_count")
            .reset_index()
        )
    else:
        grouped = pd.DataFrame(columns=["transition_category", "transition_dimension", "event_count"])

    coverage_rows = bundle.transition_summary.get("event_summaries", [])
    coverage_frame = pd.DataFrame(coverage_rows)
    if not coverage_frame.empty and {"transition_category", "transition_dimension", "coverage_status"}.issubset(coverage_frame.columns):
        counts = (
            coverage_frame.groupby(["transition_category", "transition_dimension", "coverage_status"], sort=True)
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        for status in _COVERAGE_STATUS_ORDER:
            if status in counts.columns:
                counts[f"{status}_event_count"] = counts[status].astype("int64")
            else:
                counts[f"{status}_event_count"] = 0
        counts = counts[
            [
                "transition_category",
                "transition_dimension",
                "sufficient_event_count",
                "sparse_event_count",
                "empty_event_count",
            ]
        ]
    else:
        counts = pd.DataFrame(
            columns=[
                "transition_category",
                "transition_dimension",
                "sufficient_event_count",
                "sparse_event_count",
                "empty_event_count",
            ]
        )

    if grouped.empty and counts.empty:
        result = counts.copy()
    elif grouped.empty:
        result = counts.copy()
        result["event_count"] = (
            result["sufficient_event_count"] + result["sparse_event_count"] + result["empty_event_count"]
        ).astype("int64")
    else:
        result = grouped.merge(counts, how="left", on=["transition_category", "transition_dimension"])
        for column in ("event_count", "sufficient_event_count", "sparse_event_count", "empty_event_count"):
            if column not in result.columns:
                result[column] = 0
            result[column] = result[column].fillna(0).astype("int64")
    result.attrs = {}
    return result.sort_values(["transition_category", "transition_dimension"], kind="stable").reset_index(drop=True)


def list_attribution_warnings(
    source: RegimeReviewBundle | RegimeAttributionResult | Mapping[str, Any],
) -> pd.DataFrame:
    """Return deterministic attribution caveats and fragility warnings."""

    summary = _coerce_attribution_summary(source)
    rows: list[dict[str, Any]] = []
    if summary:
        rows.extend(
            [
                {
                    "warning_type": "fragility_flag",
                    "detail": str(bool(summary.get("fragility_flag", False))).lower(),
                },
                {
                    "warning_type": "fragility_reason",
                    "detail": str(summary.get("fragility_reason", "not available")),
                },
            ]
        )
        for caveat in summary.get("coverage_caveats", []):
            rows.append({"warning_type": "coverage_caveat", "detail": str(caveat)})
        transition_summary = summary.get("transition_summary")
        if isinstance(transition_summary, dict):
            for category in transition_summary.get("sparse_transition_categories", []):
                rows.append({"warning_type": "sparse_transition_category", "detail": str(category)})
    frame = pd.DataFrame(rows, columns=["warning_type", "detail"])
    frame.attrs = {}
    return frame.drop_duplicates(ignore_index=True)


def slice_regime_table(
    frame: pd.DataFrame,
    *,
    regime_label: str | Sequence[str] | None = None,
    dimension: str | Sequence[str] | None = None,
    volatility_state: str | Sequence[str] | None = None,
    trend_state: str | Sequence[str] | None = None,
    drawdown_recovery_state: str | Sequence[str] | None = None,
    stress_state: str | Sequence[str] | None = None,
    coverage_status: str | Sequence[str] | None = None,
    alignment_status: str | Sequence[str] | None = None,
    transition_category: str | Sequence[str] | None = None,
    transition_window_role: str | Sequence[str] | None = None,
    surface: str | Sequence[str] | None = None,
    run_id: str | Sequence[str] | None = None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Filter one regime-aware table without mutating or reordering columns."""

    filtered = frame.copy(deep=True)
    filtered.attrs = {}
    filtered = _apply_filter(filtered, "regime_label", regime_label)
    filtered = _apply_filter(filtered, "dimension", dimension)
    filtered = _apply_filter(filtered, ("volatility_state", "regime_volatility_state"), volatility_state)
    filtered = _apply_filter(filtered, ("trend_state", "regime_trend_state"), trend_state)
    filtered = _apply_filter(
        filtered,
        ("drawdown_recovery_state", "regime_drawdown_recovery_state"),
        drawdown_recovery_state,
    )
    filtered = _apply_filter(filtered, ("stress_state", "regime_stress_state"), stress_state)
    filtered = _apply_filter(filtered, "coverage_status", coverage_status)
    filtered = _apply_filter(filtered, ("alignment_status", "regime_alignment_status"), alignment_status)
    filtered = _apply_filter(filtered, "transition_category", transition_category)
    filtered = _apply_filter(filtered, "transition_window_role", transition_window_role)
    filtered = _apply_filter(filtered, "surface", surface)
    filtered = _apply_filter(filtered, "run_id", run_id)
    if columns is not None:
        missing = [column for column in columns if column not in filtered.columns]
        if missing:
            raise KeyError(f"Requested display columns are missing: {missing}.")
        filtered = filtered.loc[:, list(columns)].copy()
    return filtered.reset_index(drop=True)


def slice_conditional_metrics(source: RegimeReviewBundle | pd.DataFrame, **filters: Any) -> pd.DataFrame:
    frame = _coerce_table(source, attribute="conditional_metrics", label="conditional_metrics")
    return slice_regime_table(frame, **filters)


def slice_transition_events(source: RegimeReviewBundle | pd.DataFrame, **filters: Any) -> pd.DataFrame:
    frame = _coerce_table(source, attribute="transition_events", label="transition_events")
    return slice_regime_table(frame, **filters)


def slice_transition_windows(source: RegimeReviewBundle | pd.DataFrame, **filters: Any) -> pd.DataFrame:
    frame = _coerce_table(source, attribute="transition_windows", label="transition_windows")
    return slice_regime_table(frame, **filters)


def compare_runs_by_regime(
    source: RegimeReviewBundle | RegimeComparisonResult | pd.DataFrame | Mapping[str, RegimeConditionalResult | pd.DataFrame],
    *,
    dimension: str = "composite",
    regime_label: str | None = None,
    run_id: str | Sequence[str] | None = None,
    surface: str | None = None,
    primary_metric: str | None = None,
) -> pd.DataFrame:
    """Compare runs under one selected regime using persisted or in-memory comparison surfaces."""

    if isinstance(source, RegimeComparisonResult):
        frame = source.comparison_table.copy(deep=True)
    elif isinstance(source, Mapping):
        frame = compare_regime_results(
            source,
            dimension=dimension,
            surface=surface,
            primary_metric=primary_metric,
        ).comparison_table.copy(deep=True)
    elif isinstance(source, pd.DataFrame):
        frame = source.copy(deep=True)
    else:
        bundle = _coerce_bundle(source)
        if bundle.comparison_table is not None:
            frame = bundle.comparison_table.copy(deep=True)
        elif bundle.conditional_metrics is not None:
            resolved_surface = surface
            if resolved_surface is None and bundle.conditional_summary is not None:
                resolved_surface = str(bundle.conditional_summary.get("surface"))
            frame = compare_regime_results(
                {"run": bundle.conditional_metrics.copy(deep=True)},
                dimension=dimension,
                surface=resolved_surface,
                primary_metric=primary_metric,
            ).comparison_table.copy(deep=True)
        else:
            raise ValueError("No comparison_table or conditional_metrics are available to compare.")
    frame.attrs = {}
    frame = slice_regime_table(frame, regime_label=regime_label, dimension=dimension, run_id=run_id)
    return frame.sort_values(["regime_label", "rank_within_regime", "run_id"], kind="stable").reset_index(drop=True)


def summarize_run_regime_extremes(
    source: RegimeReviewBundle | RegimeAttributionResult | pd.DataFrame | Mapping[str, Any],
) -> pd.DataFrame:
    """Return best/worst regime rows for interactive review."""

    if isinstance(source, RegimeAttributionResult):
        frame = source.attribution_table.copy(deep=True)
    elif isinstance(source, pd.DataFrame):
        frame = source.copy(deep=True)
    elif isinstance(source, Mapping):
        summary = dict(source)
        rows = [summary.get("best_regime"), summary.get("worst_regime")]
        frame = pd.DataFrame([row for row in rows if isinstance(row, dict)])
        if not frame.empty:
            frame = frame.rename(columns={"metric_value": "primary_metric_value"})
    else:
        bundle = _coerce_bundle(source)
        if bundle.attribution_table is not None:
            frame = bundle.attribution_table.copy(deep=True)
        elif bundle.attribution_summary is not None:
            return summarize_run_regime_extremes(bundle.attribution_summary)
        else:
            raise ValueError("No attribution artifacts are available to summarize.")

    if frame.empty:
        frame.attrs = {}
        return frame
    if {"is_best_regime", "is_worst_regime"}.issubset(frame.columns):
        frame = frame.loc[frame["is_best_regime"] | frame["is_worst_regime"]].copy()
    frame.attrs = {}
    return frame.sort_values(["regime_label"], kind="stable").reset_index(drop=True)


def summarize_transition_category_extremes(
    source: RegimeReviewBundle | RegimeTransitionAnalysisResult | RegimeTransitionAttributionResult,
) -> pd.DataFrame:
    """Return strongest and weakest transition categories from the existing M24.5 summary logic."""

    if isinstance(source, RegimeTransitionAnalysisResult):
        attribution = summarize_transition_attribution(source)
    elif isinstance(source, RegimeTransitionAttributionResult):
        attribution = source
    else:
        bundle = _coerce_bundle(source)
        if bundle.transition_summary is None:
            frame = pd.DataFrame(columns=["transition_category", "transition_dimension", "metric_value", "event_count"])
            frame.attrs = {}
            return frame
        attribution = summarize_transition_attribution(_transition_result_from_summary(bundle.transition_summary))

    summary = attribution.summary
    rows = [summary.get("best_transition_category"), summary.get("worst_transition_category")]
    frame = pd.DataFrame([row for row in rows if isinstance(row, dict)])
    frame.attrs = {}
    if frame.empty:
        return frame
    return frame.sort_values(["transition_category", "transition_dimension"], kind="stable").reset_index(drop=True)


def summarize_fragility_flags(
    runs: Mapping[str, RegimeReviewBundle | RegimeAttributionResult | Mapping[str, Any]],
) -> pd.DataFrame:
    """Summarize fragility flags across multiple runs using persisted summaries when available."""

    rows: list[dict[str, Any]] = []
    for run_id, source in sorted(runs.items()):
        summary = _coerce_attribution_summary(source)
        rows.append(
            {
                "run_id": str(run_id),
                "fragility_flag": bool(summary.get("fragility_flag", False)),
                "fragility_reason": summary.get("fragility_reason"),
                "dominant_regime_label": summary.get("dominant_regime_label"),
            }
        )
    frame = pd.DataFrame(rows, columns=["run_id", "fragility_flag", "fragility_reason", "dominant_regime_label"])
    frame.attrs = {}
    return frame


def render_attribution_summary_markdown(
    source: RegimeReviewBundle | RegimeAttributionResult | Mapping[str, Any],
) -> str:
    """Render a compact markdown summary for one attribution result."""

    summary = _coerce_attribution_summary(source)
    if not summary:
        return "## Attribution Summary\n\nNo attribution summary is available.\n"
    lines = [
        "## Attribution Summary",
        "",
        f"- Surface: `{summary.get('surface', 'unknown')}`",
        f"- Dimension: `{summary.get('dimension', 'unknown')}`",
        f"- Primary metric: `{summary.get('primary_metric', 'unknown')}` ({summary.get('metric_direction', 'unknown')})",
        f"- Dominant regime: `{summary.get('dominant_regime_label')}`",
        f"- Fragility flag: `{str(bool(summary.get('fragility_flag', False))).lower()}`",
        f"- Fragility reason: {summary.get('fragility_reason', 'not available')}",
    ]
    return "\n".join(lines) + "\n"


def render_comparison_summary_markdown(
    source: RegimeReviewBundle | RegimeComparisonResult | Mapping[str, Any],
) -> str:
    """Render a compact markdown summary for a regime comparison surface."""

    summary = _coerce_comparison_summary(source)
    if not summary:
        return "## Comparison Summary\n\nNo comparison summary is available.\n"
    robust_runs = summary.get("robust_run_ids", [])
    lines = [
        "## Comparison Summary",
        "",
        f"- Surface: `{summary.get('surface', 'unknown')}`",
        f"- Dimension: `{summary.get('dimension', 'unknown')}`",
        f"- Run count: `{summary.get('run_count', 0)}`",
        f"- Regime count: `{summary.get('regime_count', 0)}`",
        f"- Robust runs: {', '.join(f'`{run_id}`' for run_id in robust_runs) if robust_runs else 'none'}",
    ]
    return "\n".join(lines) + "\n"


def render_transition_highlights_markdown(
    source: RegimeReviewBundle | RegimeTransitionAnalysisResult | RegimeTransitionAttributionResult,
) -> str:
    """Render transition highlights from an existing transition attribution summary."""

    if isinstance(source, RegimeTransitionAttributionResult):
        summary = source.summary
    elif isinstance(source, RegimeTransitionAnalysisResult):
        summary = summarize_transition_attribution(source).summary
    else:
        bundle = _coerce_bundle(source)
        if bundle.transition_summary is None:
            return "## Transition Highlights\n\nNo transition summary is available.\n"
        summary = summarize_transition_attribution(_transition_result_from_summary(bundle.transition_summary)).summary
    lines = ["## Transition Highlights", ""]
    for bullet in summary.get("interpretation_highlights", []):
        lines.append(f"- {bullet}")
    return "\n".join(lines) + "\n"


def render_artifact_inventory_markdown(source: RegimeReviewBundle | str | Path | ExecutionResult) -> str:
    """Render the loaded artifact inventory as compact markdown."""

    bundle = _coerce_bundle(source)
    inventory = bundle.inventory()
    lines = ["## Artifact Inventory", ""]
    for row in inventory.itertuples(index=False):
        status = "loaded" if row.loaded else "missing"
        lines.append(f"- `{row.artifact_name}`: {status}")
    return "\n".join(lines) + "\n"


def _coerce_bundle(source: RegimeReviewBundle | str | Path | ExecutionResult) -> RegimeReviewBundle:
    if isinstance(source, RegimeReviewBundle):
        return source
    return load_regime_review_bundle(source)


def _coerce_source_root(source: str | Path | ExecutionResult | None) -> Path | None:
    if source is None:
        return None
    if isinstance(source, ExecutionResult):
        if source.artifact_dir is None:
            raise ValueError("ExecutionResult has no artifact_dir for regime review loading.")
        return source.artifact_dir
    return Path(source)


def _coerce_optional_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value)


def _require_existing_path(path: Path, *, name: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Explicit regime review path for {name!r} does not exist: {path}")
    return path


def _rows_from_metrics_like_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    required = _require_columns(frame, ("dimension", "regime_label"))
    if "observation_count" not in required.columns:
        required["observation_count"] = 0
    grouped = required.groupby(["dimension", "regime_label"], sort=True)["observation_count"].sum().reset_index()
    return [
        {
            "dimension": str(row["dimension"]),
            "regime_label": str(row["regime_label"]),
            "row_count": int(row["observation_count"]),
            "source": "conditional_metrics",
        }
        for row in grouped.to_dict(orient="records")
    ]


def _sort_coverage_frame(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy(deep=True)
    order = {value: index for index, value in enumerate((*_COVERAGE_STATUS_ORDER, "defined_rows", "undefined_rows"))}
    working["_coverage_order"] = working["coverage_status"].map(lambda value: order.get(str(value), 99))
    working = working.sort_values(["dimension", "_coverage_order", "coverage_status"], kind="stable").drop(
        columns="_coverage_order"
    )
    working.attrs = {}
    return working.reset_index(drop=True)


def _alignment_status_frame(bundle: RegimeReviewBundle) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if bundle.conditional_summary is not None:
        if "alignment_summary" in bundle.conditional_summary:
            summary = bundle.conditional_summary.get("alignment_summary", {})
            for status in _ALIGNMENT_STATUS_ORDER:
                rows.append({"dimension": bundle.conditional_summary.get("dimension", "all"), "alignment_status": status, "row_count": int(summary.get(status, 0))})
        else:
            dimensions_summary = bundle.conditional_summary.get("dimensions_summary", {})
            for dimension, dimension_summary in sorted(dimensions_summary.items()):
                alignment_summary = dict(dimension_summary.get("alignment_summary", {}))
                for status in _ALIGNMENT_STATUS_ORDER:
                    rows.append(
                        {
                            "dimension": str(dimension),
                            "alignment_status": status,
                            "row_count": int(alignment_summary.get(status, 0)),
                        }
                    )
    frame = pd.DataFrame(rows, columns=["dimension", "alignment_status", "row_count"])
    frame.attrs = {}
    return frame.sort_values(["dimension", "alignment_status"], kind="stable").reset_index(drop=True)


def _apply_filter(frame: pd.DataFrame, column_names: str | tuple[str, ...], value: Any) -> pd.DataFrame:
    if value is None:
        return frame
    if isinstance(column_names, str):
        candidates = (column_names,)
    else:
        candidates = column_names
    column = next((candidate for candidate in candidates if candidate in frame.columns), None)
    if column is None:
        requested = "/".join(candidates)
        raise KeyError(f"Cannot filter by {requested!r}; none of those columns are present.")
    values = {value} if isinstance(value, str) or not isinstance(value, Sequence) else set(value)
    return frame.loc[frame[column].isin(values)].copy()


def _coerce_table(source: RegimeReviewBundle | pd.DataFrame, *, attribute: str, label: str) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        frame = source.copy(deep=True)
    else:
        bundle = _coerce_bundle(source)
        frame = getattr(bundle, attribute)
        if frame is None:
            raise ValueError(f"Bundle has no {label} table loaded.")
        frame = frame.copy(deep=True)
    frame.attrs = {}
    return frame


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Expected regime review columns are missing: {missing}.")
    return frame.copy(deep=True)


def _coerce_attribution_summary(
    source: RegimeReviewBundle | RegimeAttributionResult | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(source, RegimeAttributionResult):
        return dict(source.summary)
    if isinstance(source, RegimeReviewBundle):
        return dict(source.attribution_summary or {})
    return dict(source)


def _coerce_comparison_summary(
    source: RegimeReviewBundle | RegimeComparisonResult | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(source, RegimeComparisonResult):
        return dict(source.summary)
    if isinstance(source, RegimeReviewBundle):
        return dict(source.comparison_summary or {})
    return dict(source)


def _transition_result_from_summary(summary: Mapping[str, Any]) -> RegimeTransitionAnalysisResult:
    rows = summary.get("event_summaries", [])
    frame = pd.DataFrame(rows)
    if "ts_utc" in frame.columns:
        frame["ts_utc"] = pd.to_datetime(frame["ts_utc"], utc=True, errors="coerce")
    metadata = summary.get("metadata", {})
    window_config = summary.get("window_configuration", {})
    config = RegimeTransitionConfig(
        min_observations=int(metadata.get("min_observations", 5)),
        pre_event_rows=int(window_config.get("pre_event_rows", 2)),
        post_event_rows=int(window_config.get("post_event_rows", 2)),
        allow_window_overlap=bool(window_config.get("allow_window_overlap", True)),
    )
    return RegimeTransitionAnalysisResult(
        surface=str(summary.get("surface", "strategy")),
        events=pd.DataFrame(),
        windows=pd.DataFrame(),
        event_summaries=frame,
        config=config,
        metadata=dict(metadata),
    )


__all__ = [
    "RegimeReviewBundle",
    "available_regime_labels",
    "compare_runs_by_regime",
    "inspect_regime_artifacts",
    "list_attribution_warnings",
    "list_transition_categories",
    "load_regime_review_bundle",
    "render_artifact_inventory_markdown",
    "render_attribution_summary_markdown",
    "render_comparison_summary_markdown",
    "render_transition_highlights_markdown",
    "slice_conditional_metrics",
    "slice_regime_table",
    "slice_transition_events",
    "slice_transition_windows",
    "summarize_fragility_flags",
    "summarize_regime_coverage",
    "summarize_run_regime_extremes",
    "summarize_transition_category_extremes",
]
