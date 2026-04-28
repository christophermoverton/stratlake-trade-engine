from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.config.regime_aware_candidate_selection import RegimeAwareCandidateSelectionConfig
from src.research.registry import canonicalize_value, serialize_canonical_json


CATEGORIES = ("global_performer", "regime_specialist", "transition_resilient", "defensive_fallback")
CANDIDATE_SELECTION_COLUMNS = (
    "candidate_id",
    "candidate_name",
    "artifact_type",
    "source_run_id",
    "global_rank",
    "global_score",
    "selected",
    "selection_category",
    "primary_regime",
    "secondary_regimes",
    "regime_specialist_score",
    "transition_resilience_score",
    "defensive_score",
    "regime_confidence_required",
    "regime_confidence_observed",
    "redundancy_group",
    "redundancy_action",
    "selection_reason",
    "source_review_run_id",
    "source_benchmark_run_id",
)
REGIME_SCORE_COLUMNS = (
    "candidate_id",
    "candidate_name",
    "artifact_type",
    "regime_id",
    "regime_label",
    "observation_count",
    "regime_score",
    "regime_rank",
    "regime_return",
    "regime_sharpe",
    "regime_max_drawdown",
    "regime_ic",
    "regime_rank_ic",
    "regime_hit_rate",
    "transition_adjacent",
    "confidence_weighted_score",
    "selected_for_regime",
    "selection_reason",
)
CATEGORY_COLUMNS = (
    "candidate_id",
    "candidate_name",
    "artifact_type",
    "category",
    "category_rank",
    "category_score",
    "selected",
    "primary_reason",
    "warning_count",
    "failed_gate_count",
    "source_metric_names",
    "source_metric_values",
)
TRANSITION_COLUMNS = (
    "candidate_id",
    "candidate_name",
    "artifact_type",
    "transition_score",
    "transition_window_return",
    "transition_window_drawdown",
    "transition_window_observation_count",
    "selected",
    "selection_reason",
)
DEFENSIVE_COLUMNS = (
    "candidate_id",
    "candidate_name",
    "artifact_type",
    "defensive_score",
    "high_vol_drawdown",
    "volatility",
    "correlation_to_selected",
    "selected",
    "selection_reason",
)
REDUNDANCY_COLUMNS = (
    "candidate_id",
    "candidate_name",
    "category",
    "redundancy_group",
    "compared_to_candidate_id",
    "redundancy_metric",
    "redundancy_value",
    "threshold",
    "redundancy_action",
    "selected",
    "reason",
)
REQUIRED_CANDIDATE_COLUMNS = ("candidate_id", "candidate_name", "artifact_type")


@dataclass(frozen=True)
class RegimeAwareCandidateSelectionRunResult:
    selection_run_id: str
    selection_name: str
    source_review_run_id: str
    source_benchmark_run_id: str
    output_dir: Path
    candidate_selection_csv_path: Path
    candidate_selection_json_path: Path
    regime_candidate_scores_csv_path: Path
    regime_candidate_scores_json_path: Path
    category_assignments_path: Path
    transition_resilience_path: Path
    defensive_fallback_candidates_path: Path
    redundancy_report_path: Path
    allocation_hints_path: Path
    selection_summary_path: Path
    config_path: Path
    manifest_path: Path
    selection_summary: dict[str, Any]
    manifest: dict[str, Any]


def run_regime_aware_candidate_selection(
    config: RegimeAwareCandidateSelectionConfig,
) -> RegimeAwareCandidateSelectionRunResult:
    review_root = Path(config.source_review_pack).resolve()
    _require_dir(review_root, "Regime review pack")
    review_metadata = _load_review_pack_metadata(review_root)
    source_review_run_id = str(review_metadata.get("review_run_id") or review_root.name)
    source_benchmark_run_id = str(review_metadata.get("source_benchmark_run_id") or "unknown")

    candidate_rows, inventory, load_warnings = _load_candidate_universe(config)
    if not candidate_rows:
        raise ValueError("Regime-aware candidate selection requires at least one candidate row.")

    enriched = _enrich_candidate_rows(candidate_rows, config)
    regime_rows = _build_regime_rows(enriched, config)
    category_rows, selected_entries = _build_category_assignments(enriched, regime_rows, config)
    selected_entries, redundancy_rows, redundancy_warnings = _apply_redundancy(selected_entries, config)
    selected_lookup = {(entry["candidate_id"], entry["category"], entry.get("regime_id")) for entry in selected_entries}
    category_rows = _mark_category_rows(category_rows, selected_lookup)
    regime_rows = _mark_regime_rows(regime_rows, selected_lookup)
    transition_rows = _build_transition_rows(enriched, selected_lookup)
    defensive_rows = _build_defensive_rows(enriched, selected_lookup)
    candidate_selection_rows = _build_candidate_selection_rows(
        enriched,
        selected_entries,
        config=config,
        source_review_run_id=source_review_run_id,
        source_benchmark_run_id=source_benchmark_run_id,
    )

    generated_files = [
        "candidate_selection.csv",
        "candidate_selection.json",
        "regime_candidate_scores.csv",
        "regime_candidate_scores.json",
        "category_assignments.csv",
        "transition_resilience.csv",
        "defensive_fallback_candidates.csv",
        "redundancy_report.csv",
        "allocation_hints.json",
        "selection_summary.json",
        "config.json",
        "manifest.json",
        "source_regime_review_pack.json",
        "candidate_input_inventory.json",
    ]
    selection_run_id = _build_selection_run_id(
        config=config,
        source_review_run_id=source_review_run_id,
        source_benchmark_run_id=source_benchmark_run_id,
        candidate_ids=[row["candidate_id"] for row in enriched],
    )
    output_dir = Path(config.output_root).resolve() / selection_run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    allocation_hints = _build_allocation_hints(selected_entries, config)
    limitations = sorted(set(load_warnings + redundancy_warnings))
    summary = _build_summary(
        selection_run_id=selection_run_id,
        config=config,
        candidate_count=len(enriched),
        selected_entries=selected_entries,
        source_review_run_id=source_review_run_id,
        source_benchmark_run_id=source_benchmark_run_id,
        source_review_pack_path=review_root,
        output_dir=output_dir,
        generated_files=generated_files,
        limitations=limitations,
    )
    manifest = _build_manifest(
        selection_run_id=selection_run_id,
        config=config,
        source_review_run_id=source_review_run_id,
        source_benchmark_run_id=source_benchmark_run_id,
        review_root=review_root,
        inventory=inventory,
        generated_files=generated_files,
        output_dir=output_dir,
        row_counts={
            "candidate_selection": len(candidate_selection_rows),
            "regime_candidate_scores": len(regime_rows),
            "category_assignments": len(category_rows),
            "transition_resilience": len(transition_rows),
            "defensive_fallback_candidates": len(defensive_rows),
            "redundancy_report": len(redundancy_rows),
        },
        selected_entries=selected_entries,
        warning_count=len(limitations),
    )

    paths = {
        "candidate_selection_csv": output_dir / "candidate_selection.csv",
        "candidate_selection_json": output_dir / "candidate_selection.json",
        "regime_candidate_scores_csv": output_dir / "regime_candidate_scores.csv",
        "regime_candidate_scores_json": output_dir / "regime_candidate_scores.json",
        "category_assignments": output_dir / "category_assignments.csv",
        "transition_resilience": output_dir / "transition_resilience.csv",
        "defensive": output_dir / "defensive_fallback_candidates.csv",
        "redundancy": output_dir / "redundancy_report.csv",
        "allocation_hints": output_dir / "allocation_hints.json",
        "summary": output_dir / "selection_summary.json",
        "config": output_dir / "config.json",
        "manifest": output_dir / "manifest.json",
        "source_review": output_dir / "source_regime_review_pack.json",
        "inventory": output_dir / "candidate_input_inventory.json",
    }
    _write_csv(paths["candidate_selection_csv"], candidate_selection_rows, CANDIDATE_SELECTION_COLUMNS)
    _write_json(paths["candidate_selection_json"], {"rows": candidate_selection_rows, "columns": list(CANDIDATE_SELECTION_COLUMNS), "row_count": len(candidate_selection_rows)})
    _write_csv(paths["regime_candidate_scores_csv"], regime_rows, REGIME_SCORE_COLUMNS)
    _write_json(paths["regime_candidate_scores_json"], {"rows": regime_rows, "columns": list(REGIME_SCORE_COLUMNS), "row_count": len(regime_rows)})
    _write_csv(paths["category_assignments"], category_rows, CATEGORY_COLUMNS)
    _write_csv(paths["transition_resilience"], transition_rows, TRANSITION_COLUMNS)
    _write_csv(paths["defensive"], defensive_rows, DEFENSIVE_COLUMNS)
    _write_csv(paths["redundancy"], redundancy_rows, REDUNDANCY_COLUMNS)
    _write_json(paths["allocation_hints"], allocation_hints)
    _write_json(paths["summary"], summary)
    _write_json(paths["config"], config.to_dict())
    _write_json(paths["manifest"], manifest)
    _write_json(paths["source_review"], review_metadata)
    _write_json(paths["inventory"], inventory)

    return RegimeAwareCandidateSelectionRunResult(
        selection_run_id=selection_run_id,
        selection_name=config.selection_name,
        source_review_run_id=source_review_run_id,
        source_benchmark_run_id=source_benchmark_run_id,
        output_dir=output_dir,
        candidate_selection_csv_path=paths["candidate_selection_csv"],
        candidate_selection_json_path=paths["candidate_selection_json"],
        regime_candidate_scores_csv_path=paths["regime_candidate_scores_csv"],
        regime_candidate_scores_json_path=paths["regime_candidate_scores_json"],
        category_assignments_path=paths["category_assignments"],
        transition_resilience_path=paths["transition_resilience"],
        defensive_fallback_candidates_path=paths["defensive"],
        redundancy_report_path=paths["redundancy"],
        allocation_hints_path=paths["allocation_hints"],
        selection_summary_path=paths["summary"],
        config_path=paths["config"],
        manifest_path=paths["manifest"],
        selection_summary=summary,
        manifest=manifest,
    )


def _load_candidate_universe(
    config: RegimeAwareCandidateSelectionConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    warnings: list[str] = []
    universe = config.source_candidate_universe
    if universe.candidate_metrics_path is None:
        raise ValueError(
            "Registry-backed regime-aware candidate selection is reserved for a later expansion; "
            "provide source_candidate_universe.candidate_metrics_path."
        )
    path = Path(universe.candidate_metrics_path)
    if not path.exists():
        raise FileNotFoundError(f"Candidate metrics input does not exist: {path.as_posix()}")
    rows = _load_rows(path)
    missing_required = [column for column in REQUIRED_CANDIDATE_COLUMNS if any(_missing(row.get(column)) for row in rows)]
    if missing_required:
        raise ValueError(f"Candidate metrics input is missing required fields: {missing_required}.")
    optional_metric_names = [
        "global_score",
        "regime_score",
        "transition_score",
        "defensive_score",
        "redundancy_group",
        "correlation_to_selected",
    ]
    for metric in optional_metric_names:
        if all(_missing(row.get(metric)) for row in rows):
            warnings.append(f"Optional candidate metric '{metric}' was not supplied; fallback scoring or relaxed pruning was used where possible.")
    inventory = {
        "input_mode": "file_backed",
        "candidate_metrics_path": _rel(path),
        "row_count": len(rows),
        "columns": sorted({key for row in rows for key in row}),
        "warnings": warnings,
    }
    return rows, canonicalize_value(inventory), warnings


def _load_review_pack_metadata(review_root: Path) -> dict[str, Any]:
    manifest_path = review_root / "manifest.json"
    summary_path = review_root / "review_summary.json"
    if not manifest_path.exists() and not summary_path.exists():
        raise FileNotFoundError(
            f"Regime review pack must contain manifest.json or review_summary.json: {review_root.as_posix()}"
        )
    metadata: dict[str, Any] = {}
    if summary_path.exists():
        metadata.update(_load_json(summary_path))
    if manifest_path.exists():
        metadata.update(_load_json(manifest_path))
    return canonicalize_value(metadata)


def _enrich_candidate_rows(
    rows: list[dict[str, Any]],
    config: RegimeAwareCandidateSelectionConfig,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        item = {str(key): (None if _missing(value) else value) for key, value in row.items()}
        item["candidate_id"] = str(item["candidate_id"])
        item["candidate_name"] = str(item.get("candidate_name") or item["candidate_id"])
        item["artifact_type"] = str(item.get("artifact_type") or "unknown")
        item["source_run_id"] = None if _missing(item.get("source_run_id")) else str(item.get("source_run_id"))
        item["global_score"] = _score_or_fallback(
            item.get("global_score"),
            (
                _scale(item.get("sharpe"), low=-1.0, high=3.0),
                _scale(item.get("total_return"), low=-0.2, high=0.5),
                _drawdown_score(item.get("max_drawdown"), floor=-0.5),
            ),
            weights=(0.45, 0.35, 0.20),
        )
        item["regime_score"] = _score_or_fallback(
            item.get("regime_score"),
            (
                _scale(item.get("regime_sharpe"), low=-1.0, high=3.0),
                _scale(item.get("regime_return"), low=-0.2, high=0.5),
                _scale(item.get("regime_ic"), low=-0.05, high=0.10),
                _scale(item.get("regime_rank_ic"), low=-0.05, high=0.10),
                _drawdown_score(item.get("regime_max_drawdown"), floor=-0.5),
            ),
            weights=(0.30, 0.25, 0.15, 0.15, 0.15),
        )
        item["transition_score"] = _score_or_fallback(
            item.get("transition_score"),
            (
                _scale(item.get("transition_window_return"), low=-0.15, high=0.20),
                _drawdown_score(item.get("transition_window_drawdown"), floor=-0.3),
            ),
            weights=(0.55, 0.45),
        )
        item["defensive_score"] = _score_or_fallback(
            item.get("defensive_score"),
            (
                _drawdown_score(item.get("high_vol_drawdown"), floor=-0.4),
                _inverse_scale(item.get("volatility"), low=0.0, high=0.6),
                _inverse_scale(item.get("correlation_to_selected"), low=0.0, high=1.0),
            ),
            weights=(0.50, 0.25, 0.25),
        )
        item["_input_order"] = index
        enriched.append(item)
    ranked = sorted(enriched, key=lambda item: (-float(item["global_score"]), str(item["candidate_id"])))
    for rank, item in enumerate(ranked, start=1):
        item["global_rank"] = rank
    return sorted(ranked, key=lambda item: str(item["candidate_id"]))


def _build_regime_rows(
    rows: list[dict[str, Any]],
    config: RegimeAwareCandidateSelectionConfig,
) -> list[dict[str, Any]]:
    regime_rows = []
    for row in rows:
        if _missing(row.get("regime_id")) and _missing(row.get("regime_label")):
            continue
        observed_confidence = _float_or_none(row.get("regime_confidence_observed"))
        confidence_weight = 1.0
        if observed_confidence is not None:
            confidence_weight = observed_confidence
        score = float(row["regime_score"])
        regime_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "candidate_name": row["candidate_name"],
                "artifact_type": row["artifact_type"],
                "regime_id": row.get("regime_id"),
                "regime_label": row.get("regime_label"),
                "observation_count": row.get("observation_count"),
                "regime_score": score,
                "regime_rank": None,
                "regime_return": row.get("regime_return"),
                "regime_sharpe": row.get("regime_sharpe"),
                "regime_max_drawdown": row.get("regime_max_drawdown"),
                "regime_ic": row.get("regime_ic"),
                "regime_rank_ic": row.get("regime_rank_ic"),
                "regime_hit_rate": row.get("regime_hit_rate"),
                "transition_adjacent": row.get("transition_adjacent"),
                "confidence_weighted_score": _round_score(score * confidence_weight),
                "selected_for_regime": False,
                "selection_reason": "Not selected for this regime.",
            }
        )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in regime_rows:
        grouped.setdefault(str(row.get("regime_id") or row.get("regime_label") or ""), []).append(row)
    for group_rows in grouped.values():
        for rank, row in enumerate(
            sorted(group_rows, key=lambda item: (-float(item["regime_score"]), str(item["candidate_id"]))),
            start=1,
        ):
            row["regime_rank"] = rank
    return sorted(regime_rows, key=lambda item: (str(item.get("regime_id")), int(item["regime_rank"] or 0), str(item["candidate_id"])))


def _build_category_assignments(
    rows: list[dict[str, Any]],
    regime_rows: list[dict[str, Any]],
    config: RegimeAwareCandidateSelectionConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assignments: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    by_id = {row["candidate_id"]: row for row in rows}
    cat = config.selection_categories
    if cat.global_performer.enabled:
        candidates = [row for row in rows if float(row["global_score"]) >= cat.global_performer.min_global_score]
        selected.extend(_take_category(candidates, "global_performer", "global_score", cat.global_performer.max_candidates))
    if cat.regime_specialist.enabled:
        by_regime: dict[str, list[dict[str, Any]]] = {}
        for regime_row in regime_rows:
            obs = _float_or_none(regime_row.get("observation_count"))
            if float(regime_row["regime_score"]) < cat.regime_specialist.min_regime_score:
                continue
            if obs is not None and obs < cat.regime_specialist.min_regime_observations:
                continue
            by_regime.setdefault(str(regime_row.get("regime_id") or regime_row.get("regime_label")), []).append(regime_row)
        for regime_key in sorted(by_regime):
            chosen = sorted(by_regime[regime_key], key=lambda item: (-float(item["regime_score"]), str(item["candidate_id"])))[: cat.regime_specialist.max_candidates_per_regime]
            for item in chosen:
                selected.append(
                    {
                        **by_id[str(item["candidate_id"])],
                        "category": "regime_specialist",
                        "category_score": float(item["regime_score"]),
                        "regime_id": item.get("regime_id"),
                    }
                )
    if cat.transition_resilient.enabled:
        candidates = [
            row
            for row in rows
            if float(row["transition_score"]) >= cat.transition_resilient.min_transition_score
            and (
                _float_or_none(row.get("transition_window_drawdown")) is None
                or _float_or_none(row.get("transition_window_drawdown")) >= cat.transition_resilient.max_transition_drawdown
            )
        ]
        selected.extend(_take_category(candidates, "transition_resilient", "transition_score", cat.transition_resilient.max_candidates))
    if cat.defensive_fallback.enabled:
        candidates = [
            row
            for row in rows
            if float(row["defensive_score"]) >= 0.0
            and (
                float(row["defensive_score"]) >= 0.55
                or _float_or_none(row.get("high_vol_drawdown")) is not None
                and _float_or_none(row.get("high_vol_drawdown")) >= cat.defensive_fallback.max_high_vol_drawdown
            )
        ]
        candidates = [
            row
            for row in candidates
            if _float_or_none(row.get("correlation_to_selected")) is None
            or _float_or_none(row.get("correlation_to_selected")) <= cat.defensive_fallback.max_correlation_to_selected
        ]
        selected.extend(_take_category(candidates, "defensive_fallback", "defensive_score", cat.defensive_fallback.max_candidates))

    selected_keys = {(item["candidate_id"], item["category"], item.get("regime_id")) for item in selected}
    for category in CATEGORIES:
        category_items = [item for item in selected if item["category"] == category]
        ranked = sorted(category_items, key=lambda item: (-float(item["category_score"]), str(item["candidate_id"]), str(item.get("regime_id") or "")))
        for rank, item in enumerate(ranked, start=1):
            assignments.append(_category_row(item, rank, True))
    for row in rows:
        for category, score_name in (
            ("global_performer", "global_score"),
            ("transition_resilient", "transition_score"),
            ("defensive_fallback", "defensive_score"),
        ):
            if (row["candidate_id"], category, None) not in selected_keys:
                assignments.append(_category_row({**row, "category": category, "category_score": row[score_name]}, None, False))
    return (
        sorted(assignments, key=lambda item: (str(item["category"]), int(item["category_rank"] or 999999), str(item["candidate_id"]))),
        selected,
    )


def _take_category(rows: list[dict[str, Any]], category: str, score_name: str, limit: int) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda item: (-float(item[score_name]), str(item["candidate_id"])))
    return [{**item, "category": category, "category_score": float(item[score_name]), "regime_id": None} for item in ranked[:limit]]


def _apply_redundancy(
    selected: list[dict[str, Any]],
    config: RegimeAwareCandidateSelectionConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    if not config.redundancy.enabled:
        rows = [
            _redundancy_row(item, action="not_applied", selected=True, reason="Redundancy pruning disabled by config.")
            for item in selected
        ]
        return selected, rows, []
    report: list[dict[str, Any]] = []
    retained: list[dict[str, Any]] = []
    warnings: list[str] = []
    has_group = any(not _missing(item.get("redundancy_group")) for item in selected)
    has_corr = any(_float_or_none(item.get("correlation_to_selected")) is not None for item in selected)
    if not has_group and not has_corr:
        warnings.append("No redundancy_group or correlation_to_selected data was supplied; correlation pruning was not applied.")
    seen_groups: dict[tuple[str, str], dict[str, Any]] = {}
    retained_ids_by_category: dict[str, set[str]] = {category: set() for category in CATEGORIES}
    for item in sorted(selected, key=lambda value: (str(value["category"]), -float(value["category_score"]), str(value["candidate_id"]), str(value.get("regime_id") or ""))):
        category = str(item["category"])
        group = None if _missing(item.get("redundancy_group")) else str(item["redundancy_group"])
        group_key = (category if config.redundancy.apply_within_category else "__all__", group or f"__candidate__:{item['candidate_id']}")
        if group is not None and group_key in seen_groups:
            winner = seen_groups[group_key]
            report.append(_redundancy_row(item, compared_to=winner["candidate_id"], action="pruned", selected=False, reason=f"Pruned redundant group '{group}' behind higher-ranked candidate."))
            continue
        corr = _float_or_none(item.get("correlation_to_selected"))
        if corr is not None and corr > config.redundancy.max_pairwise_correlation and config.redundancy.apply_across_categories:
            report.append(_redundancy_row(item, action="pruned", selected=False, metric="correlation_to_selected", value=corr, threshold=config.redundancy.max_pairwise_correlation, reason="Pruned because correlation_to_selected exceeded redundancy threshold."))
            continue
        retained.append(item)
        retained_ids_by_category.setdefault(category, set()).add(str(item["candidate_id"]))
        if group is not None:
            seen_groups[group_key] = item
        report.append(_redundancy_row(item, action="retained", selected=True, reason="Retained after deterministic redundancy checks."))
    return retained, sorted(report, key=lambda item: (str(item["category"]), str(item["candidate_id"]), str(item["redundancy_action"]))), warnings


def _build_candidate_selection_rows(
    rows: list[dict[str, Any]],
    selected_entries: list[dict[str, Any]],
    *,
    config: RegimeAwareCandidateSelectionConfig,
    source_review_run_id: str,
    source_benchmark_run_id: str,
) -> list[dict[str, Any]]:
    selected_by_id: dict[str, list[dict[str, Any]]] = {}
    for entry in selected_entries:
        selected_by_id.setdefault(str(entry["candidate_id"]), []).append(entry)
    output = []
    for row in sorted(rows, key=lambda item: (int(item["global_rank"]), str(item["candidate_id"]))):
        entries = selected_by_id.get(str(row["candidate_id"]), [])
        categories = [str(entry["category"]) for entry in entries]
        regimes = sorted({str(entry.get("regime_id")) for entry in entries if entry.get("regime_id") is not None})
        output.append(
            {
                "candidate_id": row["candidate_id"],
                "candidate_name": row["candidate_name"],
                "artifact_type": row["artifact_type"],
                "source_run_id": row.get("source_run_id"),
                "global_rank": row.get("global_rank"),
                "global_score": row.get("global_score"),
                "selected": bool(entries),
                "selection_category": "|".join(categories) if categories else None,
                "primary_regime": regimes[0] if regimes else row.get("regime_id"),
                "secondary_regimes": "|".join(regimes[1:]) if len(regimes) > 1 else None,
                "regime_specialist_score": row.get("regime_score"),
                "transition_resilience_score": row.get("transition_score"),
                "defensive_score": row.get("defensive_score"),
                "regime_confidence_required": config.regime_context.min_regime_confidence,
                "regime_confidence_observed": row.get("regime_confidence_observed"),
                "redundancy_group": row.get("redundancy_group"),
                "redundancy_action": "retained" if entries else "not_selected",
                "selection_reason": (
                    f"Selected for {', '.join(categories)}."
                    if categories
                    else "Not selected by configured category thresholds, limits, or redundancy pruning."
                ),
                "source_review_run_id": source_review_run_id,
                "source_benchmark_run_id": source_benchmark_run_id,
            }
        )
    return output


def _build_transition_rows(rows: list[dict[str, Any]], selected_lookup: set[tuple[str, str, Any]]) -> list[dict[str, Any]]:
    selected_ids = {candidate_id for candidate_id, category, _ in selected_lookup if category == "transition_resilient"}
    return [
        {
            "candidate_id": row["candidate_id"],
            "candidate_name": row["candidate_name"],
            "artifact_type": row["artifact_type"],
            "transition_score": row["transition_score"],
            "transition_window_return": row.get("transition_window_return"),
            "transition_window_drawdown": row.get("transition_window_drawdown"),
            "transition_window_observation_count": row.get("transition_window_observation_count"),
            "selected": row["candidate_id"] in selected_ids,
            "selection_reason": "Selected as transition-resilient." if row["candidate_id"] in selected_ids else "Not selected as transition-resilient.",
        }
        for row in sorted(rows, key=lambda item: (-float(item["transition_score"]), str(item["candidate_id"])))
    ]


def _build_defensive_rows(rows: list[dict[str, Any]], selected_lookup: set[tuple[str, str, Any]]) -> list[dict[str, Any]]:
    selected_ids = {candidate_id for candidate_id, category, _ in selected_lookup if category == "defensive_fallback"}
    return [
        {
            "candidate_id": row["candidate_id"],
            "candidate_name": row["candidate_name"],
            "artifact_type": row["artifact_type"],
            "defensive_score": row["defensive_score"],
            "high_vol_drawdown": row.get("high_vol_drawdown"),
            "volatility": row.get("volatility"),
            "correlation_to_selected": row.get("correlation_to_selected"),
            "selected": row["candidate_id"] in selected_ids,
            "selection_reason": "Selected as defensive fallback." if row["candidate_id"] in selected_ids else "Not selected as defensive fallback.",
        }
        for row in sorted(rows, key=lambda item: (-float(item["defensive_score"]), str(item["candidate_id"])))
    ]


def _build_allocation_hints(
    selected_entries: list[dict[str, Any]],
    config: RegimeAwareCandidateSelectionConfig,
) -> dict[str, Any]:
    budgets = config.allocation_hints.default_category_budget
    counts = {category: sum(1 for entry in selected_entries if entry["category"] == category) for category in CATEGORIES}
    hints = []
    if config.allocation_hints.write_category_weight_hints:
        for entry in sorted(selected_entries, key=lambda item: (str(item["category"]), str(item["candidate_id"]), str(item.get("regime_id") or ""))):
            count = counts.get(str(entry["category"]), 0)
            hints.append(
                {
                    "candidate_id": entry["candidate_id"],
                    "category": entry["category"],
                    "suggested_category_budget": budgets.get(str(entry["category"]), 0.0),
                    "within_category_weight_hint": None if count == 0 else _round_score(1.0 / count),
                }
            )
    return {
        "category_budgets": budgets,
        "selected_candidate_count_by_category": counts,
        "candidate_weight_hints": hints,
    }


def _build_summary(**kwargs: Any) -> dict[str, Any]:
    selected_entries = kwargs["selected_entries"]
    by_category = {category: sum(1 for entry in selected_entries if entry["category"] == category) for category in CATEGORIES}
    top = {
        category: [
            {"candidate_id": entry["candidate_id"], "category_score": entry["category_score"]}
            for entry in sorted(
                [item for item in selected_entries if item["category"] == category],
                key=lambda item: (-float(item["category_score"]), str(item["candidate_id"])),
            )[:5]
        ]
        for category in CATEGORIES
    }
    return canonicalize_value(
        {
            "selection_run_id": kwargs["selection_run_id"],
            "selection_name": kwargs["config"].selection_name,
            "source_review_run_id": kwargs["source_review_run_id"],
            "source_benchmark_run_id": kwargs["source_benchmark_run_id"],
            "candidate_count": kwargs["candidate_count"],
            "selected_count": len({entry["candidate_id"] for entry in selected_entries}),
            "selected_count_by_category": by_category,
            "rejected_or_not_selected_count": max(0, kwargs["candidate_count"] - len({entry["candidate_id"] for entry in selected_entries})),
            "warning_count": len(kwargs["limitations"]),
            "source_review_pack_path": _rel(kwargs["source_review_pack_path"]),
            "output_root": _rel(kwargs["output_dir"]),
            "generated_artifacts": kwargs["generated_files"],
            "top_candidates_by_category": top,
            "limitations": kwargs["limitations"],
        }
    )


def _build_manifest(**kwargs: Any) -> dict[str, Any]:
    selected_entries = kwargs["selected_entries"]
    category_counts = {category: sum(1 for entry in selected_entries if entry["category"] == category) for category in CATEGORIES}
    return canonicalize_value(
        {
            "artifact_type": "regime_aware_candidate_selection",
            "schema_version": 1,
            "selection_run_id": kwargs["selection_run_id"],
            "selection_name": kwargs["config"].selection_name,
            "source_review_run_id": kwargs["source_review_run_id"],
            "source_benchmark_run_id": kwargs["source_benchmark_run_id"],
            "source_paths": {
                "source_review_pack": _rel(kwargs["review_root"]),
                "candidate_metrics_path": kwargs["inventory"].get("candidate_metrics_path"),
            },
            "generated_files": kwargs["generated_files"],
            "row_counts": kwargs["row_counts"],
            "selected_counts": {
                "unique_candidates": len({entry["candidate_id"] for entry in selected_entries}),
                "category_assignments": len(selected_entries),
            },
            "category_counts": category_counts,
            "warning_count": kwargs["warning_count"],
            "file_inventory": {
                name: {"path": name, "rows": _row_count(kwargs["output_dir"] / name)}
                for name in kwargs["generated_files"]
            },
        }
    )


def _category_row(item: dict[str, Any], rank: int | None, selected: bool) -> dict[str, Any]:
    metric_names = _source_metric_names(item["category"])
    return {
        "candidate_id": item["candidate_id"],
        "candidate_name": item["candidate_name"],
        "artifact_type": item["artifact_type"],
        "category": item["category"],
        "category_rank": rank,
        "category_score": item["category_score"],
        "selected": selected,
        "primary_reason": "Selected by category threshold and limit." if selected else "Not selected by category threshold, limit, or redundancy pruning.",
        "warning_count": 0,
        "failed_gate_count": 0,
        "source_metric_names": "|".join(metric_names),
        "source_metric_values": json.dumps({name: item.get(name) for name in metric_names}, sort_keys=True),
    }


def _mark_category_rows(rows: list[dict[str, Any]], selected_lookup: set[tuple[str, str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        if (row["candidate_id"], row["category"], None) in selected_lookup:
            row["selected"] = True
            row["primary_reason"] = "Selected by category threshold and retained after redundancy pruning."
        elif any(candidate_id == row["candidate_id"] and category == row["category"] for candidate_id, category, _ in selected_lookup):
            row["selected"] = True
            row["primary_reason"] = "Selected by category threshold and retained after redundancy pruning."
    return rows


def _mark_regime_rows(rows: list[dict[str, Any]], selected_lookup: set[tuple[str, str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        key = (row["candidate_id"], "regime_specialist", row.get("regime_id"))
        if key in selected_lookup:
            row["selected_for_regime"] = True
            row["selection_reason"] = "Selected as regime specialist."
    return rows


def _redundancy_row(
    item: dict[str, Any],
    *,
    compared_to: str | None = None,
    action: str,
    selected: bool,
    reason: str,
    metric: str = "redundancy_group",
    value: Any = None,
    threshold: Any = None,
) -> dict[str, Any]:
    return {
        "candidate_id": item["candidate_id"],
        "candidate_name": item["candidate_name"],
        "category": item["category"],
        "redundancy_group": item.get("redundancy_group"),
        "compared_to_candidate_id": compared_to,
        "redundancy_metric": metric,
        "redundancy_value": value if value is not None else item.get("redundancy_group"),
        "threshold": threshold,
        "redundancy_action": action,
        "selected": selected,
        "reason": reason,
    }


def _source_metric_names(category: str) -> tuple[str, ...]:
    return {
        "global_performer": ("global_score", "sharpe", "total_return", "max_drawdown"),
        "regime_specialist": ("regime_score", "regime_sharpe", "regime_return", "regime_ic", "regime_rank_ic", "regime_max_drawdown"),
        "transition_resilient": ("transition_score", "transition_window_return", "transition_window_drawdown"),
        "defensive_fallback": ("defensive_score", "high_vol_drawdown", "volatility", "correlation_to_selected"),
    }[category]


def _build_selection_run_id(
    *,
    config: RegimeAwareCandidateSelectionConfig,
    source_review_run_id: str,
    source_benchmark_run_id: str,
    candidate_ids: list[str],
) -> str:
    payload = {
        "config": config.to_dict(),
        "source_review_run_id": source_review_run_id,
        "source_benchmark_run_id": source_benchmark_run_id,
        "candidate_ids": sorted(candidate_ids),
    }
    digest = hashlib.sha256(serialize_canonical_json(payload).encode("utf-8")).hexdigest()[:12]
    return f"{config.selection_name}_{digest}"


def _score_or_fallback(value: Any, components: tuple[float | None, ...], weights: tuple[float, ...]) -> float:
    explicit = _float_or_none(value)
    if explicit is not None:
        return _round_score(min(max(explicit, 0.0), 1.0))
    available = [(component, weight) for component, weight in zip(components, weights, strict=True) if component is not None]
    if not available:
        return 0.0
    total_weight = sum(weight for _component, weight in available)
    return _round_score(sum(float(component) * weight for component, weight in available) / total_weight)


def _scale(value: Any, *, low: float, high: float) -> float | None:
    number = _float_or_none(value)
    if number is None:
        return None
    if high == low:
        return None
    return min(max((number - low) / (high - low), 0.0), 1.0)


def _inverse_scale(value: Any, *, low: float, high: float) -> float | None:
    scaled = _scale(value, low=low, high=high)
    return None if scaled is None else 1.0 - scaled


def _drawdown_score(value: Any, *, floor: float) -> float | None:
    number = _float_or_none(value)
    if number is None:
        return None
    return min(max((number - floor) / (0.0 - floor), 0.0), 1.0)


def _float_or_none(value: Any) -> float | None:
    if _missing(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _round_score(value: float) -> float:
    return round(float(value), 12)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = _load_json(path)
        rows = payload.get("rows") if isinstance(payload, Mapping) else None
        if not isinstance(rows, list):
            raise ValueError(f"Candidate metrics JSON must contain a rows array: {path.as_posix()}")
        return [dict(row) for row in rows if isinstance(row, Mapping)]
    frame = pd.read_csv(path, keep_default_na=True)
    return [{str(key): (None if _missing(value) else value) for key, value in row.items()} for row in frame.to_dict(orient="records")]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path.as_posix()}")
    return payload


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(canonicalize_value(payload), indent=2, sort_keys=True), encoding="utf-8", newline="\n")


def _row_count(path: Path) -> int | None:
    if not path.exists():
        return None
    if path.suffix.lower() == ".csv":
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return 0
        return max(text.count("\n") - 1, 0)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            return len(payload["rows"])
    return None


def _require_dir(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path.as_posix()}")
    if not path.is_dir():
        raise ValueError(f"{label} must be a directory: {path.as_posix()}")


def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


__all__ = [
    "RegimeAwareCandidateSelectionRunResult",
    "run_regime_aware_candidate_selection",
]
