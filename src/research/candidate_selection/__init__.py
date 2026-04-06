"""Candidate selection module for Milestone 15.

This module implements the core deterministic candidate selection pipeline
that bridges alpha evaluation/review to portfolio allocation.

Main public API:
    - load_candidate_universe(): Load candidates from alpha registry with filters
    - rank_candidates(): Deterministically rank candidates
    - select_top_candidates(): Select subset of ranked candidates
    - run_candidate_selection(): End-to-end pipeline (with optional eligibility gating)
    - CandidateRecord: Canonical candidate data schema
    - EligibilityThresholds: Configurable threshold configuration
    - EligibilityResult: Per-candidate gate outcome record
    - evaluate_eligibility(): Evaluate all candidates against thresholds
    - filter_by_eligibility(): Split candidates into eligible / rejected
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.research.candidate_selection.eligibility import (
    EligibilityResult,
    evaluate_eligibility,
    filter_by_eligibility,
    summarize_eligibility,
)
from src.research.candidate_selection.gates import (
    EligibilityThresholds,
    resolve_eligibility_thresholds,
)
from src.research.candidate_selection.loader import (
    CandidateSelectionError,
    load_candidate_universe,
)
from src.research.candidate_selection.persistence import (
    CandidatePersistenceError,
    build_candidate_selection_run_id,
    resolve_candidate_selection_artifact_dir,
    write_candidate_selection_artifacts,
    write_eligibility_artifacts,
)
from src.research.candidate_selection.ranker import (
    rank_candidates,
    select_top_candidates,
)
from src.research.candidate_selection.schema import CandidateRecord
from src.research.candidate_selection.validation import (
    CandidateValidationError,
    validate_candidate_universe,
    validate_ranked_universe,
)

__all__ = [
    # Schema
    "CandidateRecord",
    "EligibilityResult",
    "EligibilityThresholds",
    # Errors
    "CandidateSelectionError",
    "CandidatePersistenceError",
    "CandidateValidationError",
    # Loaders
    "load_candidate_universe",
    # Eligibility
    "evaluate_eligibility",
    "filter_by_eligibility",
    "summarize_eligibility",
    "resolve_eligibility_thresholds",
    # Ranking
    "rank_candidates",
    "select_top_candidates",
    # Persistence
    "build_candidate_selection_run_id",
    "resolve_candidate_selection_artifact_dir",
    "write_candidate_selection_artifacts",
    "write_eligibility_artifacts",
    # Validation
    "validate_candidate_universe",
    "validate_ranked_universe",
    # Pipeline
    "run_candidate_selection",
]


def run_candidate_selection(
    *,
    artifacts_root: str | Path = Path("artifacts") / "alpha",
    alpha_name: str | None = None,
    dataset: str | None = None,
    timeframe: str | None = None,
    evaluation_horizon: int | None = None,
    mapping_name: str | None = None,
    primary_metric: str = "ic_ir",
    max_candidate_count: int | None = None,
    output_artifacts_root: str | Path = Path("artifacts") / "candidate_selection",
    # Eligibility gate thresholds (all default to None = disabled)
    min_mean_ic: float | None = None,
    min_mean_rank_ic: float | None = None,
    min_ic_ir: float | None = None,
    min_rank_ic_ir: float | None = None,
    min_history_length: int | None = None,
    require_mean_ic: bool = False,
    require_ic_ir: bool = False,
) -> dict[str, Any]:
    """Execute end-to-end candidate selection pipeline with optional eligibility gating.

    Pipeline stages:
        1. Load candidate universe from the alpha evaluation registry.
        2. Evaluate eligibility gates (if any threshold is configured).
        3. Rank eligible candidates deterministically.
        4. Select top N candidates.
        5. Validate and persist all artifacts.

    When no eligibility thresholds are configured (all ``None``), every
    candidate passes gating and the behavior is identical to Milestone 15
    Issue 1.

    Args:
        artifacts_root: Root of alpha evaluation artifacts.
        alpha_name: Optional alpha name filter.
        dataset: Optional dataset filter.
        timeframe: Optional timeframe filter.
        evaluation_horizon: Optional evaluation horizon filter.
        mapping_name: Optional mapping name filter.
        primary_metric: Primary ranking metric (default: ic_ir).
        max_candidate_count: Max candidates to select after gating (None = all).
        output_artifacts_root: Root for candidate selection artifacts.
        min_mean_ic: Minimum mean IC threshold (None = disabled).
        min_mean_rank_ic: Minimum mean Rank IC threshold (None = disabled).
        min_ic_ir: Minimum IC IR threshold (None = disabled).
        min_rank_ic_ir: Minimum Rank IC IR threshold (None = disabled).
        min_history_length: Minimum observation count (None = disabled).
        require_mean_ic: Fail candidates whose mean_ic is missing.
        require_ic_ir: Fail candidates whose ic_ir is missing.

    Returns:
        Result dict with run_id, counts, artifact paths, and eligibility summary.

    Raises:
        CandidateSelectionError: If loading/processing fails.
        CandidateValidationError: If validation fails.
    """
    filters = {
        "alpha_name": alpha_name,
        "dataset": dataset,
        "timeframe": timeframe,
        "evaluation_horizon": evaluation_horizon,
        "mapping_name": mapping_name,
    }

    # Stage 1: Load candidate universe
    universe = load_candidate_universe(
        artifacts_root=artifacts_root,
        alpha_name=alpha_name,
        dataset=dataset,
        timeframe=timeframe,
        evaluation_horizon=evaluation_horizon,
        mapping_name=mapping_name,
    )

    if not universe:
        raise CandidateSelectionError("No candidates matched filter criteria.")

    validate_candidate_universe(universe)

    # Stage 2: Eligibility gating
    thresholds = resolve_eligibility_thresholds(
        min_mean_ic=min_mean_ic,
        min_mean_rank_ic=min_mean_rank_ic,
        min_ic_ir=min_ic_ir,
        min_rank_ic_ir=min_rank_ic_ir,
        min_history_length=min_history_length,
        require_mean_ic=require_mean_ic,
        require_ic_ir=require_ic_ir,
    )

    eligibility_results = evaluate_eligibility(universe, thresholds)
    eligible_candidates, rejected_candidates = filter_by_eligibility(universe, eligibility_results)
    elig_summary = summarize_eligibility(eligibility_results, thresholds)

    # Stage 3: Rank eligible candidates
    ranked_universe = rank_candidates(eligible_candidates, primary_metric=primary_metric)
    validate_ranked_universe(ranked_universe)

    # Stage 4: Select top N
    selected = select_top_candidates(ranked_universe, max_count=max_candidate_count)

    # Stage 5: Persist artifacts
    run_id = build_candidate_selection_run_id(
        filters=filters,
        candidate_ids=[c.candidate_id for c in universe],  # hash over full universe for stability
        primary_metric=primary_metric,
    )

    # Write base universe artifacts
    universe_csv, _selected_csv, summary_json, manifest_json = write_candidate_selection_artifacts(
        universe=ranked_universe,
        selected=selected,
        run_id=run_id,
        filters=filters,
        primary_metric=primary_metric,
        artifacts_root=output_artifacts_root,
    )

    # Write eligibility artifacts (overwrites selected_candidates.csv with gate-filtered version)
    eligibility_csv, selected_csv, rejected_csv, summary_json = write_eligibility_artifacts(
        eligibility_results=eligibility_results,
        eligible_candidates=ranked_universe,
        rejected_candidates=rejected_candidates,
        eligibility_summary={
            **elig_summary,
            "run_id": run_id,
            "universe_count": len(universe),
            "selected_count": len(selected),
            "primary_metric": primary_metric,
            "filters": filters,
        },
        run_id=run_id,
        artifacts_root=output_artifacts_root,
    )

    return {
        "run_id": run_id,
        "universe_count": len(universe),
        "eligible_count": len(eligible_candidates),
        "rejected_count": len(rejected_candidates),
        "selected_count": len(selected),
        "primary_metric": primary_metric,
        "filters": filters,
        "thresholds": thresholds.to_dict(),
        "eligibility_summary": elig_summary,
        "universe_csv": str(universe_csv),
        "selected_csv": str(selected_csv),
        "rejected_csv": str(rejected_csv),
        "eligibility_csv": str(eligibility_csv),
        "summary_json": str(summary_json),
        "manifest_json": str(manifest_json),
    }
