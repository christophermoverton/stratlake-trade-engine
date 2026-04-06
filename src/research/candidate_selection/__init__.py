"""Candidate selection module for Milestone 15.

This module implements the core deterministic candidate selection pipeline
that bridges alpha evaluation/review to portfolio allocation.

Main public API:
    - load_candidate_universe(): Load candidates from alpha registry with filters
    - rank_candidates(): Deterministically rank candidates
    - select_top_candidates(): Select subset of ranked candidates
    - run_candidate_selection(): End-to-end pipeline
    - CandidateRecord: Canonical candidate data schema
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.research.candidate_selection.loader import (
    CandidateSelectionError,
    load_candidate_universe,
)
from src.research.candidate_selection.persistence import (
    CandidatePersistenceError,
    build_candidate_selection_run_id,
    resolve_candidate_selection_artifact_dir,
    write_candidate_selection_artifacts,
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
    "CandidateRecord",
    "CandidateSelectionError",
    "CandidatePersistenceError",
    "CandidateValidationError",
    "load_candidate_universe",
    "rank_candidates",
    "select_top_candidates",
    "build_candidate_selection_run_id",
    "resolve_candidate_selection_artifact_dir",
    "write_candidate_selection_artifacts",
    "validate_candidate_universe",
    "validate_ranked_universe",
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
) -> dict[str, Any]:
    """Execute end-to-end candidate selection pipeline.
    
    Loads candidates from registry, ranks, selects, validates, and persists artifacts.
    
    Args:
        artifacts_root: Root of alpha evaluation artifacts.
        alpha_name: Optional alpha name filter.
        dataset: Optional dataset filter.
        timeframe: Optional timeframe filter.
        evaluation_horizon: Optional evaluation horizon filter.
        mapping_name: Optional mapping name filter.
        primary_metric: Primary ranking metric (default: ic_ir).
        max_candidate_count: Max candidates to select (None = all).
        output_artifacts_root: Root for candidate selection artifacts.
    
    Returns:
        Result dict with run_id, counts, artifact paths.
    
    Raises:
        CandidateSelectionError: If loading/processing fails.
        CandidateValidationError: If validation fails.
    """
    # Load candidates from registry
    filters = {
        "alpha_name": alpha_name,
        "dataset": dataset,
        "timeframe": timeframe,
        "evaluation_horizon": evaluation_horizon,
        "mapping_name": mapping_name,
    }

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

    # Rank candidates
    ranked_universe = rank_candidates(universe, primary_metric=primary_metric)
    validate_ranked_universe(ranked_universe)

    # Select top N
    selected = select_top_candidates(ranked_universe, max_count=max_candidate_count)

    # Persist artifacts
    run_id = build_candidate_selection_run_id(
        filters=filters,
        candidate_ids=[c.candidate_id for c in ranked_universe],
        primary_metric=primary_metric,
    )

    universe_csv, selected_csv, summary_json, manifest_json = write_candidate_selection_artifacts(
        universe=ranked_universe,
        selected=selected,
        run_id=run_id,
        filters=filters,
        primary_metric=primary_metric,
        artifacts_root=output_artifacts_root,
    )

    return {
        "run_id": run_id,
        "universe_count": len(ranked_universe),
        "selected_count": len(selected),
        "primary_metric": primary_metric,
        "filters": filters,
        "universe_csv": str(universe_csv),
        "selected_csv": str(selected_csv),
        "summary_json": str(summary_json),
        "manifest_json": str(manifest_json),
    }
