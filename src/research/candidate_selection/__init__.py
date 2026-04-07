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
from src.research.candidate_selection.allocation import (
    AllocationConstraints,
    AllocationDecision,
    AllocationError,
    allocate_candidates,
    resolve_allocation_constraints,
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
    write_allocation_artifacts,
    write_redundancy_artifacts,
)
from src.research.candidate_selection.redundancy import (
    RedundancyError,
    RedundancyRejection,
    RedundancyThresholds,
    apply_redundancy_filter,
    resolve_redundancy_thresholds,
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
    "RedundancyThresholds",
    "RedundancyRejection",
    "AllocationConstraints",
    "AllocationDecision",
    # Errors
    "CandidateSelectionError",
    "CandidatePersistenceError",
    "CandidateValidationError",
    "RedundancyError",
    "AllocationError",
    # Loaders
    "load_candidate_universe",
    # Eligibility
    "evaluate_eligibility",
    "filter_by_eligibility",
    "summarize_eligibility",
    "resolve_eligibility_thresholds",
    "resolve_redundancy_thresholds",
    "resolve_allocation_constraints",
    "apply_redundancy_filter",
    "allocate_candidates",
    # Ranking
    "rank_candidates",
    "select_top_candidates",
    # Persistence
    "build_candidate_selection_run_id",
    "resolve_candidate_selection_artifact_dir",
    "write_candidate_selection_artifacts",
    "write_eligibility_artifacts",
    "write_allocation_artifacts",
    "write_redundancy_artifacts",
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
    # Redundancy filtering thresholds
    max_pairwise_correlation: float | None = None,
    min_overlap_observations: int | None = None,
    # Allocation governance
    allocation_method: str = "equal_weight",
    max_weight_per_candidate: float | None = None,
    min_allocation_candidate_count: int | None = None,
    min_allocation_weight: float | None = None,
    allocation_weight_sum_tolerance: float = 1e-12,
    allocation_rounding_decimals: int = 12,
    allocation_enabled: bool = True,
) -> dict[str, Any]:
    """Execute end-to-end candidate selection pipeline with optional eligibility gating.

    Pipeline stages:
        1. Load candidate universe from the alpha evaluation registry.
        2. Evaluate eligibility gates (if any threshold is configured).
        3. Rank eligible candidates deterministically.
        4. Apply deterministic redundancy filtering over ranked eligible candidates.
        5. Select top N candidates from redundancy-filtered candidates.
        6. Validate and persist all artifacts.

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
        max_pairwise_correlation: Absolute pairwise sleeve return correlation threshold.
            None disables redundancy pruning.
        min_overlap_observations: Minimum overlapping timestamps required for
            valid pairwise correlation.

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

    # Stage 4: Redundancy filtering
    redundancy_thresholds = resolve_redundancy_thresholds(
        max_pairwise_correlation=max_pairwise_correlation,
        min_overlap_observations=min_overlap_observations,
    )
    filtered_candidates, redundancy_rejections, correlation_matrix, overlap_matrix, redundancy_summary = (
        apply_redundancy_filter(
            ranked_universe,
            thresholds=redundancy_thresholds,
        )
    )

    # Stage 5: Select top N
    selected = select_top_candidates(filtered_candidates, max_count=max_candidate_count)

    # Stage 6: Persist artifacts
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

    correlation_csv, selected_csv, rejected_csv, summary_json, manifest_json = write_redundancy_artifacts(
        eligible_candidates=ranked_universe,
        selected_candidates=selected,
        rejected_candidates_eligibility=rejected_candidates,
        eligibility_results=eligibility_results,
        redundancy_rejections=redundancy_rejections,
        correlation_matrix=correlation_matrix,
        redundancy_summary={
            **{k: v for k, v in redundancy_summary.items() if k != "thresholds"},
            "redundancy_thresholds": redundancy_thresholds.to_dict(),
            "overlap_matrix_rows": int(len(overlap_matrix.index)),
        },
        run_id=run_id,
        artifacts_root=output_artifacts_root,
    )

    allocation_csv: str | None = None
    allocation_summary: dict[str, Any] = {
        "allocation_enabled": False,
        "allocation_method": None,
        "allocated_candidates": 0,
    }
    allocation_constraints = resolve_allocation_constraints(
        max_weight_per_candidate=max_weight_per_candidate,
        min_candidate_count=min_allocation_candidate_count,
        min_weight_threshold=min_allocation_weight,
        weight_sum_tolerance=allocation_weight_sum_tolerance,
        rounding_decimals=allocation_rounding_decimals,
    )
    if allocation_enabled and selected:
        allocation_decisions, allocation_stage_summary = allocate_candidates(
            selected,
            method=allocation_method,
            constraints=allocation_constraints,
        )
        allocation_csv_path, selected_csv, summary_json, manifest_json = write_allocation_artifacts(
            selected_candidates=selected,
            allocation_decisions=allocation_decisions,
            allocation_summary=allocation_stage_summary,
            allocation_constraints=allocation_constraints.to_dict(),
            run_id=run_id,
            artifacts_root=output_artifacts_root,
        )
        allocated_ids = {decision.candidate_id for decision in allocation_decisions}
        selected = [
            candidate
            for candidate in selected
            if candidate.candidate_id in allocated_ids
        ]
        allocation_csv = str(allocation_csv_path)
        allocation_summary = {
            **allocation_stage_summary,
            "allocation_enabled": True,
        }
    elif allocation_enabled:
        allocation_csv_path, selected_csv, summary_json, manifest_json = write_allocation_artifacts(
            selected_candidates=selected,
            allocation_decisions=[],
            allocation_summary={
                "allocation_method": str(allocation_method),
                "allocated_candidates": 0,
                "constraint_adjusted_candidates": 0,
                "weight_sum": None,
                "weight_min": None,
                "weight_max": None,
                "concentration_hhi": None,
            },
            allocation_constraints=allocation_constraints.to_dict(),
            run_id=run_id,
            artifacts_root=output_artifacts_root,
        )
        allocation_csv = str(allocation_csv_path)
        allocation_summary = {
            "allocation_enabled": True,
            "allocation_method": str(allocation_method),
            "allocated_candidates": 0,
            "constraint_adjusted_candidates": 0,
        }
    else:
        allocation_summary = {
            **allocation_summary,
            "allocation_constraints": allocation_constraints.to_dict(),
        }

    return {
        "run_id": run_id,
        "universe_count": len(universe),
        "eligible_count": len(eligible_candidates),
        "rejected_count": len(rejected_candidates) + len(redundancy_rejections),
        "selected_count": len(selected),
        "primary_metric": primary_metric,
        "filters": filters,
        "thresholds": thresholds.to_dict(),
        "redundancy_thresholds": redundancy_thresholds.to_dict(),
        "allocation_constraints": allocation_constraints.to_dict(),
        "allocation_summary": allocation_summary,
        "eligibility_summary": elig_summary,
        "redundancy_summary": redundancy_summary,
        "universe_csv": str(universe_csv),
        "selected_csv": str(selected_csv),
        "rejected_csv": str(rejected_csv),
        "eligibility_csv": str(eligibility_csv),
        "correlation_csv": str(correlation_csv),
        "allocation_csv": allocation_csv,
        "summary_json": str(summary_json),
        "manifest_json": str(manifest_json),
    }
