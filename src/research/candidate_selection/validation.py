"""Validation logic for candidate selection."""

from __future__ import annotations

import math

from src.research.candidate_selection.schema import CandidateRecord


class CandidateValidationError(ValueError):
    """Raised when candidate validation fails."""


def validate_candidate_universe(candidates: list[CandidateRecord]) -> None:
    """Validate a list of candidates for data quality and uniqueness.
    
    Checks:
        - No duplicate candidate IDs
        - Required fields present
        - Ranking metrics are finite where required
        - Selection ranks assigned correctly
    
    Raises:
        CandidateValidationError: If validation fails.
    """
    if not candidates:
        return

    # Check for duplicate IDs
    seen_ids = set()
    for c in candidates:
        if c.candidate_id in seen_ids:
            raise CandidateValidationError(
                f"Duplicate candidate_id: {c.candidate_id}"
            )
        seen_ids.add(c.candidate_id)

    # Check that required fields are present
    for c in candidates:
        if not c.alpha_name or not c.alpha_name.strip():
            raise CandidateValidationError(
                f"Candidate {c.candidate_id} missing alpha_name"
            )
        if not c.alpha_run_id or not c.alpha_run_id.strip():
            raise CandidateValidationError(
                f"Candidate {c.candidate_id} missing alpha_run_id"
            )
        if not c.dataset or not c.dataset.strip():
            raise CandidateValidationError(
                f"Candidate {c.candidate_id} missing dataset"
            )
        if not c.timeframe or not c.timeframe.strip():
            raise CandidateValidationError(
                f"Candidate {c.candidate_id} missing timeframe"
            )
        if c.evaluation_horizon <= 0:
            raise CandidateValidationError(
                f"Candidate {c.candidate_id} has invalid evaluation_horizon: {c.evaluation_horizon}"
            )

    # Check that ranking metrics are finite (if provided)
    for c in candidates:
        for metric_name in ("ic_ir", "mean_ic", "mean_rank_ic", "rank_ic_ir"):
            value = getattr(c, metric_name)
            if value is not None:
                if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
                    raise CandidateValidationError(
                        f"Candidate {c.candidate_id} has invalid {metric_name}: {value}"
                    )

    # Check that sleeve metrics are finite (if provided)
    for c in candidates:
        for metric_name in ("sharpe_ratio", "annualized_return", "total_return", "max_drawdown", "average_turnover"):
            value = getattr(c, metric_name)
            if value is not None:
                if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
                    raise CandidateValidationError(
                        f"Candidate {c.candidate_id} has invalid {metric_name}: {value}"
                    )


def validate_ranked_universe(ranked_candidates: list[CandidateRecord]) -> None:
    """Validate that candidates are correctly ranked with valid rank assignments.
    
    Checks:
        - selection_rank is 1-indexed and sequential
        - No rank is zero or negative
        - Ranks match list position
    
    Raises:
        CandidateValidationError: If validation fails.
    """
    if not ranked_candidates:
        return

    validate_candidate_universe(ranked_candidates)

    for idx, candidate in enumerate(ranked_candidates, start=1):
        if candidate.selection_rank != idx:
            raise CandidateValidationError(
                f"Candidate at position {idx} has incorrect rank {candidate.selection_rank}"
            )
        if candidate.selection_rank <= 0:
            raise CandidateValidationError(
                f"Invalid selection_rank {candidate.selection_rank} for {candidate.candidate_id}"
            )
