"""Deterministic ranking and selection of candidates."""

from __future__ import annotations

from typing import Any

from src.research.candidate_selection.schema import CandidateRecord

# Metrics that are "higher is better"
_DESC_METRICS = {
    "mean_ic",
    "ic_ir",
    "mean_rank_ic",
    "rank_ic_ir",
    "n_periods",
    "sharpe_ratio",
    "annualized_return",
    "total_return",
}

# Metrics that are "lower is better"
_ASC_METRICS = {
    "max_drawdown",
    "average_turnover",
}


def rank_candidates(
    candidates: list[CandidateRecord],
    *,
    primary_metric: str = "ic_ir",
) -> list[CandidateRecord]:
    """Deterministically rank candidates using a primary metric and tiebreakers.
    
    Args:
        candidates: List of candidates to rank (in any order).
        primary_metric: Primary ranking metric. Defaults to "ic_ir".
    
    Returns:
        List of CandidateRecord objects with selection_rank assigned (1-indexed).
    
    Raises:
        ValueError: If primary_metric is unknown or no candidates provided.
    """
    if not candidates:
        return []

    if primary_metric not in {"mean_ic", "ic_ir", "mean_rank_ic", "rank_ic_ir"}:
        raise ValueError(f"Unknown primary_metric: {primary_metric}")

    ranked = sorted(
        candidates,
        key=lambda c: _sort_key(c, primary_metric=primary_metric),
    )

    return [
        CandidateRecord(
            candidate_id=c.candidate_id,
            alpha_name=c.alpha_name,
            alpha_run_id=c.alpha_run_id,
            sleeve_run_id=c.sleeve_run_id,
            mapping_name=c.mapping_name,
            dataset=c.dataset,
            timeframe=c.timeframe,
            evaluation_horizon=c.evaluation_horizon,
            mean_ic=c.mean_ic,
            ic_ir=c.ic_ir,
            mean_rank_ic=c.mean_rank_ic,
            rank_ic_ir=c.rank_ic_ir,
            n_periods=c.n_periods,
            sharpe_ratio=c.sharpe_ratio,
            annualized_return=c.annualized_return,
            total_return=c.total_return,
            max_drawdown=c.max_drawdown,
            average_turnover=c.average_turnover,
            selection_rank=rank,
            promotion_status=c.promotion_status,
            review_status=c.review_status,
            artifact_path=c.artifact_path,
        )
        for rank, c in enumerate(ranked, start=1)
    ]


def _sort_key(candidate: CandidateRecord, *, primary_metric: str) -> tuple[Any, ...]:
    """Return deterministic sort key for candidate."""
    
    # Build ordered metric list with primary first
    metric_order = [
        primary_metric,
        *[m for m in ("ic_ir", "mean_ic", "mean_rank_ic", "rank_ic_ir") if m != primary_metric],
    ]

    sort_parts: list[Any] = []
    
    # Add each metric in order
    for metric in metric_order:
        value = getattr(candidate, metric)
        sort_parts.extend(_metric_sort_components(value, metric))

    # Add final deterministic tiebreakers
    sort_parts.extend([
        candidate.mapping_name is None,
        "" if candidate.mapping_name is None else candidate.mapping_name,
        candidate.alpha_name,
        candidate.alpha_run_id,
    ])

    return tuple(sort_parts)


def _metric_sort_components(value: float | int | None, metric: str) -> tuple[bool, float]:
    """Return sort components for a metric value.
    
    Returns tuple of (is_null, normalized_value).
    None values sort last. Direction depends on metric type.
    """
    
    # Null values always sort last
    if value is None:
        return (True, 0.0)

    normalized = float(value)
    
    # Determine sort direction
    if metric in _DESC_METRICS:
        # Higher is better: negate for descending sort
        return (False, -normalized)
    elif metric in _ASC_METRICS:
        # Lower is better: use as-is for ascending sort
        return (False, normalized)
    else:
        # Unknown metric: treat as desc (safest default)
        return (False, -normalized)


def select_top_candidates(
    ranked_candidates: list[CandidateRecord],
    *,
    max_count: int | None = None,
) -> list[CandidateRecord]:
    """Select top N candidates from ranked list.
    
    Args:
        ranked_candidates: Already-ranked candidates (from rank_candidates).
        max_count: Maximum number of candidates to select. None = select all.
    
    Returns:
        Subset of ranked candidates (selection_rank preserved).
    """
    if max_count is None:
        return ranked_candidates
    
    return ranked_candidates[:max_count]
