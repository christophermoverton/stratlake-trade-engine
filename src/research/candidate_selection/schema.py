"""Canonical schema for candidate selection records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CandidateRecord:
    """Immutable record representing one candidate alpha sleeve.
    
    This schema is the canonical representation used throughout the candidate
    selection pipeline and serves as the basis for ranking, filtering, and
    allocation workflows in Milestone 15.
    """

    # Identification
    candidate_id: str
    """Unique deterministic identifier for candidate, e.g., alpha_run_id."""

    alpha_name: str
    """Alpha model name."""

    alpha_run_id: str
    """Run ID of the alpha evaluation."""

    sleeve_run_id: str | None
    """Run ID of the sleeve (implementation), if created."""

    mapping_name: str | None
    """Name of signal mapping applied."""

    # Dataset Context
    dataset: str
    """Feature dataset used for training and evaluation."""

    timeframe: str
    """Timeframe (e.g., 'daily')."""

    # Evaluation Context
    evaluation_horizon: int
    """Forward return horizon in bars."""

    # Forecast Quality Metrics (all higher-is-better)
    mean_ic: float | None
    """Mean Information Coefficient."""

    ic_ir: float | None
    """Information Ratio of IC series (primary ranking metric)."""

    mean_rank_ic: float | None
    """Mean Rank Information Coefficient."""

    rank_ic_ir: float | None
    """Information Ratio of Rank IC series."""

    n_periods: int | None
    """Number of evaluation periods."""

    # Implementation/Sleeve Metrics (mixed direction)
    sharpe_ratio: float | None
    """Sharpe ratio of sleeve returns."""

    annualized_return: float | None
    """Annualized return of sleeve implementation."""

    total_return: float | None
    """Total return over evaluation period."""

    max_drawdown: float | None
    """Maximum drawdown observed."""

    average_turnover: float | None
    """Average turnover of sleeve."""

    # Selection and Status
    selection_rank: int
    """Rank position after deterministic ranking (1-indexed)."""

    promotion_status: str
    """Promotion gate status: eligible, blocked, promoted, etc."""

    review_status: str
    """Review status: candidate, promoted, rejected, needs_review."""

    # Provenance
    artifact_path: str
    """Path to source alpha evaluation artifact directory."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for CSV/JSON persistence."""
        return {
            "candidate_id": self.candidate_id,
            "alpha_name": self.alpha_name,
            "alpha_run_id": self.alpha_run_id,
            "sleeve_run_id": self.sleeve_run_id,
            "mapping_name": self.mapping_name,
            "dataset": self.dataset,
            "timeframe": self.timeframe,
            "evaluation_horizon": self.evaluation_horizon,
            "mean_ic": self.mean_ic,
            "ic_ir": self.ic_ir,
            "mean_rank_ic": self.mean_rank_ic,
            "rank_ic_ir": self.rank_ic_ir,
            "n_periods": self.n_periods,
            "sharpe_ratio": self.sharpe_ratio,
            "annualized_return": self.annualized_return,
            "total_return": self.total_return,
            "max_drawdown": self.max_drawdown,
            "average_turnover": self.average_turnover,
            "selection_rank": self.selection_rank,
            "promotion_status": self.promotion_status,
            "review_status": self.review_status,
            "artifact_path": self.artifact_path,
        }

    @staticmethod
    def csv_columns() -> list[str]:
        """Return ordered column list for CSV export."""
        return [
            "selection_rank",
            "candidate_id",
            "alpha_name",
            "alpha_run_id",
            "sleeve_run_id",
            "mapping_name",
            "dataset",
            "timeframe",
            "evaluation_horizon",
            "mean_ic",
            "ic_ir",
            "mean_rank_ic",
            "rank_ic_ir",
            "n_periods",
            "sharpe_ratio",
            "annualized_return",
            "total_return",
            "max_drawdown",
            "average_turnover",
            "promotion_status",
            "review_status",
            "artifact_path",
        ]
