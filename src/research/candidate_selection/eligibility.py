"""Eligibility gate pipeline stage for candidate selection.

This module implements the eligibility evaluation stage that sits between
candidate universe construction and downstream ranked selection::

    Candidate Universe → Eligibility / Quality Gates → Eligible Candidates → Ranked Selection

Public API:
    - :class:`EligibilityResult` — per-candidate gate outcome record
    - :func:`evaluate_eligibility` — evaluate all candidates against thresholds
    - :func:`filter_by_eligibility` — split into eligible / rejected lists
    - :func:`build_eligibility_rejection_index` — map candidate_id → failed_checks
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.research.candidate_selection.gates import EligibilityThresholds, evaluate_candidate_gates
from src.research.candidate_selection.schema import CandidateRecord


# ---------------------------------------------------------------------------
# Eligibility result schema
# ---------------------------------------------------------------------------

_NO_FAILURES: str = ""
"""Canonical representation of 'no failed checks' in the CSV output."""


@dataclass(frozen=True)
class EligibilityResult:
    """Immutable per-candidate eligibility gate outcome.

    Schema design notes:
        - ``failed_checks`` stores ``"|"``-delimited, deterministically sorted
          gate failure labels, or the empty string for eligible candidates.
        - Threshold snapshot columns record the threshold values that were
          active at evaluation time to make artifacts fully self-describing.
        - Fields that were ``None`` on the candidate are persisted as ``None``
          here; downstream code must not infer eligibility from their absence.
    """

    # Candidate identity
    candidate_id: str
    alpha_name: str
    sleeve_run_id: str | None
    dataset: str
    timeframe: str
    evaluation_horizon: int

    # Gate outcome
    is_eligible: bool
    failed_checks: str
    """``"|"``-delimited sorted failure labels, or ``""`` if eligible."""

    # Observed metric values
    mean_ic: float | None
    mean_rank_ic: float | None
    ic_ir: float | None
    rank_ic_ir: float | None
    history_length: int | None
    """Mirrors ``CandidateRecord.n_periods``."""

    # Threshold snapshot (values active during evaluation)
    threshold_min_mean_ic: float | None
    threshold_min_mean_rank_ic: float | None
    threshold_min_ic_ir: float | None
    threshold_min_rank_ic_ir: float | None
    threshold_min_history_length: int | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for CSV/JSON persistence."""
        return {
            "candidate_id": self.candidate_id,
            "alpha_name": self.alpha_name,
            "sleeve_run_id": self.sleeve_run_id,
            "dataset": self.dataset,
            "timeframe": self.timeframe,
            "evaluation_horizon": self.evaluation_horizon,
            "is_eligible": self.is_eligible,
            "failed_checks": self.failed_checks,
            "mean_ic": self.mean_ic,
            "mean_rank_ic": self.mean_rank_ic,
            "ic_ir": self.ic_ir,
            "rank_ic_ir": self.rank_ic_ir,
            "history_length": self.history_length,
            "threshold_min_mean_ic": self.threshold_min_mean_ic,
            "threshold_min_mean_rank_ic": self.threshold_min_mean_rank_ic,
            "threshold_min_ic_ir": self.threshold_min_ic_ir,
            "threshold_min_rank_ic_ir": self.threshold_min_rank_ic_ir,
            "threshold_min_history_length": self.threshold_min_history_length,
        }

    @staticmethod
    def csv_columns() -> list[str]:
        """Return the ordered column list for CSV export."""
        return [
            "candidate_id",
            "alpha_name",
            "sleeve_run_id",
            "dataset",
            "timeframe",
            "evaluation_horizon",
            "is_eligible",
            "failed_checks",
            "mean_ic",
            "mean_rank_ic",
            "ic_ir",
            "rank_ic_ir",
            "history_length",
            "threshold_min_mean_ic",
            "threshold_min_mean_rank_ic",
            "threshold_min_ic_ir",
            "threshold_min_rank_ic_ir",
            "threshold_min_history_length",
        ]


# ---------------------------------------------------------------------------
# Pipeline stage functions
# ---------------------------------------------------------------------------


def evaluate_eligibility(
    candidates: list[CandidateRecord],
    thresholds: EligibilityThresholds,
) -> list[EligibilityResult]:
    """Evaluate eligibility for all candidates against the configured thresholds.

    Evaluation preserves input order to ensure deterministic, auditable
    processing.  Each candidate is evaluated independently.

    Args:
        candidates: Candidate universe to evaluate (any order).
        thresholds: Gate threshold configuration.

    Returns:
        List of :class:`EligibilityResult` in the same order as ``candidates``.
    """
    results: list[EligibilityResult] = []
    for candidate in candidates:
        failed_checks = evaluate_candidate_gates(candidate, thresholds)
        is_eligible = len(failed_checks) == 0
        results.append(
            EligibilityResult(
                candidate_id=candidate.candidate_id,
                alpha_name=candidate.alpha_name,
                sleeve_run_id=candidate.sleeve_run_id,
                dataset=candidate.dataset,
                timeframe=candidate.timeframe,
                evaluation_horizon=candidate.evaluation_horizon,
                is_eligible=is_eligible,
                failed_checks="|".join(failed_checks) if failed_checks else _NO_FAILURES,
                mean_ic=candidate.mean_ic,
                mean_rank_ic=candidate.mean_rank_ic,
                ic_ir=candidate.ic_ir,
                rank_ic_ir=candidate.rank_ic_ir,
                history_length=candidate.n_periods,
                threshold_min_mean_ic=thresholds.min_mean_ic,
                threshold_min_mean_rank_ic=thresholds.min_mean_rank_ic,
                threshold_min_ic_ir=thresholds.min_ic_ir,
                threshold_min_rank_ic_ir=thresholds.min_rank_ic_ir,
                threshold_min_history_length=thresholds.min_history_length,
            )
        )
    return results


def filter_by_eligibility(
    candidates: list[CandidateRecord],
    eligibility_results: list[EligibilityResult],
) -> tuple[list[CandidateRecord], list[CandidateRecord]]:
    """Split candidates into eligible and rejected lists.

    Preserves the original order within each partition.

    Args:
        candidates: Full candidate list (must correspond to ``eligibility_results``).
        eligibility_results: Gate evaluation outcomes (parallel to ``candidates``).

    Returns:
        ``(eligible_candidates, rejected_candidates)`` — both in stable input order.
    """
    eligible_ids: frozenset[str] = frozenset(
        r.candidate_id for r in eligibility_results if r.is_eligible
    )
    eligible = [c for c in candidates if c.candidate_id in eligible_ids]
    rejected = [c for c in candidates if c.candidate_id not in eligible_ids]
    return eligible, rejected


def build_eligibility_rejection_index(
    eligibility_results: list[EligibilityResult],
) -> dict[str, list[str]]:
    """Build a mapping from ``candidate_id`` to its list of failed check labels.

    Only rejected candidates appear in the returned dict; eligible candidates
    are absent (not mapped to an empty list) so callers can use ``in`` membership
    to check eligibility status efficiently.

    Args:
        eligibility_results: Gate evaluation outcomes.

    Returns:
        ``{candidate_id: [failed_check, ...]}`` for all rejected candidates.
    """
    index: dict[str, list[str]] = {}
    for result in eligibility_results:
        if not result.is_eligible:
            checks = result.failed_checks.split("|") if result.failed_checks else []
            index[result.candidate_id] = checks
    return index


def summarize_eligibility(
    eligibility_results: list[EligibilityResult],
    thresholds: EligibilityThresholds,
) -> dict[str, Any]:
    """Build an aggregate eligibility summary dict for JSON persistence.

    Args:
        eligibility_results: Full list of gate evaluation outcomes.
        thresholds: Threshold configuration snapshot.

    Returns:
        Dict with totals, breakdown by rejection reason, and threshold snapshot.
    """
    total = len(eligibility_results)
    eligible_count = sum(1 for r in eligibility_results if r.is_eligible)
    rejected_count = total - eligible_count

    # Count rejection reasons (a candidate failing multiple gates contributes
    # to each reason's count independently)
    rejection_counts: dict[str, int] = {}
    for result in eligibility_results:
        if not result.is_eligible and result.failed_checks:
            for check in result.failed_checks.split("|"):
                rejection_counts[check] = rejection_counts.get(check, 0) + 1

    return {
        "total_candidates": total,
        "eligible_candidates": eligible_count,
        "rejected_candidates": rejected_count,
        "rejection_counts_by_reason": dict(sorted(rejection_counts.items())),
        "thresholds": thresholds.to_dict(),
    }
