"""Deterministic cross-candidate redundancy filtering for candidate selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.research.candidate_selection.schema import CandidateRecord


class RedundancyError(ValueError):
    """Raised when redundancy inputs are malformed or unusable."""


@dataclass(frozen=True)
class RedundancyThresholds:
    """Configurable thresholds for redundancy filtering.

    Attributes:
        max_pairwise_correlation:
            Absolute correlation threshold. If ``None``, redundancy pruning is disabled.
        min_overlap_observations:
            Minimum overlapping timestamps required to compute a valid pairwise
            correlation. Pairs with fewer observations are treated as undefined.
    """

    max_pairwise_correlation: float | None = None
    min_overlap_observations: int = 10

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_pairwise_correlation": self.max_pairwise_correlation,
            "min_overlap_observations": self.min_overlap_observations,
        }


@dataclass(frozen=True)
class RedundancyRejection:
    """Structured rejection metadata for one candidate pruned by redundancy."""

    candidate_id: str
    rejected_against_candidate_id: str
    observed_correlation: float
    overlap_observations: int
    configured_max_correlation: float
    rejection_reason: str = "correlation_above_threshold"
    rejected_stage: str = "redundancy_filter"


def resolve_redundancy_thresholds(
    *,
    max_pairwise_correlation: float | None = None,
    min_overlap_observations: int | None = None,
) -> RedundancyThresholds:
    """Resolve redundancy threshold settings.

    Args:
        max_pairwise_correlation: Absolute correlation threshold in [0, 1].
        min_overlap_observations: Minimum overlapping observations for a valid correlation.

    Returns:
        Normalized :class:`RedundancyThresholds`.
    """
    resolved_overlap = 10 if min_overlap_observations is None else int(min_overlap_observations)
    if resolved_overlap <= 1:
        raise RedundancyError("min_overlap_observations must be >= 2.")

    if max_pairwise_correlation is None:
        return RedundancyThresholds(
            max_pairwise_correlation=None,
            min_overlap_observations=resolved_overlap,
        )

    threshold = float(max_pairwise_correlation)
    if threshold < 0.0 or threshold > 1.0:
        raise RedundancyError("max_pairwise_correlation must be between 0 and 1 inclusive.")

    return RedundancyThresholds(
        max_pairwise_correlation=threshold,
        min_overlap_observations=resolved_overlap,
    )


def load_candidate_sleeve_returns(candidate: CandidateRecord) -> pd.Series:
    """Load one candidate sleeve return series indexed by UTC timestamp.

    Returns:
        ``pd.Series`` with datetime index (UTC), sorted by timestamp, name set to candidate_id.

    Raises:
        RedundancyError: if required sleeve artifacts are missing or malformed.
    """
    returns_path = Path(candidate.artifact_path) / "sleeve_returns.csv"
    if not returns_path.exists():
        raise RedundancyError(
            f"Missing sleeve_returns.csv for candidate {candidate.candidate_id}: {returns_path}"
        )

    frame = pd.read_csv(returns_path)
    required = {"ts_utc", "sleeve_return"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RedundancyError(
            "Invalid sleeve_returns.csv for candidate "
            f"{candidate.candidate_id}: missing columns {missing!r}."
        )

    normalized = frame.loc[:, ["ts_utc", "sleeve_return"]].copy()
    normalized["ts_utc"] = pd.to_datetime(normalized["ts_utc"], utc=True, errors="coerce")
    normalized["sleeve_return"] = pd.to_numeric(normalized["sleeve_return"], errors="coerce")
    normalized = normalized.dropna(axis="index", how="any")

    if normalized.empty:
        raise RedundancyError(
            f"Sleeve returns are empty after normalization for candidate {candidate.candidate_id}."
        )

    if normalized["ts_utc"].duplicated().any():
        dup_count = int(normalized["ts_utc"].duplicated().sum())
        raise RedundancyError(
            "Sleeve returns contain duplicate ts_utc values for candidate "
            f"{candidate.candidate_id} (duplicates={dup_count})."
        )

    normalized = normalized.sort_values("ts_utc", kind="stable")
    series = pd.Series(
        normalized["sleeve_return"].to_numpy(dtype="float64", copy=True),
        index=pd.DatetimeIndex(normalized["ts_utc"]),
        name=candidate.candidate_id,
        dtype="float64",
    )
    return series


def compute_redundancy_matrix(
    candidates: list[CandidateRecord],
    *,
    min_overlap_observations: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Series], dict[str, int]]:
    """Compute deterministic pairwise correlation and overlap matrices.

    Args:
        candidates: Candidates in deterministic processing order.
        min_overlap_observations: Minimum overlap for valid correlation.

    Returns:
        Tuple of:
            - correlation matrix (float, NaN for undefined pairs),
            - overlap count matrix (int),
            - loaded series by candidate_id,
            - undefined reason counts.
    """
    ordered_ids = [c.candidate_id for c in candidates]
    series_by_id = {c.candidate_id: load_candidate_sleeve_returns(c) for c in candidates}

    corr = pd.DataFrame(index=ordered_ids, columns=ordered_ids, dtype="float64")
    overlap = pd.DataFrame(index=ordered_ids, columns=ordered_ids, dtype="int64")

    undefined_reason_counts = {
        "insufficient_overlap": 0,
        "zero_variance": 0,
    }

    for left_id in ordered_ids:
        for right_id in ordered_ids:
            if left_id == right_id:
                corr.loc[left_id, right_id] = 1.0
                overlap.loc[left_id, right_id] = int(len(series_by_id[left_id]))
                continue

            joined = pd.concat([series_by_id[left_id], series_by_id[right_id]], axis=1, join="inner")
            joined.columns = [left_id, right_id]
            joined = joined.dropna(axis="index", how="any")
            obs = int(len(joined))
            overlap.loc[left_id, right_id] = obs

            if obs < min_overlap_observations:
                corr.loc[left_id, right_id] = float("nan")
                undefined_reason_counts["insufficient_overlap"] += 1
                continue

            left_std = float(joined[left_id].std(ddof=0))
            right_std = float(joined[right_id].std(ddof=0))
            if left_std == 0.0 or right_std == 0.0:
                corr.loc[left_id, right_id] = float("nan")
                undefined_reason_counts["zero_variance"] += 1
                continue

            value = float(joined[left_id].corr(joined[right_id], method="pearson"))
            if pd.isna(value):
                corr.loc[left_id, right_id] = float("nan")
                undefined_reason_counts["zero_variance"] += 1
                continue
            corr.loc[left_id, right_id] = value

    corr.index.name = "candidate_id"
    corr.columns.name = None
    overlap.index.name = "candidate_id"
    overlap.columns.name = None

    return corr, overlap, series_by_id, undefined_reason_counts


def apply_redundancy_filter(
    ranked_candidates: list[CandidateRecord],
    *,
    thresholds: RedundancyThresholds,
) -> tuple[list[CandidateRecord], list[RedundancyRejection], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Apply deterministic greedy redundancy pruning over ranked candidates.

    Policy:
        - Traverse candidates in ascending ``selection_rank`` order.
        - Keep a candidate unless it breaches the configured absolute correlation
          threshold against an already-kept candidate.
        - On breach, reject against the first kept candidate encountered in rank order.

    Returns:
        ``(kept, rejected, correlation_matrix, overlap_matrix, diagnostics_summary)``.
    """
    if not ranked_candidates:
        empty_corr = pd.DataFrame()
        empty_overlap = pd.DataFrame()
        return [], [], empty_corr, empty_overlap, {
            "eligible_before_redundancy": 0,
            "retained_after_redundancy": 0,
            "pruned_by_redundancy": 0,
            "redundancy_enabled": thresholds.max_pairwise_correlation is not None,
            "thresholds": thresholds.to_dict(),
            "undefined_pairwise_correlation": {
                "insufficient_overlap": 0,
                "zero_variance": 0,
            },
        }

    if thresholds.max_pairwise_correlation is None:
        empty_corr = pd.DataFrame()
        empty_overlap = pd.DataFrame()
        return list(ranked_candidates), [], empty_corr, empty_overlap, {
            "eligible_before_redundancy": len(ranked_candidates),
            "retained_after_redundancy": len(ranked_candidates),
            "pruned_by_redundancy": 0,
            "redundancy_enabled": False,
            "thresholds": thresholds.to_dict(),
            "rejection_counts_by_reason": {},
            "undefined_pairwise_correlation": {
                "insufficient_overlap": 0,
                "zero_variance": 0,
            },
        }

    corr, overlap, _series, undefined_reason_counts = compute_redundancy_matrix(
        ranked_candidates,
        min_overlap_observations=thresholds.min_overlap_observations,
    )

    # Count each unordered pair once for summary statistics.
    pair_undefined_summary = {
        "insufficient_overlap": undefined_reason_counts["insufficient_overlap"] // 2,
        "zero_variance": undefined_reason_counts["zero_variance"] // 2,
    }

    kept: list[CandidateRecord] = []
    rejected: list[RedundancyRejection] = []

    for candidate in ranked_candidates:
        rejected_against: RedundancyRejection | None = None
        for survivor in kept:
            observed = corr.loc[candidate.candidate_id, survivor.candidate_id]
            if pd.isna(observed):
                continue

            observed_correlation = float(observed)
            if abs(observed_correlation) > float(thresholds.max_pairwise_correlation):
                rejected_against = RedundancyRejection(
                    candidate_id=candidate.candidate_id,
                    rejected_against_candidate_id=survivor.candidate_id,
                    observed_correlation=observed_correlation,
                    overlap_observations=int(overlap.loc[candidate.candidate_id, survivor.candidate_id]),
                    configured_max_correlation=float(thresholds.max_pairwise_correlation),
                )
                break

        if rejected_against is None:
            kept.append(candidate)
        else:
            rejected.append(rejected_against)

    rejection_counts_by_reason: dict[str, int] = {}
    for item in rejected:
        rejection_counts_by_reason[item.rejection_reason] = rejection_counts_by_reason.get(item.rejection_reason, 0) + 1

    return kept, rejected, corr, overlap, {
        "eligible_before_redundancy": len(ranked_candidates),
        "retained_after_redundancy": len(kept),
        "pruned_by_redundancy": len(rejected),
        "redundancy_enabled": True,
        "thresholds": thresholds.to_dict(),
        "rejection_counts_by_reason": dict(sorted(rejection_counts_by_reason.items())),
        "undefined_pairwise_correlation": pair_undefined_summary,
    }


def serialize_correlation_matrix(correlation: pd.DataFrame) -> pd.DataFrame:
    """Return correlation matrix in deterministic CSV-ready shape.

    The first column is ``candidate_id`` followed by candidate IDs in the
    same deterministic order as the matrix index.
    """
    if correlation.empty:
        return pd.DataFrame(columns=["candidate_id"])

    ordered_ids = list(correlation.index)
    matrix = correlation.loc[ordered_ids, ordered_ids]
    out = matrix.reset_index()
    out.columns = ["candidate_id", *ordered_ids]
    return out
