from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.research.candidate_selection.allocation import AllocationDecision
from src.research.candidate_selection.eligibility import EligibilityResult
from src.research.candidate_selection.schema import CandidateRecord


class CandidateArtifactConsistencyError(ValueError):
    """Raised when candidate-selection artifacts are internally inconsistent."""


def validate_candidate_selection_artifact_payload_consistency(
    *,
    universe: list[CandidateRecord],
    eligibility_results: list[EligibilityResult],
    selected: list[CandidateRecord],
    rejected_rows: list[dict[str, object]],
    allocation_decisions: list[AllocationDecision],
    allocation_enabled: bool,
    allocation_weight_sum_tolerance: float,
) -> None:
    """Validate structural and cross-artifact consistency before persistence."""

    errors: list[str] = []

    _require_unique_ids([candidate.candidate_id for candidate in universe], "candidate_universe", errors)
    _require_unique_ids([result.candidate_id for result in eligibility_results], "eligibility_filter_results", errors)
    _require_unique_ids([candidate.candidate_id for candidate in selected], "selected_candidates", errors)
    _require_unique_ids([str(row.get("candidate_id")) for row in rejected_rows], "rejected_candidates", errors)
    _require_unique_ids([decision.candidate_id for decision in allocation_decisions], "allocation_weights", errors)

    universe_ids = {candidate.candidate_id for candidate in universe}
    eligibility_ids = {result.candidate_id for result in eligibility_results}
    selected_ids = {candidate.candidate_id for candidate in selected}
    rejected_ids = {str(row.get("candidate_id")) for row in rejected_rows}
    allocation_ids = {decision.candidate_id for decision in allocation_decisions}

    if universe_ids != eligibility_ids:
        errors.append("candidate_id mismatch between candidate_universe and eligibility_filter_results")

    overlap = selected_ids.intersection(rejected_ids)
    if overlap:
        errors.append(f"selected_candidates and rejected_candidates overlap on candidate_id values: {sorted(overlap)}")

    if selected_ids.union(rejected_ids) != universe_ids:
        errors.append("selected_candidates + rejected_candidates must equal candidate_universe")

    if allocation_enabled:
        if selected_ids != allocation_ids:
            errors.append("allocation_weights candidate_id set must match selected_candidates when allocation is enabled")

        if allocation_decisions:
            weight_sum = float(sum(float(decision.allocation_weight) for decision in allocation_decisions))
            if abs(weight_sum - 1.0) > float(allocation_weight_sum_tolerance):
                errors.append(
                    "allocation_weights must sum to 1.0 within tolerance "
                    f"(observed={weight_sum}, tolerance={allocation_weight_sum_tolerance})"
                )
            negative = [
                decision.candidate_id
                for decision in allocation_decisions
                if float(decision.allocation_weight) < -float(allocation_weight_sum_tolerance)
            ]
            if negative:
                errors.append(f"allocation_weights contains negative values for candidate_ids: {sorted(negative)}")

    valid_stages = {"eligibility_gate", "redundancy_filter"}
    for row in rejected_rows:
        candidate_id = str(row.get("candidate_id"))
        stage = row.get("rejected_stage")
        reason = row.get("rejection_reason")
        if stage not in valid_stages:
            errors.append(
                f"rejected_candidates row for candidate_id={candidate_id!r} has invalid rejected_stage={stage!r}"
            )
            continue

        if stage == "eligibility_gate" and not str(row.get("failed_checks") or "").strip():
            errors.append(
                f"eligibility rejection for candidate_id={candidate_id!r} must include failed_checks"
            )

        if stage == "redundancy_filter":
            valid_reasons = {"correlation_above_threshold", "max_candidate_count_limit"}
            if reason not in valid_reasons:
                errors.append(
                    f"redundancy rejection for candidate_id={candidate_id!r} has invalid reason={reason!r}"
                )

    if errors:
        raise CandidateArtifactConsistencyError("; ".join(sorted(errors)))


def ensure_dataframe_columns(
    frame: pd.DataFrame,
    *,
    required_columns: Iterable[str],
    owner: str,
) -> None:
    """Assert required columns are present in deterministic order checks."""

    required = list(required_columns)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise CandidateArtifactConsistencyError(
            f"{owner} missing required columns: {sorted(missing)}"
        )


def _require_unique_ids(candidate_ids: list[str], owner: str, errors: list[str]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for candidate_id in candidate_ids:
        if candidate_id in seen:
            duplicates.add(candidate_id)
        seen.add(candidate_id)
    if duplicates:
        errors.append(f"{owner} contains duplicate candidate_id values: {sorted(duplicates)}")
