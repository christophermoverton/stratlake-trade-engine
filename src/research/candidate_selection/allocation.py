"""Deterministic allocation governance for candidate selection outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.research.candidate_selection.schema import CandidateRecord


class AllocationError(ValueError):
    """Raised when allocation governance input or constraints are invalid."""


@dataclass(frozen=True)
class AllocationConstraints:
    """Explicit deterministic allocation governance constraints."""

    max_weight_per_candidate: float | None = None
    min_candidate_count: int = 1
    min_weight_threshold: float | None = None
    weight_sum_tolerance: float = 1e-12
    rounding_decimals: int = 12

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_weight_per_candidate": self.max_weight_per_candidate,
            "min_candidate_count": self.min_candidate_count,
            "min_weight_threshold": self.min_weight_threshold,
            "weight_sum_tolerance": self.weight_sum_tolerance,
            "rounding_decimals": self.rounding_decimals,
        }


@dataclass(frozen=True)
class AllocationDecision:
    """One deterministic candidate allocation decision."""

    candidate_id: str
    alpha_name: str
    sleeve_run_id: str | None
    allocation_weight: float
    allocation_method: str
    pre_constraint_weight: float | None = None
    constraint_adjusted_flag: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "alpha_name": self.alpha_name,
            "sleeve_run_id": self.sleeve_run_id,
            "allocation_weight": self.allocation_weight,
            "allocation_method": self.allocation_method,
            "pre_constraint_weight": self.pre_constraint_weight,
            "constraint_adjusted_flag": self.constraint_adjusted_flag,
        }

    @staticmethod
    def csv_columns() -> list[str]:
        return [
            "candidate_id",
            "alpha_name",
            "sleeve_run_id",
            "allocation_weight",
            "allocation_method",
            "pre_constraint_weight",
            "constraint_adjusted_flag",
        ]


def resolve_allocation_constraints(
    *,
    max_weight_per_candidate: float | None = None,
    min_candidate_count: int | None = None,
    min_weight_threshold: float | None = None,
    weight_sum_tolerance: float = 1e-12,
    rounding_decimals: int = 12,
) -> AllocationConstraints:
    """Resolve and validate allocation constraint settings."""

    resolved_min_candidates = 1 if min_candidate_count is None else int(min_candidate_count)
    if resolved_min_candidates <= 0:
        raise AllocationError("min_candidate_count must be >= 1.")

    resolved_max_weight = None if max_weight_per_candidate is None else float(max_weight_per_candidate)
    if resolved_max_weight is not None and (resolved_max_weight <= 0.0 or resolved_max_weight > 1.0):
        raise AllocationError("max_weight_per_candidate must be in (0, 1].")

    resolved_min_weight = None if min_weight_threshold is None else float(min_weight_threshold)
    if resolved_min_weight is not None and (resolved_min_weight <= 0.0 or resolved_min_weight >= 1.0):
        raise AllocationError("min_weight_threshold must be in (0, 1).")

    tolerance = float(weight_sum_tolerance)
    if tolerance < 0.0:
        raise AllocationError("weight_sum_tolerance must be non-negative.")

    resolved_rounding = int(rounding_decimals)
    if resolved_rounding < 0 or resolved_rounding > 15:
        raise AllocationError("rounding_decimals must be in [0, 15].")

    return AllocationConstraints(
        max_weight_per_candidate=resolved_max_weight,
        min_candidate_count=resolved_min_candidates,
        min_weight_threshold=resolved_min_weight,
        weight_sum_tolerance=tolerance,
        rounding_decimals=resolved_rounding,
    )


def allocate_candidates(
    candidates: list[CandidateRecord],
    *,
    method: str = "equal_weight",
    constraints: AllocationConstraints | None = None,
) -> tuple[list[AllocationDecision], dict[str, Any]]:
    """Allocate candidate weights deterministically under explicit constraints."""

    if not candidates:
        raise AllocationError("Cannot allocate an empty candidate set.")

    candidate_ids = [candidate.candidate_id for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise AllocationError("Allocation input contains duplicate candidate_id values.")

    resolved_constraints = constraints or resolve_allocation_constraints()
    if len(candidates) < resolved_constraints.min_candidate_count:
        raise AllocationError(
            "Allocation requires at least "
            f"{resolved_constraints.min_candidate_count} candidates, found {len(candidates)}."
        )

    normalized_method = str(method).strip().lower()
    if normalized_method in {"max_sharpe", "risk_parity"}:
        raise AllocationError(
            "Allocation method "
            f"'{normalized_method}' is not enabled in candidate-selection governance yet. "
            "Use equal_weight for Issue 4 and integrate optimizer-backed methods in a follow-up."
        )
    if normalized_method != "equal_weight":
        raise AllocationError(f"Unsupported allocation method: {normalized_method}")

    ordered = sorted(
        candidates,
        key=lambda candidate: (
            int(candidate.selection_rank),
            str(candidate.alpha_name),
            str(candidate.candidate_id),
        ),
    )

    raw_weight = 1.0 / float(len(ordered))
    raw_weights = {candidate.candidate_id: raw_weight for candidate in ordered}
    adjusted_weights = dict(raw_weights)

    if resolved_constraints.max_weight_per_candidate is not None:
        adjusted_weights = _apply_max_weight_constraint(
            raw_weights=raw_weights,
            ordered_ids=[candidate.candidate_id for candidate in ordered],
            max_weight=resolved_constraints.max_weight_per_candidate,
            tolerance=resolved_constraints.weight_sum_tolerance,
        )

    if resolved_constraints.min_weight_threshold is not None:
        adjusted_weights = _drop_and_renormalize_by_min_weight(
            weights=adjusted_weights,
            ordered_ids=[candidate.candidate_id for candidate in ordered],
            min_weight=resolved_constraints.min_weight_threshold,
            tolerance=resolved_constraints.weight_sum_tolerance,
        )

    retained_candidates = [candidate for candidate in ordered if candidate.candidate_id in adjusted_weights]
    if len(retained_candidates) < resolved_constraints.min_candidate_count:
        raise AllocationError(
            "Allocation constraints remove too many candidates. "
            f"Retained {len(retained_candidates)} < min_candidate_count {resolved_constraints.min_candidate_count}."
        )

    rounded_weights = _round_weights_with_residual(
        adjusted_weights,
        ordered_ids=[candidate.candidate_id for candidate in retained_candidates],
        decimals=resolved_constraints.rounding_decimals,
    )
    _validate_weight_sum(
        rounded_weights,
        tolerance=resolved_constraints.weight_sum_tolerance,
    )

    decisions: list[AllocationDecision] = []
    for candidate in retained_candidates:
        candidate_id = candidate.candidate_id
        pre_constraint = raw_weights.get(candidate_id)
        final_weight = rounded_weights[candidate_id]
        adjusted_flag = abs(final_weight - float(pre_constraint)) > resolved_constraints.weight_sum_tolerance
        decisions.append(
            AllocationDecision(
                candidate_id=candidate_id,
                alpha_name=candidate.alpha_name,
                sleeve_run_id=candidate.sleeve_run_id,
                allocation_weight=final_weight,
                allocation_method=normalized_method,
                pre_constraint_weight=pre_constraint,
                constraint_adjusted_flag=bool(adjusted_flag),
            )
        )

    final_weights = [decision.allocation_weight for decision in decisions]
    summary = {
        "allocation_method": normalized_method,
        "constraints": resolved_constraints.to_dict(),
        "allocated_candidates": len(decisions),
        "constraint_adjusted_candidates": sum(1 for decision in decisions if decision.constraint_adjusted_flag),
        "weight_sum": float(sum(final_weights)),
        "weight_min": float(min(final_weights)) if final_weights else None,
        "weight_max": float(max(final_weights)) if final_weights else None,
        "concentration_hhi": float(sum(weight * weight for weight in final_weights)) if final_weights else None,
    }
    return decisions, summary


def _apply_max_weight_constraint(
    *,
    raw_weights: dict[str, float],
    ordered_ids: list[str],
    max_weight: float,
    tolerance: float,
) -> dict[str, float]:
    if float(len(ordered_ids)) * max_weight + tolerance < 1.0:
        raise AllocationError(
            "max_weight_per_candidate constraint is infeasible for the candidate count: "
            f"n={len(ordered_ids)}, max_weight={max_weight}."
        )

    fixed: dict[str, float] = {}
    unresolved = list(ordered_ids)

    while unresolved:
        remaining_total = 1.0 - sum(fixed.values())
        unresolved_raw_total = sum(raw_weights[candidate_id] for candidate_id in unresolved)
        if unresolved_raw_total <= 0.0:
            raise AllocationError("Cannot distribute remaining weight under max-weight constraint.")

        scale = remaining_total / unresolved_raw_total
        proposed = {candidate_id: raw_weights[candidate_id] * scale for candidate_id in unresolved}
        violating = [candidate_id for candidate_id in unresolved if proposed[candidate_id] > max_weight + tolerance]
        if not violating:
            fixed.update(proposed)
            break

        for candidate_id in violating:
            fixed[candidate_id] = max_weight
        unresolved = [candidate_id for candidate_id in unresolved if candidate_id not in set(violating)]

    return fixed


def _drop_and_renormalize_by_min_weight(
    *,
    weights: dict[str, float],
    ordered_ids: list[str],
    min_weight: float,
    tolerance: float,
) -> dict[str, float]:
    retained_ids = [candidate_id for candidate_id in ordered_ids if weights[candidate_id] + tolerance >= min_weight]
    if not retained_ids:
        raise AllocationError(
            "min_weight_threshold removes all candidates. Reduce threshold or disable this constraint."
        )

    retained_total = sum(weights[candidate_id] for candidate_id in retained_ids)
    if retained_total <= 0.0:
        raise AllocationError("Cannot renormalize after min_weight_threshold filtering.")
    return {candidate_id: weights[candidate_id] / retained_total for candidate_id in retained_ids}


def _round_weights_with_residual(
    weights: dict[str, float],
    *,
    ordered_ids: list[str],
    decimals: int,
) -> dict[str, float]:
    rounded = {candidate_id: round(float(weights[candidate_id]), decimals) for candidate_id in ordered_ids}
    residual = round(1.0 - sum(rounded.values()), decimals)
    if ordered_ids:
        anchor_id = ordered_ids[0]
        rounded[anchor_id] = round(rounded[anchor_id] + residual, decimals)
    return rounded


def _validate_weight_sum(weights: dict[str, float], *, tolerance: float) -> None:
    if not weights:
        raise AllocationError("Allocation produced an empty weight set.")

    total = float(sum(weights.values()))
    if abs(total - 1.0) > tolerance:
        raise AllocationError(f"Allocation weights must sum to 1.0 (observed={total}).")

    for candidate_id, weight in weights.items():
        if weight < -tolerance:
            raise AllocationError(f"Allocation weight for {candidate_id} is negative ({weight}).")


__all__ = [
    "AllocationConstraints",
    "AllocationDecision",
    "AllocationError",
    "allocate_candidates",
    "resolve_allocation_constraints",
]
