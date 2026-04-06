"""Configurable gate checks for candidate eligibility evaluation.

Each gate check evaluates one aspect of candidate quality and returns a
failure label if the candidate does not meet the configured threshold.

Design notes:
    - All checks are deterministic: same inputs → same failure set.
    - Gate evaluation order is fixed by GATE_CHECK_ORDER for auditability.
    - Checks are independent: a candidate may fail multiple gates.
    - Disabled thresholds (None) are silently skipped, not errors.
    - Missing metric values are not penalised by threshold gates unless
      ``require_*`` flags are explicitly set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Canonical failure label constants
# ---------------------------------------------------------------------------

MISSING_REQUIRED_METRIC = "missing_required_metric"
NON_FINITE_METRIC = "non_finite_metric"
MEAN_IC_BELOW_THRESHOLD = "mean_ic_below_threshold"
MEAN_RANK_IC_BELOW_THRESHOLD = "mean_rank_ic_below_threshold"
IC_IR_BELOW_THRESHOLD = "ic_ir_below_threshold"
RANK_IC_IR_BELOW_THRESHOLD = "rank_ic_ir_below_threshold"
HISTORY_LENGTH_BELOW_THRESHOLD = "history_length_below_threshold"

# Ordered sequence used to sort failure labels deterministically
GATE_CHECK_ORDER: tuple[str, ...] = (
    MISSING_REQUIRED_METRIC,
    NON_FINITE_METRIC,
    MEAN_IC_BELOW_THRESHOLD,
    MEAN_RANK_IC_BELOW_THRESHOLD,
    IC_IR_BELOW_THRESHOLD,
    RANK_IC_IR_BELOW_THRESHOLD,
    HISTORY_LENGTH_BELOW_THRESHOLD,
)

# Map each base label to its canonical sort index for stable ordering
_GATE_ORDER_INDEX: dict[str, int] = {label: idx for idx, label in enumerate(GATE_CHECK_ORDER)}


# ---------------------------------------------------------------------------
# Threshold configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EligibilityThresholds:
    """Configurable thresholds for the eligibility gate framework.

    All numeric thresholds default to ``None`` (disabled).  Setting a threshold
    to ``None`` means that gate check is simply skipped for every candidate.

    The ``require_*`` flags can be used to fail candidates whose metric value
    is entirely missing (``None``), independent of any numeric threshold.
    """

    min_mean_ic: float | None = None
    """Minimum acceptable mean IC.  ``None`` = gate disabled."""

    min_mean_rank_ic: float | None = None
    """Minimum acceptable mean Rank IC.  ``None`` = gate disabled."""

    min_ic_ir: float | None = None
    """Minimum acceptable IC information ratio.  ``None`` = gate disabled."""

    min_rank_ic_ir: float | None = None
    """Minimum acceptable Rank IC information ratio.  ``None`` = gate disabled."""

    min_history_length: int | None = None
    """Minimum acceptable observation count (n_periods).  ``None`` = gate disabled."""

    require_mean_ic: bool = False
    """Fail candidates whose ``mean_ic`` is ``None``."""

    require_ic_ir: bool = False
    """Fail candidates whose ``ic_ir`` is ``None``."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize thresholds to a plain dict suitable for JSON persistence."""
        return {
            "min_mean_ic": self.min_mean_ic,
            "min_mean_rank_ic": self.min_mean_rank_ic,
            "min_ic_ir": self.min_ic_ir,
            "min_rank_ic_ir": self.min_rank_ic_ir,
            "min_history_length": self.min_history_length,
            "require_mean_ic": self.require_mean_ic,
            "require_ic_ir": self.require_ic_ir,
        }


_DEFAULT_THRESHOLDS = EligibilityThresholds()


def resolve_eligibility_thresholds(
    *,
    min_mean_ic: float | None = None,
    min_mean_rank_ic: float | None = None,
    min_ic_ir: float | None = None,
    min_rank_ic_ir: float | None = None,
    min_history_length: int | None = None,
    require_mean_ic: bool = False,
    require_ic_ir: bool = False,
) -> EligibilityThresholds:
    """Build an :class:`EligibilityThresholds` from keyword arguments.

    This is the preferred factory for constructing thresholds from CLI or
    config overrides; it mirrors the field names exactly so callers can use
    ``**kwargs`` forwarding.
    """
    return EligibilityThresholds(
        min_mean_ic=min_mean_ic,
        min_mean_rank_ic=min_mean_rank_ic,
        min_ic_ir=min_ic_ir,
        min_rank_ic_ir=min_rank_ic_ir,
        min_history_length=min_history_length,
        require_mean_ic=require_mean_ic,
        require_ic_ir=require_ic_ir,
    )


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------


def evaluate_candidate_gates(
    candidate: Any,  # CandidateRecord — typed as Any to avoid circular import
    thresholds: EligibilityThresholds,
) -> list[str]:
    """Evaluate all configured gate checks for one candidate.

    Args:
        candidate: A :class:`~src.research.candidate_selection.schema.CandidateRecord`.
        thresholds: Threshold configuration.

    Returns:
        Deterministically sorted list of failed check labels.
        Empty list means the candidate passes all enabled gates.
    """
    failed: list[str] = []

    # ------------------------------------------------------------------
    # 1. Missing required metrics
    # ------------------------------------------------------------------
    if thresholds.require_mean_ic and candidate.mean_ic is None:
        failed.append(f"{MISSING_REQUIRED_METRIC}:mean_ic")
    if thresholds.require_ic_ir and candidate.ic_ir is None:
        failed.append(f"{MISSING_REQUIRED_METRIC}:ic_ir")

    # ------------------------------------------------------------------
    # 2. Non-finite metric checks
    #    Any provided metric that is NaN or ±Inf is an invalid data quality
    #    issue and always constitutes a failure.
    # ------------------------------------------------------------------
    for metric_name in ("mean_ic", "ic_ir", "mean_rank_ic", "rank_ic_ir"):
        value = getattr(candidate, metric_name, None)
        if value is not None:
            try:
                fv = float(value)
            except (TypeError, ValueError):
                failed.append(f"{NON_FINITE_METRIC}:{metric_name}")
                continue
            if not math.isfinite(fv):
                failed.append(f"{NON_FINITE_METRIC}:{metric_name}")

    # ------------------------------------------------------------------
    # 3. Numeric threshold gates
    #    Each gate only applies when:
    #      (a) the threshold is set (not None), and
    #      (b) the candidate's metric is available (not None).
    #    Candidates with a missing metric are not penalised by threshold
    #    gates unless a require_* flag has been set (see step 1).
    # ------------------------------------------------------------------
    if thresholds.min_mean_ic is not None and candidate.mean_ic is not None:
        if float(candidate.mean_ic) < thresholds.min_mean_ic:
            failed.append(MEAN_IC_BELOW_THRESHOLD)

    if thresholds.min_mean_rank_ic is not None and candidate.mean_rank_ic is not None:
        if float(candidate.mean_rank_ic) < thresholds.min_mean_rank_ic:
            failed.append(MEAN_RANK_IC_BELOW_THRESHOLD)

    if thresholds.min_ic_ir is not None and candidate.ic_ir is not None:
        if float(candidate.ic_ir) < thresholds.min_ic_ir:
            failed.append(IC_IR_BELOW_THRESHOLD)

    if thresholds.min_rank_ic_ir is not None and candidate.rank_ic_ir is not None:
        if float(candidate.rank_ic_ir) < thresholds.min_rank_ic_ir:
            failed.append(RANK_IC_IR_BELOW_THRESHOLD)

    if thresholds.min_history_length is not None and candidate.n_periods is not None:
        if int(candidate.n_periods) < thresholds.min_history_length:
            failed.append(HISTORY_LENGTH_BELOW_THRESHOLD)

    return _sort_failures(failed)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sort_failures(failures: list[str]) -> list[str]:
    """Return failures in deterministic canonical order.

    Primary sort key: position in GATE_CHECK_ORDER for the base label prefix.
    Secondary sort key: alphabetic on the full label string (handles suffixes
    like ``:mean_ic`` vs ``:ic_ir``).
    """
    def _key(label: str) -> tuple[int, str]:
        base = label.split(":")[0]
        return (_GATE_ORDER_INDEX.get(base, len(GATE_CHECK_ORDER)), label)

    return sorted(failures, key=_key)
