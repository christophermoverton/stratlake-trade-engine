from __future__ import annotations

from typing import Any

CANONICAL_PROMOTION_STATUSES = frozenset({"eligible", "warn", "needs_review", "rejected", "blocked"})
CANONICAL_REVIEW_STATUSES = frozenset({"candidate", "needs_review", "rejected"})
PROMOTION_STATUS_ALIASES = {
    "review": "needs_review",
    "needs work": "needs_review",
    "needs_work": "needs_review",
    "needs-work": "needs_review",
}
REVIEW_STATUS_ALIASES = {
    "review": "needs_review",
    "needs work": "needs_review",
    "needs_work": "needs_review",
    "needs-work": "needs_review",
}
PROMOTION_TO_REVIEW_STATUS = {
    "eligible": "candidate",
    "warn": "needs_review",
    "needs_review": "needs_review",
    "rejected": "rejected",
    "blocked": "rejected",
}


def normalize_promotion_status(value: Any) -> str | None:
    """Normalize governance boundary labels without defining promotion policy."""

    normalized = _coerce_status(value)
    if normalized is None:
        return None
    return PROMOTION_STATUS_ALIASES.get(normalized, normalized)


def normalize_review_status(value: Any) -> str | None:
    """Normalize review labels for governance comparison only."""

    normalized = _coerce_status(value)
    if normalized is None:
        return None
    return REVIEW_STATUS_ALIASES.get(normalized, normalized)


def status_was_normalized(raw_status: str | None, normalized_status: str | None) -> bool:
    return raw_status is not None and normalized_status is not None and raw_status != normalized_status


def _coerce_status(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.lower()


__all__ = [
    "CANONICAL_PROMOTION_STATUSES",
    "CANONICAL_REVIEW_STATUSES",
    "PROMOTION_STATUS_ALIASES",
    "PROMOTION_TO_REVIEW_STATUS",
    "REVIEW_STATUS_ALIASES",
    "normalize_promotion_status",
    "normalize_review_status",
    "status_was_normalized",
]
