from __future__ import annotations

from collections.abc import Iterable

FEATURE_ALIASES: dict[str, tuple[str, ...]] = {
    "feature_sma_20": ("feature_sma20",),
    "feature_sma_50": ("feature_sma50",),
}

LEGACY_FEATURE_ALIASES: dict[str, str] = {
    alias: canonical
    for canonical, aliases in FEATURE_ALIASES.items()
    for alias in aliases
}


def canonicalize_feature_name(name: str) -> str:
    """Return the canonical feature name for one config-facing feature identifier."""

    return LEGACY_FEATURE_ALIASES.get(name, name)


def resolve_feature_names(names: Iterable[str], available_columns: Iterable[str]) -> list[str]:
    """
    Resolve requested feature names against available dataframe columns.

    Legacy aliases remain accepted when the canonical column exists, but callers
    still receive the column names that are actually present in the dataset.
    """

    available = set(available_columns)
    resolved: list[str] = []
    for name in names:
        canonical = canonicalize_feature_name(name)
        if canonical in available:
            resolved.append(canonical)
            continue
        if name in available:
            resolved.append(name)
            continue
        aliases = FEATURE_ALIASES.get(canonical, ())
        legacy_match = next((alias for alias in aliases if alias in available), None)
        if legacy_match is not None:
            resolved.append(legacy_match)
            continue
        resolved.append(canonical if canonical != name else name)
    return resolved
