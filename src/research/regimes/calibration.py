from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.research.regimes.taxonomy import (
    REGIME_DIMENSIONS,
    REGIME_OUTPUT_COLUMNS,
    REGIME_STATE_COLUMNS,
    REGIME_TAXONOMY,
    TAXONOMY_VERSION,
    UNDEFINED_REGIME_LABEL,
)
from src.research.regimes.validation import RegimeValidationError, validate_regime_labels
from src.research.registry import canonicalize_value

DEFAULT_REGIME_CALIBRATION_PROFILE = "baseline"
DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.50


class RegimeCalibrationError(ValueError):
    """Raised when regime calibration configuration or execution is invalid."""


@dataclass(frozen=True)
class RegimeCalibrationProfile:
    """Deterministic calibration controls applied to canonical regime labels."""

    name: str
    min_regime_duration_days: int
    transition_smoothing_window: int
    allow_single_day_flips: bool
    max_flip_rate: float
    max_single_day_flip_share: float
    min_observations_per_regime: int
    min_observations_for_attribution: int
    low_confidence_share_threshold: float
    unstable_regime_fallback: str | None = None
    unknown_regime_fallback: str | None = None
    require_stability_for_attribution: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allow_single_day_flips": self.allow_single_day_flips,
            "low_confidence_share_threshold": self.low_confidence_share_threshold,
            "max_flip_rate": self.max_flip_rate,
            "max_single_day_flip_share": self.max_single_day_flip_share,
            "metadata": dict(sorted(self.metadata.items())),
            "min_observations_for_attribution": self.min_observations_for_attribution,
            "min_observations_per_regime": self.min_observations_per_regime,
            "min_regime_duration_days": self.min_regime_duration_days,
            "name": self.name,
            "require_stability_for_attribution": self.require_stability_for_attribution,
            "transition_smoothing_window": self.transition_smoothing_window,
            "unknown_regime_fallback": self.unknown_regime_fallback,
            "unstable_regime_fallback": self.unstable_regime_fallback,
        }


@dataclass(frozen=True)
class RegimeCalibrationResult:
    """Structured result for deterministic regime calibration."""

    labels: pd.DataFrame
    audit: pd.DataFrame
    profile: RegimeCalibrationProfile
    stability_metrics: dict[str, Any]
    warnings: tuple[str, ...]
    profile_flags: dict[str, bool]
    attribution_summary: pd.DataFrame
    fallback_summary: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    taxonomy_version: str = TAXONOMY_VERSION


def builtin_regime_calibration_profiles() -> dict[str, RegimeCalibrationProfile]:
    return {
        profile.name: profile
        for profile in (
            RegimeCalibrationProfile(
                name="baseline",
                min_regime_duration_days=2,
                transition_smoothing_window=1,
                allow_single_day_flips=False,
                max_flip_rate=0.25,
                max_single_day_flip_share=0.10,
                min_observations_per_regime=3,
                min_observations_for_attribution=5,
                low_confidence_share_threshold=0.25,
                unstable_regime_fallback=None,
                unknown_regime_fallback=_undefined_composite_label(),
                require_stability_for_attribution=True,
                metadata={"intent": "balanced default sensitivity"},
            ),
            RegimeCalibrationProfile(
                name="conservative",
                min_regime_duration_days=3,
                transition_smoothing_window=3,
                allow_single_day_flips=False,
                max_flip_rate=0.15,
                max_single_day_flip_share=0.05,
                min_observations_per_regime=5,
                min_observations_for_attribution=8,
                low_confidence_share_threshold=0.15,
                unstable_regime_fallback=_undefined_composite_label(),
                unknown_regime_fallback=_undefined_composite_label(),
                require_stability_for_attribution=True,
                metadata={"intent": "suppress short-lived noise"},
            ),
            RegimeCalibrationProfile(
                name="reactive",
                min_regime_duration_days=1,
                transition_smoothing_window=1,
                allow_single_day_flips=True,
                max_flip_rate=0.50,
                max_single_day_flip_share=0.30,
                min_observations_per_regime=2,
                min_observations_for_attribution=4,
                low_confidence_share_threshold=0.40,
                unstable_regime_fallback=None,
                unknown_regime_fallback=_undefined_composite_label(),
                require_stability_for_attribution=False,
                metadata={"intent": "maximize responsiveness"},
            ),
            RegimeCalibrationProfile(
                name="crisis_sensitive",
                min_regime_duration_days=1,
                transition_smoothing_window=1,
                allow_single_day_flips=True,
                max_flip_rate=0.60,
                max_single_day_flip_share=0.35,
                min_observations_per_regime=2,
                min_observations_for_attribution=5,
                low_confidence_share_threshold=0.30,
                unstable_regime_fallback=None,
                unknown_regime_fallback=_undefined_composite_label(),
                require_stability_for_attribution=True,
                metadata={"intent": "preserve fast stress transitions"},
            ),
        )
    }


def resolve_regime_calibration_profile(
    profile: RegimeCalibrationProfile | dict[str, Any] | str | None,
) -> RegimeCalibrationProfile:
    builtins = builtin_regime_calibration_profiles()
    if profile is None:
        return builtins[DEFAULT_REGIME_CALIBRATION_PROFILE]
    if isinstance(profile, str):
        resolved = builtins.get(profile)
        if resolved is None:
            raise RegimeCalibrationError(
                f"Unknown regime calibration profile {profile!r}. Supported profiles: {sorted(builtins)}."
            )
        return resolved
    if isinstance(profile, RegimeCalibrationProfile):
        _validate_profile(profile)
        return profile
    if not isinstance(profile, dict):
        raise RegimeCalibrationError("Regime calibration profile must be a profile name, dict, or dataclass.")

    required_fields = {
        "allow_single_day_flips",
        "low_confidence_share_threshold",
        "max_flip_rate",
        "max_single_day_flip_share",
        "min_observations_for_attribution",
        "min_observations_per_regime",
        "min_regime_duration_days",
        "name",
        "require_stability_for_attribution",
        "transition_smoothing_window",
        "unknown_regime_fallback",
        "unstable_regime_fallback",
    }
    missing = sorted(required_fields - set(profile))
    if missing:
        raise RegimeCalibrationError(f"Regime calibration profile is missing required fields: {missing}.")
    allowed_fields = set(RegimeCalibrationProfile.__dataclass_fields__)
    unknown = sorted(set(profile) - allowed_fields)
    if unknown:
        raise RegimeCalibrationError(f"Regime calibration profile contains unsupported fields: {unknown}.")

    resolved = RegimeCalibrationProfile(**profile)
    _validate_profile(resolved)
    return resolved


def apply_regime_calibration(
    regime_labels: pd.DataFrame,
    *,
    profile: RegimeCalibrationProfile | dict[str, Any] | str | None = None,
    confidence_column: str | None = None,
    low_confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    metadata: dict[str, Any] | None = None,
) -> RegimeCalibrationResult:
    resolved_profile = resolve_regime_calibration_profile(profile)
    _validate_profile(resolved_profile)
    labels = validate_regime_labels(regime_labels)
    confidence = _resolve_confidence_series(
        labels=regime_labels,
        normalized_labels=labels,
        confidence_column=confidence_column,
        low_confidence_threshold=low_confidence_threshold,
    )

    raw_labels = labels["regime_label"].astype("string").tolist()
    smoothed_labels = _causal_mode_smooth(raw_labels, window=resolved_profile.transition_smoothing_window)
    stabilized_labels = _stabilize_labels(
        smoothed_labels,
        min_regime_duration_days=resolved_profile.min_regime_duration_days,
    )

    calibrated = _labels_frame_from_composite_labels(labels, stabilized_labels)
    audit = pd.DataFrame(
        {
            "ts_utc": calibrated["ts_utc"].to_numpy(copy=True),
            "raw_regime_label": pd.Series(raw_labels, dtype="string"),
            "smoothed_regime_label": pd.Series(smoothed_labels, dtype="string"),
            "stabilized_regime_label": pd.Series(stabilized_labels, dtype="string"),
            "confidence_value": confidence["value"],
            "is_low_confidence": confidence["low_mask"],
            "used_unknown_fallback": pd.Series(False, index=calibrated.index, dtype="bool"),
            "used_low_confidence_fallback": pd.Series(False, index=calibrated.index, dtype="bool"),
            "used_unstable_profile_fallback": pd.Series(False, index=calibrated.index, dtype="bool"),
        }
    )

    if resolved_profile.unknown_regime_fallback is not None:
        unknown_mask = ~calibrated["is_defined"].astype("bool")
        if unknown_mask.any():
            calibrated = _apply_fallback(calibrated, unknown_mask, fallback_label=resolved_profile.unknown_regime_fallback)
            audit.loc[unknown_mask, "used_unknown_fallback"] = True

    if resolved_profile.unstable_regime_fallback is not None and confidence["low_mask"].any():
        calibrated = _apply_fallback(
            calibrated,
            confidence["low_mask"],
            fallback_label=resolved_profile.unstable_regime_fallback,
        )
        audit.loc[confidence["low_mask"], "used_low_confidence_fallback"] = True

    gating_metrics, attribution_summary = _compute_stability_metrics(
        labels=calibrated,
        min_regime_duration_days=resolved_profile.min_regime_duration_days,
        min_observations_per_regime=resolved_profile.min_observations_per_regime,
        min_observations_for_attribution=resolved_profile.min_observations_for_attribution,
        require_stability_for_attribution=resolved_profile.require_stability_for_attribution,
        low_confidence_mask=confidence["low_mask"],
    )
    profile_flags = _profile_flags(resolved_profile, gating_metrics)

    if profile_flags["is_unstable_profile"] and resolved_profile.unstable_regime_fallback is not None:
        unstable_mask = pd.Series(True, index=calibrated.index, dtype="bool")
        calibrated = _apply_fallback(
            calibrated,
            unstable_mask,
            fallback_label=resolved_profile.unstable_regime_fallback,
        )
        audit.loc[:, "used_unstable_profile_fallback"] = True

    final_metrics, final_attribution_summary = _compute_stability_metrics(
        labels=calibrated,
        min_regime_duration_days=resolved_profile.min_regime_duration_days,
        min_observations_per_regime=resolved_profile.min_observations_per_regime,
        min_observations_for_attribution=resolved_profile.min_observations_for_attribution,
        require_stability_for_attribution=resolved_profile.require_stability_for_attribution,
        low_confidence_mask=confidence["low_mask"],
    )
    warnings = _warnings_for_metrics(resolved_profile, gating_metrics, final_attribution_summary)

    result_metadata = canonicalize_value(
        {
            "confidence_column": confidence_column,
            "confidence_threshold": low_confidence_threshold if confidence_column is not None else None,
            "input_row_count": int(len(labels)),
            "metadata": dict(metadata or {}),
            "taxonomy_version": TAXONOMY_VERSION,
        }
    )
    fallback_summary = canonicalize_value(
        {
            "low_confidence_fallback_rows": int(audit["used_low_confidence_fallback"].sum()),
            "unknown_fallback_rows": int(audit["used_unknown_fallback"].sum()),
            "unstable_profile_fallback_rows": int(audit["used_unstable_profile_fallback"].sum()),
            "unknown_regime_fallback": resolved_profile.unknown_regime_fallback,
            "unstable_regime_fallback": resolved_profile.unstable_regime_fallback,
        }
    )

    return RegimeCalibrationResult(
        labels=calibrated,
        audit=audit,
        profile=resolved_profile,
        stability_metrics=canonicalize_value(final_metrics),
        warnings=tuple(warnings),
        profile_flags=canonicalize_value(profile_flags),
        attribution_summary=final_attribution_summary,
        fallback_summary=fallback_summary,
        metadata=result_metadata,
    )


def _validate_profile(profile: RegimeCalibrationProfile) -> None:
    if not profile.name.strip():
        raise RegimeCalibrationError("Regime calibration profile name must be a non-empty string.")
    for field_name in (
        "min_regime_duration_days",
        "transition_smoothing_window",
        "min_observations_per_regime",
        "min_observations_for_attribution",
    ):
        value = getattr(profile, field_name)
        if not isinstance(value, int) or value < 1:
            raise RegimeCalibrationError(f"Regime calibration field {field_name!r} must be a positive integer.")
    for field_name in (
        "max_flip_rate",
        "max_single_day_flip_share",
        "low_confidence_share_threshold",
    ):
        value = float(getattr(profile, field_name))
        if value < 0.0 or value > 1.0:
            raise RegimeCalibrationError(
                f"Regime calibration field {field_name!r} must be between 0.0 and 1.0."
            )
    if profile.min_observations_for_attribution < profile.min_observations_per_regime:
        raise RegimeCalibrationError(
            "Regime calibration field 'min_observations_for_attribution' must be >= "
            "'min_observations_per_regime'."
        )
    if profile.allow_single_day_flips and profile.min_regime_duration_days != 1:
        raise RegimeCalibrationError(
            "Regime calibration profiles that allow single-day flips must use min_regime_duration_days=1."
        )
    _validate_fallback_label(profile.unknown_regime_fallback, field_name="unknown_regime_fallback")
    _validate_fallback_label(profile.unstable_regime_fallback, field_name="unstable_regime_fallback")


def _resolve_confidence_series(
    *,
    labels: pd.DataFrame,
    normalized_labels: pd.DataFrame,
    confidence_column: str | None,
    low_confidence_threshold: float,
) -> dict[str, pd.Series]:
    if low_confidence_threshold < 0.0 or low_confidence_threshold > 1.0:
        raise RegimeCalibrationError("low_confidence_threshold must be between 0.0 and 1.0.")
    if confidence_column is None:
        empty_value = pd.Series([None] * len(normalized_labels), index=normalized_labels.index, dtype="object")
        low_mask = pd.Series(False, index=normalized_labels.index, dtype="bool")
        return {"low_mask": low_mask, "value": empty_value}
    if confidence_column not in labels.columns:
        raise RegimeCalibrationError(f"Confidence column {confidence_column!r} is not present in the input frame.")

    confidence_values = pd.to_numeric(labels[confidence_column], errors="coerce")
    invalid = confidence_values.notna() & ~confidence_values.between(0.0, 1.0)
    if invalid.any():
        bad_value = float(confidence_values.loc[invalid].iloc[0])
        raise RegimeCalibrationError(
            f"Confidence column {confidence_column!r} must contain values between 0.0 and 1.0. "
            f"First invalid value={bad_value}."
        )
    low_mask = confidence_values.lt(low_confidence_threshold).fillna(True).astype("bool")
    value = confidence_values.astype("float64").where(confidence_values.notna(), None)
    return {"low_mask": low_mask, "value": value.astype("object")}


def _causal_mode_smooth(labels: list[str], *, window: int) -> list[str]:
    if window == 1:
        return list(labels)
    smoothed: list[str] = []
    for position in range(len(labels)):
        start = max(0, position + 1 - window)
        history = labels[start : position + 1]
        counts = pd.Series(history, dtype="string").value_counts(sort=False)
        max_count = int(counts.max())
        tied = {str(label) for label, count in counts.items() if int(count) == max_count}
        choice = next(label for label in reversed(history) if label in tied)
        smoothed.append(choice)
    return smoothed


def _stabilize_labels(labels: list[str], *, min_regime_duration_days: int) -> list[str]:
    if not labels:
        return []
    active_label = labels[0]
    candidate_label: str | None = None
    candidate_count = 0
    stabilized: list[str] = [active_label]
    for label in labels[1:]:
        if label == active_label:
            candidate_label = None
            candidate_count = 0
            stabilized.append(active_label)
            continue
        if candidate_label == label:
            candidate_count += 1
        else:
            candidate_label = label
            candidate_count = 1
        if candidate_count >= min_regime_duration_days:
            active_label = label
            candidate_label = None
            candidate_count = 0
        stabilized.append(active_label)
    return stabilized


def _labels_frame_from_composite_labels(base: pd.DataFrame, composite_labels: list[str]) -> pd.DataFrame:
    if len(base) != len(composite_labels):
        raise RegimeCalibrationError("Composite label sequence length must match the regime label frame length.")
    calibrated = base.copy(deep=True)
    calibrated.attrs = {}
    parsed = [_parse_composite_label(label) for label in composite_labels]
    for dimension in REGIME_DIMENSIONS:
        state_column = REGIME_STATE_COLUMNS[dimension]
        calibrated[state_column] = pd.Series(
            [mapping[dimension] for mapping in parsed],
            index=calibrated.index,
            dtype="string",
        )
    calibrated["regime_label"] = pd.Series(composite_labels, index=calibrated.index, dtype="string")
    calibrated["is_defined"] = calibrated.loc[:, list(REGIME_STATE_COLUMNS.values())].ne(UNDEFINED_REGIME_LABEL).all(axis=1)
    return validate_regime_labels(calibrated.loc[:, list(REGIME_OUTPUT_COLUMNS)])


def _apply_fallback(labels: pd.DataFrame, mask: pd.Series, *, fallback_label: str) -> pd.DataFrame:
    if len(mask) != len(labels):
        raise RegimeCalibrationError("Fallback mask length must match the regime label frame length.")
    _validate_fallback_label(fallback_label, field_name="fallback_label")
    result = labels.copy(deep=True)
    result.attrs = {}
    parsed = _parse_composite_label(fallback_label)
    for dimension in REGIME_DIMENSIONS:
        result.loc[mask, REGIME_STATE_COLUMNS[dimension]] = parsed[dimension]
    result.loc[mask, "regime_label"] = fallback_label
    result.loc[mask, "is_defined"] = all(parsed[dimension] != UNDEFINED_REGIME_LABEL for dimension in REGIME_DIMENSIONS)
    try:
        return validate_regime_labels(result.loc[:, list(REGIME_OUTPUT_COLUMNS)])
    except RegimeValidationError as exc:
        raise RegimeCalibrationError(
            "Fallback label is incompatible with the underlying metric availability for one or more rows."
        ) from exc


def _compute_stability_metrics(
    *,
    labels: pd.DataFrame,
    min_regime_duration_days: int,
    min_observations_per_regime: int,
    min_observations_for_attribution: int,
    require_stability_for_attribution: bool,
    low_confidence_mask: pd.Series,
) -> tuple[dict[str, Any], pd.DataFrame]:
    composite = labels["regime_label"].astype("string")
    runs = _run_table(composite)
    durations = [run["duration"] for run in runs]
    transition_count = max(len(runs) - 1, 0)
    single_day_flip_count = sum(
        1
        for index, run in enumerate(runs)
        if run["duration"] == 1
        and 0 < index < len(runs) - 1
        and runs[index - 1]["label"] == runs[index + 1]["label"]
    )
    unstable_rows = sum(run["duration"] for run in runs if run["duration"] < min_regime_duration_days)
    counts = composite.value_counts(sort=False).astype("int64")
    per_label_has_short_run = {
        str(run["label"]): True
        for run in runs
        if run["duration"] < min_regime_duration_days
    }
    attribution_rows: list[dict[str, Any]] = []
    for label in sorted(str(key) for key in counts.index.tolist()):
        observation_count = int(counts.loc[label])
        has_short_run = per_label_has_short_run.get(label, False)
        meets_min_regime_observations = observation_count >= min_observations_per_regime
        eligible = observation_count >= min_observations_for_attribution and meets_min_regime_observations
        if require_stability_for_attribution and has_short_run:
            eligible = False
        attribution_rows.append(
            {
                "meets_min_observations_per_regime": bool(meets_min_regime_observations),
                "regime_label": label,
                "is_attribution_eligible": bool(eligible),
                "has_unstable_run": bool(has_short_run),
                "observation_count": observation_count,
            }
        )
    attribution_summary = pd.DataFrame(
        attribution_rows,
        columns=[
            "regime_label",
            "observation_count",
            "meets_min_observations_per_regime",
            "has_unstable_run",
            "is_attribution_eligible",
        ],
    )
    metrics = {
        "average_regime_duration": float(pd.Series(durations, dtype="float64").mean()) if durations else 0.0,
        "attribution_eligible_regime_count": int(attribution_summary["is_attribution_eligible"].sum())
        if not attribution_summary.empty
        else 0,
        "attribution_ineligible_regime_count": int(len(attribution_summary))
        - (int(attribution_summary["is_attribution_eligible"].sum()) if not attribution_summary.empty else 0),
        "flip_rate": float(transition_count / max(len(labels) - 1, 1)),
        "low_confidence_share": float(low_confidence_mask.astype("float64").mean()) if len(low_confidence_mask) else 0.0,
        "maximum_regime_duration": int(max(durations)) if durations else 0,
        "median_regime_duration": float(pd.Series(durations, dtype="float64").median()) if durations else 0.0,
        "minimum_regime_duration": int(min(durations)) if durations else 0,
        "regime_count": int(len(runs)),
        "single_day_flip_count": int(single_day_flip_count),
        "single_day_flip_share": float(single_day_flip_count / max(len(runs), 1)),
        "total_observations": int(len(labels)),
        "transition_count": int(transition_count),
        "unstable_regime_share": float(unstable_rows / max(len(labels), 1)),
    }
    return canonicalize_value(metrics), attribution_summary


def _run_table(labels: pd.Series) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    start = 0
    values = labels.astype("string").tolist()
    for position in range(1, len(values) + 1):
        if position < len(values) and values[position] == values[start]:
            continue
        runs.append(
            {
                "duration": position - start,
                "end": position - 1,
                "label": values[start],
                "start": start,
            }
        )
        start = position
    return runs


def _profile_flags(profile: RegimeCalibrationProfile, metrics: dict[str, Any]) -> dict[str, bool]:
    exceeds_flip_rate = float(metrics["flip_rate"]) > profile.max_flip_rate
    exceeds_single_day_share = float(metrics["single_day_flip_share"]) > profile.max_single_day_flip_share
    exceeds_low_confidence_share = float(metrics["low_confidence_share"]) > profile.low_confidence_share_threshold
    has_unstable_runs = float(metrics["unstable_regime_share"]) > 0.0
    return {
        "exceeds_low_confidence_share": exceeds_low_confidence_share,
        "exceeds_max_flip_rate": exceeds_flip_rate,
        "exceeds_max_single_day_flip_share": exceeds_single_day_share,
        "has_unstable_runs": has_unstable_runs,
        "is_unstable_profile": any(
            (exceeds_flip_rate, exceeds_single_day_share, exceeds_low_confidence_share, has_unstable_runs)
        ),
    }


def _warnings_for_metrics(
    profile: RegimeCalibrationProfile,
    metrics: dict[str, Any],
    attribution_summary: pd.DataFrame,
) -> list[str]:
    warnings: list[str] = []
    if float(metrics["flip_rate"]) > profile.max_flip_rate:
        warnings.append(
            "Calibration output exceeds max_flip_rate and should be treated as unstable for decision support."
        )
    if float(metrics["single_day_flip_share"]) > profile.max_single_day_flip_share:
        warnings.append(
            "Calibration output exceeds max_single_day_flip_share and contains too many one-observation reversals."
        )
    if float(metrics["low_confidence_share"]) > profile.low_confidence_share_threshold:
        warnings.append(
            "Calibration output exceeds low_confidence_share_threshold and confidence-aware downstream usage should be gated."
        )
    ineligible_count = (
        int((~attribution_summary["is_attribution_eligible"]).sum()) if not attribution_summary.empty else 0
    )
    if ineligible_count > 0:
        warnings.append(
            f"{ineligible_count} regime label(s) do not meet attribution eligibility requirements for this profile."
        )
    sparse_count = (
        int((~attribution_summary["meets_min_observations_per_regime"]).sum())
        if not attribution_summary.empty
        else 0
    )
    if sparse_count > 0:
        warnings.append(
            f"{sparse_count} regime label(s) do not meet min_observations_per_regime for stable review."
        )
    return warnings


def _validate_fallback_label(label: str | None, *, field_name: str) -> None:
    if label is None:
        return
    _parse_composite_label(label, field_name=field_name)


def _parse_composite_label(label: str, *, field_name: str = "regime_label") -> dict[str, str]:
    if not isinstance(label, str) or not label.strip():
        raise RegimeCalibrationError(f"{field_name} must be a non-empty composite regime label string.")
    parts = label.split("|")
    if len(parts) != len(REGIME_DIMENSIONS):
        raise RegimeCalibrationError(
            f"{field_name} must include exactly {len(REGIME_DIMENSIONS)} taxonomy dimensions in canonical order."
        )
    mapping: dict[str, str] = {}
    for expected_dimension, part in zip(REGIME_DIMENSIONS, parts, strict=False):
        if "=" not in part:
            raise RegimeCalibrationError(
                f"{field_name} must use 'dimension=state' segments. Invalid segment: {part!r}."
            )
        dimension, state = part.split("=", 1)
        if dimension != expected_dimension:
            raise RegimeCalibrationError(
                f"{field_name} must follow canonical dimension order {REGIME_DIMENSIONS}."
            )
        allowed_states = set(REGIME_TAXONOMY[dimension].labels)
        if state not in allowed_states:
            raise RegimeCalibrationError(
                f"{field_name} contains unsupported taxonomy label {state!r} for dimension {dimension!r}."
            )
        mapping[dimension] = state
    return mapping


def _undefined_composite_label() -> str:
    return "|".join(f"{dimension}={UNDEFINED_REGIME_LABEL}" for dimension in REGIME_DIMENSIONS)


__all__ = [
    "DEFAULT_LOW_CONFIDENCE_THRESHOLD",
    "DEFAULT_REGIME_CALIBRATION_PROFILE",
    "RegimeCalibrationError",
    "RegimeCalibrationProfile",
    "RegimeCalibrationResult",
    "apply_regime_calibration",
    "builtin_regime_calibration_profiles",
    "resolve_regime_calibration_profile",
]