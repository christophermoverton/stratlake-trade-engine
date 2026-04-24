from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any, Literal, Mapping

import pandas as pd

from src.portfolio.optimizer import SUPPORTED_PORTFOLIO_OPTIMIZERS
from src.research.metrics import (
    infer_periods_per_year,
    max_drawdown,
    sharpe_ratio,
    total_return,
    volatility,
)
from src.research.regimes.calibration import _parse_composite_label
from src.research.regimes.taxonomy import REGIME_DIMENSIONS, REGIME_STATE_COLUMNS, TAXONOMY_VERSION
from src.research.regimes.validation import validate_regime_labels
from src.research.registry import canonicalize_value

PolicySurface = Literal["strategy", "portfolio", "alpha"]
FallbackPolicy = Literal["neutral", "baseline", "cash_proxy", "reduce_exposure", "skip_adaptation"]

SUPPORTED_FALLBACK_POLICIES: tuple[str, ...] = (
    "neutral",
    "baseline",
    "cash_proxy",
    "reduce_exposure",
    "skip_adaptation",
)
SUPPORTED_CONFIDENCE_REASONS: tuple[str, ...] = ("ambiguous", "low_confidence", "unsupported")
DEFAULT_POLICY_RETURN_COLUMNS: dict[str, str] = {
    "alpha": "alpha_return",
    "portfolio": "portfolio_return",
    "strategy": "strategy_return",
}
DEFAULT_POLICY_WEIGHT_COLUMNS: tuple[str, ...] = ("weight", "baseline_weight", "portfolio_weight")

_POLICY_DECISION_COLUMNS: tuple[str, ...] = (
    "symbol",
    "ts_utc",
    "regime_label",
    "matched_policy_key",
    "matched_alias",
    "confidence_score",
    "confidence_bucket",
    "confidence_fallback_reason",
    "fallback_policy_applied",
    "policy_reason",
    "signal_scale",
    "allocation_scale",
    "alpha_weight_multiplier",
    "volatility_target",
    "gross_exposure_cap",
    "max_component_weight",
    "rebalance_enabled",
    "optimizer_override",
    "allocation_rule_override",
    "calibration_profile",
    "is_unstable_profile",
    "eligible_for_downstream_decisioning",
    "policy_source",
    "baseline_weight",
    "adaptive_weight",
    "baseline_return",
    "adaptive_return",
)


class RegimePolicyError(ValueError):
    """Raised when regime policy configuration or evaluation is invalid."""


@dataclass(frozen=True)
class RegimePolicyRule:
    signal_scale: float
    allocation_scale: float
    alpha_weight_multiplier: float
    volatility_target: float
    gross_exposure_cap: float
    max_component_weight: float
    rebalance_enabled: bool
    optimizer_override: str | None
    allocation_rule_override: str | None
    fallback_policy: FallbackPolicy | str

    def to_dict(self) -> dict[str, Any]:
        return {
            "allocation_rule_override": self.allocation_rule_override,
            "allocation_scale": self.allocation_scale,
            "alpha_weight_multiplier": self.alpha_weight_multiplier,
            "fallback_policy": self.fallback_policy,
            "gross_exposure_cap": self.gross_exposure_cap,
            "max_component_weight": self.max_component_weight,
            "optimizer_override": self.optimizer_override,
            "rebalance_enabled": self.rebalance_enabled,
            "signal_scale": self.signal_scale,
            "volatility_target": self.volatility_target,
        }


@dataclass(frozen=True)
class RegimePolicyAlias:
    name: str
    match: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"match": canonicalize_value(dict(self.match)), "name": self.name}


@dataclass(frozen=True)
class RegimePolicyConfidenceConfig:
    min_confidence: float
    low_confidence_fallback: FallbackPolicy | str
    ambiguous_fallback: FallbackPolicy | str
    unsupported_fallback: FallbackPolicy | str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ambiguous_fallback": self.ambiguous_fallback,
            "low_confidence_fallback": self.low_confidence_fallback,
            "min_confidence": self.min_confidence,
            "unsupported_fallback": self.unsupported_fallback,
        }


@dataclass(frozen=True)
class RegimePolicyConfig:
    default: RegimePolicyRule
    confidence: RegimePolicyConfidenceConfig
    regimes: dict[str, RegimePolicyRule] = field(default_factory=dict)
    regime_aliases: dict[str, RegimePolicyAlias] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return canonicalize_value(
            {
                "metadata": dict(sorted(self.metadata.items())),
                "regime_aliases": {
                    name: alias.to_dict()
                    for name, alias in self.regime_aliases.items()
                },
                "regime_policy": {
                    "confidence": self.confidence.to_dict(),
                    "default": self.default.to_dict(),
                    "regimes": {
                        name: rule.to_dict()
                        for name, rule in self.regimes.items()
                    },
                },
            }
        )


@dataclass(frozen=True)
class RegimePolicyDecisionResult:
    decisions: pd.DataFrame
    summary: dict[str, Any]
    config: RegimePolicyConfig
    comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: dict[str, Any] = field(default_factory=dict)
    taxonomy_version: str = TAXONOMY_VERSION


def resolve_regime_policy_config(
    payload: RegimePolicyConfig | Mapping[str, Any] | None,
) -> RegimePolicyConfig:
    if payload is None:
        return _default_policy_config()
    if isinstance(payload, RegimePolicyConfig):
        _validate_policy_config(payload)
        return payload
    if not isinstance(payload, Mapping):
        raise RegimePolicyError("Regime policy config must be a mapping when provided.")

    payload_dict = dict(payload)
    if "regime_policy" in payload_dict:
        regime_policy_payload = payload_dict["regime_policy"]
        alias_payload = payload_dict.get("regime_aliases", {})
        metadata = payload_dict.get("metadata", {})
    else:
        regime_policy_payload = {
            key: payload_dict[key]
            for key in ("default", "confidence", "regimes")
            if key in payload_dict
        }
        alias_payload = payload_dict.get("regime_aliases", {})
        metadata = payload_dict.get("metadata", {})

    if not isinstance(regime_policy_payload, Mapping):
        raise RegimePolicyError("Config key 'regime_policy' must be a mapping.")

    required_policy_sections = {"confidence", "default", "regimes"}
    missing_sections = sorted(required_policy_sections - set(regime_policy_payload))
    if missing_sections:
        raise RegimePolicyError(
            f"Regime policy config is missing required sections: {missing_sections}."
        )

    default_rule = _resolve_rule_payload(
        regime_policy_payload["default"],
        field_name="regime_policy.default",
    )
    confidence = _resolve_confidence_config(regime_policy_payload["confidence"])
    aliases = _resolve_policy_aliases(alias_payload)
    regimes = _resolve_policy_rules(regime_policy_payload["regimes"], aliases=aliases)
    config = RegimePolicyConfig(
        default=default_rule,
        confidence=confidence,
        regimes=regimes,
        regime_aliases=aliases,
        metadata=canonicalize_value(dict(metadata or {})),
    )
    _validate_policy_config(config)
    return config


def apply_regime_policy(
    regime_labels: pd.DataFrame,
    *,
    config: RegimePolicyConfig | Mapping[str, Any] | None = None,
    confidence_frame: pd.DataFrame | None = None,
    baseline_frame: pd.DataFrame | None = None,
    surface: PolicySurface = "strategy",
    return_column: str | None = None,
    weight_column: str | None = None,
    calibration_profile: str | None = None,
    is_unstable_profile: bool | None = None,
    eligible_for_downstream_decisioning: bool | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RegimePolicyDecisionResult:
    resolved_config = resolve_regime_policy_config(config)
    labels = _normalize_policy_labels(regime_labels)
    working = labels.copy(deep=True)
    working.attrs = {}

    if confidence_frame is not None:
        working = _join_confidence_frame(working, confidence_frame)
    else:
        working["confidence_score"] = pd.Series([None] * len(working), index=working.index, dtype="object")
        working["confidence_bucket"] = pd.Series([None] * len(working), index=working.index, dtype="object")
        working["fallback_flag"] = pd.Series(False, index=working.index, dtype="bool")
        working["fallback_reason"] = pd.Series([None] * len(working), index=working.index, dtype="object")

    resolved_calibration_profile = (
        calibration_profile
        if calibration_profile is not None
        else _first_present_value(working, "calibration_profile")
    )
    resolved_is_unstable = (
        bool(is_unstable_profile)
        if is_unstable_profile is not None
        else _first_present_bool(working, "is_unstable_profile")
    )
    resolved_eligible = (
        bool(eligible_for_downstream_decisioning)
        if eligible_for_downstream_decisioning is not None
        else _first_present_bool(working, "eligible_for_downstream_decisioning")
    )

    decisions = []
    for _, row in working.iterrows():
        decisions.append(
            _decision_row(
                row=row,
                config=resolved_config,
                calibration_profile=resolved_calibration_profile,
                is_unstable_profile=resolved_is_unstable,
                eligible_for_downstream_decisioning=resolved_eligible,
            )
        )
    decisions_frame = pd.DataFrame(decisions)
    decisions_frame = _sort_policy_decisions(decisions_frame)

    result = RegimePolicyDecisionResult(
        decisions=decisions_frame,
        summary=_build_policy_summary(
            decisions_frame,
            config=resolved_config,
            comparison=None,
            metadata=metadata,
        ),
        config=resolved_config,
        metadata=canonicalize_value(dict(metadata or {})),
    )
    if baseline_frame is not None:
        return compare_adaptive_vs_static(
            result,
            baseline_frame,
            surface=surface,
            return_column=return_column,
            weight_column=weight_column,
        )
    return result


def compare_adaptive_vs_static(
    result: RegimePolicyDecisionResult,
    baseline_frame: pd.DataFrame,
    *,
    surface: PolicySurface = "strategy",
    return_column: str | None = None,
    weight_column: str | None = None,
) -> RegimePolicyDecisionResult:
    if not isinstance(result, RegimePolicyDecisionResult):
        raise TypeError("compare_adaptive_vs_static expects a RegimePolicyDecisionResult.")
    baseline = _normalize_baseline_frame(baseline_frame)
    decisions = result.decisions.copy(deep=True)
    decisions.attrs = {}

    resolved_return_column = _resolve_return_column(baseline, surface=surface, return_column=return_column)
    resolved_weight_column = _resolve_weight_column(baseline, requested=weight_column)
    merged = _join_baseline_to_decisions(
        decisions,
        baseline,
        return_column=resolved_return_column,
        weight_column=resolved_weight_column,
    )
    scale_column = _scale_column_for_surface(surface)
    merged["adaptive_return"] = pd.to_numeric(merged["baseline_return"], errors="coerce") * pd.to_numeric(
        merged[scale_column],
        errors="coerce",
    )
    if "baseline_weight" in merged.columns and merged["baseline_weight"].notna().any():
        merged["adaptive_weight"] = pd.to_numeric(merged["baseline_weight"], errors="coerce") * pd.to_numeric(
            merged["allocation_scale"],
            errors="coerce",
        )
    else:
        merged["baseline_weight"] = pd.Series([None] * len(merged), index=merged.index, dtype="object")
        merged["adaptive_weight"] = pd.Series([None] * len(merged), index=merged.index, dtype="object")

    merged = _add_equity_curves(merged)
    comparison = _build_comparison_frame(merged, surface=surface)
    summary = _build_policy_summary(
        merged,
        config=result.config,
        comparison=comparison,
        metadata=result.metadata,
    )
    return RegimePolicyDecisionResult(
        decisions=merged,
        summary=summary,
        config=result.config,
        comparison=comparison,
        metadata=result.metadata,
        taxonomy_version=result.taxonomy_version,
    )


def _default_policy_config() -> RegimePolicyConfig:
    return RegimePolicyConfig(
        default=RegimePolicyRule(
            signal_scale=1.0,
            allocation_scale=1.0,
            alpha_weight_multiplier=1.0,
            volatility_target=0.10,
            gross_exposure_cap=1.0,
            max_component_weight=1.0,
            rebalance_enabled=True,
            optimizer_override=None,
            allocation_rule_override=None,
            fallback_policy="baseline",
        ),
        confidence=RegimePolicyConfidenceConfig(
            min_confidence=0.60,
            low_confidence_fallback="neutral",
            ambiguous_fallback="neutral",
            unsupported_fallback="baseline",
        ),
    )


def _resolve_policy_rules(
    payload: Any,
    *,
    aliases: Mapping[str, RegimePolicyAlias],
) -> dict[str, RegimePolicyRule]:
    if not isinstance(payload, Mapping):
        raise RegimePolicyError("Config key 'regime_policy.regimes' must be a mapping.")
    resolved: dict[str, RegimePolicyRule] = {}
    for key, value in payload.items():
        normalized_key = _normalize_non_empty_string(key, field_name="regime_policy.regimes key")
        if not _is_canonical_regime_label(normalized_key) and normalized_key not in aliases:
            raise RegimePolicyError(
                "Regime policy rules must be keyed by canonical regime labels or configured aliases. "
                f"Unsupported key: {normalized_key!r}."
            )
        resolved[normalized_key] = _resolve_rule_payload(
            value,
            field_name=f"regime_policy.regimes.{normalized_key}",
        )
    return resolved


def _resolve_policy_aliases(payload: Any) -> dict[str, RegimePolicyAlias]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise RegimePolicyError("Config key 'regime_aliases' must be a mapping when provided.")
    aliases: dict[str, RegimePolicyAlias] = {}
    for key, value in payload.items():
        name = _normalize_non_empty_string(key, field_name="regime_aliases key")
        if _is_canonical_regime_label(name):
            raise RegimePolicyError("Alias names must not reuse canonical composite regime labels.")
        if not isinstance(value, Mapping):
            raise RegimePolicyError(f"Alias {name!r} must map to a configuration object.")
        match = value.get("match")
        if not isinstance(match, Mapping) or not match:
            raise RegimePolicyError(f"Alias {name!r} must define a non-empty 'match' mapping.")
        aliases[name] = RegimePolicyAlias(name=name, match=_normalize_alias_match(match, alias_name=name))
    return aliases


def _normalize_alias_match(match: Mapping[str, Any], *, alias_name: str) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in match.items():
        normalized_key = _normalize_alias_match_key(key)
        if normalized_key == "is_defined":
            if not isinstance(value, bool):
                raise RegimePolicyError(f"Alias {alias_name!r} field 'is_defined' must be boolean.")
            normalized[normalized_key] = value
            continue
        normalized_value = _normalize_non_empty_string(
            value,
            field_name=f"regime_aliases.{alias_name}.match.{normalized_key}",
        )
        if normalized_key == "regime_label":
            _parse_composite_label(
                normalized_value,
                field_name=f"regime_aliases.{alias_name}.match.regime_label",
            )
        else:
            dimension = _dimension_for_state_column(normalized_key)
            allowed = set(_allowed_states_for_dimension(dimension))
            if normalized_value not in allowed:
                raise RegimePolicyError(
                    f"Alias {alias_name!r} match value {normalized_value!r} is not valid for {normalized_key!r}."
                )
        normalized[normalized_key] = normalized_value
    return normalized


def _resolve_confidence_config(payload: Any) -> RegimePolicyConfidenceConfig:
    if not isinstance(payload, Mapping):
        raise RegimePolicyError("Config key 'regime_policy.confidence' must be a mapping.")
    required = {
        "ambiguous_fallback",
        "low_confidence_fallback",
        "min_confidence",
        "unsupported_fallback",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise RegimePolicyError(f"Regime policy confidence config is missing required fields: {missing}.")
    resolved = RegimePolicyConfidenceConfig(
        min_confidence=_coerce_probability(payload["min_confidence"], field_name="regime_policy.confidence.min_confidence"),
        low_confidence_fallback=_normalize_fallback_policy(
            payload["low_confidence_fallback"],
            field_name="regime_policy.confidence.low_confidence_fallback",
        ),
        ambiguous_fallback=_normalize_fallback_policy(
            payload["ambiguous_fallback"],
            field_name="regime_policy.confidence.ambiguous_fallback",
        ),
        unsupported_fallback=_normalize_fallback_policy(
            payload["unsupported_fallback"],
            field_name="regime_policy.confidence.unsupported_fallback",
        ),
    )
    return resolved


def _resolve_rule_payload(payload: Any, *, field_name: str) -> RegimePolicyRule:
    if not isinstance(payload, Mapping):
        raise RegimePolicyError(f"{field_name} must be a mapping.")
    required = {
        "allocation_rule_override",
        "allocation_scale",
        "alpha_weight_multiplier",
        "fallback_policy",
        "gross_exposure_cap",
        "max_component_weight",
        "optimizer_override",
        "rebalance_enabled",
        "signal_scale",
        "volatility_target",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise RegimePolicyError(f"{field_name} is missing required fields: {missing}.")
    return RegimePolicyRule(
        signal_scale=_coerce_non_negative_float(payload["signal_scale"], field_name=f"{field_name}.signal_scale"),
        allocation_scale=_coerce_non_negative_float(
            payload["allocation_scale"],
            field_name=f"{field_name}.allocation_scale",
        ),
        alpha_weight_multiplier=_coerce_non_negative_float(
            payload["alpha_weight_multiplier"],
            field_name=f"{field_name}.alpha_weight_multiplier",
        ),
        volatility_target=_coerce_positive_float(
            payload["volatility_target"],
            field_name=f"{field_name}.volatility_target",
        ),
        gross_exposure_cap=_coerce_non_negative_float(
            payload["gross_exposure_cap"],
            field_name=f"{field_name}.gross_exposure_cap",
        ),
        max_component_weight=_coerce_bounded_float(
            payload["max_component_weight"],
            lower=0.0,
            upper=1.0,
            lower_inclusive=False,
            upper_inclusive=True,
            field_name=f"{field_name}.max_component_weight",
        ),
        rebalance_enabled=_coerce_bool(payload["rebalance_enabled"], field_name=f"{field_name}.rebalance_enabled"),
        optimizer_override=_normalize_optimizer_override(
            payload["optimizer_override"],
            field_name=f"{field_name}.optimizer_override",
        ),
        allocation_rule_override=_normalize_optional_string(
            payload["allocation_rule_override"],
            field_name=f"{field_name}.allocation_rule_override",
        ),
        fallback_policy=_normalize_fallback_policy(
            payload["fallback_policy"],
            field_name=f"{field_name}.fallback_policy",
        ),
    )


def _validate_policy_config(config: RegimePolicyConfig) -> None:
    _validate_rule(config.default, field_name="regime_policy.default")
    if not isinstance(config.metadata, dict):
        raise RegimePolicyError("Regime policy config metadata must be a dictionary.")
    for reason, fallback in (
        ("low_confidence", config.confidence.low_confidence_fallback),
        ("ambiguous", config.confidence.ambiguous_fallback),
        ("unsupported", config.confidence.unsupported_fallback),
    ):
        if fallback not in SUPPORTED_FALLBACK_POLICIES:
            raise RegimePolicyError(f"Unsupported {reason} fallback policy {fallback!r}.")
    for alias_name, alias in config.regime_aliases.items():
        if alias_name != alias.name:
            raise RegimePolicyError("Regime policy alias map keys must match alias names.")
        if not alias.match:
            raise RegimePolicyError(f"Regime policy alias {alias_name!r} must include at least one match rule.")
    for key, rule in config.regimes.items():
        if not _is_canonical_regime_label(key) and key not in config.regime_aliases:
            raise RegimePolicyError(
                f"Regime policy rule key {key!r} does not map to a canonical label or configured alias."
            )
        _validate_rule(rule, field_name=f"regime_policy.regimes.{key}")


def _validate_rule(rule: RegimePolicyRule, *, field_name: str) -> None:
    for numeric_field in (
        "signal_scale",
        "allocation_scale",
        "alpha_weight_multiplier",
        "volatility_target",
        "gross_exposure_cap",
        "max_component_weight",
    ):
        value = getattr(rule, numeric_field)
        if not isinstance(value, int | float) or not math.isfinite(float(value)):
            raise RegimePolicyError(f"{field_name}.{numeric_field} must be a finite numeric value.")
    if rule.signal_scale < 0.0 or rule.allocation_scale < 0.0 or rule.alpha_weight_multiplier < 0.0:
        raise RegimePolicyError(f"{field_name} scale fields must be non-negative.")
    if rule.volatility_target <= 0.0:
        raise RegimePolicyError(f"{field_name}.volatility_target must be positive.")
    if rule.gross_exposure_cap < 0.0:
        raise RegimePolicyError(f"{field_name}.gross_exposure_cap must be non-negative.")
    if rule.max_component_weight <= 0.0 or rule.max_component_weight > 1.0:
        raise RegimePolicyError(f"{field_name}.max_component_weight must be > 0 and <= 1.")
    if not isinstance(rule.rebalance_enabled, bool):
        raise RegimePolicyError(f"{field_name}.rebalance_enabled must be boolean.")
    _normalize_fallback_policy(rule.fallback_policy, field_name=f"{field_name}.fallback_policy")
    _normalize_optimizer_override(rule.optimizer_override, field_name=f"{field_name}.optimizer_override")


def _normalize_policy_labels(regime_labels: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(regime_labels, pd.DataFrame):
        raise TypeError("regime_labels must be a pandas DataFrame.")
    normalized = regime_labels.copy(deep=True)
    normalized.attrs = {}
    required_columns = (
        "ts_utc",
        *REGIME_STATE_COLUMNS.values(),
        "regime_label",
        "is_defined",
        "volatility_metric",
        "trend_metric",
        "drawdown_metric",
        "stress_correlation_metric",
        "stress_dispersion_metric",
    )
    label_contract = validate_regime_labels(normalized.loc[:, list(required_columns)])
    for column in label_contract.columns:
        normalized[column] = label_contract[column]
    if "symbol" in normalized.columns:
        normalized["symbol"] = normalized["symbol"].astype("string")
    return normalized


def _join_confidence_frame(labels: pd.DataFrame, confidence_frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(confidence_frame, pd.DataFrame):
        raise TypeError("confidence_frame must be a pandas DataFrame.")
    required = {"confidence_bucket", "confidence_score", "fallback_flag", "fallback_reason", "ts_utc"}
    if "symbol" in labels.columns:
        required.add("symbol")
    missing = sorted(required - set(confidence_frame.columns))
    if missing:
        raise RegimePolicyError(f"confidence_frame is missing required columns: {missing}.")
    normalized = confidence_frame.copy(deep=True)
    normalized.attrs = {}
    normalized["ts_utc"] = pd.to_datetime(normalized["ts_utc"], utc=True, errors="coerce")
    if normalized["ts_utc"].isna().any():
        raise RegimePolicyError("confidence_frame contains invalid ts_utc values.")
    if "symbol" in labels.columns:
        normalized["symbol"] = normalized["symbol"].astype("string")
        join_keys = ["symbol", "ts_utc"]
    else:
        join_keys = ["ts_utc"]
    if normalized.duplicated(subset=join_keys).any():
        raise RegimePolicyError("confidence_frame must be unique on its policy join keys.")
    normalized["confidence_score"] = pd.to_numeric(normalized["confidence_score"], errors="coerce")
    invalid_score = normalized["confidence_score"].notna() & ~normalized["confidence_score"].between(0.0, 1.0)
    if invalid_score.any():
        raise RegimePolicyError("confidence_frame confidence_score values must be between 0 and 1.")
    normalized["confidence_bucket"] = normalized["confidence_bucket"].astype("string")
    normalized["fallback_flag"] = normalized["fallback_flag"].astype("bool")
    normalized["fallback_reason"] = normalized["fallback_reason"].astype("string")
    invalid_reason = normalized["fallback_reason"].dropna().loc[
        ~normalized["fallback_reason"].dropna().isin(SUPPORTED_CONFIDENCE_REASONS)
    ]
    if not invalid_reason.empty:
        raise RegimePolicyError(
            f"confidence_frame contains unsupported fallback_reason {str(invalid_reason.iloc[0])!r}."
        )
    merged = labels.merge(
        normalized.loc[:, join_keys + ["confidence_score", "confidence_bucket", "fallback_flag", "fallback_reason"]],
        on=join_keys,
        how="left",
        sort=False,
    )
    merged["fallback_flag"] = merged["fallback_flag"].fillna(False).astype("bool")
    return merged


def _decision_row(
    *,
    row: pd.Series,
    config: RegimePolicyConfig,
    calibration_profile: str | None,
    is_unstable_profile: bool,
    eligible_for_downstream_decisioning: bool | None,
) -> dict[str, Any]:
    matched_alias = _match_alias(row, config.regime_aliases)
    exact_label = str(row["regime_label"])
    base_rule = config.default
    matched_policy_key = "default"
    policy_source = "default"
    if exact_label in config.regimes:
        base_rule = config.regimes[exact_label]
        matched_policy_key = exact_label
        policy_source = "regime_label"
    elif matched_alias is not None and matched_alias in config.regimes:
        base_rule = config.regimes[matched_alias]
        matched_policy_key = matched_alias
        policy_source = "alias"

    fallback_policy = None
    policy_reason_parts = [f"policy_source={policy_source}"]
    confidence_reason = _confidence_reason(row, config.confidence)
    if confidence_reason is not None:
        fallback_policy = {
            "ambiguous": config.confidence.ambiguous_fallback,
            "low_confidence": config.confidence.low_confidence_fallback,
            "unsupported": config.confidence.unsupported_fallback,
        }[confidence_reason]
        policy_reason_parts.append(f"confidence_fallback={confidence_reason}")
    elif not bool(row["is_defined"]):
        fallback_policy = base_rule.fallback_policy
        policy_reason_parts.append("undefined_regime_fallback")
    elif eligible_for_downstream_decisioning is False:
        fallback_policy = base_rule.fallback_policy
        policy_reason_parts.append("ineligible_profile_fallback")
    elif is_unstable_profile:
        fallback_policy = base_rule.fallback_policy
        policy_reason_parts.append("unstable_profile_fallback")

    applied_rule = _apply_fallback_policy(base_rule, config.default, fallback_policy)

    decision = {
        "symbol": None if "symbol" not in row.index or pd.isna(row["symbol"]) else str(row["symbol"]),
        "ts_utc": pd.Timestamp(row["ts_utc"]),
        "regime_label": exact_label,
        "matched_policy_key": matched_policy_key,
        "matched_alias": matched_alias,
        "confidence_score": None if pd.isna(row.get("confidence_score")) else float(row["confidence_score"]),
        "confidence_bucket": None if pd.isna(row.get("confidence_bucket")) else str(row["confidence_bucket"]),
        "confidence_fallback_reason": confidence_reason,
        "fallback_policy_applied": fallback_policy,
        "policy_reason": ";".join(policy_reason_parts),
        "signal_scale": float(applied_rule.signal_scale),
        "allocation_scale": float(applied_rule.allocation_scale),
        "alpha_weight_multiplier": float(applied_rule.alpha_weight_multiplier),
        "volatility_target": float(applied_rule.volatility_target),
        "gross_exposure_cap": float(applied_rule.gross_exposure_cap),
        "max_component_weight": float(applied_rule.max_component_weight),
        "rebalance_enabled": bool(applied_rule.rebalance_enabled),
        "optimizer_override": applied_rule.optimizer_override,
        "allocation_rule_override": applied_rule.allocation_rule_override,
        "calibration_profile": calibration_profile,
        "is_unstable_profile": bool(is_unstable_profile),
        "eligible_for_downstream_decisioning": eligible_for_downstream_decisioning,
        "policy_source": policy_source,
        "baseline_weight": None,
        "adaptive_weight": None,
        "baseline_return": None,
        "adaptive_return": None,
    }
    return decision


def _confidence_reason(row: pd.Series, confidence: RegimePolicyConfidenceConfig) -> str | None:
    fallback_reason = row.get("fallback_reason")
    if pd.notna(fallback_reason):
        return str(fallback_reason)
    score = row.get("confidence_score")
    if pd.notna(score) and float(score) < float(confidence.min_confidence):
        return "low_confidence"
    return None


def _apply_fallback_policy(
    rule: RegimePolicyRule,
    default_rule: RegimePolicyRule,
    fallback_policy: str | None,
) -> RegimePolicyRule:
    if fallback_policy is None or fallback_policy == "baseline":
        return default_rule if fallback_policy == "baseline" else rule
    if fallback_policy == "skip_adaptation":
        return replace(
            default_rule,
            signal_scale=1.0,
            allocation_scale=1.0,
            alpha_weight_multiplier=1.0,
            optimizer_override=None,
            allocation_rule_override=None,
        )
    if fallback_policy == "neutral":
        return replace(
            default_rule,
            signal_scale=0.0,
            allocation_scale=0.0,
            alpha_weight_multiplier=0.0,
            gross_exposure_cap=0.0,
            max_component_weight=1.0e-12,
            rebalance_enabled=False,
            optimizer_override=None,
            allocation_rule_override=None,
        )
    if fallback_policy == "cash_proxy":
        return replace(
            default_rule,
            signal_scale=0.0,
            allocation_scale=0.0,
            alpha_weight_multiplier=0.0,
            gross_exposure_cap=0.0,
            max_component_weight=1.0e-12,
            rebalance_enabled=True,
            optimizer_override="equal_weight" if default_rule.optimizer_override is not None else None,
            allocation_rule_override=None,
        )
    if fallback_policy == "reduce_exposure":
        return replace(
            rule,
            signal_scale=min(rule.signal_scale, 0.5),
            allocation_scale=min(rule.allocation_scale, 0.5),
            alpha_weight_multiplier=min(rule.alpha_weight_multiplier, 0.5),
            volatility_target=min(rule.volatility_target, default_rule.volatility_target),
            gross_exposure_cap=min(rule.gross_exposure_cap, 0.5),
            max_component_weight=min(rule.max_component_weight, 0.5),
        )
    raise RegimePolicyError(f"Unsupported fallback policy {fallback_policy!r}.")


def _match_alias(row: pd.Series, aliases: Mapping[str, RegimePolicyAlias]) -> str | None:
    for alias_name, alias in aliases.items():
        if _row_matches_alias(row, alias):
            return alias_name
    return None


def _row_matches_alias(row: pd.Series, alias: RegimePolicyAlias) -> bool:
    for key, expected in alias.match.items():
        if key == "is_defined":
            if bool(row["is_defined"]) is not bool(expected):
                return False
            continue
        actual = row.get(key)
        if pd.isna(actual):
            return False
        if str(actual) != str(expected):
            return False
    return True


def _normalize_baseline_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("baseline_frame must be a pandas DataFrame.")
    if "ts_utc" not in frame.columns:
        raise RegimePolicyError("baseline_frame must include 'ts_utc'.")
    normalized = frame.copy(deep=True)
    normalized.attrs = {}
    normalized["ts_utc"] = pd.to_datetime(normalized["ts_utc"], utc=True, errors="coerce")
    if normalized["ts_utc"].isna().any():
        raise RegimePolicyError("baseline_frame contains invalid ts_utc values.")
    if "symbol" in normalized.columns:
        normalized["symbol"] = normalized["symbol"].astype("string")
        join_keys = ["symbol", "ts_utc"]
    else:
        join_keys = ["ts_utc"]
    if normalized.duplicated(subset=join_keys).any():
        raise RegimePolicyError("baseline_frame must be unique on its policy join keys.")
    return normalized


def _resolve_return_column(
    baseline: pd.DataFrame,
    *,
    surface: PolicySurface,
    return_column: str | None,
) -> str | None:
    if return_column is not None:
        if return_column not in baseline.columns:
            raise RegimePolicyError(f"baseline_frame is missing return column {return_column!r}.")
        return return_column
    candidate = DEFAULT_POLICY_RETURN_COLUMNS[surface]
    return candidate if candidate in baseline.columns else None


def _resolve_weight_column(baseline: pd.DataFrame, *, requested: str | None) -> str | None:
    if requested is not None:
        if requested not in baseline.columns:
            raise RegimePolicyError(f"baseline_frame is missing weight column {requested!r}.")
        return requested
    for candidate in DEFAULT_POLICY_WEIGHT_COLUMNS:
        if candidate in baseline.columns:
            return candidate
    return None


def _join_baseline_to_decisions(
    decisions: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    return_column: str | None,
    weight_column: str | None,
) -> pd.DataFrame:
    join_keys = ["symbol", "ts_utc"] if "symbol" in decisions.columns and decisions["symbol"].notna().any() and "symbol" in baseline.columns else ["ts_utc"]
    columns = list(join_keys)
    if return_column is not None:
        columns.append(return_column)
    if weight_column is not None:
        columns.append(weight_column)
    merged = decisions.merge(
        baseline.loc[:, columns],
        on=join_keys,
        how="left",
        sort=False,
    )
    if return_column is not None:
        merged["baseline_return"] = pd.to_numeric(merged[return_column], errors="coerce")
        merged = merged.drop(columns=return_column)
    else:
        merged["baseline_return"] = pd.Series([None] * len(merged), index=merged.index, dtype="object")
    if weight_column is not None:
        merged["baseline_weight"] = pd.to_numeric(merged[weight_column], errors="coerce")
        merged = merged.drop(columns=weight_column)
    else:
        merged["baseline_weight"] = pd.Series([None] * len(merged), index=merged.index, dtype="object")
    return merged


def _add_equity_curves(decisions: pd.DataFrame) -> pd.DataFrame:
    enriched = decisions.copy(deep=True)
    enriched.attrs = {}
    valid_returns = pd.to_numeric(enriched["baseline_return"], errors="coerce").fillna(0.0).astype("float64")
    valid_adaptive = pd.to_numeric(enriched["adaptive_return"], errors="coerce").fillna(0.0).astype("float64")
    enriched["baseline_equity"] = (1.0 + valid_returns).cumprod()
    enriched["adaptive_equity"] = (1.0 + valid_adaptive).cumprod()
    return enriched


def _build_comparison_frame(decisions: pd.DataFrame, *, surface: PolicySurface) -> pd.DataFrame:
    baseline = pd.to_numeric(decisions["baseline_return"], errors="coerce").dropna().astype("float64")
    adaptive = pd.to_numeric(decisions["adaptive_return"], errors="coerce").dropna().astype("float64")
    periods_per_year = infer_periods_per_year(
        decisions.loc[:, [column for column in ("ts_utc",) if column in decisions.columns]]
    )
    comparison = pd.DataFrame(
        [
            canonicalize_value(
                {
                    "surface": surface,
                    "baseline_total_return": total_return(baseline),
                    "adaptive_total_return": total_return(adaptive),
                    "baseline_volatility": volatility(baseline),
                    "adaptive_volatility": volatility(adaptive),
                    "baseline_max_drawdown": max_drawdown(baseline),
                    "adaptive_max_drawdown": max_drawdown(adaptive),
                    "baseline_sharpe": sharpe_ratio(baseline, periods_per_year=periods_per_year),
                    "adaptive_sharpe": sharpe_ratio(adaptive, periods_per_year=periods_per_year),
                    "average_signal_scale": float(pd.to_numeric(decisions["signal_scale"], errors="coerce").mean()),
                    "average_allocation_scale": float(pd.to_numeric(decisions["allocation_scale"], errors="coerce").mean()),
                    "fallback_row_count": int(decisions["fallback_policy_applied"].notna().sum()),
                    "low_confidence_fallback_count": int(decisions["confidence_fallback_reason"].eq("low_confidence").sum()),
                    "unknown_regime_fallback_count": int(decisions["policy_reason"].str.contains("undefined_regime_fallback", regex=False).sum()),
                }
            )
        ]
    )
    comparison.attrs = {}
    return comparison


def _build_policy_summary(
    decisions: pd.DataFrame,
    *,
    config: RegimePolicyConfig,
    comparison: pd.DataFrame | None,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    fallback_counts = {
        "ambiguous": int(decisions["confidence_fallback_reason"].eq("ambiguous").sum()),
        "low_confidence": int(decisions["confidence_fallback_reason"].eq("low_confidence").sum()),
        "policy_fallback_rows": int(decisions["fallback_policy_applied"].notna().sum()),
        "undefined_regime": int(decisions["policy_reason"].str.contains("undefined_regime_fallback", regex=False).sum()),
        "unsupported": int(decisions["confidence_fallback_reason"].eq("unsupported").sum()),
    }
    payload: dict[str, Any] = {
        "artifact_type": "regime_policy_summary",
        "schema_version": 1,
        "taxonomy_version": TAXONOMY_VERSION,
        "row_count": int(len(decisions)),
        "policy_keys": sorted(decisions["matched_policy_key"].astype("string").unique().tolist()),
        "decision_counts_by_policy_key": {
            str(key): int(value)
            for key, value in decisions["matched_policy_key"].astype("string").value_counts(sort=False).sort_index().items()
        },
        "fallback_counts": fallback_counts,
        "policy_config": config.to_dict(),
        "columns": list(decisions.columns),
        "metadata": canonicalize_value(dict(metadata or {})),
    }
    if comparison is not None and not comparison.empty:
        payload["comparison_metrics"] = canonicalize_value(comparison.to_dict(orient="records"))
    return canonicalize_value(payload)


def _sort_policy_decisions(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.copy(deep=True)
    ordered.attrs = {}
    if "symbol" in ordered.columns:
        ordered = ordered.sort_values(["ts_utc", "symbol"], kind="stable", na_position="last")
    else:
        ordered = ordered.sort_values(["ts_utc"], kind="stable")
    for column in _POLICY_DECISION_COLUMNS:
        if column not in ordered.columns:
            ordered[column] = None
    if "ts_utc" in ordered.columns:
        ordered["ts_utc"] = pd.to_datetime(ordered["ts_utc"], utc=True, errors="raise")
    for column in (
        "symbol",
        "matched_policy_key",
        "matched_alias",
        "confidence_bucket",
        "confidence_fallback_reason",
        "fallback_policy_applied",
        "policy_reason",
        "optimizer_override",
        "allocation_rule_override",
        "calibration_profile",
        "policy_source",
    ):
        if column in ordered.columns:
            ordered[column] = ordered[column].astype("object").where(pd.notna(ordered[column]), None)
    extra_columns = [
        column
        for column in ordered.columns
        if column not in _POLICY_DECISION_COLUMNS and column not in {"baseline_equity", "adaptive_equity"}
    ]
    ordered_columns = list(_POLICY_DECISION_COLUMNS)
    for column in ("baseline_equity", "adaptive_equity"):
        if column in ordered.columns:
            ordered_columns.append(column)
    ordered_columns.extend(sorted(extra_columns))
    return ordered.loc[:, ordered_columns].reset_index(drop=True)


def _scale_column_for_surface(surface: PolicySurface) -> str:
    return {
        "alpha": "alpha_weight_multiplier",
        "portfolio": "allocation_scale",
        "strategy": "signal_scale",
    }[surface]


def _normalize_alias_match_key(value: Any) -> str:
    key = _normalize_non_empty_string(value, field_name="alias match key")
    if key in REGIME_DIMENSIONS:
        return REGIME_STATE_COLUMNS[key]
    if key in {"is_defined", "regime_label"}:
        return key
    if key in REGIME_STATE_COLUMNS.values():
        return key
    raise RegimePolicyError(
        "Alias matches must target existing regime dimensions, state columns, 'regime_label', or 'is_defined'. "
        f"Unsupported key: {key!r}."
    )


def _dimension_for_state_column(column: str) -> str:
    for dimension, state_column in REGIME_STATE_COLUMNS.items():
        if state_column == column:
            return dimension
    raise RegimePolicyError(f"Unsupported regime state column {column!r}.")


def _allowed_states_for_dimension(dimension: str) -> tuple[str, ...]:
    from src.research.regimes.taxonomy import REGIME_TAXONOMY

    return REGIME_TAXONOMY[dimension].labels


def _is_canonical_regime_label(value: str) -> bool:
    try:
        _parse_composite_label(value, field_name="regime_policy rule key")
    except Exception:
        return False
    return True


def _first_present_value(frame: pd.DataFrame, column: str) -> str | None:
    if column not in frame.columns:
        return None
    non_null = frame[column].dropna()
    return None if non_null.empty else str(non_null.iloc[0])


def _first_present_bool(frame: pd.DataFrame, column: str) -> bool | None:
    if column not in frame.columns:
        return None
    non_null = frame[column].dropna()
    return None if non_null.empty else bool(non_null.iloc[0])


def _normalize_non_empty_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RegimePolicyError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _normalize_optional_string(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _normalize_non_empty_string(value, field_name=field_name)


def _normalize_optimizer_override(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = _normalize_non_empty_string(value, field_name=field_name).lower()
    if normalized not in SUPPORTED_PORTFOLIO_OPTIMIZERS:
        supported = ", ".join(SUPPORTED_PORTFOLIO_OPTIMIZERS)
        raise RegimePolicyError(f"{field_name} must be one of {supported} when provided.")
    return normalized


def _normalize_fallback_policy(value: Any, *, field_name: str) -> str:
    normalized = _normalize_non_empty_string(value, field_name=field_name)
    if normalized not in SUPPORTED_FALLBACK_POLICIES:
        raise RegimePolicyError(
            f"{field_name} must be one of {list(SUPPORTED_FALLBACK_POLICIES)!r}."
        )
    return normalized


def _coerce_bool(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise RegimePolicyError(f"{field_name} must be boolean.")
    return value


def _coerce_probability(value: Any, *, field_name: str) -> float:
    return _coerce_bounded_float(
        value,
        lower=0.0,
        upper=1.0,
        lower_inclusive=True,
        upper_inclusive=True,
        field_name=field_name,
    )


def _coerce_positive_float(value: Any, *, field_name: str) -> float:
    numeric = _coerce_finite_float(value, field_name=field_name)
    if numeric <= 0.0:
        raise RegimePolicyError(f"{field_name} must be positive.")
    return numeric


def _coerce_non_negative_float(value: Any, *, field_name: str) -> float:
    numeric = _coerce_finite_float(value, field_name=field_name)
    if numeric < 0.0:
        raise RegimePolicyError(f"{field_name} must be non-negative.")
    return numeric


def _coerce_bounded_float(
    value: Any,
    *,
    lower: float,
    upper: float,
    lower_inclusive: bool,
    upper_inclusive: bool,
    field_name: str,
) -> float:
    numeric = _coerce_finite_float(value, field_name=field_name)
    lower_ok = numeric >= lower if lower_inclusive else numeric > lower
    upper_ok = numeric <= upper if upper_inclusive else numeric < upper
    if not lower_ok or not upper_ok:
        raise RegimePolicyError(
            f"{field_name} must be within bounds "
            f"{'[' if lower_inclusive else '('}{lower}, {upper}{']' if upper_inclusive else ')'}."
        )
    return numeric


def _coerce_finite_float(value: Any, *, field_name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise RegimePolicyError(f"{field_name} must be a finite numeric value.") from exc
    if not math.isfinite(numeric):
        raise RegimePolicyError(f"{field_name} must be a finite numeric value.")
    return numeric


__all__ = [
    "DEFAULT_POLICY_RETURN_COLUMNS",
    "PolicySurface",
    "RegimePolicyAlias",
    "RegimePolicyConfidenceConfig",
    "RegimePolicyConfig",
    "RegimePolicyDecisionResult",
    "RegimePolicyError",
    "RegimePolicyRule",
    "SUPPORTED_FALLBACK_POLICIES",
    "apply_regime_policy",
    "compare_adaptive_vs_static",
    "resolve_regime_policy_config",
]
