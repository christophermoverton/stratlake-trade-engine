from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.research.registry import canonicalize_value


class RegimePromotionGateConfigError(ValueError):
    """Raised when a regime promotion-gate configuration is malformed."""


_ROOT_KEYS = frozenset(
    {
        "gate_name",
        "decision_policy",
        "outcomes",
        "missing_metric_policy",
        "confidence",
        "entropy",
        "stability",
        "transition_behavior",
        "adaptive_uplift",
        "drawdown",
        "policy_turnover",
        "metadata",
    }
)
_OUTCOME_KEYS = frozenset({"pass", "warn", "review", "fail"})
_MISSING_POLICY_KEYS = frozenset(
    {"required_metric_missing", "optional_metric_missing", "not_applicable_metric"}
)
_CATEGORY_KEYS = frozenset({"enabled", "required_for_variants", "failure_impact"})
_CATEGORY_THRESHOLD_KEYS: dict[str, frozenset[str]] = {
    "confidence": frozenset(
        {"min_mean_confidence", "min_median_confidence", "max_low_confidence_pct"}
    ),
    "entropy": frozenset({"max_mean_entropy", "max_high_entropy_pct"}),
    "stability": frozenset(
        {"min_avg_regime_duration", "max_dominant_regime_share", "max_transition_rate"}
    ),
    "transition_behavior": frozenset(
        {
            "max_transition_rate",
            "max_transition_instability_score",
            "max_transition_concentration",
        }
    ),
    "adaptive_uplift": frozenset(
        {"min_return_delta", "min_sharpe_delta", "min_drawdown_delta"}
    ),
    "drawdown": frozenset({"max_allowed_drawdown", "max_high_vol_drawdown"}),
    "policy_turnover": frozenset({"max_policy_turnover", "max_policy_state_changes"}),
}
_VALID_IMPACTS = frozenset({"pass", "warn", "review", "fail", "ignore"})
_VALID_OUTCOMES = frozenset(
    {"accepted", "accepted_with_warnings", "needs_review", "rejected"}
)
_VALID_VARIANTS = frozenset(
    {
        "static_baseline",
        "taxonomy_only",
        "calibrated_taxonomy",
        "gmm_classifier",
        "gmm_calibrated_overlay",
        "policy_optimized",
    }
)


@dataclass(frozen=True)
class MissingMetricPolicy:
    required_metric_missing: str = "fail"
    optional_metric_missing: str = "warn"
    not_applicable_metric: str = "ignore"

    def to_dict(self) -> dict[str, Any]:
        return {
            "required_metric_missing": self.required_metric_missing,
            "optional_metric_missing": self.optional_metric_missing,
            "not_applicable_metric": self.not_applicable_metric,
        }


@dataclass(frozen=True)
class GateCategoryConfig:
    name: str
    enabled: bool = True
    thresholds: dict[str, float] = field(default_factory=dict)
    required_for_variants: tuple[str, ...] = ()
    failure_impact: str = "fail"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "enabled": self.enabled,
            "required_for_variants": list(self.required_for_variants),
            "failure_impact": self.failure_impact,
        }
        payload.update(self.thresholds)
        return canonicalize_value(payload)


@dataclass(frozen=True)
class RegimePromotionGateConfig:
    gate_name: str
    decision_policy: str
    outcomes: dict[str, str]
    missing_metric_policy: MissingMetricPolicy
    categories: dict[str, GateCategoryConfig]
    metadata: dict[str, Any] = field(default_factory=dict)

    def category(self, name: str) -> GateCategoryConfig:
        return self.categories.get(name, GateCategoryConfig(name=name, enabled=False))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "gate_name": self.gate_name,
            "decision_policy": self.decision_policy,
            "outcomes": dict(self.outcomes),
            "missing_metric_policy": self.missing_metric_policy.to_dict(),
            "metadata": dict(self.metadata),
        }
        for category_name in _CATEGORY_THRESHOLD_KEYS:
            payload[category_name] = self.category(category_name).to_dict()
        return canonicalize_value(payload)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RegimePromotionGateConfig":
        unknown_keys = sorted(set(payload) - _ROOT_KEYS)
        if unknown_keys:
            raise RegimePromotionGateConfigError(
                f"Regime promotion-gate config contains unsupported keys: {unknown_keys}."
            )

        outcomes = _resolve_outcomes(payload.get("outcomes"))
        missing_policy = _resolve_missing_policy(payload.get("missing_metric_policy"))
        categories = {
            name: _resolve_category(name, payload.get(name))
            for name in _CATEGORY_THRESHOLD_KEYS
        }
        return cls(
            gate_name=_normalize_required_string(
                payload.get("gate_name"), field_name="gate_name"
            ),
            decision_policy=_normalize_required_string(
                payload.get("decision_policy", "strict_with_warnings"),
                field_name="decision_policy",
            ),
            outcomes=outcomes,
            missing_metric_policy=missing_policy,
            categories=categories,
            metadata=_resolve_optional_mapping(payload.get("metadata"), field_name="metadata"),
        )


def load_regime_promotion_gate_config(path: Path) -> RegimePromotionGateConfig:
    if not path.exists():
        raise RegimePromotionGateConfigError(
            f"Regime promotion-gate config does not exist: {path.as_posix()}"
        )

    with path.open("r", encoding="utf-8") as handle:
        suffix = path.suffix.lower()
        if suffix == ".json":
            payload = json.load(handle)
        elif suffix in {".yml", ".yaml"}:
            payload = yaml.safe_load(handle) or {}
        else:
            raise RegimePromotionGateConfigError(
                f"Unsupported regime promotion-gate config format {path.suffix!r}. "
                "Use JSON, YAML, or YML."
            )
    if not isinstance(payload, Mapping):
        raise RegimePromotionGateConfigError(
            "Regime promotion-gate config must contain a top-level mapping."
        )
    return RegimePromotionGateConfig.from_mapping(payload)


def _resolve_outcomes(value: Any) -> dict[str, str]:
    default = {
        "pass": "accepted",
        "warn": "accepted_with_warnings",
        "review": "needs_review",
        "fail": "rejected",
    }
    if value is None:
        return default
    if not isinstance(value, Mapping):
        raise RegimePromotionGateConfigError(
            "Regime promotion-gate field 'outcomes' must be a mapping."
        )
    unknown_keys = sorted(set(value) - _OUTCOME_KEYS)
    if unknown_keys:
        raise RegimePromotionGateConfigError(
            f"Regime promotion-gate field 'outcomes' contains unsupported keys: {unknown_keys}."
        )
    resolved = dict(default)
    for key, outcome in value.items():
        resolved[key] = _normalize_required_string(outcome, field_name=f"outcomes.{key}")
        if resolved[key] not in _VALID_OUTCOMES:
            expected = ", ".join(sorted(_VALID_OUTCOMES))
            raise RegimePromotionGateConfigError(
                f"Regime promotion-gate field 'outcomes.{key}' must be one of: {expected}."
            )
    return resolved


def _resolve_missing_policy(value: Any) -> MissingMetricPolicy:
    payload = {} if value is None else value
    if not isinstance(payload, Mapping):
        raise RegimePromotionGateConfigError(
            "Regime promotion-gate field 'missing_metric_policy' must be a mapping."
        )
    unknown_keys = sorted(set(payload) - _MISSING_POLICY_KEYS)
    if unknown_keys:
        raise RegimePromotionGateConfigError(
            "Regime promotion-gate field 'missing_metric_policy' contains "
            f"unsupported keys: {unknown_keys}."
        )
    values = {
        "required_metric_missing": payload.get("required_metric_missing", "fail"),
        "optional_metric_missing": payload.get("optional_metric_missing", "warn"),
        "not_applicable_metric": payload.get("not_applicable_metric", "ignore"),
    }
    for key, impact in values.items():
        values[key] = _normalize_impact(impact, field_name=f"missing_metric_policy.{key}")
    return MissingMetricPolicy(**values)


def _resolve_category(name: str, value: Any) -> GateCategoryConfig:
    if value is None:
        return GateCategoryConfig(name=name, enabled=False)
    if not isinstance(value, Mapping):
        raise RegimePromotionGateConfigError(
            f"Regime promotion-gate field '{name}' must be a mapping."
        )
    supported = _CATEGORY_KEYS | _CATEGORY_THRESHOLD_KEYS[name]
    unknown_keys = sorted(set(value) - supported)
    if unknown_keys:
        raise RegimePromotionGateConfigError(
            f"Regime promotion-gate field '{name}' contains unsupported keys: {unknown_keys}."
        )
    enabled = _coerce_bool(value.get("enabled", True), field_name=f"{name}.enabled")
    thresholds: dict[str, float] = {}
    for threshold_name in sorted(_CATEGORY_THRESHOLD_KEYS[name]):
        if threshold_name not in value:
            continue
        thresholds[threshold_name] = _normalize_numeric(
            value[threshold_name], field_name=f"{name}.{threshold_name}"
        )
    return GateCategoryConfig(
        name=name,
        enabled=enabled,
        thresholds=thresholds,
        required_for_variants=_normalize_variant_sequence(
            value.get("required_for_variants", ()),
            field_name=f"{name}.required_for_variants",
        ),
        failure_impact=_normalize_impact(
            value.get("failure_impact", "fail"), field_name=f"{name}.failure_impact"
        ),
    )


def _normalize_variant_sequence(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise RegimePromotionGateConfigError(
            f"Regime promotion-gate field '{field_name}' must be a sequence."
        )
    variants = tuple(_normalize_required_string(item, field_name=field_name) for item in value)
    unknown = sorted(set(variants) - _VALID_VARIANTS)
    if unknown:
        expected = ", ".join(sorted(_VALID_VARIANTS))
        raise RegimePromotionGateConfigError(
            f"Regime promotion-gate field '{field_name}' contains unsupported variants "
            f"{unknown}. Expected one of: {expected}."
        )
    return variants


def _normalize_impact(value: Any, *, field_name: str) -> str:
    impact = _normalize_required_string(value, field_name=field_name)
    if impact not in _VALID_IMPACTS:
        expected = ", ".join(sorted(_VALID_IMPACTS))
        raise RegimePromotionGateConfigError(
            f"Regime promotion-gate field '{field_name}' must be one of: {expected}."
        )
    return impact


def _normalize_numeric(value: Any, *, field_name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise RegimePromotionGateConfigError(
            f"Regime promotion-gate field '{field_name}' must be numeric."
        )
    return float(value)


def _normalize_required_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise RegimePromotionGateConfigError(
            f"Regime promotion-gate field '{field_name}' must be a non-empty string."
        )
    normalized = value.strip()
    if not normalized:
        raise RegimePromotionGateConfigError(
            f"Regime promotion-gate field '{field_name}' must be a non-empty string."
        )
    return normalized


def _resolve_optional_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RegimePromotionGateConfigError(
            f"Regime promotion-gate field '{field_name}' must be a mapping."
        )
    return canonicalize_value(dict(value))


def _coerce_bool(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise RegimePromotionGateConfigError(
            f"Regime promotion-gate field '{field_name}' must be boolean."
        )
    return value


__all__ = [
    "GateCategoryConfig",
    "MissingMetricPolicy",
    "RegimePromotionGateConfig",
    "RegimePromotionGateConfigError",
    "load_regime_promotion_gate_config",
]
