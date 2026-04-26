from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.research.registry import canonicalize_value


class RegimeBenchmarkPackConfigError(ValueError):
    """Raised when a regime benchmark-pack configuration is malformed."""


_ROOT_KEYS = frozenset(
    {
        "benchmark_name",
        "dataset",
        "timeframe",
        "start",
        "end",
        "symbols",
        "features_root",
        "variants",
        "metrics",
        "output_root",
        "classification",
        "gmm",
        "transition",
        "policy_surface",
        "policy_return_column",
        "policy_weight_column",
        "metadata",
    }
)
_VARIANT_KEYS = frozenset(
    {
        "name",
        "regime_source",
        "calibration_profile",
        "policy_config",
    }
)
_METRIC_KEYS = frozenset(
    {
        "include_transition_metrics",
        "include_stability_metrics",
        "include_policy_metrics",
        "include_conditional_performance",
        "high_entropy_threshold",
    }
)
_SUPPORTED_REGIME_SOURCES = frozenset({"static", "taxonomy", "gmm", "policy"})
_SUPPORTED_POLICY_SURFACES = frozenset({"alpha", "portfolio", "strategy"})


@dataclass(frozen=True)
class RegimeBenchmarkVariantConfig:
    name: str
    regime_source: str
    calibration_profile: str | None = None
    policy_config: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "regime_source": self.regime_source,
            "calibration_profile": self.calibration_profile,
            "policy_config": self.policy_config,
        }


@dataclass(frozen=True)
class RegimeBenchmarkMetricsConfig:
    include_transition_metrics: bool = True
    include_stability_metrics: bool = True
    include_policy_metrics: bool = True
    include_conditional_performance: bool = True
    high_entropy_threshold: float = 0.75

    def to_dict(self) -> dict[str, Any]:
        return {
            "include_transition_metrics": self.include_transition_metrics,
            "include_stability_metrics": self.include_stability_metrics,
            "include_policy_metrics": self.include_policy_metrics,
            "include_conditional_performance": self.include_conditional_performance,
            "high_entropy_threshold": self.high_entropy_threshold,
        }


@dataclass(frozen=True)
class RegimeBenchmarkPackConfig:
    benchmark_name: str
    dataset: str
    timeframe: str
    start: str
    end: str
    symbols: tuple[str, ...]
    variants: tuple[RegimeBenchmarkVariantConfig, ...]
    metrics: RegimeBenchmarkMetricsConfig
    features_root: str = "data"
    output_root: str = "artifacts/regime_benchmarks"
    classification: dict[str, Any] = field(default_factory=dict)
    gmm: dict[str, Any] = field(default_factory=dict)
    transition: dict[str, Any] = field(default_factory=dict)
    policy_surface: str = "portfolio"
    policy_return_column: str | None = "portfolio_return"
    policy_weight_column: str | None = "weight"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return canonicalize_value(
            {
                "benchmark_name": self.benchmark_name,
                "dataset": self.dataset,
                "timeframe": self.timeframe,
                "start": self.start,
                "end": self.end,
                "symbols": list(self.symbols),
                "features_root": self.features_root,
                "variants": [variant.to_dict() for variant in self.variants],
                "metrics": self.metrics.to_dict(),
                "output_root": self.output_root,
                "classification": dict(self.classification),
                "gmm": dict(self.gmm),
                "transition": dict(self.transition),
                "policy_surface": self.policy_surface,
                "policy_return_column": self.policy_return_column,
                "policy_weight_column": self.policy_weight_column,
                "metadata": dict(self.metadata),
            }
        )


def load_regime_benchmark_pack_config(path: Path) -> RegimeBenchmarkPackConfig:
    if not path.exists():
        raise RegimeBenchmarkPackConfigError(
            f"Regime benchmark-pack config does not exist: {path.as_posix()}"
        )

    with path.open("r", encoding="utf-8") as handle:
        suffix = path.suffix.lower()
        if suffix == ".json":
            payload = json.load(handle)
        elif suffix in {".yml", ".yaml"}:
            payload = yaml.safe_load(handle) or {}
        else:
            raise RegimeBenchmarkPackConfigError(
                f"Unsupported regime benchmark-pack config format {path.suffix!r}. "
                "Use JSON, YAML, or YML."
            )

    if not isinstance(payload, Mapping):
        raise RegimeBenchmarkPackConfigError(
            "Regime benchmark-pack config must contain a top-level mapping."
        )
    return RegimeBenchmarkPackConfig.from_mapping(payload, config_dir=path.parent)


@classmethod
def _from_mapping(
    cls,
    payload: Mapping[str, Any],
    *,
    config_dir: Path | None = None,
) -> RegimeBenchmarkPackConfig:
    unknown_keys = sorted(set(payload) - _ROOT_KEYS)
    if unknown_keys:
        raise RegimeBenchmarkPackConfigError(
            f"Regime benchmark-pack config contains unsupported keys: {unknown_keys}."
        )

    variants = _resolve_variants(payload.get("variants"), config_dir=config_dir)
    if len(variants) < 1:
        raise RegimeBenchmarkPackConfigError(
            "Regime benchmark-pack config must define at least one variant."
        )

    benchmark_name = _normalize_required_string(
        payload.get("benchmark_name"),
        field_name="benchmark_name",
    )
    dataset = _normalize_required_string(payload.get("dataset"), field_name="dataset")
    timeframe = _normalize_required_string(payload.get("timeframe"), field_name="timeframe")
    start = _normalize_required_string(payload.get("start"), field_name="start")
    end = _normalize_required_string(payload.get("end"), field_name="end")
    features_root = _normalize_relative_path_string(
        payload.get("features_root", "data"),
        field_name="features_root",
    )
    output_root = _normalize_relative_path_string(
        payload.get("output_root", "artifacts/regime_benchmarks"),
        field_name="output_root",
    )
    symbols = _normalize_optional_string_sequence(payload.get("symbols"), field_name="symbols")
    metrics = _resolve_metrics(payload.get("metrics"))
    classification = _resolve_optional_mapping(payload.get("classification"), field_name="classification")
    gmm = _resolve_optional_mapping(payload.get("gmm"), field_name="gmm")
    transition = _resolve_optional_mapping(payload.get("transition"), field_name="transition")
    policy_surface = _normalize_required_string(
        payload.get("policy_surface", "portfolio"),
        field_name="policy_surface",
    )
    if policy_surface not in _SUPPORTED_POLICY_SURFACES:
        expected = ", ".join(sorted(_SUPPORTED_POLICY_SURFACES))
        raise RegimeBenchmarkPackConfigError(
            f"Regime benchmark-pack field 'policy_surface' must be one of: {expected}."
        )
    policy_return_column = _normalize_optional_string(
        payload.get("policy_return_column", "portfolio_return"),
        field_name="policy_return_column",
    )
    policy_weight_column = _normalize_optional_string(
        payload.get("policy_weight_column", "weight"),
        field_name="policy_weight_column",
    )
    metadata = _resolve_optional_mapping(payload.get("metadata"), field_name="metadata")

    return RegimeBenchmarkPackConfig(
        benchmark_name=benchmark_name,
        dataset=dataset,
        timeframe=timeframe,
        start=start,
        end=end,
        symbols=symbols,
        variants=variants,
        metrics=metrics,
        features_root=features_root,
        output_root=output_root,
        classification=classification,
        gmm=gmm,
        transition=transition,
        policy_surface=policy_surface,
        policy_return_column=policy_return_column,
        policy_weight_column=policy_weight_column,
        metadata=metadata,
    )


RegimeBenchmarkPackConfig.from_mapping = _from_mapping  # type: ignore[attr-defined]


def _resolve_variants(
    payload: Any,
    *,
    config_dir: Path | None,
) -> tuple[RegimeBenchmarkVariantConfig, ...]:
    if not isinstance(payload, list):
        raise RegimeBenchmarkPackConfigError(
            "Regime benchmark-pack field 'variants' must be a sequence."
        )

    variants: list[RegimeBenchmarkVariantConfig] = []
    seen_names: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise RegimeBenchmarkPackConfigError(
                f"Regime benchmark-pack variant at index {index} must be a mapping."
            )
        unknown_keys = sorted(set(item) - _VARIANT_KEYS)
        if unknown_keys:
            raise RegimeBenchmarkPackConfigError(
                f"Regime benchmark-pack variant {index} contains unsupported keys: {unknown_keys}."
            )

        name = _normalize_required_string(item.get("name"), field_name=f"variants[{index}].name")
        if name in seen_names:
            raise RegimeBenchmarkPackConfigError(
                f"Regime benchmark-pack variant names must be unique. Duplicate: {name!r}."
            )
        seen_names.add(name)

        regime_source = _normalize_required_string(
            item.get("regime_source"),
            field_name=f"variants[{index}].regime_source",
        )
        if regime_source not in _SUPPORTED_REGIME_SOURCES:
            expected = ", ".join(sorted(_SUPPORTED_REGIME_SOURCES))
            raise RegimeBenchmarkPackConfigError(
                f"Regime benchmark-pack field 'variants[{index}].regime_source' must be one of: {expected}."
            )

        calibration_profile = _normalize_optional_string(
            item.get("calibration_profile"),
            field_name=f"variants[{index}].calibration_profile",
        )
        policy_config = item.get("policy_config")
        resolved_policy_config = None
        if policy_config is not None:
            resolved_policy_config = _normalize_existing_path_string(
                policy_config,
                field_name=f"variants[{index}].policy_config",
                config_dir=config_dir,
            )
        if regime_source == "policy" and resolved_policy_config is None:
            raise RegimeBenchmarkPackConfigError(
                f"Regime benchmark-pack variant {name!r} requires 'policy_config' when regime_source='policy'."
            )

        variants.append(
            RegimeBenchmarkVariantConfig(
                name=name,
                regime_source=regime_source,
                calibration_profile=calibration_profile,
                policy_config=resolved_policy_config,
            )
        )
    return tuple(variants)


def _resolve_metrics(payload: Any) -> RegimeBenchmarkMetricsConfig:
    if payload is None:
        return RegimeBenchmarkMetricsConfig()
    if not isinstance(payload, Mapping):
        raise RegimeBenchmarkPackConfigError(
            "Regime benchmark-pack field 'metrics' must be a mapping."
        )
    unknown_keys = sorted(set(payload) - _METRIC_KEYS)
    if unknown_keys:
        raise RegimeBenchmarkPackConfigError(
            f"Regime benchmark-pack field 'metrics' contains unsupported keys: {unknown_keys}."
        )
    high_entropy_threshold = payload.get("high_entropy_threshold", 0.75)
    if not isinstance(high_entropy_threshold, int | float):
        raise RegimeBenchmarkPackConfigError(
            "Regime benchmark-pack field 'metrics.high_entropy_threshold' must be numeric."
        )
    high_entropy_threshold = float(high_entropy_threshold)
    if high_entropy_threshold < 0.0 or high_entropy_threshold > 1.0:
        raise RegimeBenchmarkPackConfigError(
            "Regime benchmark-pack field 'metrics.high_entropy_threshold' must be between 0.0 and 1.0."
        )
    return RegimeBenchmarkMetricsConfig(
        include_transition_metrics=_coerce_bool(
            payload.get("include_transition_metrics", True),
            field_name="metrics.include_transition_metrics",
        ),
        include_stability_metrics=_coerce_bool(
            payload.get("include_stability_metrics", True),
            field_name="metrics.include_stability_metrics",
        ),
        include_policy_metrics=_coerce_bool(
            payload.get("include_policy_metrics", True),
            field_name="metrics.include_policy_metrics",
        ),
        include_conditional_performance=_coerce_bool(
            payload.get("include_conditional_performance", True),
            field_name="metrics.include_conditional_performance",
        ),
        high_entropy_threshold=high_entropy_threshold,
    )


def _resolve_optional_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RegimeBenchmarkPackConfigError(
            f"Regime benchmark-pack field '{field_name}' must be a mapping."
        )
    return canonicalize_value(dict(value))


def _normalize_required_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise RegimeBenchmarkPackConfigError(
            f"Regime benchmark-pack field '{field_name}' must be a non-empty string."
        )
    normalized = value.strip()
    if not normalized:
        raise RegimeBenchmarkPackConfigError(
            f"Regime benchmark-pack field '{field_name}' must be a non-empty string."
        )
    return normalized


def _normalize_optional_string(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _normalize_required_string(value, field_name=field_name)


def _normalize_optional_string_sequence(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise RegimeBenchmarkPackConfigError(
            f"Regime benchmark-pack field '{field_name}' must be a sequence when provided."
        )
    return tuple(
        _normalize_required_string(item, field_name=field_name)
        for item in value
    )


def _normalize_relative_path_string(value: Any, *, field_name: str) -> str:
    normalized = _normalize_required_string(value, field_name=field_name)
    return Path(normalized).as_posix()


def _normalize_existing_path_string(
    value: Any,
    *,
    field_name: str,
    config_dir: Path | None,
) -> str:
    normalized = _normalize_required_string(value, field_name=field_name)
    path = Path(normalized)
    if not path.is_absolute() and config_dir is not None:
        path = (config_dir / path).resolve()
    if not path.exists():
        raise RegimeBenchmarkPackConfigError(
            f"Regime benchmark-pack field '{field_name}' points to a missing path: {path.as_posix()}."
        )
    return path.as_posix()


def _coerce_bool(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise RegimeBenchmarkPackConfigError(
            f"Regime benchmark-pack field '{field_name}' must be boolean."
        )
    return value


__all__ = [
    "RegimeBenchmarkMetricsConfig",
    "RegimeBenchmarkPackConfig",
    "RegimeBenchmarkPackConfigError",
    "RegimeBenchmarkVariantConfig",
    "load_regime_benchmark_pack_config",
]
