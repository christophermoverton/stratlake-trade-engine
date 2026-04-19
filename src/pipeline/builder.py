from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from src.config.robustness import RobustnessConfig
from src.portfolio import SUPPORTED_PORTFOLIO_OPTIMIZERS, validate_portfolio_config
from src.research.position_constructors import load_position_constructor_registry
from src.research.position_constructors.base import validate_asymmetry_parameters
from src.research.signal_semantics import (
    Signal,
    SignalSemanticsError,
    ensure_signal_type_compatible,
    load_signal_type_registry,
    resolve_signal_type_definition,
    score_to_binary_long_only,
    score_to_cross_section_rank,
    score_to_signed_zscore,
    score_to_ternary_long_short,
    spread_to_zscore,
    rank_to_percentile,
    percentile_to_quantile_bucket,
    validate_directional_asymmetry_compatibility,
)
from src.research.strategies.validation import load_strategies_registry

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STRATEGY_CATALOG_PATH = REPO_ROOT / "configs" / "strategies.yml"
DEFAULT_PORTFOLIO_CATALOG_PATH = REPO_ROOT / "configs" / "portfolios.yml"
_SUPPORTED_PORTFOLIO_BUILDERS = frozenset({"equal_weight", *SUPPORTED_PORTFOLIO_OPTIMIZERS})
_PORTFOLIO_PARAM_KEYS = frozenset(
    {
        "alignment_policy",
        "initial_capital",
        "optimizer",
        "portfolio_name",
        "promotion_gates",
        "risk",
        "sanity",
        "simulation",
        "timeframe",
        "validation",
        "volatility_targeting",
        "output_dir",
        "evaluation",
    }
)
_TRANSFORM_EDGE_ORDER: tuple[tuple[str, str], ...] = (
    ("prediction_score", "signed_zscore"),
    ("prediction_score", "cross_section_rank"),
    ("prediction_score", "cross_section_percentile"),
    ("prediction_score", "ternary_quantile"),
    ("prediction_score", "binary_signal"),
    ("prediction_score", "spread_zscore"),
    ("cross_section_rank", "cross_section_percentile"),
    ("cross_section_rank", "ternary_quantile"),
    ("cross_section_rank", "binary_signal"),
    ("cross_section_percentile", "ternary_quantile"),
    ("cross_section_percentile", "binary_signal"),
)


class PipelineBuilderError(ValueError):
    """Raised when declarative pipeline authoring is invalid."""


@dataclass(frozen=True)
class RegistryComponent:
    """One resolved registry-backed component."""

    name: str
    version: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class StrategySelection:
    """Declarative strategy selection."""

    name: str
    version: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SignalSelection:
    """Declarative signal semantic selection."""

    signal_type: str
    version: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConstructorSelection:
    """Declarative position constructor selection."""

    name: str
    version: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AsymmetrySelection:
    """Declarative long/short asymmetry selection."""

    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PortfolioSelection:
    """Declarative portfolio selection."""

    name: str
    version: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BuiltPipeline:
    """Rendered pipeline YAML plus any supporting config files."""

    pipeline_id: str
    pipeline_yaml: str
    support_files: dict[str, str]

    def write(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        for relative_name, content in self.support_files.items():
            target = output_path.parent / relative_name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        output_path.write_text(self.pipeline_yaml, encoding="utf-8")
        return output_path


def resolve_strategy_registry_path() -> Path:
    return REPO_ROOT / "artifacts" / "registry" / "strategies.jsonl"


def resolve_signal_registry_path() -> Path:
    return REPO_ROOT / "artifacts" / "registry" / "signal_types.jsonl"


def resolve_constructor_registry_path() -> Path:
    return REPO_ROOT / "artifacts" / "registry" / "position_constructors.jsonl"


def _normalize_mapping(value: Mapping[str, Any] | None, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise PipelineBuilderError(f"{field_name} must be a mapping when provided.")
    return {str(key): value[key] for key in sorted(value)}


def _require_non_empty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise PipelineBuilderError(f"{field_name} must be a non-empty string.")
    normalized = value.strip()
    if not normalized:
        raise PipelineBuilderError(f"{field_name} must be a non-empty string.")
    return normalized


def _coerce_optional_string(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _version_sort_key(value: str) -> tuple[Any, ...]:
    normalized = value.strip()
    parts = re.split(r"[^0-9A-Za-z]+", normalized)
    key: list[Any] = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part.lower()))
    return tuple(key) or ((1, normalized.lower()),)


def _slug(value: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z]+", "_", value.strip().lower())
    collapsed = re.sub(r"_+", "_", normalized).strip("_")
    return collapsed or "pipeline"


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _load_optional_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise PipelineBuilderError(f"Expected YAML object at {path.as_posix()}.")
    return payload


def _select_versioned_payload(
    matches: Sequence[Mapping[str, Any]],
    *,
    version: str | None,
    kind: str,
    component_name: str,
) -> Mapping[str, Any]:
    if version is not None:
        for payload in matches:
            if str(payload.get("version", "")).strip() == version:
                return payload
        available = ", ".join(sorted({str(item.get("version", "")).strip() for item in matches}))
        raise PipelineBuilderError(
            f"Unknown {kind} version for {component_name!r}: {version!r}. Available versions: {available or '<none>'}."
        )
    ordered = sorted(matches, key=lambda payload: _version_sort_key(str(payload.get("version", ""))))
    return ordered[-1]


def _resolve_named_registry_component(
    entries: Sequence[dict[str, Any]],
    *,
    name_field: str,
    component_name: str,
    version: str | None,
    kind: str,
) -> RegistryComponent:
    normalized_name = _require_non_empty_string(component_name, field_name=f"{kind}.name")
    matches = [entry for entry in entries if str(entry.get(name_field, "")).strip() == normalized_name]
    if not matches:
        available = ", ".join(sorted({str(entry.get(name_field, "")).strip() for entry in entries if entry.get(name_field)}))
        raise PipelineBuilderError(
            f"Unknown {kind} {normalized_name!r}. Available {kind}s: {available or '<none>'}."
        )

    selected = _select_versioned_payload(matches, version=version, kind=kind, component_name=normalized_name)
    resolved_version = _require_non_empty_string(selected.get("version"), field_name=f"{kind}.version")
    return RegistryComponent(name=normalized_name, version=resolved_version, payload=dict(selected))


def _resolve_signal_registry_component(
    registry: Mapping[str, Any],
    *,
    signal_type: str,
    version: str | None,
) -> RegistryComponent:
    normalized_name = _require_non_empty_string(signal_type, field_name="signal.type")
    if normalized_name not in registry:
        available = ", ".join(sorted(registry)) or "<none>"
        raise PipelineBuilderError(
            f"Unknown signal type {normalized_name!r}. Available signal types: {available}."
        )
    definition = registry[normalized_name]
    resolved_version = str(getattr(definition, "version"))
    if version is not None and version != resolved_version:
        raise PipelineBuilderError(
            f"Unsupported signal type version for {normalized_name!r}: {version!r}. "
            f"Registry defines version {resolved_version!r}."
        )
    return RegistryComponent(
        name=normalized_name,
        version=resolved_version,
        payload={
            "signal_type_id": getattr(definition, "signal_type_id"),
            "version": resolved_version,
        },
    )


def _resolve_constructor_registry_component(
    registry: Mapping[str, Any],
    *,
    constructor_name: str,
    version: str | None,
) -> RegistryComponent:
    normalized_name = _require_non_empty_string(constructor_name, field_name="constructor.name")
    if normalized_name not in registry:
        available = ", ".join(sorted(registry)) or "<none>"
        raise PipelineBuilderError(
            f"Unknown position constructor {normalized_name!r}. Available constructors: {available}."
        )
    definition = registry[normalized_name]
    resolved_version = str(getattr(definition, "version"))
    if version is not None and version != resolved_version:
        raise PipelineBuilderError(
            f"Unsupported position constructor version for {normalized_name!r}: {version!r}. "
            f"Registry defines version {resolved_version!r}."
        )
    return RegistryComponent(
        name=normalized_name,
        version=resolved_version,
        payload={
            "constructor_id": getattr(definition, "constructor_id"),
            "version": resolved_version,
        },
    )


def _resolve_strategy_template(strategy_name: str, strategy_catalog: Mapping[str, Any]) -> dict[str, Any]:
    template = strategy_catalog.get(strategy_name, {})
    if not isinstance(template, dict):
        raise PipelineBuilderError(
            f"Strategy catalog entry for {strategy_name!r} must be a mapping when present."
        )
    return dict(template)


def _resolve_portfolio_template_definition(
    portfolio_name: str,
    portfolio_catalog: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not portfolio_catalog:
        return None
    raw_portfolios = portfolio_catalog.get("portfolios", portfolio_catalog)
    if isinstance(raw_portfolios, dict):
        template = raw_portfolios.get(portfolio_name)
        if template is None:
            return None
        if not isinstance(template, dict):
            raise PipelineBuilderError(f"Portfolio template {portfolio_name!r} must be a mapping.")
        return dict(template)
    return None


def _supported_portfolio_templates(portfolio_catalog: Mapping[str, Any]) -> list[str]:
    names = set(_SUPPORTED_PORTFOLIO_BUILDERS)
    raw_portfolios = portfolio_catalog.get("portfolios", portfolio_catalog)
    if isinstance(raw_portfolios, dict):
        names.update(str(name) for name in raw_portfolios)
    return sorted(names)


def _expand_options(
    params: Mapping[str, Sequence[Any]],
    options: Mapping[str, Sequence[Any]],
) -> list[dict[str, Any]]:
    axes: list[tuple[str, Sequence[Any]]] = []
    for key in sorted(params):
        axes.append((key, tuple(params[key])))
    for key in sorted(options):
        axes.append((key, tuple(options[key])))
    if not axes:
        return []

    expanded: list[dict[str, Any]] = [{}]
    for key, values in axes:
        if not values:
            raise PipelineBuilderError(f"Sweep axis {key!r} must not be empty.")
        next_rows: list[dict[str, Any]] = []
        for row in expanded:
            for value in values:
                candidate = dict(row)
                candidate[key] = value
                next_rows.append(candidate)
        expanded = next_rows
    return expanded


def load_builder_config(path: str | Path) -> dict[str, Any]:
    """Load one declarative builder config from JSON or YAML."""

    resolved_path = Path(path)
    if not resolved_path.exists():
        raise PipelineBuilderError(f"Builder config does not exist: {resolved_path.as_posix()}.")

    suffix = resolved_path.suffix.lower()
    with resolved_path.open("r", encoding="utf-8") as handle:
        if suffix == ".json":
            payload = json.load(handle)
        elif suffix in {".yaml", ".yml"}:
            payload = yaml.safe_load(handle) or {}
        else:
            raise PipelineBuilderError(
                f"Unsupported builder config format {resolved_path.suffix!r}. Use JSON, YAML, or YML."
            )

    if not isinstance(payload, dict):
        raise PipelineBuilderError("Builder config must deserialize to an object mapping.")
    nested = payload.get("pipeline")
    if nested is not None:
        if not isinstance(nested, dict):
            raise PipelineBuilderError("pipeline section must be a mapping when provided.")
        payload = dict(nested)
    return payload


def _ordered_signal_neighbors(
    signal_type: str,
    signal_registry: Mapping[str, Any],
) -> list[str]:
    neighbors: list[str] = []
    for source_name, destination_name in _TRANSFORM_EDGE_ORDER:
        if source_name != signal_type:
            continue
        if source_name not in signal_registry or destination_name not in signal_registry:
            continue
        if destination_name not in neighbors:
            neighbors.append(destination_name)

    definition = signal_registry.get(signal_type)
    transformation_policies = getattr(definition, "transformation_policies", {})
    allowed_outputs = transformation_policies.get("allowed_outputs")
    if isinstance(allowed_outputs, list):
        for candidate in allowed_outputs:
            text = str(candidate).strip()
            if text and text in signal_registry and text not in neighbors:
                neighbors.append(text)
    for candidate_name, candidate_definition in sorted(signal_registry.items()):
        derived_from = getattr(candidate_definition, "transformation_policies", {}).get("derived_from")
        if isinstance(derived_from, list) and signal_type in {str(value) for value in derived_from}:
            if candidate_name not in neighbors:
                neighbors.append(candidate_name)
    return neighbors


def supports_signal_transform(
    source_signal_type: str,
    target_signal_type: str,
    *,
    signal_registry: Mapping[str, Any] | None = None,
) -> bool:
    """Return True when the signal registry allows the requested transform."""

    normalized_source = _require_non_empty_string(source_signal_type, field_name="source_signal_type")
    normalized_target = _require_non_empty_string(target_signal_type, field_name="target_signal_type")
    if normalized_source == normalized_target:
        return True

    registry = (
        load_signal_type_registry(resolve_signal_registry_path())
        if signal_registry is None
        else signal_registry
    )
    visited = {normalized_source}
    frontier = [normalized_source]
    while frontier:
        current = frontier.pop(0)
        for candidate in _ordered_signal_neighbors(current, registry):
            if candidate == normalized_target:
                return True
            if candidate in visited:
                continue
            visited.add(candidate)
            frontier.append(candidate)
    return False


def _resolve_transform_path(
    source_signal_type: str,
    target_signal_type: str,
    signal_registry: Mapping[str, Any],
) -> list[tuple[str, str]]:
    normalized_source = _require_non_empty_string(source_signal_type, field_name="source_signal_type")
    normalized_target = _require_non_empty_string(target_signal_type, field_name="target_signal_type")
    if normalized_source == normalized_target:
        return []

    frontier: list[str] = [normalized_source]
    parents: dict[str, str | None] = {normalized_source: None}
    while frontier:
        current = frontier.pop(0)
        for neighbor in _ordered_signal_neighbors(current, signal_registry):
            if neighbor in parents:
                continue
            parents[neighbor] = current
            if neighbor == normalized_target:
                frontier.clear()
                break
            frontier.append(neighbor)

    if normalized_target not in parents:
        return []

    path_nodes = [normalized_target]
    while parents[path_nodes[-1]] is not None:
        parent = parents[path_nodes[-1]]
        assert parent is not None
        path_nodes.append(parent)
    path_nodes.reverse()
    return list(zip(path_nodes[:-1], path_nodes[1:], strict=True))


def _apply_transform_edge(
    signal: Signal,
    source_type: str,
    destination_type: str,
    params: Mapping[str, Any],
) -> Signal:
    if source_type != signal.signal_type:
        raise PipelineBuilderError(
            f"Signal transform graph is inconsistent: expected {source_type!r}, received {signal.signal_type!r}."
        )

    if source_type == "prediction_score" and destination_type == "signed_zscore":
        return score_to_signed_zscore(signal, clip=params.get("clip"))
    if source_type == "prediction_score" and destination_type == "cross_section_rank":
        return score_to_cross_section_rank(signal)
    if source_type == "prediction_score" and destination_type == "cross_section_percentile":
        return rank_to_percentile(score_to_cross_section_rank(signal))
    if source_type == "prediction_score" and destination_type == "ternary_quantile":
        return score_to_ternary_long_short(signal, quantile=float(params.get("quantile", 0.2)))
    if source_type == "prediction_score" and destination_type == "binary_signal":
        return score_to_binary_long_only(signal, quantile=float(params.get("quantile", 0.2)))
    if source_type == "prediction_score" and destination_type == "spread_zscore":
        return spread_to_zscore(
            signal,
            clip=params.get("clip"),
            group_columns=tuple(params.get("group_columns", ("symbol",))),
        )
    if source_type == "cross_section_rank" and destination_type == "cross_section_percentile":
        return rank_to_percentile(signal)
    if source_type == "cross_section_rank" and destination_type == "ternary_quantile":
        return percentile_to_quantile_bucket(
            rank_to_percentile(signal),
            quantile=float(params.get("quantile", 0.2)),
            long_only=False,
        )
    if source_type == "cross_section_rank" and destination_type == "binary_signal":
        return percentile_to_quantile_bucket(
            rank_to_percentile(signal),
            quantile=float(params.get("quantile", 0.2)),
            long_only=True,
        )
    if source_type == "cross_section_percentile" and destination_type == "ternary_quantile":
        return percentile_to_quantile_bucket(
            signal,
            quantile=float(params.get("quantile", 0.2)),
            long_only=False,
        )
    if source_type == "cross_section_percentile" and destination_type == "binary_signal":
        return percentile_to_quantile_bucket(
            signal,
            quantile=float(params.get("quantile", 0.2)),
            long_only=True,
        )
    raise PipelineBuilderError(
        f"Unsupported signal transformation edge from {source_type!r} to {destination_type!r}."
    )


def transform_signal(
    signal: Signal,
    target_signal_type: str,
    *,
    signal_params: Mapping[str, Any] | None = None,
    signal_registry: Mapping[str, Any] | None = None,
) -> Signal:
    """Transform one managed signal to a compatible target semantic."""

    params = dict(signal_params or {})
    registry = (
        load_signal_type_registry(resolve_signal_registry_path())
        if signal_registry is None
        else signal_registry
    )
    target = _require_non_empty_string(target_signal_type, field_name="target_signal_type")
    if signal.signal_type == target:
        return signal

    path = _resolve_transform_path(signal.signal_type, target, registry)
    if not path:
        raise PipelineBuilderError(
            f"Unsupported signal transformation from {signal.signal_type!r} to {target!r}."
        )

    current = signal
    for source_type, destination_type in path:
        current = _apply_transform_edge(current, source_type, destination_type, params)
    return current


def validate_declarative_strategy_mapping(
    payload: Mapping[str, Any],
    *,
    strategy_registry_path: str | Path | None = None,
    signal_registry_path: str | Path | None = None,
    constructor_registry_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate one rendered declarative strategy spec."""

    strategy_section = payload.get("strategy")
    signal_section = payload.get("signal")
    constructor_section = payload.get("constructor")
    if not isinstance(strategy_section, Mapping):
        raise PipelineBuilderError("Declarative strategy config must include a strategy mapping.")
    if not isinstance(signal_section, Mapping):
        raise PipelineBuilderError("Declarative strategy config must include a signal mapping.")
    if not isinstance(constructor_section, Mapping):
        raise PipelineBuilderError("Declarative strategy config must include a constructor mapping.")

    strategy_name = _require_non_empty_string(strategy_section.get("name"), field_name="strategy.name")
    strategy_id = _require_non_empty_string(
        strategy_section.get("strategy_id", strategy_name),
        field_name="strategy.strategy_id",
    )
    dataset = _require_non_empty_string(strategy_section.get("dataset"), field_name="strategy.dataset")
    source_signal_type = _require_non_empty_string(
        strategy_section.get("source_signal_type"),
        field_name="strategy.source_signal_type",
    )
    strategy_params = _normalize_mapping(strategy_section.get("params"), field_name="strategy.params")
    signal_type = _require_non_empty_string(signal_section.get("type"), field_name="signal.type")
    signal_params = _normalize_mapping(signal_section.get("params"), field_name="signal.params")
    constructor_name = _require_non_empty_string(constructor_section.get("name"), field_name="constructor.name")
    constructor_params = _normalize_mapping(constructor_section.get("params"), field_name="constructor.params")

    asymmetry_section = payload.get("asymmetry")
    asymmetry_params = {}
    if asymmetry_section is not None:
        if not isinstance(asymmetry_section, Mapping):
            raise PipelineBuilderError("asymmetry must be a mapping when provided.")
        asymmetry_params = _normalize_mapping(
            asymmetry_section.get("params", asymmetry_section),
            field_name="asymmetry.params",
        )

    signal_registry = load_signal_type_registry(
        signal_registry_path if signal_registry_path is not None else resolve_signal_registry_path()
    )
    constructor_registry = load_position_constructor_registry(
        constructor_registry_path if constructor_registry_path is not None else resolve_constructor_registry_path()
    )
    strategies = load_strategies_registry(
        Path(strategy_registry_path)
        if strategy_registry_path is not None
        else resolve_strategy_registry_path()
    )
    _resolve_named_registry_component(
        strategies,
        name_field="strategy_id",
        component_name=strategy_id,
        version=_coerce_optional_string(strategy_section.get("version")),
        kind="strategy",
    )
    signal_component = _resolve_signal_registry_component(
        signal_registry,
        signal_type=signal_type,
        version=_coerce_optional_string(signal_section.get("version")),
    )
    _resolve_constructor_registry_component(
        constructor_registry,
        constructor_name=constructor_name,
        version=_coerce_optional_string(constructor_section.get("version")),
    )

    if not supports_signal_transform(source_signal_type, signal_type, signal_registry=signal_registry):
        raise PipelineBuilderError(
            f"Strategy output signal {source_signal_type!r} cannot be transformed to {signal_type!r}."
        )
    try:
        ensure_signal_type_compatible(signal_type, position_constructor=constructor_name)
    except SignalSemanticsError as exc:
        raise PipelineBuilderError(str(exc)) from exc
    signal_definition = resolve_signal_type_definition(signal_type)
    asymmetry_signal_validation = validate_directional_asymmetry_compatibility(
        signal_definition,
        constructor_name,
        asymmetry_config=asymmetry_params,
    )
    if asymmetry_params and not asymmetry_signal_validation["valid"]:
        raise PipelineBuilderError(
            str(asymmetry_signal_validation.get("error_message") or "Invalid asymmetry configuration.")
        )
    asymmetry_validation = validate_asymmetry_parameters(
        {**constructor_params, **asymmetry_params},
        constructor_name,
    )
    if not asymmetry_validation["valid"]:
        raise PipelineBuilderError(
            str(asymmetry_validation.get("error_message") or "Invalid asymmetry configuration.")
        )

    return {
        "strategy": {
            "name": strategy_name,
            "strategy_id": strategy_id,
            "version": _coerce_optional_string(strategy_section.get("version")) or "",
            "dataset": dataset,
            "params": strategy_params,
            "source_signal_type": source_signal_type,
        },
        "signal": {
            "type": signal_type,
            "version": signal_component.version,
            "params": signal_params,
        },
        "constructor": {
            "name": constructor_name,
            "version": _coerce_optional_string(constructor_section.get("version")) or "",
            "params": constructor_params,
        },
        "asymmetry": {
            "params": asymmetry_params,
        },
    }


class PipelineBuilder:
    """Fluent high-level builder for M20-compatible research pipelines."""

    def __init__(
        self,
        pipeline_id: str,
        *,
        start: str | None = None,
        end: str | None = None,
        strict: bool = False,
        strategy_registry_path: str | Path | None = None,
        signal_registry_path: str | Path | None = None,
        constructor_registry_path: str | Path | None = None,
        strategy_catalog_path: str | Path | None = None,
        portfolio_catalog_path: str | Path | None = None,
    ) -> None:
        self._pipeline_id = _slug(_require_non_empty_string(pipeline_id, field_name="pipeline_id"))
        self._start = start
        self._end = end
        self._strict = bool(strict)
        self._strategy_registry_path = Path(strategy_registry_path) if strategy_registry_path is not None else resolve_strategy_registry_path()
        self._signal_registry_path = Path(signal_registry_path) if signal_registry_path is not None else resolve_signal_registry_path()
        self._constructor_registry_path = Path(constructor_registry_path) if constructor_registry_path is not None else resolve_constructor_registry_path()
        self._strategy_catalog_path = Path(strategy_catalog_path) if strategy_catalog_path is not None else DEFAULT_STRATEGY_CATALOG_PATH
        self._portfolio_catalog_path = Path(portfolio_catalog_path) if portfolio_catalog_path is not None else DEFAULT_PORTFOLIO_CATALOG_PATH
        self._strategy: StrategySelection | None = None
        self._signal: SignalSelection | None = None
        self._constructor: ConstructorSelection | None = None
        self._asymmetry: AsymmetrySelection | None = None
        self._portfolio: PortfolioSelection | None = None
        self._sweep: dict[str, Any] | None = None

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        strategy_registry_path: str | Path | None = None,
        signal_registry_path: str | Path | None = None,
        constructor_registry_path: str | Path | None = None,
        strategy_catalog_path: str | Path | None = None,
        portfolio_catalog_path: str | Path | None = None,
    ) -> "PipelineBuilder":
        pipeline_id = payload.get("pipeline_id", payload.get("id", "pipeline"))
        builder = cls(
            str(pipeline_id),
            start=_coerce_optional_string(payload.get("start")),
            end=_coerce_optional_string(payload.get("end")),
            strict=bool(payload.get("strict", False)),
            strategy_registry_path=strategy_registry_path,
            signal_registry_path=signal_registry_path,
            constructor_registry_path=constructor_registry_path,
            strategy_catalog_path=strategy_catalog_path,
            portfolio_catalog_path=portfolio_catalog_path,
        )

        strategy_section = payload.get("strategy")
        if isinstance(strategy_section, Mapping):
            builder.strategy(
                str(strategy_section.get("name")),
                version=_coerce_optional_string(strategy_section.get("version")),
                params=_normalize_mapping(strategy_section.get("params"), field_name="strategy.params"),
                config=_normalize_mapping(strategy_section.get("config"), field_name="strategy.config"),
            )

        signal_section = payload.get("signal")
        if isinstance(signal_section, Mapping):
            builder.signal(
                str(signal_section.get("type", signal_section.get("signal_type"))),
                version=_coerce_optional_string(signal_section.get("version")),
                params=_normalize_mapping(signal_section.get("params"), field_name="signal.params"),
            )

        constructor_section = payload.get("constructor", payload.get("construct_positions"))
        if isinstance(constructor_section, Mapping):
            builder.construct_positions(
                str(constructor_section.get("name")),
                version=_coerce_optional_string(constructor_section.get("version")),
                params=_normalize_mapping(constructor_section.get("params"), field_name="constructor.params"),
            )

        asymmetry_section = payload.get("asymmetry")
        if isinstance(asymmetry_section, Mapping):
            builder.asymmetry(
                params=_normalize_mapping(
                    asymmetry_section.get("params", asymmetry_section),
                    field_name="asymmetry.params",
                )
            )

        portfolio_section = payload.get("portfolio")
        if isinstance(portfolio_section, Mapping):
            builder.portfolio(
                str(portfolio_section.get("name")),
                version=_coerce_optional_string(portfolio_section.get("version")),
                params=_normalize_mapping(portfolio_section.get("params"), field_name="portfolio.params"),
            )

        if payload.get("sweep") is not None:
            sweep_payload = payload.get("sweep")
            if not isinstance(sweep_payload, Mapping):
                raise PipelineBuilderError("sweep must be a mapping when provided.")
            builder.sweep(dict(sweep_payload))

        return builder

    def strategy(
        self,
        name: str,
        version: str | None = None,
        params: Mapping[str, Any] | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> "PipelineBuilder":
        self._strategy = StrategySelection(
            name=_require_non_empty_string(name, field_name="strategy.name"),
            version=_coerce_optional_string(version),
            params=_normalize_mapping(params, field_name="strategy.params"),
            config=_normalize_mapping(config, field_name="strategy.config"),
        )
        return self

    def signal(
        self,
        signal_type: str,
        version: str | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> "PipelineBuilder":
        self._signal = SignalSelection(
            signal_type=_require_non_empty_string(signal_type, field_name="signal.type"),
            version=_coerce_optional_string(version),
            params=_normalize_mapping(params, field_name="signal.params"),
        )
        return self

    def construct_positions(
        self,
        name: str,
        version: str | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> "PipelineBuilder":
        self._constructor = ConstructorSelection(
            name=_require_non_empty_string(name, field_name="constructor.name"),
            version=_coerce_optional_string(version),
            params=_normalize_mapping(params, field_name="constructor.params"),
        )
        return self

    def asymmetry(self, params: Mapping[str, Any] | None = None) -> "PipelineBuilder":
        self._asymmetry = AsymmetrySelection(
            params=_normalize_mapping(params, field_name="asymmetry.params"),
        )
        return self

    def portfolio(
        self,
        name: str,
        version: str | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> "PipelineBuilder":
        self._portfolio = PortfolioSelection(
            name=_require_non_empty_string(name, field_name="portfolio.name"),
            version=_coerce_optional_string(version),
            params=_normalize_mapping(params, field_name="portfolio.params"),
        )
        return self

    def sweep(self, config: Mapping[str, Any]) -> "PipelineBuilder":
        self._sweep = _normalize_mapping(config, field_name="sweep")
        return self

    def build(self) -> BuiltPipeline:
        return self._render(Path.cwd() / f"{self._pipeline_id}.pipeline.yml")

    def to_yaml(self, path: str | Path | None = None) -> str:
        built = self._render(
            Path(path) if path is not None else (Path.cwd() / f"{self._pipeline_id}.pipeline.yml")
        )
        if path is not None:
            built.write(path)
        return built.pipeline_yaml

    def run(self, path: str | Path | None = None):
        from src.cli.run_pipeline import run_cli as run_pipeline_cli

        if path is not None:
            output_path = Path(path)
            self._render(output_path).write(output_path)
            return run_pipeline_cli(["--config", str(output_path)])

        with tempfile.TemporaryDirectory(prefix=f"{self._pipeline_id}_", dir=Path.cwd()) as temp_dir:
            output_path = Path(temp_dir) / f"{self._pipeline_id}.pipeline.yml"
            self._render(output_path).write(output_path)
            return run_pipeline_cli(["--config", str(output_path)])

    def _render(self, output_path: Path) -> BuiltPipeline:
        strategy_registry = load_strategies_registry(self._strategy_registry_path)
        signal_registry = load_signal_type_registry(self._signal_registry_path)
        constructor_registry = load_position_constructor_registry(self._constructor_registry_path)
        strategy_catalog = _load_optional_yaml(self._strategy_catalog_path)
        portfolio_catalog = _load_optional_yaml(self._portfolio_catalog_path)

        if self._strategy is None:
            raise PipelineBuilderError("A declarative pipeline must define a strategy.")
        if self._sweep is not None and self._portfolio is not None:
            raise PipelineBuilderError(
                "Sweep pipelines cannot include a portfolio stage in the first builder pass."
            )
        if self._sweep is None and (self._signal is None or self._constructor is None):
            raise PipelineBuilderError(
                "Non-sweep pipelines must define both a signal semantic and a position constructor."
            )

        if self._sweep is not None:
            return self._render_sweep_pipeline(
                output_path,
                strategy_registry=strategy_registry,
                signal_registry=signal_registry,
                constructor_registry=constructor_registry,
                strategy_catalog=strategy_catalog,
            )
        return self._render_single_pipeline(
            output_path,
            strategy_registry=strategy_registry,
            signal_registry=signal_registry,
            constructor_registry=constructor_registry,
            strategy_catalog=strategy_catalog,
            portfolio_catalog=portfolio_catalog,
        )

    def _strategy_alias(self) -> str:
        return f"{self._pipeline_id}__strategy"

    def _strategy_step_argv(self, strategy_config_reference: str) -> list[str]:
        argv = ["--config", strategy_config_reference]
        if self._start is not None:
            argv.extend(["--start", self._start])
        if self._end is not None:
            argv.extend(["--end", self._end])
        if self._strict:
            argv.append("--strict")
        return argv

    def _render_single_pipeline(
        self,
        output_path: Path,
        *,
        strategy_registry: Sequence[dict[str, Any]],
        signal_registry: Mapping[str, Any],
        constructor_registry: Mapping[str, Any],
        strategy_catalog: Mapping[str, Any],
        portfolio_catalog: Mapping[str, Any],
    ) -> BuiltPipeline:
        assert self._strategy is not None
        assert self._signal is not None
        assert self._constructor is not None

        resolved_strategy = _resolve_named_registry_component(
            strategy_registry,
            name_field="strategy_id",
            component_name=self._strategy.name,
            version=self._strategy.version,
            kind="strategy",
        )
        strategy_template = _resolve_strategy_template(self._strategy.name, strategy_catalog)
        dataset = _require_non_empty_string(
            self._strategy.config.get("dataset", strategy_template.get("dataset")),
            field_name="strategy.dataset",
        )
        source_signal_type = _require_non_empty_string(
            strategy_template.get(
                "signal_type",
                resolved_strategy.payload.get("output_signal", {}).get("signal_type"),
            ),
            field_name="strategy.source_signal_type",
        )
        resolved_signal = _resolve_signal_registry_component(
            signal_registry,
            signal_type=self._signal.signal_type,
            version=self._signal.version,
        )
        resolved_constructor = _resolve_constructor_registry_component(
            constructor_registry,
            constructor_name=self._constructor.name,
            version=self._constructor.version,
        )

        if not supports_signal_transform(source_signal_type, resolved_signal.name, signal_registry=signal_registry):
            raise PipelineBuilderError(
                f"Strategy output signal {source_signal_type!r} cannot be transformed to {resolved_signal.name!r}."
            )
        try:
            ensure_signal_type_compatible(resolved_signal.name, position_constructor=resolved_constructor.name)
        except SignalSemanticsError as exc:
            raise PipelineBuilderError(str(exc)) from exc

        asymmetry_params = {} if self._asymmetry is None else dict(self._asymmetry.params)
        asymmetry_signal_validation = validate_directional_asymmetry_compatibility(
            resolve_signal_type_definition(resolved_signal.name),
            resolved_constructor.name,
            asymmetry_config=asymmetry_params,
        )
        if asymmetry_params and not asymmetry_signal_validation["valid"]:
            raise PipelineBuilderError(
                str(asymmetry_signal_validation.get("error_message") or "Invalid asymmetry configuration.")
            )
        asymmetry_validation = validate_asymmetry_parameters(
            {**self._constructor.params, **asymmetry_params},
            resolved_constructor.name,
        )
        if not asymmetry_validation["valid"]:
            raise PipelineBuilderError(
                str(asymmetry_validation.get("error_message") or "Invalid asymmetry configuration.")
            )

        strategy_alias = self._strategy_alias()
        strategy_spec_name = f"{self._pipeline_id}.strategy.builder.yml"
        strategy_spec_path = output_path.parent / strategy_spec_name
        strategy_spec_reference = _display_path(strategy_spec_path)
        strategy_spec_payload = {
            "strategy": {
                "name": strategy_alias,
                "strategy_id": resolved_strategy.name,
                "version": resolved_strategy.version,
                "dataset": dataset,
                "params": dict(self._strategy.params),
                "source_signal_type": source_signal_type,
            },
            "signal": {
                "type": resolved_signal.name,
                "version": resolved_signal.version,
                "params": dict(self._signal.params),
            },
            "constructor": {
                "name": resolved_constructor.name,
                "version": resolved_constructor.version,
                "params": dict(self._constructor.params),
            },
            "asymmetry": {
                "params": asymmetry_params,
            },
        }
        validate_declarative_strategy_mapping(
            strategy_spec_payload,
            strategy_registry_path=self._strategy_registry_path,
            signal_registry_path=self._signal_registry_path,
            constructor_registry_path=self._constructor_registry_path,
        )

        pipeline_steps: list[dict[str, Any]] = [
            {
                "id": "run_strategy",
                "adapter": "python_module",
                "module": "src.cli.run_builder_strategy",
                "argv": self._strategy_step_argv(strategy_spec_reference),
            }
        ]
        support_files = {
            strategy_spec_name: yaml.safe_dump(strategy_spec_payload, sort_keys=False),
        }

        if self._portfolio is not None:
            portfolio_payload, portfolio_name = self._build_portfolio_payload(
                portfolio_catalog=portfolio_catalog,
                component_strategy_name=strategy_alias,
            )
            portfolio_file_name = f"{self._pipeline_id}.portfolio.builder.yml"
            portfolio_path = output_path.parent / portfolio_file_name
            portfolio_reference = _display_path(portfolio_path)
            support_files[portfolio_file_name] = yaml.safe_dump(portfolio_payload, sort_keys=False)
            portfolio_timeframe = str(self._portfolio.params.get("timeframe", "1D"))
            portfolio_step = {
                "id": "run_portfolio",
                "depends_on": ["run_strategy"],
                "adapter": "python_module",
                "module": "src.cli.run_portfolio",
                "argv": [
                    "--portfolio-config",
                    portfolio_reference,
                    "--portfolio-name",
                    portfolio_name,
                    "--from-registry",
                    "--timeframe",
                    portfolio_timeframe,
                ],
            }
            output_dir = self._portfolio.params.get("output_dir")
            if output_dir is not None:
                portfolio_step["argv"].extend(["--output-dir", str(output_dir)])
            evaluation = self._portfolio.params.get("evaluation")
            if evaluation is not None:
                portfolio_step["argv"].extend(["--evaluation", str(evaluation)])
            if self._strict:
                portfolio_step["argv"].append("--strict")
            pipeline_steps.append(portfolio_step)

        return BuiltPipeline(
            pipeline_id=self._pipeline_id,
            pipeline_yaml=yaml.safe_dump({"id": self._pipeline_id, "steps": pipeline_steps}, sort_keys=False),
            support_files=support_files,
        )

    def _build_portfolio_payload(
        self,
        *,
        portfolio_catalog: Mapping[str, Any],
        component_strategy_name: str,
    ) -> tuple[dict[str, Any], str]:
        assert self._portfolio is not None

        unknown_params = sorted(set(self._portfolio.params) - _PORTFOLIO_PARAM_KEYS)
        if unknown_params:
            raise PipelineBuilderError(
                f"Unsupported portfolio params for builder-generated portfolios: {', '.join(unknown_params)}."
            )

        portfolio_name = str(self._portfolio.params.get("portfolio_name", f"{self._pipeline_id}_portfolio"))
        template_definition = _resolve_portfolio_template_definition(self._portfolio.name, portfolio_catalog)
        if template_definition is None and self._portfolio.name not in _SUPPORTED_PORTFOLIO_BUILDERS:
            available = ", ".join(_supported_portfolio_templates(portfolio_catalog))
            raise PipelineBuilderError(
                f"Unknown portfolio builder {self._portfolio.name!r}. Available templates/builders: {available}."
            )

        if template_definition is None:
            definition = {
                "portfolio_name": portfolio_name,
                "allocator": self._portfolio.name,
                "components": [{"strategy_name": component_strategy_name}],
                "initial_capital": float(self._portfolio.params.get("initial_capital", 1.0)),
                "alignment_policy": str(self._portfolio.params.get("alignment_policy", "intersection")),
            }
        else:
            definition = {
                "portfolio_name": portfolio_name,
                "allocator": template_definition.get("allocator", self._portfolio.name),
                "components": [{"strategy_name": component_strategy_name}],
                "initial_capital": float(self._portfolio.params.get("initial_capital", template_definition.get("initial_capital", 1.0))),
                "alignment_policy": str(self._portfolio.params.get("alignment_policy", template_definition.get("alignment_policy", "intersection"))),
            }
            for key in ("optimizer", "validation", "risk", "volatility_targeting", "sanity", "simulation", "promotion_gates"):
                if template_definition.get(key) is not None:
                    definition[key] = template_definition.get(key)

        for key in ("optimizer", "validation", "risk", "volatility_targeting", "sanity", "simulation", "promotion_gates"):
            if self._portfolio.params.get(key) is not None:
                definition[key] = self._portfolio.params.get(key)

        validate_portfolio_config(
            {
                **definition,
                "components": [{"strategy_name": component_strategy_name, "run_id": f"{component_strategy_name}_placeholder"}],
            }
        )
        return {"portfolios": {portfolio_name: definition}}, portfolio_name

    def _sweep_step_argv(
        self,
        *,
        strategy_name: str,
        strategies_reference: str,
        robustness_reference: str,
    ) -> list[str]:
        argv = [
            "--strategy",
            strategy_name,
            "--strategies-config",
            strategies_reference,
            "--robustness",
            robustness_reference,
        ]
        if self._start is not None:
            argv.extend(["--start", self._start])
        if self._end is not None:
            argv.extend(["--end", self._end])
        if self._strict:
            argv.append("--strict")
        return argv

    def _render_sweep_pipeline(
        self,
        output_path: Path,
        *,
        strategy_registry: Sequence[dict[str, Any]],
        signal_registry: Mapping[str, Any],
        constructor_registry: Mapping[str, Any],
        strategy_catalog: Mapping[str, Any],
    ) -> BuiltPipeline:
        assert self._strategy is not None
        assert self._sweep is not None

        resolved_strategy = _resolve_named_registry_component(
            strategy_registry,
            name_field="strategy_id",
            component_name=self._strategy.name,
            version=self._strategy.version,
            kind="strategy",
        )
        strategy_template = _resolve_strategy_template(self._strategy.name, strategy_catalog)
        dataset = _require_non_empty_string(
            self._strategy.config.get("dataset", strategy_template.get("dataset")),
            field_name="strategy.dataset",
        )
        source_signal_type = _require_non_empty_string(
            strategy_template.get(
                "signal_type",
                resolved_strategy.payload.get("output_signal", {}).get("signal_type"),
            ),
            field_name="strategy.source_signal_type",
        )

        strategies_file_name = f"{self._pipeline_id}.strategies.builder.yml"
        strategies_file_path = output_path.parent / strategies_file_name
        strategies_reference = _display_path(strategies_file_path)
        base_strategy_config = {
            self._strategy.name: {
                "dataset": dataset,
                "signal_type": source_signal_type,
                "signal_params": {},
                "position_constructor": {
                    "name": self._constructor.name if self._constructor is not None else "identity_weights",
                    "params": dict(self._constructor.params) if self._constructor is not None else {},
                },
                "parameters": dict(self._strategy.params),
            }
        }

        robustness_payload = self._build_robustness_payload(
            strategy_name=resolved_strategy.name,
            source_signal_type=source_signal_type,
            signal_registry=signal_registry,
            constructor_registry=constructor_registry,
            strategy_registry=strategy_registry,
        )
        robustness_file_name = f"{self._pipeline_id}.robustness.builder.yml"
        robustness_path = output_path.parent / robustness_file_name
        robustness_reference = _display_path(robustness_path)

        return BuiltPipeline(
            pipeline_id=self._pipeline_id,
            pipeline_yaml=yaml.safe_dump(
                {
                    "id": self._pipeline_id,
                    "steps": [
                        {
                            "id": "research_sweep",
                            "adapter": "python_module",
                            "module": "src.cli.run_strategy",
                            "argv": self._sweep_step_argv(
                                strategy_name=self._strategy.name,
                                strategies_reference=strategies_reference,
                                robustness_reference=robustness_reference,
                            ),
                        }
                    ],
                },
                sort_keys=False,
            ),
            support_files={
                strategies_file_name: yaml.safe_dump(base_strategy_config, sort_keys=False),
                robustness_file_name: yaml.safe_dump({"robustness": robustness_payload}, sort_keys=False),
            },
        )

    def _build_robustness_payload(
        self,
        *,
        strategy_name: str,
        source_signal_type: str,
        signal_registry: Mapping[str, Any],
        constructor_registry: Mapping[str, Any],
        strategy_registry: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        assert self._sweep is not None

        raw_payload = json.loads(json.dumps(self._sweep))
        sweep_payload = {
            key: raw_payload[key]
            for key in sorted(raw_payload)
            if key
            not in {
                "ranking",
                "thresholds",
                "stability",
                "statistical_controls",
            }
        }
        strategy_section = sweep_payload.setdefault("strategy", {})
        if not isinstance(strategy_section, dict):
            raise PipelineBuilderError("sweep.strategy must be a mapping when provided.")
        strategy_section.setdefault("name", [strategy_name])

        if self._signal is not None:
            signal_section = sweep_payload.setdefault("signal", {})
            if not isinstance(signal_section, dict):
                raise PipelineBuilderError("sweep.signal must be a mapping when provided.")
            signal_section.setdefault("type", [self._signal.signal_type])
            if self._signal.version is not None:
                signal_section.setdefault("version", self._signal.version)
            base_params = signal_section.setdefault("params", {})
            if not isinstance(base_params, dict):
                raise PipelineBuilderError("sweep.signal.params must be a mapping when provided.")
            for key, value in self._signal.params.items():
                base_params.setdefault(key, [value])

        if self._constructor is not None:
            constructor_section = sweep_payload.setdefault("constructor", {})
            if not isinstance(constructor_section, dict):
                raise PipelineBuilderError("sweep.constructor must be a mapping when provided.")
            constructor_section.setdefault("name", [self._constructor.name])
            if self._constructor.version is not None:
                constructor_section.setdefault("version", self._constructor.version)
            constructor_names = constructor_section.get("name", [self._constructor.name])
            if not isinstance(constructor_names, list):
                raise PipelineBuilderError("sweep.constructor.name must be a list when provided.")
            if constructor_names == [self._constructor.name]:
                base_params = constructor_section.setdefault("params", {})
                if not isinstance(base_params, dict):
                    raise PipelineBuilderError("sweep.constructor.params must be a mapping when provided.")
                for key, value in self._constructor.params.items():
                    base_params.setdefault(key, [value])

        if self._asymmetry is not None:
            asymmetry_section = sweep_payload.setdefault("asymmetry", {})
            if not isinstance(asymmetry_section, dict):
                raise PipelineBuilderError("sweep.asymmetry must be a mapping when provided.")
            for key, value in self._asymmetry.params.items():
                asymmetry_section.setdefault(key, [value])

        robustness_payload = {
            "strategy_name": strategy_name,
            "sweep": sweep_payload,
        }
        for key in ("ranking", "thresholds", "stability", "statistical_controls"):
            if raw_payload.get(key) is not None:
                robustness_payload[key] = raw_payload[key]
        robustness_config = RobustnessConfig.from_mapping(robustness_payload)
        self._validate_sweep_space(
            strategy_name=strategy_name,
            source_signal_type=source_signal_type,
            robustness_config=robustness_config,
            signal_registry=signal_registry,
            constructor_registry=constructor_registry,
            strategy_registry=strategy_registry,
        )
        return robustness_payload

    def _validate_sweep_space(
        self,
        *,
        strategy_name: str,
        source_signal_type: str,
        robustness_config: RobustnessConfig,
        signal_registry: Mapping[str, Any],
        constructor_registry: Mapping[str, Any],
        strategy_registry: Sequence[dict[str, Any]],
    ) -> None:
        sweep = robustness_config.research_sweep
        strategy_names = sweep.strategy.names or (strategy_name,)
        signal_types = sweep.signal.types or ((self._signal.signal_type,) if self._signal is not None else (source_signal_type,))
        constructor_names = sweep.constructor.names or ((self._constructor.name,) if self._constructor is not None else ())
        if not constructor_names:
            raise PipelineBuilderError("Sweep pipelines must define at least one constructor.")

        asymmetry_space = _expand_options(sweep.asymmetry.params, sweep.asymmetry.options) or [dict(self._asymmetry.params if self._asymmetry is not None else {})]
        constructor_space = _expand_options(sweep.constructor.params, sweep.constructor.options) or [dict(self._constructor.params if self._constructor is not None else {})]

        valid_count = 0
        reasons: set[str] = set()
        for current_strategy in strategy_names:
            _resolve_named_registry_component(
                strategy_registry,
                name_field="strategy_id",
                component_name=current_strategy,
                version=sweep.strategy.version or self._strategy.version,
                kind="strategy",
            )
            for signal_type in signal_types:
                _resolve_signal_registry_component(
                    signal_registry,
                    signal_type=signal_type,
                    version=sweep.signal.version or (None if self._signal is None else self._signal.version),
                )
                if not supports_signal_transform(source_signal_type, signal_type, signal_registry=signal_registry):
                    reasons.add(
                        f"strategy output signal '{source_signal_type}' is not compatible with requested signal semantics '{signal_type}'."
                    )
                    continue
                for constructor_name in constructor_names:
                    _resolve_constructor_registry_component(
                        constructor_registry,
                        constructor_name=constructor_name,
                        version=sweep.constructor.version or (None if self._constructor is None else self._constructor.version),
                    )
                    try:
                        ensure_signal_type_compatible(signal_type, position_constructor=constructor_name)
                    except SignalSemanticsError as exc:
                        reasons.add(str(exc))
                        continue
                    signal_definition = resolve_signal_type_definition(signal_type)
                    for constructor_params in constructor_space:
                        for asymmetry_params in asymmetry_space:
                            validation = validate_directional_asymmetry_compatibility(
                                signal_definition,
                                constructor_name,
                                asymmetry_config=asymmetry_params,
                            )
                            if asymmetry_params and not validation["valid"]:
                                reasons.add(str(validation.get("error_message") or "Invalid asymmetry configuration."))
                                continue
                            asymmetry_validation = validate_asymmetry_parameters(
                                {**constructor_params, **asymmetry_params},
                                constructor_name,
                            )
                            if not asymmetry_validation["valid"]:
                                reasons.add(str(asymmetry_validation.get("error_message") or "Invalid asymmetry configuration."))
                                continue
                            valid_count += 1

        if valid_count == 0:
            formatted = ", ".join(sorted(reasons)) or "no valid configurations"
            raise PipelineBuilderError(
                f"Extended robustness sweep produced no valid configurations. Reasons: {formatted}."
            )


__all__ = [
    "BuiltPipeline",
    "PipelineBuilder",
    "PipelineBuilderError",
    "load_builder_config",
    "supports_signal_transform",
    "transform_signal",
    "validate_declarative_strategy_mapping",
]
