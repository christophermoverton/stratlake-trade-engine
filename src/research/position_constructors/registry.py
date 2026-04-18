from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from src.research.position_constructors.base import PositionConstructor, PositionConstructorError
from src.research.position_constructors.identity import IdentityWeightsPositionConstructor
from src.research.position_constructors.rank_dollar_neutral import RankDollarNeutralPositionConstructor
from src.research.position_constructors.softmax_long_only import SoftmaxLongOnlyPositionConstructor
from src.research.position_constructors.top_bottom_equal_weight import TopBottomEqualWeightPositionConstructor
from src.research.position_constructors.zscore_clip_scale import ZScoreClipScalePositionConstructor
from src.research.registry import canonicalize_value

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POSITION_CONSTRUCTORS_REGISTRY = (
    REPO_ROOT / "artifacts" / "registry" / "position_constructors.jsonl"
)

_IMPLEMENTATIONS: dict[str, type[PositionConstructor]] = {
    "identity_weights": IdentityWeightsPositionConstructor,
    "rank_dollar_neutral": RankDollarNeutralPositionConstructor,
    "softmax_long_only": SoftmaxLongOnlyPositionConstructor,
    "top_bottom_equal_weight": TopBottomEqualWeightPositionConstructor,
    "zscore_clip_scale": ZScoreClipScalePositionConstructor,
}


@dataclass(frozen=True)
class PositionConstructorDefinition:
    constructor_id: str
    version: str
    inputs: tuple[str, ...]
    parameters: dict[str, dict[str, Any]]
    tags: tuple[str, ...] = ()


def _normalize_non_empty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise PositionConstructorError(f"{field_name} must be a non-empty string.")
    normalized = value.strip()
    if not normalized:
        raise PositionConstructorError(f"{field_name} must be a non-empty string.")
    return normalized


def _definition_from_payload(payload: Mapping[str, Any]) -> PositionConstructorDefinition:
    constructor_id = _normalize_non_empty_string(payload.get("constructor_id"), field_name="constructor_id")
    version = _normalize_non_empty_string(payload.get("version"), field_name="version")
    inputs = payload.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise PositionConstructorError(f"Position constructor {constructor_id!r} must define one or more inputs.")
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise PositionConstructorError(
            f"Position constructor {constructor_id!r} must define a parameters object."
        )
    normalized_parameters: dict[str, dict[str, Any]] = {}
    for name, schema in sorted(parameters.items()):
        if not isinstance(schema, dict):
            raise PositionConstructorError(
                f"Position constructor {constructor_id!r} parameter {name!r} must define an object schema."
            )
        parameter_type = schema.get("type")
        if parameter_type not in {"bool", "float", "int", "object", "string"}:
            raise PositionConstructorError(
                "Position constructor "
                f"{constructor_id!r} parameter {name!r} must declare one of ['bool', 'float', 'int', 'object', 'string']."
            )
        normalized_parameters[str(name)] = canonicalize_value(dict(schema))
    return PositionConstructorDefinition(
        constructor_id=constructor_id,
        version=version,
        inputs=tuple(str(value) for value in inputs if str(value).strip()),
        parameters=normalized_parameters,
        tags=tuple(str(value) for value in payload.get("tags", []) if str(value).strip()),
    )


def load_position_constructor_registry(
    path: str | Path = DEFAULT_POSITION_CONSTRUCTORS_REGISTRY,
) -> dict[str, PositionConstructorDefinition]:
    registry_path = Path(path)
    if not registry_path.exists():
        raise PositionConstructorError(f"Position constructor registry not found: {registry_path.as_posix()}")

    definitions: dict[str, PositionConstructorDefinition] = {}
    with registry_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PositionConstructorError(
                    f"Position constructor registry contains invalid JSON on line {line_number}."
                ) from exc
            if not isinstance(payload, dict):
                raise PositionConstructorError(
                    f"Position constructor registry line {line_number} must deserialize to an object."
                )
            definition = _definition_from_payload(payload)
            definitions[definition.constructor_id] = definition

    if not definitions:
        raise PositionConstructorError("Position constructor registry is empty.")
    return definitions


def normalize_position_constructor_config(
    config: Mapping[str, Any] | None,
    *,
    field_name: str = "position_constructor",
) -> dict[str, Any] | None:
    if config is None:
        return None
    if not isinstance(config, Mapping):
        raise PositionConstructorError(f"{field_name} must be a mapping when provided.")
    name = config.get("name", config.get("constructor_id"))
    normalized_name = _normalize_non_empty_string(name, field_name=f"{field_name}.name")
    params = config.get("params", config.get("constructor_params", {}))
    if params is None:
        params = {}
    if not isinstance(params, Mapping):
        raise PositionConstructorError(f"{field_name}.params must be a mapping when provided.")
    return canonicalize_value({"name": normalized_name, "params": dict(params)})


def position_constructor_metadata_payload(
    *,
    name: str,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_name = _normalize_non_empty_string(name, field_name="constructor_id")
    normalized_params = {} if params is None else dict(params)
    return canonicalize_value(
        {
            "constructor_id": normalized_name,
            "constructor_params": normalized_params,
        }
    )


def extract_position_constructor_config(
    payload: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if payload is None or not isinstance(payload, Mapping):
        return None
    nested = payload.get("position_constructor")
    if isinstance(nested, Mapping):
        return normalize_position_constructor_config(nested)

    name = payload.get("constructor_id")
    if name is None:
        return None
    params = payload.get("constructor_params", {})
    return normalize_position_constructor_config({"name": name, "params": params})


def _validate_constructor_params(
    definition: PositionConstructorDefinition,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_params = canonicalize_value(dict(params))
    missing = [
        name
        for name, schema in definition.parameters.items()
        if schema.get("required", True) and name not in normalized_params
    ]
    if missing:
        formatted = ", ".join(sorted(missing))
        raise PositionConstructorError(
            f"Position constructor {definition.constructor_id!r} requires parameters: {formatted}."
        )
    unexpected = [name for name in normalized_params if name not in definition.parameters]
    if unexpected:
        formatted = ", ".join(sorted(unexpected))
        raise PositionConstructorError(
            f"Position constructor {definition.constructor_id!r} does not allow parameters: {formatted}."
        )
    for name, value in normalized_params.items():
        parameter_type = definition.parameters[name].get("type")
        if parameter_type == "bool":
            if not isinstance(value, bool):
                raise PositionConstructorError(
                    f"Position constructor {definition.constructor_id!r} parameter {name!r} must be boolean."
                )
        if parameter_type == "float":
            try:
                float(value)
            except (TypeError, ValueError) as exc:
                raise PositionConstructorError(
                    f"Position constructor {definition.constructor_id!r} parameter {name!r} must be numeric."
                ) from exc
        if parameter_type == "int":
            if isinstance(value, bool):
                raise PositionConstructorError(
                    f"Position constructor {definition.constructor_id!r} parameter {name!r} must be an integer."
                )
            try:
                converted = int(value)
            except (TypeError, ValueError) as exc:
                raise PositionConstructorError(
                    f"Position constructor {definition.constructor_id!r} parameter {name!r} must be an integer."
                ) from exc
            if converted != value and not (isinstance(value, float) and value.is_integer()):
                raise PositionConstructorError(
                    f"Position constructor {definition.constructor_id!r} parameter {name!r} must be an integer."
                )
        if parameter_type == "object" and not isinstance(value, Mapping):
            raise PositionConstructorError(
                f"Position constructor {definition.constructor_id!r} parameter {name!r} must be an object mapping."
            )
        if parameter_type == "string" and not isinstance(value, str):
            raise PositionConstructorError(
                f"Position constructor {definition.constructor_id!r} parameter {name!r} must be a string."
            )
    return normalized_params


def resolve_constructor(name: str, params: dict[str, Any]) -> PositionConstructor:
    normalized_name = _normalize_non_empty_string(name, field_name="constructor_id")
    if not isinstance(params, dict):
        raise PositionConstructorError("Position constructor params must be a dictionary.")

    definitions = load_position_constructor_registry()
    try:
        definition = definitions[normalized_name]
    except KeyError as exc:
        available = ", ".join(sorted(definitions)) or "<none>"
        raise PositionConstructorError(
            f"Undefined position constructor {normalized_name!r}. Available constructors: {available}."
        ) from exc

    validated_params = _validate_constructor_params(definition, params)
    try:
        implementation = _IMPLEMENTATIONS[normalized_name]
    except KeyError as exc:
        raise PositionConstructorError(
            f"Position constructor {normalized_name!r} is registered but has no implementation."
        ) from exc
    return implementation(params=validated_params, version=definition.version)


__all__ = [
    "DEFAULT_POSITION_CONSTRUCTORS_REGISTRY",
    "PositionConstructorDefinition",
    "extract_position_constructor_config",
    "load_position_constructor_registry",
    "normalize_position_constructor_config",
    "position_constructor_metadata_payload",
    "resolve_constructor",
]
