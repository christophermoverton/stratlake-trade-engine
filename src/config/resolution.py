from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from src.config.profiles import (
    RuntimeProfile,
    RuntimeProfileError,
    SUPPORTED_RUNTIME_PROFILES,
    load_runtime_profile,
    validate_runtime_profile,
)
from src.config.review import ReviewConfig
from src.config.runtime import resolve_runtime_config


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE_DIR = REPO_ROOT / "configs" / "profiles"

_SETTING_DEFAULTS: dict[str, Any] = {
    "marketlake_root": None,
    "features_root": "data",
    "artifacts_root": "artifacts",
    "duckdb_path": ":memory:",
    "log_level": "INFO",
    "default_timezone": "UTC",
}
_WORKFLOW_CONFIG_DEFAULTS: dict[str, str] = {
    "config_dir": "configs",
    "execution_config": "configs/execution.yml",
    "sanity_config": "configs/sanity.yml",
    "review_config": "configs/review.yml",
    "universe_config": "configs/universe.yml",
    "features_config": "configs/features.yml",
}
_BOUNDARY_DEFAULTS: dict[str, bool] = {
    "direct_scan": True,
    "derived_outputs_authoritative": False,
    "mutates_canonical_artifacts": False,
    "requires_network": False,
    "requires_credentials": False,
    "requires_live_market_data": False,
}
_ENVIRONMENT_SETTING_KEYS: dict[str, str] = {
    "MARKETLAKE_ROOT": "marketlake_root",
    "FEATURES_ROOT": "features_root",
    "ARTIFACTS_ROOT": "artifacts_root",
    "DUCKDB_PATH": "duckdb_path",
    "LOG_LEVEL": "log_level",
    "DEFAULT_TIMEZONE": "default_timezone",
}
_TOP_LEVEL_OVERRIDE_KEYS = frozenset(
    {
        "settings",
        "workflow_configs",
        "runtime",
        "review",
        "boundaries",
    }
)
_PATH_SETTING_KEYS = frozenset({"marketlake_root", "features_root", "artifacts_root"})
_SOURCE_RANK = {"default": 0, "profile": 1, "environment": 2, "cli_override": 3}


class ConfigResolutionError(ValueError):
    """Raised when M39 configuration resolution receives unsupported values."""


@dataclass(frozen=True)
class ConfigProvenanceEntry:
    value: Any
    source: str
    source_detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": _canonicalize(self.value),
            "source": self.source,
            "source_detail": self.source_detail,
        }


@dataclass(frozen=True)
class ResolvedRuntimeConfig:
    settings: dict[str, Any]
    workflow_configs: dict[str, str]
    runtime: dict[str, Any]
    review: dict[str, Any]
    boundaries: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return _canonicalize(
            {
                "settings": self.settings,
                "workflow_configs": self.workflow_configs,
                "runtime": self.runtime,
                "review": self.review,
                "boundaries": self.boundaries,
            }
        )

    def to_json_dict(self) -> dict[str, Any]:
        return self.to_dict()


@dataclass(frozen=True)
class ConfigResolutionResult:
    config: ResolvedRuntimeConfig
    provenance: dict[str, ConfigProvenanceEntry]
    profile_name: str | None
    profile_path: str | None
    precedence: tuple[str, ...] = ("default", "profile", "environment", "cli_override")

    def to_dict(self) -> dict[str, Any]:
        return _canonicalize(
            {
                "config": self.config.to_dict(),
                "provenance": {
                    field: self.provenance[field].to_dict()
                    for field in sorted(self.provenance)
                },
                "profile": {
                    "name": self.profile_name,
                    "path": self.profile_path,
                },
                "precedence": list(self.precedence),
                "authoritative": False,
                "artifact_boundaries": {
                    "direct_scan": self.config.boundaries["direct_scan"],
                    "derived_outputs_authoritative": self.config.boundaries[
                        "derived_outputs_authoritative"
                    ],
                    "mutates_canonical_artifacts": self.config.boundaries[
                        "mutates_canonical_artifacts"
                    ],
                },
            }
        )

    def to_json_dict(self) -> dict[str, Any]:
        return self.to_dict()

    def to_json(self) -> str:
        return json.dumps(self.to_json_dict(), indent=2, sort_keys=True)


def resolve_runtime_profile_config(
    profile: str | RuntimeProfile | None = None,
    *,
    profile_path: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
) -> ConfigResolutionResult:
    """
    Resolve M39 runtime configuration with deterministic provenance.

    Precedence is: defaults < profile config < environment variables < CLI overrides.
    The resolver does not load .env, execute workflows, scan artifacts, or write files.
    """

    selected_profile, selected_profile_path = _load_selected_profile(profile, profile_path)
    profile_detail = _source_detail(selected_profile_path) if selected_profile_path is not None else None

    settings, settings_provenance = _seed_layer("settings", _SETTING_DEFAULTS)
    workflow_configs, workflow_provenance = _seed_layer(
        "workflow_configs",
        _WORKFLOW_CONFIG_DEFAULTS,
    )
    boundaries, boundaries_provenance = _seed_layer("boundaries", _BOUNDARY_DEFAULTS)
    runtime_default = resolve_runtime_config().to_dict()
    runtime, runtime_provenance = _seed_layer("runtime", runtime_default)
    runtime_sources: list[Mapping[str, Any]] = []
    review_default = ReviewConfig.default().to_dict()
    review, review_provenance = _seed_layer("review", review_default)
    review_sources: list[Mapping[str, Any]] = []

    if selected_profile is not None:
        settings = _apply_section(
            settings,
            settings_provenance,
            "settings",
            selected_profile.settings,
            source="profile",
            source_detail=profile_detail or f"profile:{selected_profile.profile}",
        )
        workflow_configs = _apply_section(
            workflow_configs,
            workflow_provenance,
            "workflow_configs",
            selected_profile.workflow_configs,
            source="profile",
            source_detail=profile_detail or f"profile:{selected_profile.profile}",
        )
        boundaries = _apply_section(
            boundaries,
            boundaries_provenance,
            "boundaries",
            selected_profile.boundaries,
            source="profile",
            source_detail=profile_detail or f"profile:{selected_profile.profile}",
        )
        runtime = _apply_section(
            runtime,
            runtime_provenance,
            "runtime",
            selected_profile.runtime,
            source="profile",
            source_detail=profile_detail or f"profile:{selected_profile.profile}",
        )
        if selected_profile.runtime:
            runtime_sources.append(selected_profile.runtime)
        review = _apply_section(
            review,
            review_provenance,
            "review",
            selected_profile.review,
            source="profile",
            source_detail=profile_detail or f"profile:{selected_profile.profile}",
        )
        if selected_profile.review:
            review_sources.append(selected_profile.review)

    if environment is not None:
        env_settings = _settings_from_environment(environment)
        settings = _apply_environment_settings(settings, settings_provenance, env_settings)

    cli_payload = _validate_cli_overrides(cli_overrides)
    if cli_payload:
        settings = _apply_section(
            settings,
            settings_provenance,
            "settings",
            cli_payload.get("settings", {}),
            source="cli_override",
            source_detail="cli_overrides.settings",
        )
        workflow_configs = _apply_section(
            workflow_configs,
            workflow_provenance,
            "workflow_configs",
            cli_payload.get("workflow_configs", {}),
            source="cli_override",
            source_detail="cli_overrides.workflow_configs",
        )
        boundaries = _apply_section(
            boundaries,
            boundaries_provenance,
            "boundaries",
            cli_payload.get("boundaries", {}),
            source="cli_override",
            source_detail="cli_overrides.boundaries",
        )
        runtime = _apply_section(
            runtime,
            runtime_provenance,
            "runtime",
            cli_payload.get("runtime", {}),
            source="cli_override",
            source_detail="cli_overrides.runtime",
        )
        if cli_payload.get("runtime"):
            runtime_sources.append(cli_payload["runtime"])
        review = _apply_section(
            review,
            review_provenance,
            "review",
            cli_payload.get("review", {}),
            source="cli_override",
            source_detail="cli_overrides.review",
        )
        if cli_payload.get("review"):
            review_sources.append(cli_payload["review"])

    settings = _validate_resolved_settings(settings)
    workflow_configs = _validate_resolved_workflow_configs(workflow_configs)
    boundaries = _validate_resolved_boundaries(boundaries)
    runtime, runtime_provenance = _normalize_runtime(runtime_sources, runtime_default, runtime_provenance)
    review, review_provenance = _normalize_review(review_sources, review_default, review_provenance)

    config = ResolvedRuntimeConfig(
        settings=dict(settings),
        workflow_configs=dict(workflow_configs),
        runtime=dict(runtime),
        review=dict(review),
        boundaries=dict(boundaries),
    )
    provenance = {
        **settings_provenance,
        **workflow_provenance,
        **boundaries_provenance,
        **runtime_provenance,
        **review_provenance,
    }
    return ConfigResolutionResult(
        config=config,
        provenance={field: provenance[field] for field in sorted(provenance)},
        profile_name=None if selected_profile is None else selected_profile.profile,
        profile_path=None if selected_profile_path is None else _source_detail(selected_profile_path),
    )


def _load_selected_profile(
    profile: str | RuntimeProfile | None,
    profile_path: str | Path | None,
) -> tuple[RuntimeProfile | None, Path | None]:
    if isinstance(profile, RuntimeProfile):
        if profile_path is not None:
            raise ConfigResolutionError("Provide either a RuntimeProfile object or profile_path, not both.")
        return profile, None
    if profile is not None and profile_path is not None:
        raise ConfigResolutionError("Provide either profile name or profile_path, not both.")
    if profile_path is not None:
        path = Path(profile_path)
        return load_runtime_profile(str(path)), path
    if profile is None:
        return None, None
    profile_name = profile.strip().lower()
    if profile_name not in SUPPORTED_RUNTIME_PROFILES:
        raise ConfigResolutionError(
            f"Runtime profile must be one of {sorted(SUPPORTED_RUNTIME_PROFILES)}."
        )
    path = DEFAULT_PROFILE_DIR / f"{profile_name}.yml"
    return load_runtime_profile(str(path)), path


def _seed_layer(
    section: str,
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, ConfigProvenanceEntry]]:
    values = _flatten({section: payload})
    unflattened = _unflatten({key.removeprefix(f"{section}."): value for key, value in values.items()})
    provenance = {
        field: ConfigProvenanceEntry(
            value=value,
            source="default",
            source_detail="repository defaults",
        )
        for field, value in sorted(values.items())
    }
    return unflattened, provenance


def _apply_section(
    current: Mapping[str, Any],
    provenance: dict[str, ConfigProvenanceEntry],
    section: str,
    payload: Mapping[str, Any],
    *,
    source: str,
    source_detail: str,
) -> dict[str, Any]:
    if not payload:
        return dict(current)
    flat_current = _flatten({section: current})
    flat_payload = _flatten({section: payload})
    flat_current.update(flat_payload)
    for field, value in sorted(flat_payload.items()):
        provenance[field] = ConfigProvenanceEntry(
            value=value,
            source=source,
            source_detail=source_detail,
        )
    return _unflatten({key.removeprefix(f"{section}."): value for key, value in flat_current.items()})


def _settings_from_environment(environment: Mapping[str, str]) -> dict[str, tuple[str, str]]:
    settings: dict[str, tuple[str, str]] = {}
    for env_key in sorted(_ENVIRONMENT_SETTING_KEYS):
        value = environment.get(env_key)
        if value is None or str(value).strip() == "":
            continue
        settings[_ENVIRONMENT_SETTING_KEYS[env_key]] = (str(value), env_key)
    return settings


def _apply_environment_settings(
    current: Mapping[str, Any],
    provenance: dict[str, ConfigProvenanceEntry],
    env_settings: Mapping[str, tuple[str, str]],
) -> dict[str, Any]:
    resolved = dict(current)
    for field_name, (raw_value, env_key) in sorted(env_settings.items()):
        value = _validate_setting_value(field_name, raw_value)
        resolved[field_name] = value
        field = f"settings.{field_name}"
        provenance[field] = ConfigProvenanceEntry(
            value=value,
            source="environment",
            source_detail=env_key,
        )
    return resolved


def _validate_cli_overrides(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ConfigResolutionError("CLI overrides must be a mapping.")
    unknown_keys = sorted(set(payload) - _TOP_LEVEL_OVERRIDE_KEYS)
    if unknown_keys:
        raise ConfigResolutionError(f"CLI overrides contain unsupported keys: {unknown_keys}.")
    try:
        validated = validate_runtime_profile(
            {
                "schema_version": 1,
                "profile": "local",
                **dict(payload),
            }
        )
    except RuntimeProfileError as exc:
        raise ConfigResolutionError(str(exc)) from exc
    return {
        "settings": validated.settings,
        "workflow_configs": validated.workflow_configs,
        "runtime": validated.runtime,
        "review": validated.review,
        "boundaries": {
            key: value
            for key, value in validated.boundaries.items()
            if key in dict(payload).get("boundaries", {})
        },
    }


def _validate_resolved_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for key in sorted(_SETTING_DEFAULTS):
        value = settings.get(key)
        if value is None:
            resolved[key] = None
        else:
            resolved[key] = _validate_setting_value(key, value)
    return resolved


def _validate_setting_value(field_name: str, value: Any) -> Any:
    if field_name in _PATH_SETTING_KEYS:
        return _portable_path(value, field_name=f"settings.{field_name}")
    if field_name == "duckdb_path":
        text = _required_string(value, field_name="settings.duckdb_path")
        return text if text == ":memory:" else _portable_path(text, field_name="settings.duckdb_path")
    if field_name in {"log_level", "default_timezone"}:
        return _required_string(value, field_name=f"settings.{field_name}")
    raise ConfigResolutionError(f"Unsupported settings field: {field_name}.")


def _validate_resolved_workflow_configs(workflow_configs: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(key): _portable_path(value, field_name=f"workflow_configs.{key}")
        for key, value in sorted(workflow_configs.items())
    }


def _validate_resolved_boundaries(boundaries: Mapping[str, Any]) -> dict[str, bool]:
    try:
        validated = validate_runtime_profile(
            {
                "schema_version": 1,
                "profile": "local",
                "boundaries": dict(boundaries),
            }
        )
    except RuntimeProfileError as exc:
        raise ConfigResolutionError(str(exc)) from exc
    return dict(validated.boundaries)


def _normalize_runtime(
    runtime_sources: Sequence[Mapping[str, Any]],
    runtime_default: Mapping[str, Any],
    provenance: dict[str, ConfigProvenanceEntry],
) -> tuple[dict[str, Any], dict[str, ConfigProvenanceEntry]]:
    try:
        normalized = resolve_runtime_config(
            *({"runtime": source} for source in runtime_sources)
        ).to_dict()
    except ValueError as exc:
        raise ConfigResolutionError(str(exc)) from exc
    return _normalize_section(
        "runtime",
        normalized,
        runtime_default,
        provenance,
        derived_detail="runtime resolver",
    )


def _normalize_review(
    review_sources: Sequence[Mapping[str, Any]],
    review_default: Mapping[str, Any],
    provenance: dict[str, ConfigProvenanceEntry],
) -> tuple[dict[str, Any], dict[str, ConfigProvenanceEntry]]:
    try:
        resolved = ReviewConfig.default()
        for source in review_sources:
            resolved = ReviewConfig.from_mapping({"review": source}, base=resolved)
        normalized = resolved.to_dict()
    except ValueError as exc:
        raise ConfigResolutionError(str(exc)) from exc
    output = normalized.get("output")
    if isinstance(output, Mapping) and output.get("path") is not None:
        normalized = {
            **normalized,
            "output": {
                **dict(output),
                "path": _portable_path(output.get("path"), field_name="review.output.path"),
            },
        }
    return _normalize_section(
        "review",
        normalized,
        review_default,
        provenance,
        derived_detail="review resolver",
    )


def _normalize_section(
    section: str,
    normalized: Mapping[str, Any],
    defaults: Mapping[str, Any],
    provenance: dict[str, ConfigProvenanceEntry],
    *,
    derived_detail: str,
) -> tuple[dict[str, Any], dict[str, ConfigProvenanceEntry]]:
    normalized_flat = _flatten({section: normalized})
    default_flat = _flatten({section: defaults})
    highest = _highest_explicit_source(provenance, section)
    for field, value in sorted(normalized_flat.items()):
        existing = provenance.get(field)
        if (
            existing is not None
            and value != existing.value
            and highest is not None
            and _SOURCE_RANK.get(highest.source, -1) > _SOURCE_RANK.get(existing.source, -1)
        ):
            provenance[field] = ConfigProvenanceEntry(
                value=value,
                source=highest.source,
                source_detail=f"{highest.source_detail} via {derived_detail}",
            )
        elif existing is not None and (existing.source != "default" or value == default_flat.get(field)):
            provenance[field] = ConfigProvenanceEntry(
                value=value,
                source=existing.source,
                source_detail=existing.source_detail,
            )
        elif value == default_flat.get(field):
            provenance[field] = ConfigProvenanceEntry(
                value=value,
                source="default",
                source_detail="repository defaults",
            )
        elif highest is not None:
            provenance[field] = ConfigProvenanceEntry(
                value=value,
                source=highest.source,
                source_detail=f"{highest.source_detail} via {derived_detail}",
            )
        else:
            provenance[field] = ConfigProvenanceEntry(
                value=value,
                source="default",
                source_detail=f"repository defaults via {derived_detail}",
            )

    for field in sorted(set(provenance) - set(normalized_flat)):
        if field.startswith(f"{section}."):
            provenance.pop(field)

    return _unflatten(
        {key.removeprefix(f"{section}."): value for key, value in normalized_flat.items()}
    ), provenance


def _highest_explicit_source(
    provenance: Mapping[str, ConfigProvenanceEntry],
    section: str,
) -> ConfigProvenanceEntry | None:
    candidates = [
        entry
        for field, entry in provenance.items()
        if field.startswith(f"{section}.") and entry.source != "default"
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda entry: (_SOURCE_RANK.get(entry.source, -1), entry.source_detail))[-1]


def _flatten(payload: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key in sorted(payload):
        value = payload[key]
        field = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(_flatten(value, field))
        else:
            flat[field] = value
    return flat


def _unflatten(flat: Mapping[str, Any]) -> dict[str, Any]:
    root: dict[str, Any] = {}
    for field, value in sorted(flat.items()):
        cursor = root
        parts = field.split(".")
        for part in parts[:-1]:
            child = cursor.setdefault(part, {})
            if not isinstance(child, dict):
                raise ConfigResolutionError(f"Conflicting config field path: {field}.")
            cursor = child
        cursor[parts[-1]] = value
    return root


def _portable_path(value: Any, *, field_name: str) -> str:
    text = _required_string(value, field_name=field_name)
    normalized = text.replace("\\", "/")
    if normalized != text:
        raise ConfigResolutionError(
            f"Config field '{field_name}' must use POSIX-style '/' separators."
        )
    if "://" in normalized:
        raise ConfigResolutionError(f"Config field '{field_name}' must not be a URI.")
    if normalized.startswith("~"):
        raise ConfigResolutionError(f"Config field '{field_name}' must not use a home shortcut.")
    path = PurePosixPath(normalized)
    if path.is_absolute() or (len(normalized) >= 2 and normalized[1] == ":"):
        raise ConfigResolutionError(f"Config field '{field_name}' must be repository-relative.")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ConfigResolutionError(
            f"Config field '{field_name}' must be a normalized repository-relative path."
        )
    return path.as_posix()


def _required_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ConfigResolutionError(f"Config field '{field_name}' must be a non-empty string.")
    normalized = value.strip()
    if not normalized:
        raise ConfigResolutionError(f"Config field '{field_name}' must be a non-empty string.")
    return normalized


def _source_detail(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return f"<external>/{path.name}"


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    if isinstance(value, tuple):
        return [_canonicalize(item) for item in value]
    return value
