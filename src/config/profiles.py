from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import yaml

from src.config.review import ReviewConfig
from src.config.runtime import resolve_runtime_config
from src.config.settings import load_yaml_config


SUPPORTED_RUNTIME_PROFILES = frozenset({"local", "ci", "notebook", "pipeline"})
PROFILE_SCHEMA_VERSION = 1

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "profile",
        "description",
        "use_case",
        "settings",
        "workflow_configs",
        "runtime",
        "review",
        "boundaries",
    }
)
_SETTING_KEYS = frozenset(
    {
        "marketlake_root",
        "features_root",
        "artifacts_root",
        "duckdb_path",
        "log_level",
        "default_timezone",
    }
)
_PATH_SETTING_KEYS = frozenset({"marketlake_root", "features_root", "artifacts_root"})
_WORKFLOW_CONFIG_KEYS = frozenset(
    {
        "config_dir",
        "execution_config",
        "sanity_config",
        "review_config",
        "universe_config",
        "features_config",
        "strategies_config",
        "portfolios_config",
        "evaluation_config",
        "pipeline_config",
        "benchmark_pack_config",
        "research_campaign_config",
    }
)
_BOUNDARY_KEYS = frozenset(
    {
        "direct_scan",
        "derived_outputs_authoritative",
        "mutates_canonical_artifacts",
        "requires_network",
        "requires_credentials",
        "requires_live_market_data",
    }
)
_FALSE_BOUNDARY_KEYS = frozenset(
    {
        "derived_outputs_authoritative",
        "mutates_canonical_artifacts",
        "requires_network",
        "requires_credentials",
        "requires_live_market_data",
    }
)


class RuntimeProfileError(ValueError):
    """Raised when a runtime profile does not satisfy the M39 profile contract."""


@dataclass(frozen=True)
class RuntimeProfile:
    """Validated, non-secret runtime profile metadata for selecting workflow defaults."""

    schema_version: int
    profile: str
    description: str | None
    use_case: str | None
    settings: dict[str, Any]
    workflow_configs: dict[str, str]
    runtime: dict[str, Any]
    review: dict[str, Any]
    boundaries: dict[str, bool]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RuntimeProfile":
        if not isinstance(payload, Mapping):
            raise RuntimeProfileError("Runtime profile must be a top-level mapping.")

        unknown_keys = sorted(set(payload) - _TOP_LEVEL_KEYS)
        if unknown_keys:
            raise RuntimeProfileError(f"Runtime profile contains unsupported keys: {unknown_keys}.")

        schema_version = payload.get("schema_version")
        if schema_version != PROFILE_SCHEMA_VERSION:
            raise RuntimeProfileError(
                f"Runtime profile field 'schema_version' must be {PROFILE_SCHEMA_VERSION}."
            )

        profile_name = _required_string(payload.get("profile"), field_name="profile").lower()
        if profile_name not in SUPPORTED_RUNTIME_PROFILES:
            raise RuntimeProfileError(
                "Runtime profile field 'profile' must be one of "
                f"{sorted(SUPPORTED_RUNTIME_PROFILES)}."
            )

        description = _optional_string(payload.get("description"), field_name="description")
        use_case = _optional_string(payload.get("use_case"), field_name="use_case")
        settings = _validate_settings(payload.get("settings"))
        workflow_configs = _validate_workflow_configs(payload.get("workflow_configs"))
        runtime = _validate_runtime(payload.get("runtime"))
        review = _validate_review(payload.get("review"))
        boundaries = _validate_boundaries(payload.get("boundaries"))

        return cls(
            schema_version=schema_version,
            profile=profile_name,
            description=description,
            use_case=use_case,
            settings=settings,
            workflow_configs=workflow_configs,
            runtime=runtime,
            review=review,
            boundaries=boundaries,
        )

    @property
    def explicit_settings(self) -> dict[str, Any]:
        return dict(self.settings)

    @property
    def explicit_workflow_configs(self) -> dict[str, str]:
        return dict(self.workflow_configs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile": self.profile,
            "description": self.description,
            "use_case": self.use_case,
            "settings": dict(self.settings),
            "workflow_configs": dict(self.workflow_configs),
            "runtime": dict(self.runtime),
            "review": dict(self.review),
            "boundaries": dict(self.boundaries),
        }


def load_runtime_profile(path: str) -> RuntimeProfile:
    """Load and validate a profile file without resolving data or mutating artifacts."""

    payload = load_yaml_config(Path(path))
    return RuntimeProfile.from_mapping(payload)


def load_runtime_profile_text(text: str) -> RuntimeProfile:
    """Validate a profile payload from YAML text."""

    payload = yaml.safe_load(text) or {}
    return RuntimeProfile.from_mapping(payload)


def validate_runtime_profile(payload: Mapping[str, Any]) -> RuntimeProfile:
    """Validate a mapping against the M39 runtime profile contract."""

    return RuntimeProfile.from_mapping(payload)


def _validate_settings(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise RuntimeProfileError("Runtime profile field 'settings' must be a mapping.")
    unknown_keys = sorted(set(payload) - _SETTING_KEYS)
    if unknown_keys:
        raise RuntimeProfileError(f"Runtime profile field 'settings' contains unsupported keys: {unknown_keys}.")

    resolved: dict[str, Any] = {}
    for key, value in payload.items():
        if key in _PATH_SETTING_KEYS:
            resolved[key] = _portable_path(value, field_name=f"settings.{key}")
        elif key == "duckdb_path":
            resolved[key] = _duckdb_path(value)
        elif key in {"log_level", "default_timezone"}:
            resolved[key] = _required_string(value, field_name=f"settings.{key}")
    return resolved


def _validate_workflow_configs(payload: Any) -> dict[str, str]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise RuntimeProfileError("Runtime profile field 'workflow_configs' must be a mapping.")
    unknown_keys = sorted(set(payload) - _WORKFLOW_CONFIG_KEYS)
    if unknown_keys:
        raise RuntimeProfileError(
            f"Runtime profile field 'workflow_configs' contains unsupported keys: {unknown_keys}."
        )
    return {
        str(key): _portable_path(value, field_name=f"workflow_configs.{key}")
        for key, value in payload.items()
    }


def _validate_runtime(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise RuntimeProfileError("Runtime profile field 'runtime' must be a mapping.")
    try:
        resolve_runtime_config({"runtime": payload})
    except ValueError as exc:
        raise RuntimeProfileError(str(exc)) from exc
    return dict(payload)


def _validate_review(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise RuntimeProfileError("Runtime profile field 'review' must be a mapping.")
    try:
        ReviewConfig.from_mapping({"review": payload})
    except ValueError as exc:
        raise RuntimeProfileError(str(exc)) from exc
    resolved = dict(payload)
    output = resolved.get("output")
    if isinstance(output, Mapping) and output.get("path") is not None:
        resolved["output"] = {
            **dict(output),
            "path": _portable_path(output.get("path"), field_name="review.output.path"),
        }
    return resolved


def _validate_boundaries(payload: Any) -> dict[str, bool]:
    defaults = {
        "direct_scan": True,
        "derived_outputs_authoritative": False,
        "mutates_canonical_artifacts": False,
        "requires_network": False,
        "requires_credentials": False,
        "requires_live_market_data": False,
    }
    if payload is None:
        return defaults
    if not isinstance(payload, Mapping):
        raise RuntimeProfileError("Runtime profile field 'boundaries' must be a mapping.")
    unknown_keys = sorted(set(payload) - _BOUNDARY_KEYS)
    if unknown_keys:
        raise RuntimeProfileError(f"Runtime profile field 'boundaries' contains unsupported keys: {unknown_keys}.")

    resolved = dict(defaults)
    for key, value in payload.items():
        if not isinstance(value, bool):
            raise RuntimeProfileError(f"Runtime profile field 'boundaries.{key}' must be a boolean.")
        resolved[str(key)] = value

    if resolved["direct_scan"] is not True:
        raise RuntimeProfileError("Runtime profiles must keep direct_scan true; direct scan is canonical.")
    invalid_true = sorted(key for key in _FALSE_BOUNDARY_KEYS if resolved[key] is True)
    if invalid_true:
        raise RuntimeProfileError(
            "Runtime profiles cannot enable non-portable or authoritative boundary flags: "
            f"{invalid_true}."
        )
    return resolved


def _duckdb_path(value: Any) -> str:
    text = _required_string(value, field_name="settings.duckdb_path")
    if text == ":memory:":
        return text
    return _portable_path(text, field_name="settings.duckdb_path")


def _portable_path(value: Any, *, field_name: str) -> str:
    text = _required_string(value, field_name=field_name)
    normalized = text.replace("\\", "/")
    if normalized != text:
        raise RuntimeProfileError(
            f"Runtime profile field '{field_name}' must use POSIX-style '/' separators."
        )
    if "://" in normalized:
        raise RuntimeProfileError(f"Runtime profile field '{field_name}' must not be a URI.")
    if normalized.startswith("~"):
        raise RuntimeProfileError(f"Runtime profile field '{field_name}' must not use a home shortcut.")
    path = PurePosixPath(normalized)
    if path.is_absolute() or (len(normalized) >= 2 and normalized[1] == ":"):
        raise RuntimeProfileError(
            f"Runtime profile field '{field_name}' must be repository-relative."
        )
    if any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeProfileError(
            f"Runtime profile field '{field_name}' must be a normalized repository-relative path."
        )
    return path.as_posix()


def _required_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise RuntimeProfileError(f"Runtime profile field '{field_name}' must be a non-empty string.")
    normalized = value.strip()
    if not normalized:
        raise RuntimeProfileError(f"Runtime profile field '{field_name}' must be a non-empty string.")
    return normalized


def _optional_string(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, field_name=field_name)
