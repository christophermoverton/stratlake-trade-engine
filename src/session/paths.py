from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

from src.session.contracts import (
    DEFAULT_ARTIFACTS_ROOT,
    DEFAULT_CONFIGS_ROOT,
    DEFAULT_FEATURES_ROOT,
    DEFAULT_MARKETLAKE_ROOT,
    SESSION_SCHEMA_VERSION,
    NotebookProjectSession,
    PathKind,
    PathSource,
    ResolvedSessionPath,
)
from src.session.io import PATH_RESOLUTION_FILE_NAME, SESSION_DIR_NAME, SESSION_FILE_NAME

SESSION_PATH_KEYS = (
    "notebook_cwd",
    "project_root",
    "configs_root",
    "artifacts_root",
    "features_root",
    "marketlake_root",
    "drive_root",
)
REQUIRED_SESSION_FIELDS = (
    "schema_version",
    "project_name",
    "notebook_cwd",
    "project_root",
    "configs_root",
    "artifacts_root",
    "features_root",
    "marketlake_root",
)
ENVIRONMENT_FALLBACKS = {
    "artifacts_root": "ARTIFACTS_ROOT",
    "features_root": "FEATURES_ROOT",
    "marketlake_root": "MARKETLAKE_ROOT",
}
DEFAULT_PATHS = {
    "configs_root": DEFAULT_CONFIGS_ROOT,
    "artifacts_root": DEFAULT_ARTIFACTS_ROOT,
    "features_root": DEFAULT_FEATURES_ROOT,
    "marketlake_root": DEFAULT_MARKETLAKE_ROOT,
}


def find_session_root(start: Path | str | None = None) -> Path:
    candidate = Path.cwd() if start is None else Path(start).expanduser()
    candidate = candidate.resolve()
    if candidate.is_file():
        candidate = candidate.parent

    for current in (candidate, *candidate.parents):
        if (current / SESSION_DIR_NAME / SESSION_FILE_NAME).is_file():
            return current
    raise FileNotFoundError(
        f"No StratLake session root found from {candidate.as_posix()}. "
        f"Expected {SESSION_DIR_NAME}/{SESSION_FILE_NAME} in this directory or a parent."
    )


def load_session(root: Path | str) -> NotebookProjectSession:
    session_root = find_session_root(root)
    session_path = session_root / SESSION_DIR_NAME / SESSION_FILE_NAME
    try:
        raw = json.loads(session_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid StratLake session JSON: {session_path.as_posix()}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"Invalid StratLake session JSON object: {session_path.as_posix()}")
    _validate_session_payload(raw, session_path)
    return _session_from_payload(raw, session_root)


def resolve_session_paths(
    session_or_root: NotebookProjectSession | Path | str | Mapping[str, object],
    overrides: Mapping[str, Path | str | None] | None = None,
) -> dict[str, ResolvedSessionPath]:
    session, session_root = _coerce_session(session_or_root)
    override_values = dict(overrides or {})
    unknown = sorted(set(override_values) - set(SESSION_PATH_KEYS))
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"Unknown session path override(s): {joined}")

    resolved: dict[str, ResolvedSessionPath] = {}
    resolved["project_root"] = _resolve_path_value(
        value=override_values.get("project_root", "."),
        project_root=session_root,
        source=PathSource.EXPLICIT_OVERRIDE
        if "project_root" in override_values
        else PathSource.SESSION_METADATA,
        input_path=_input_value(override_values, "project_root", session.project_root.path),
    )

    for key in ("notebook_cwd", "configs_root", "artifacts_root", "features_root", "marketlake_root"):
        value, source, input_path = _select_path_value(
            key=key,
            session=session,
            overrides=override_values,
        )
        resolved[key] = _resolve_path_value(
            value=value,
            project_root=session_root,
            source=source,
            input_path=input_path,
        )

    if "drive_root" in override_values:
        drive_override = override_values["drive_root"]
        if drive_override is not None:
            resolved["drive_root"] = _resolve_path_value(
                value=drive_override,
                project_root=session_root,
                source=PathSource.EXPLICIT_OVERRIDE,
                input_path=str(drive_override),
            )
    elif session.drive_root is not None:
        resolved["drive_root"] = _resolve_path_value(
            value=session.drive_root.path,
            project_root=session_root,
            source=PathSource.SESSION_METADATA,
            input_path=session.drive_root.path,
        )

    return resolved


def write_path_resolution_report(
    session_or_root: NotebookProjectSession | Path | str | Mapping[str, object],
    *,
    overrides: Mapping[str, Path | str | None] | None = None,
    overwrite: bool = True,
) -> Path:
    session, session_root = _coerce_session(session_or_root)
    path_resolution_path = (
        session_root / SESSION_DIR_NAME / PATH_RESOLUTION_FILE_NAME
    ).resolve()
    _ensure_under_project_root(path_resolution_path, session_root)
    if path_resolution_path.exists() and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing path resolution report without overwrite=True: "
            f"{path_resolution_path.as_posix()}"
        )

    paths = resolve_session_paths(session, overrides=overrides)
    data = {
        "schema_version": session.schema_version,
        "project_name": session.project_name,
        "paths": {key: value.to_resolution_dict() for key, value in paths.items()},
    }
    path_resolution_path.parent.mkdir(parents=True, exist_ok=True)
    path_resolution_path.write_text(
        json.dumps(data, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return path_resolution_path


def _validate_session_payload(payload: Mapping[str, object], session_path: Path) -> None:
    missing = [key for key in REQUIRED_SESSION_FIELDS if key not in payload]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            f"StratLake session file is missing required field(s) {joined}: "
            f"{session_path.as_posix()}"
        )
    if payload["schema_version"] != SESSION_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported StratLake session schema version "
            f"{payload['schema_version']!r}; expected {SESSION_SCHEMA_VERSION}."
        )
    if not isinstance(payload["project_name"], str) or not payload["project_name"]:
        raise ValueError(f"StratLake session project_name must be a non-empty string: {session_path.as_posix()}")
    for key in REQUIRED_SESSION_FIELDS:
        if key in ("schema_version", "project_name"):
            continue
        _validate_path_payload(key, payload[key], session_path)
    if "drive_root" in payload:
        _validate_path_payload("drive_root", payload["drive_root"], session_path)


def _validate_path_payload(key: str, value: object, session_path: Path) -> None:
    if not isinstance(value, dict):
        raise ValueError(
            f"StratLake session field {key} must be a path object: {session_path.as_posix()}"
        )
    for field in ("path", "kind", "source"):
        if field not in value:
            raise ValueError(
                f"StratLake session field {key} is missing {field}: {session_path.as_posix()}"
            )
        if not isinstance(value[field], str):
            raise ValueError(
                f"StratLake session field {key}.{field} must be a string: {session_path.as_posix()}"
            )
    if not value["path"]:
        raise ValueError(
            f"StratLake session field {key}.path must be non-empty: {session_path.as_posix()}"
        )
    _coerce_path_kind(value["kind"])
    _coerce_path_source(value["source"])


def _session_from_payload(payload: Mapping[str, object], session_root: Path) -> NotebookProjectSession:
    drive_persistence = payload.get("drive_persistence", {})
    drive_persistence_enabled = False
    if isinstance(drive_persistence, dict):
        drive_persistence_enabled = bool(drive_persistence.get("enabled", False))

    return NotebookProjectSession(
        schema_version=SESSION_SCHEMA_VERSION,
        project_name=str(payload["project_name"]),
        notebook_cwd=_loaded_path("notebook_cwd", payload["notebook_cwd"], session_root),
        project_root=_loaded_path("project_root", payload["project_root"], session_root),
        configs_root=_loaded_path("configs_root", payload["configs_root"], session_root),
        artifacts_root=_loaded_path("artifacts_root", payload["artifacts_root"], session_root),
        features_root=_loaded_path("features_root", payload["features_root"], session_root),
        marketlake_root=_loaded_path("marketlake_root", payload["marketlake_root"], session_root),
        drive_root=_loaded_path("drive_root", payload["drive_root"], session_root)
        if "drive_root" in payload
        else None,
        drive_persistence_enabled=drive_persistence_enabled,
    )


def _loaded_path(
    key: str,
    value: object,
    session_root: Path,
) -> ResolvedSessionPath:
    if not isinstance(value, dict):
        raise ValueError(f"Invalid loaded path for {key}")
    path_text = str(value["path"])
    if not path_text:
        return ResolvedSessionPath(
            path="",
            kind=_coerce_path_kind(str(value["kind"])),
            source=_coerce_path_source(str(value["source"])),
            resolved_path="",
            input_path="",
            base=None,
        )
    resolved = _resolve_path_value(
        value=path_text,
        project_root=session_root,
        source=_coerce_path_source(str(value["source"])),
        input_path=path_text,
    )
    expected_kind = _coerce_path_kind(str(value["kind"]))
    if resolved.kind is not expected_kind:
        return ResolvedSessionPath(
            path=resolved.path,
            kind=expected_kind,
            source=resolved.source,
            resolved_path=resolved.resolved_path,
            input_path=resolved.input_path,
            base=resolved.base,
        )
    return resolved


def _coerce_session(
    session_or_root: NotebookProjectSession | Path | str | Mapping[str, object],
) -> tuple[NotebookProjectSession, Path]:
    if isinstance(session_or_root, NotebookProjectSession):
        session_root = Path(session_or_root.project_root.resolved_path).resolve()
        return session_or_root, session_root
    if isinstance(session_or_root, Mapping):
        project_root_value = session_or_root.get("project_root")
        if not isinstance(project_root_value, Mapping):
            raise ValueError("Session mapping must include project_root metadata")
        root_value = project_root_value.get("resolved_path")
        if not isinstance(root_value, str):
            raise ValueError("Session mapping must include project_root.resolved_path")
        root = Path(root_value).resolve()
        return _session_from_payload(session_or_root, root), root
    session = load_session(session_or_root)
    return session, Path(session.project_root.resolved_path).resolve()


def _select_path_value(
    *,
    key: str,
    session: NotebookProjectSession,
    overrides: Mapping[str, Path | str | None],
) -> tuple[Path | str, PathSource, str | None]:
    if key in overrides and overrides[key] is not None:
        return overrides[key], PathSource.EXPLICIT_OVERRIDE, str(overrides[key])

    session_value = getattr(session, key)
    if isinstance(session_value, ResolvedSessionPath) and session_value.path:
        return session_value.path, PathSource.SESSION_METADATA, session_value.path

    env_name = ENVIRONMENT_FALLBACKS.get(key)
    if env_name and os.environ.get(env_name):
        value = os.environ[env_name]
        return value, PathSource.ENVIRONMENT_VARIABLE, f"{env_name}={value}"

    if key in DEFAULT_PATHS:
        value = DEFAULT_PATHS[key]
        return value, PathSource.DEFAULT, value

    raise ValueError(f"No value available for required session path: {key}")


def _input_value(
    overrides: Mapping[str, Path | str | None],
    key: str,
    fallback: str,
) -> str | None:
    if key in overrides:
        return None if overrides[key] is None else str(overrides[key])
    return fallback


def _resolve_path_value(
    *,
    value: Path | str,
    project_root: Path,
    source: PathSource,
    input_path: str | None,
) -> ResolvedSessionPath:
    candidate = Path(value).expanduser()
    base = None if candidate.is_absolute() else project_root.as_posix()
    resolved = candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()
    kind = _classify_path(candidate=candidate, resolved=resolved, project_root=project_root)
    return ResolvedSessionPath(
        path=_serialize_path(candidate=candidate, resolved=resolved, project_root=project_root, kind=kind),
        kind=kind,
        source=source,
        input_path=input_path,
        base=base,
        resolved_path=resolved.as_posix(),
    )


def _classify_path(*, candidate: Path, resolved: Path, project_root: Path) -> PathKind:
    if _is_relative_to(resolved, project_root):
        return PathKind.PROJECT_INTERNAL
    if candidate.is_absolute():
        return PathKind.EXTERNAL_ABSOLUTE
    return PathKind.EXTERNAL_OR_PROJECT_RELATIVE


def _serialize_path(
    *,
    candidate: Path,
    resolved: Path,
    project_root: Path,
    kind: PathKind,
) -> str:
    if kind is PathKind.PROJECT_INTERNAL:
        relative = resolved.relative_to(project_root)
        serialized = relative.as_posix()
        return serialized or "."
    if candidate.is_absolute():
        return resolved.as_posix()
    return candidate.as_posix()


def _coerce_path_kind(value: str) -> PathKind:
    try:
        return PathKind(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported StratLake session path kind: {value}") from exc


def _coerce_path_source(value: str) -> PathSource:
    try:
        return PathSource(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported StratLake session path source: {value}") from exc


def _ensure_under_project_root(path: Path, project_root: Path) -> None:
    try:
        path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(
            f"Refusing to write path resolution report outside project root: {path.as_posix()}"
        ) from exc


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
    except ValueError:
        return False
    return True
