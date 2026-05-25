from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping

SESSION_SCHEMA_VERSION = 1
DEFAULT_CONFIGS_ROOT = "configs"
DEFAULT_ARTIFACTS_ROOT = "artifacts"
DEFAULT_FEATURES_ROOT = "data/curated"
DEFAULT_MARKETLAKE_ROOT = "data/curated"


class PathKind(str, Enum):
    PROJECT_INTERNAL = "project_internal"
    EXTERNAL_ABSOLUTE = "external_absolute"
    EXTERNAL_OR_PROJECT_RELATIVE = "external_or_project_relative"


class PathSource(str, Enum):
    CURRENT_WORKING_DIRECTORY = "current_working_directory"
    DEFAULT = "default"
    EXPLICIT_ARTIFACTS_ROOT = "explicit_artifacts_root"
    EXPLICIT_CONFIGS_ROOT = "explicit_configs_root"
    EXPLICIT_DRIVE_ROOT = "explicit_drive_root"
    EXPLICIT_FEATURES_ROOT = "explicit_features_root"
    EXPLICIT_MARKETLAKE_ROOT = "explicit_marketlake_root"
    EXPLICIT_ROOT = "explicit_root"


@dataclass(frozen=True)
class ResolvedSessionPath:
    path: str
    kind: PathKind
    source: PathSource
    resolved_path: str
    input_path: str | None = None
    base: str | None = None

    def to_session_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "kind": self.kind.value,
            "source": self.source.value,
        }

    def to_resolution_dict(self) -> dict[str, str | None]:
        return {
            "input_path": self.input_path,
            "path": self.path,
            "kind": self.kind.value,
            "source": self.source.value,
            "base": self.base,
            "resolved_path": self.resolved_path,
        }


@dataclass(frozen=True)
class PathResolutionReport:
    schema_version: int
    project_name: str
    paths: Mapping[str, ResolvedSessionPath]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "project_name": self.project_name,
            "paths": {
                name: resolved.to_resolution_dict()
                for name, resolved in self.paths.items()
            },
        }


@dataclass(frozen=True)
class NotebookProjectSession:
    schema_version: int
    project_name: str
    notebook_cwd: ResolvedSessionPath
    project_root: ResolvedSessionPath
    configs_root: ResolvedSessionPath
    artifacts_root: ResolvedSessionPath
    features_root: ResolvedSessionPath
    marketlake_root: ResolvedSessionPath
    drive_root: ResolvedSessionPath | None = None

    def paths(self) -> dict[str, ResolvedSessionPath]:
        values = {
            "notebook_cwd": self.notebook_cwd,
            "project_root": self.project_root,
            "configs_root": self.configs_root,
            "artifacts_root": self.artifacts_root,
            "features_root": self.features_root,
            "marketlake_root": self.marketlake_root,
        }
        if self.drive_root is not None:
            values["drive_root"] = self.drive_root
        return values

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "schema_version": self.schema_version,
            "project_name": self.project_name,
            "notebook_cwd": self.notebook_cwd.to_session_dict(),
            "project_root": self.project_root.to_session_dict(),
            "configs_root": self.configs_root.to_session_dict(),
            "artifacts_root": self.artifacts_root.to_session_dict(),
            "features_root": self.features_root.to_session_dict(),
            "marketlake_root": self.marketlake_root.to_session_dict(),
        }
        if self.drive_root is not None:
            data["drive_root"] = self.drive_root.to_session_dict()
        return data

    def resolution_report(self) -> PathResolutionReport:
        return PathResolutionReport(
            schema_version=self.schema_version,
            project_name=self.project_name,
            paths=self.paths(),
        )


def create_notebook_project_session(
    *,
    project_root: Path | str,
    project_name: str | None = None,
    notebook_cwd: Path | str | None = None,
    configs_root: Path | str = DEFAULT_CONFIGS_ROOT,
    artifacts_root: Path | str = DEFAULT_ARTIFACTS_ROOT,
    features_root: Path | str = DEFAULT_FEATURES_ROOT,
    marketlake_root: Path | str = DEFAULT_MARKETLAKE_ROOT,
    drive_root: Path | str | None = None,
) -> NotebookProjectSession:
    root_input = Path(project_root).expanduser()
    resolved_project_root = root_input.resolve()
    effective_project_name = project_name or resolved_project_root.name
    effective_notebook_cwd = Path.cwd() if notebook_cwd is None else Path(notebook_cwd)

    notebook_cwd_path = _resolve_session_path(
        value=effective_notebook_cwd,
        project_root=resolved_project_root,
        source=PathSource.CURRENT_WORKING_DIRECTORY,
        input_path=None if notebook_cwd is None else str(notebook_cwd),
    )
    project_root_path = ResolvedSessionPath(
        path=".",
        kind=PathKind.PROJECT_INTERNAL,
        source=PathSource.EXPLICIT_ROOT,
        input_path=str(project_root),
        base=None,
        resolved_path=resolved_project_root.as_posix(),
    )
    drive_root_path = None
    if drive_root is not None:
        drive_root_path = _resolve_session_path(
            value=drive_root,
            project_root=resolved_project_root,
            source=PathSource.EXPLICIT_DRIVE_ROOT,
            input_path=str(drive_root),
        )

    return NotebookProjectSession(
        schema_version=SESSION_SCHEMA_VERSION,
        project_name=effective_project_name,
        notebook_cwd=notebook_cwd_path,
        project_root=project_root_path,
        configs_root=_resolve_session_path(
            value=configs_root,
            project_root=resolved_project_root,
            source=_source_for_path(
                value=configs_root,
                default=DEFAULT_CONFIGS_ROOT,
                explicit_source=PathSource.EXPLICIT_CONFIGS_ROOT,
            ),
            input_path=str(configs_root),
        ),
        artifacts_root=_resolve_session_path(
            value=artifacts_root,
            project_root=resolved_project_root,
            source=_source_for_path(
                value=artifacts_root,
                default=DEFAULT_ARTIFACTS_ROOT,
                explicit_source=PathSource.EXPLICIT_ARTIFACTS_ROOT,
            ),
            input_path=str(artifacts_root),
        ),
        features_root=_resolve_session_path(
            value=features_root,
            project_root=resolved_project_root,
            source=_source_for_path(
                value=features_root,
                default=DEFAULT_FEATURES_ROOT,
                explicit_source=PathSource.EXPLICIT_FEATURES_ROOT,
            ),
            input_path=str(features_root),
        ),
        marketlake_root=_resolve_session_path(
            value=marketlake_root,
            project_root=resolved_project_root,
            source=_source_for_path(
                value=marketlake_root,
                default=DEFAULT_MARKETLAKE_ROOT,
                explicit_source=PathSource.EXPLICIT_MARKETLAKE_ROOT,
            ),
            input_path=str(marketlake_root),
        ),
        drive_root=drive_root_path,
    )


def _source_for_path(
    *,
    value: Path | str,
    default: str,
    explicit_source: PathSource,
) -> PathSource:
    if Path(value).as_posix() == default:
        return PathSource.DEFAULT
    return explicit_source


def _resolve_session_path(
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


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
    except ValueError:
        return False
    return True
