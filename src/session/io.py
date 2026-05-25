from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from src.session.contracts import NotebookProjectSession

SESSION_DIR_NAME = ".stratlake"
SESSION_FILE_NAME = "session.json"
PATH_RESOLUTION_FILE_NAME = "path_resolution.json"


@dataclass(frozen=True)
class SessionWriteResult:
    session_path: Path
    path_resolution_path: Path


def write_session_files(
    session: NotebookProjectSession,
    *,
    overwrite: bool = False,
) -> SessionWriteResult:
    project_root = Path(session.project_root.resolved_path).resolve()
    session_dir = (project_root / SESSION_DIR_NAME).resolve()
    _ensure_under_project_root(session_dir, project_root)
    session_path = (session_dir / SESSION_FILE_NAME).resolve()
    resolution_path = (session_dir / PATH_RESOLUTION_FILE_NAME).resolve()
    _ensure_under_project_root(session_path, project_root)
    _ensure_under_project_root(resolution_path, project_root)

    existing = [path for path in (session_path, resolution_path) if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(path.as_posix() for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing session metadata without overwrite=True: "
            f"{joined}"
        )

    project_root.mkdir(parents=True, exist_ok=True)
    session_dir.mkdir(parents=True, exist_ok=True)
    _write_json(session_path, session.to_dict())
    _write_json(resolution_path, session.resolution_report().to_dict())
    return SessionWriteResult(
        session_path=session_path,
        path_resolution_path=resolution_path,
    )


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.write_text(
        json.dumps(data, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _ensure_under_project_root(path: Path, project_root: Path) -> None:
    try:
        path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(
            f"Refusing to write session metadata outside project root: {path.as_posix()}"
        ) from exc
