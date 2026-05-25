"""Notebook project-session contracts for StratLake workflows."""

from src.session.contracts import (
    NotebookProjectSession,
    PathKind,
    PathResolutionReport,
    PathSource,
    ResolvedSessionPath,
    create_notebook_project_session,
)
from src.session.drive_adapter import (
    SessionCopyItem,
    SessionCopyPlan,
    SessionCopyResult,
    export_session_to_drive,
    import_session_from_drive,
    plan_session_copy,
    write_drive_sync_manifest,
)
from src.session.io import SessionWriteResult, write_session_files
from src.session.paths import (
    find_session_root,
    load_session,
    resolve_session_paths,
    write_path_resolution_report,
)

__all__ = [
    "NotebookProjectSession",
    "PathKind",
    "PathResolutionReport",
    "PathSource",
    "ResolvedSessionPath",
    "SessionCopyItem",
    "SessionCopyPlan",
    "SessionCopyResult",
    "SessionWriteResult",
    "create_notebook_project_session",
    "export_session_to_drive",
    "find_session_root",
    "import_session_from_drive",
    "load_session",
    "plan_session_copy",
    "resolve_session_paths",
    "write_drive_sync_manifest",
    "write_path_resolution_report",
    "write_session_files",
]
