from __future__ import annotations

from src.session.contracts import NotebookProjectSession, PathResolutionReport


def session_to_dict(session: NotebookProjectSession) -> dict[str, object]:
    return session.to_dict()


def path_resolution_to_dict(report: PathResolutionReport) -> dict[str, object]:
    return report.to_dict()
