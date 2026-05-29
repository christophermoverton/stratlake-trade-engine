from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
COLAB_DOCS = (
    REPO_ROOT / "docs" / "colab_project_sessions.md",
    REPO_ROOT / "src" / "resources" / "notebook_workspace" / "docs" / "colab_project_sessions.md",
)
OTHER_DOCS = (
    REPO_ROOT / "docs" / "notebook_integration.md",
    REPO_ROOT / "docs" / "notebook_workspace_bootstrap.md",
    REPO_ROOT / "README.md",
)


def test_m44_notebook_doctor_colab_docs_cover_command_and_boundaries() -> None:
    expected_tokens = (
        "## Notebook Doctor Preflight (Read-Only)",
        "stratlake-notebook-doctor",
        '--root "{STRATLAKE_ROOT}"',
        '--marketlake-root "{MARKETLAKE_ROOT}"',
        '--drive-root "{DRIVE_ROOT}"',
        '--archive-root "{ARCHIVE_ROOT}"',
        "--check-configs",
        "--check-universe",
        "--check-drive",
        "--check-archives",
        "--check-secrets",
        "--json",
        "no .env or os.environ mutation",
        "Google API calls",
        "no hidden sync",
        "Secret values are never printed",
        "stratlake-validate-marketlake-handoff",
        "stratlake-session-archive-restore-bootstrap",
    )

    for path in COLAB_DOCS:
        text = path.read_text(encoding="utf-8")
        for token in expected_tokens:
            assert token in text, f"{path} is missing notebook doctor token: {token}"


def test_m44_notebook_doctor_cross_docs_reference_command() -> None:
    for path in OTHER_DOCS:
        text = path.read_text(encoding="utf-8")
        assert "stratlake-notebook-doctor" in text
        assert "read-only" in text.lower()
