from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
COLAB_DOCS = (
    REPO_ROOT / "docs" / "colab_project_sessions.md",
    REPO_ROOT / "src" / "resources" / "notebook_workspace" / "docs" / "colab_project_sessions.md",
)


def test_restore_first_colab_docs_include_restore_bootstrap_dry_run_and_execution() -> None:
    expected_tokens = (
        "## Restore-First Colab Workflow (Fresh Runtime)",
        "stratlake-session-archive-restore-bootstrap",
        '--archive-root "{ARCHIVE_ROOT}"',
        '--target-root "{STRATLAKE_ROOT}"',
        "--validate-before-restore",
        "--inspect-before-restore",
        "--dry-run",
        "--json",
        "--overwrite-policy fail_if_exists",
        'ARCHIVE_ID = "notebook-session-001"',
        "ARCHIVE_ROOT = SESSION_ARCHIVES_ROOT / ARCHIVE_ID",
    )
    for path in COLAB_DOCS:
        source = path.read_text(encoding="utf-8")
        for token in expected_tokens:
            assert token in source, f"{path} is missing restore-first token: {token}"


def test_restore_first_colab_docs_cover_boundaries_and_handoff_chaining() -> None:
    expected_tokens = (
        "not canonical StratLake state",
        "Neither is hidden sync",
        "Google APIs",
        "commands do not change notebook CWD, mutate `.env`, mutate `os.environ`",
        "stratlake-validate-marketlake-handoff",
        "before feature builds",
    )
    for path in COLAB_DOCS:
        source = path.read_text(encoding="utf-8")
        for token in expected_tokens:
            assert token in source, f"{path} is missing boundary token: {token}"
        assert "/content/drive/MyDrive/" in source
        assert "/content/gdrive/" not in source
        assert "restore-on-import" not in source
        assert "background restore" not in source
