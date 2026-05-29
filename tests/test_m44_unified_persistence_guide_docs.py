from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GUIDE_PATH = REPO_ROOT / "docs" / "colab_persistence_guide.md"
COLAB_DOCS = (
    REPO_ROOT / "docs" / "colab_project_sessions.md",
    REPO_ROOT / "src" / "resources" / "notebook_workspace" / "docs" / "colab_project_sessions.md",
)


def test_unified_persistence_guide_exists_and_covers_core_categories() -> None:
    assert GUIDE_PATH.exists()
    text = GUIDE_PATH.read_text(encoding="utf-8")

    expected_tokens = (
        "fintech-market-ingestion",
        "StratLake Trade Engine",
        "Local runtime state",
        "Lightweight session persistence",
        "Archive and backup packs",
        "Mounted Drive paths",
        "Persistence Categories",
        "Decision Table",
        "Fresh Runtime Restore Sequence",
        "End-of-Session Persistence Sequence",
        "Canonical and Derived Boundaries",
    )
    for token in expected_tokens:
        assert token in text, f"guide missing token: {token}"


def test_unified_persistence_guide_covers_boundaries_and_no_hidden_sync() -> None:
    text = GUIDE_PATH.read_text(encoding="utf-8")

    expected_tokens = (
        "not canonical",
        "derived",
        "no hidden sync",
        "no Google API calls or OAuth behavior",
        "no automatic persistence or restore-on-import",
        "restore locally first",
        "mounted filesystem",
        "/content/drive/MyDrive",
    )
    for token in expected_tokens:
        assert token in text, f"guide missing boundary token: {token}"

    assert "/content/gdrive" not in text


def test_unified_persistence_guide_references_stratlake_commands_and_fintech_examples() -> None:
    text = GUIDE_PATH.read_text(encoding="utf-8")

    stratlake_tokens = (
        "stratlake-init-session",
        "stratlake-session-export",
        "stratlake-session-import",
        "stratlake-session-archive-bootstrap",
        "stratlake-session-archive-restore-bootstrap",
        "stratlake-notebook-doctor",
        "stratlake-validate-marketlake-handoff",
    )
    for token in stratlake_tokens:
        assert token in text, f"guide missing StratLake command token: {token}"

    fintech_example_tokens = (
        "fintech-session-archive-bootstrap",
        "fintech-session-archive-restore-bootstrap",
        "fintech-save-session",
        "fintech-restore-session",
        "fintech-notebook-doctor",
        "naming may vary",
    )
    for token in fintech_example_tokens:
        assert token in text, f"guide missing fintech companion token: {token}"


def test_colab_docs_link_unified_persistence_guide_and_keep_drive_path_shape() -> None:
    for path in COLAB_DOCS:
        text = path.read_text(encoding="utf-8")
        assert "## Unified Persistence Choices" in text
        assert "colab_persistence_guide.md" in text
        assert "mounted Drive as persistence and transport only" in text
        assert "/content/drive/MyDrive/" in text
        assert "/content/gdrive/" not in text
