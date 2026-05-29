from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_NOTES = REPO_ROOT / "docs" / "m44_release_notes.md"
RELEASE_CHECKLIST = REPO_ROOT / "docs" / "m44_release_validation_checklist.md"
README = REPO_ROOT / "README.md"


def test_m44_release_notes_exist_and_identify_release_metadata() -> None:
    assert RELEASE_NOTES.exists()
    text = RELEASE_NOTES.read_text(encoding="utf-8")

    expected_tokens = (
        "M44 - Colab Notebook Session Ergonomics and MarketLake Restore Handoff",
        "0.44.0",
        "v0.44.0-colab-notebook-session-ergonomics-marketlake-handoff",
        "feature/m44-unified-colab-session-drive-persistence",
        "Target branch:",
        "`main`",
        "`#483` through `#491`",
    )
    for token in expected_tokens:
        assert token in text, f"M44 release notes missing token: {token}"


def test_m44_release_notes_cover_commands_and_architecture_boundaries() -> None:
    text = RELEASE_NOTES.read_text(encoding="utf-8")

    expected_tokens = (
        "stratlake-init-session",
        "--notebook-configs",
        "stratlake-validate-marketlake-handoff",
        "stratlake-notebook-doctor",
        "Drive remains mounted filesystem persistence/transport only",
        "archive and backup packs remain derived and non-canonical",
        "local runtime roots remain active working state",
        "no hidden sync",
        "no Google API or OAuth behavior",
        "no restore-on-import behavior",
        "no background persistence",
        "no fintech ingestion behavior added to StratLake",
    )
    for token in expected_tokens:
        assert token in text, f"M44 release notes missing boundary token: {token}"


def test_m44_release_notes_include_merge_readiness_evidence() -> None:
    text = RELEASE_NOTES.read_text(encoding="utf-8")

    expected_tokens = (
        "Recommended PR title:",
        "M44: Colab notebook session ergonomics and MarketLake handoff",
        "Generated-output cleanup status:",
        "docs/examples/output/m38_static_evidence_review_pack_example/",
        "python -m src.cli.run_docs_path_lint",
        "python -m pytest -q",
        "Post-Merge Validation Checklist",
    )
    for token in expected_tokens:
        assert token in text, f"M44 release notes missing readiness token: {token}"


def test_m44_release_checklist_and_readme_link_release_notes() -> None:
    checklist = RELEASE_CHECKLIST.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")

    assert "v0.44.0-colab-notebook-session-ergonomics-marketlake-handoff" in checklist
    assert "python -m src.cli.run_docs_path_lint" in checklist
    assert "docs/m44_release_notes.md" in readme
    assert "docs/m44_release_validation_checklist.md" in readme
    assert "Release 0.44.0: Colab Notebook Session Ergonomics" in readme


def test_m44_docs_references_are_relative_and_lint_friendly() -> None:
    for path in (RELEASE_NOTES, RELEASE_CHECKLIST):
        text = path.read_text(encoding="utf-8")
        assert "file://" not in text
        assert "C:\\" not in text
        assert "/Users/" not in text
        assert "/home/" not in text
