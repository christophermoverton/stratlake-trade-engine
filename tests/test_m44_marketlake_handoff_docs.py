from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
COLAB_DOCS = (
    REPO_ROOT / "docs" / "colab_project_sessions.md",
    REPO_ROOT / "src" / "resources" / "notebook_workspace" / "docs" / "colab_project_sessions.md",
)
NOTEBOOK_DOCS = (
    REPO_ROOT / "docs" / "notebook_integration.md",
    REPO_ROOT / "src" / "resources" / "notebook_workspace" / "docs" / "notebook_integration.md",
)


def test_marketlake_handoff_docs_mention_validator_command_and_profile_variables() -> None:
    notebook_tokens = (
        "stratlake-validate-marketlake-handoff",
        "UNIVERSE_CONFIG",
        "START",
        "END",
        "The validator is read-only.",
        "missing date-window coverage",
    )
    readme_tokens = (
        "stratlake-validate-marketlake-handoff",
        "The validator is read-only.",
        "requested symbols",
        "requested date window",
    )
    colab_tokens = (
        "stratlake-validate-marketlake-handoff",
        "UNIVERSE_CONFIG",
        "START",
        "END",
        "This check is read-only.",
        'MARKETLAKE_ROOT = FINTECH_ROOT / "data" / "curated"',
    )
    for path in NOTEBOOK_DOCS:
        source = path.read_text(encoding="utf-8")
        for token in notebook_tokens:
            assert token in source, f"{path} is missing validator token: {token}"
    readme_source = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for token in readme_tokens:
        assert token in readme_source, f"README.md is missing validator token: {token}"
    for path in COLAB_DOCS:
        source = path.read_text(encoding="utf-8")
        for token in colab_tokens:
            assert token in source, f"{path} is missing validator token: {token}"
