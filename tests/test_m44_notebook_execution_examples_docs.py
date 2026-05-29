from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
COLAB_DOCS = (
    REPO_ROOT / "docs" / "colab_project_sessions.md",
    REPO_ROOT / "src" / "resources" / "notebook_workspace" / "docs" / "colab_project_sessions.md",
)
CROSS_DOCS = (
    REPO_ROOT / "docs" / "notebook_integration.md",
    REPO_ROOT / "src" / "resources" / "notebook_workspace" / "docs" / "notebook_integration.md",
    REPO_ROOT / "docs" / "notebook_workspace_bootstrap.md",
    REPO_ROOT / "README.md",
)


def test_m44_notebook_execution_section_and_boundary_tokens_present() -> None:
    expected_tokens = (
        "## Notebook-Native Execution After Restore",
        "Use CLI for workflow boundaries where command behavior is the point",
        "stratlake-init-session",
        "stratlake-session-archive-bootstrap",
        "stratlake-session-archive-restore-bootstrap",
        "stratlake-notebook-doctor",
        "stratlake-validate-marketlake-handoff",
        "Use Python execution APIs for interactive research after readiness passes",
        "same execution system and canonical artifact",
        "Do not duplicate strategy logic in notebook code",
        "do not run workflows directly from Drive archive-pack",
    )

    for path in COLAB_DOCS:
        text = path.read_text(encoding="utf-8")
        for token in expected_tokens:
            assert token in text, f"{path} is missing #488 boundary token: {token}"


def test_m44_notebook_execution_examples_use_current_api_shape_and_profile_vars() -> None:
    expected_tokens = (
        "from src.execution import run_strategy",
        "strategy_result = run_strategy(",
        '"momentum_v1"',
        "start=START",
        "end=END",
        "strategies_config_path=STRATEGIES_CONFIG",
        'STRATEGIES_CONFIG = STRATLAKE_ROOT / "configs" / "strategies.yml"',
        'EVALUATION_CONFIG = STRATLAKE_ROOT / "configs" / "evaluation.yml"',
        "strategy_result.notebook_summary()",
        "strategy_result.load_metrics_json()",
        "strategy_result.load_manifest()",
        "strategy_result.output_keys()",
        'strategy_result.output_path("metrics_json", must_exist=True)',
        "window_runs = []",
        "window_start",
        "window_end",
    )

    for path in COLAB_DOCS:
        text = path.read_text(encoding="utf-8")
        for token in expected_tokens:
            assert token in text, f"{path} is missing #488 execution example token: {token}"
        assert "run_id" in text
        assert "hard-coded run" in text


def test_m44_notebook_execution_examples_preserve_cli_examples_for_restore_and_validation() -> None:
    for path in COLAB_DOCS:
        text = path.read_text(encoding="utf-8")
        assert "!stratlake-session-archive-restore-bootstrap" in text
        assert "!stratlake-notebook-doctor" in text
        assert "!stratlake-validate-marketlake-handoff" in text
        assert "canonical artifact" in text


def test_m44_notebook_execution_cross_docs_reference_new_pattern() -> None:
    for path in CROSS_DOCS:
        text = path.read_text(encoding="utf-8")
        assert "src.execution" in text
        assert "restore" in text.lower()
        assert "CLI" in text or "cli" in text
