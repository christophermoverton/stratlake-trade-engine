from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MILESTONE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "milestone_validation.yml"


def test_milestone_validation_keeps_manual_dispatch() -> None:
    text = MILESTONE_WORKFLOW.read_text(encoding="utf-8")

    assert "workflow_dispatch:" in text


def test_milestone_validation_covers_current_milestone_branch_pattern() -> None:
    text = MILESTONE_WORKFLOW.read_text(encoding="utf-8")

    assert '- "feature/m*"' in text
    assert "startsWith(github.head_ref, 'feature/m')" in text


def test_milestone_validation_preserves_legacy_branch_patterns() -> None:
    text = MILESTONE_WORKFLOW.read_text(encoding="utf-8")

    assert '- "milestone/**"' in text
    assert '- "m22/**"' in text
    assert "startsWith(github.head_ref, 'milestone/')" in text
    assert "startsWith(github.head_ref, 'm22/')" in text
