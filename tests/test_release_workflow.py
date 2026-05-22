from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"
README = REPO_ROOT / "README.md"


def test_release_workflow_is_tag_driven_only() -> None:
    text = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    assert "on:" in text
    assert "push:" in text
    assert "tags:" in text
    assert '- "v*"' in text
    assert "pull_request:" not in text
    assert "workflow_dispatch:" not in text


def test_release_workflow_uses_release_permissions_and_concurrency() -> None:
    text = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    assert "permissions:" in text
    assert "contents: write" in text
    assert "concurrency:" in text
    assert "group: ${{ github.workflow }}-${{ github.ref }}" in text
    assert "cancel-in-progress: false" in text


def test_release_workflow_uses_constrained_install_and_cache_policy() -> None:
    text = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    assert "actions/setup-python@v5" in text
    assert "cache: pip" in text
    assert "cache-dependency-path: |" in text
    assert "pyproject.toml" in text
    assert "requirements-dev.lock" in text
    assert 'python -m pip install -e ".[dev]" -c requirements-dev.lock' in text


def test_release_workflow_validates_before_publication() -> None:
    text = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    release_step = text.index("Create GitHub Release")

    for command in (
        "tests/test_packaging_readiness.py",
        "tests/test_dependency_reproducibility.py",
        "tests/test_line_ending_policy.py",
        "tests/test_cross_platform_reproducibility_audit.py",
        "tests/test_portable_paths.py",
        "tests/test_docs_path_portability.py",
        "python -m src.cli.run_docs_path_lint --output artifacts/qa/docs_path_lint_release.json",
        "python -m build",
    ):
        assert command in text
        assert text.index(command) < release_step


def test_release_workflow_supports_optional_wheel_install_smoke_toggle() -> None:
    text = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    assert "STRATLAKE_RUN_WHEEL_INSTALL_SMOKE" in text
    assert "Optional wheel install smoke test" in text
    assert "env.STRATLAKE_RUN_WHEEL_INSTALL_SMOKE == '1'" in text
    assert "tests/test_wheel_install_smoke.py" in text

    release_step = text.index("Create GitHub Release")
    assert text.index("Optional wheel install smoke test") < release_step


def test_release_workflow_uses_deterministic_release_notes_and_github_token() -> None:
    text = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    assert "artifacts/qa/release_notes.md" in text
    assert "GITHUB_REF_NAME" in text
    assert "Package publication to PyPI/TestPyPI is out of scope" in text
    assert "softprops/action-gh-release@v2" in text
    assert "GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}" in text
    assert "gh release" not in text


def test_release_workflow_keeps_package_distributions_as_workflow_artifacts() -> None:
    text = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    assert "Upload package build artifacts" in text
    assert "name: package-build" in text
    assert "path: dist/*" in text

    release_step = text[text.index("Create GitHub Release") :]
    assert "dist/*" not in release_step


def test_release_workflow_documentation_describes_tag_driven_scope() -> None:
    readme = README.read_text(encoding="utf-8")

    assert "Release" in readme
    assert "`v*` tag" in readme
    assert "GITHUB_TOKEN" in readme
    assert "Local `gh auth` is not required" in readme
    assert "python -m build" in readme
    assert "does not publish to PyPI/TestPyPI" in readme
