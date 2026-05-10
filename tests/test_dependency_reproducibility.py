from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_FILE = REPO_ROOT / "requirements-dev.lock"
README = REPO_ROOT / "README.md"
WORKFLOWS = [
    REPO_ROOT / ".github" / "workflows" / "ci.yml",
    REPO_ROOT / ".github" / "workflows" / "milestone_validation.yml",
]

FORBIDDEN_LOCAL_PATH_PATTERNS = (
    "C:\\",
    "C:/Users/",
    "/Users/",
    "/home/",
    "file://",
)

EXPECTED_LOCKED_PACKAGES = {
    "build",
    "duckdb",
    "jsonschema",
    "lightgbm",
    "matplotlib",
    "pandas",
    "pyarrow",
    "pytest",
    "python-dotenv",
    "pyyaml",
    "ruff",
    "scikit-learn",
    "scipy",
    "xgboost",
}


def test_dev_constraints_lock_exists_and_uses_lf_newline() -> None:
    raw = LOCK_FILE.read_bytes()

    assert raw
    assert raw.endswith(b"\n")
    assert b"\r\n" not in raw


def test_dev_constraints_lock_is_portable() -> None:
    text = LOCK_FILE.read_text(encoding="utf-8")

    for forbidden in FORBIDDEN_LOCAL_PATH_PATTERNS:
        assert forbidden not in text

    for line in _lock_lines():
        assert not line.startswith("-e ")
        assert "@ file:" not in line
        assert "\\" not in line


def test_dev_constraints_lock_covers_project_and_dev_dependencies() -> None:
    locked_names = {_package_name(line) for line in _lock_lines()}

    assert EXPECTED_LOCKED_PACKAGES <= locked_names


def test_dependency_reproducibility_documentation_describes_refresh_workflow() -> None:
    readme = README.read_text(encoding="utf-8")

    assert "requirements-dev.lock" in readme
    assert "pyproject.toml" in readme
    assert "python -m pip freeze --all --exclude-editable" in readme
    assert 'python -m pip install -e ".[dev]" -c requirements-dev.lock' in readme
    assert "not package publication" in readme


def test_ci_install_commands_use_dev_constraints_lock() -> None:
    for workflow in WORKFLOWS:
        text = workflow.read_text(encoding="utf-8")

        assert 'python -m pip install -e ".[dev]" -c requirements-dev.lock' in text
        assert "requirements-dev.lock" in text


def test_ci_pip_cache_keys_include_dev_constraints_lock() -> None:
    for workflow in WORKFLOWS:
        text = workflow.read_text(encoding="utf-8")

        assert "cache-dependency-path: |" in text
        assert "pyproject.toml" in text
        assert "requirements-dev.lock" in text


def _lock_lines() -> list[str]:
    return [
        line.strip()
        for line in LOCK_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def _package_name(requirement: str) -> str:
    match = re.match(r"^([A-Za-z0-9_.-]+)", requirement)
    assert match is not None, requirement
    return match.group(1).replace("_", "-").lower()
