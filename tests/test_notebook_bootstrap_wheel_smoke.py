from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_WHEEL_NOTEBOOK_SMOKE_ENV_VAR = "STRATLAKE_RUN_NOTEBOOK_BOOTSTRAP_WHEEL_SMOKE"


@pytest.mark.skipif(
    os.environ.get(RUN_WHEEL_NOTEBOOK_SMOKE_ENV_VAR) != "1",
    reason=(
        "Notebook bootstrap wheel smoke test is opt-in. "
        f"Set {RUN_WHEEL_NOTEBOOK_SMOKE_ENV_VAR}=1 to run."
    ),
)
def test_wheel_install_supports_stratlake_init_notebook(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)

    _run([sys.executable, "-m", "build", "--wheel", "--outdir", str(dist_dir)], cwd=REPO_ROOT)

    wheel_paths = sorted(dist_dir.glob("*.whl"))
    assert len(wheel_paths) == 1

    venv_dir = tmp_path / "wheel-notebook-smoke-venv"
    _run([sys.executable, "-m", "venv", str(venv_dir)], cwd=REPO_ROOT)

    venv_python = _venv_python(venv_dir)
    _run(
        [str(venv_python), "-m", "pip", "install", "--force-reinstall", "--no-deps", str(wheel_paths[0])],
        cwd=REPO_ROOT,
    )

    workspace_root = tmp_path / "stratlake-notebook-smoke"
    script_name = "stratlake-init-notebook.exe" if os.name == "nt" else "stratlake-init-notebook"
    script_path = _venv_scripts_dir(venv_dir) / script_name

    _run([str(script_path), "--root", str(workspace_root)], cwd=tmp_path)

    for relative in (
        "notebooks",
        "configs",
        "docs",
        "contracts",
        "artifacts",
    ):
        assert (workspace_root / relative).is_dir()

    assert (workspace_root / "configs" / "features.yml").is_file()
    assert (workspace_root / "configs" / "profiles" / "notebook.yml").is_file()
    assert (workspace_root / "docs" / "notebook_integration.md").is_file()
    assert (workspace_root / "docs" / "examples" / "notebook_execution_api_examples.py").is_file()


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _venv_scripts_dir(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts"
    return venv_dir / "bin"


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
