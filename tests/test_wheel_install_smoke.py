from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_WHEEL_SMOKE_ENV_VAR = "STRATLAKE_RUN_WHEEL_INSTALL_SMOKE"


@pytest.mark.skipif(
    os.environ.get(RUN_WHEEL_SMOKE_ENV_VAR) != "1",
    reason=(
        "Wheel install smoke test is opt-in. "
        f"Set {RUN_WHEEL_SMOKE_ENV_VAR}=1 to run."
    ),
)
def test_built_wheel_installs_and_imports_from_site_packages(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)

    _run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--outdir",
            str(dist_dir),
        ],
        cwd=REPO_ROOT,
    )

    wheel_paths = sorted(dist_dir.glob("*.whl"))
    assert len(wheel_paths) == 1

    venv_dir = tmp_path / "wheel-smoke-venv"
    _run([sys.executable, "-m", "venv", str(venv_dir)], cwd=REPO_ROOT)

    venv_python = _venv_python(venv_dir)
    _run([str(venv_python), "-m", "pip", "install", "--no-deps", str(wheel_paths[0])], cwd=REPO_ROOT)

    smoke_cwd = tmp_path / "smoke-cwd"
    smoke_cwd.mkdir(parents=True, exist_ok=True)
    completed = _run(
        [
            str(venv_python),
            "-c",
            (
                "import importlib.metadata as m, json, src, cli;"
                "d = m.distribution('stratlake-trade-engine');"
                "print(json.dumps({"
                "'version': m.version('stratlake-trade-engine'),"
                "'src_path': src.__file__,"
                "'cli_path': cli.__file__,"
                "'dist_src': str(d.locate_file('src/__init__.py')),"
                "'dist_cli': str(d.locate_file('cli/__init__.py'))"
                "}))"
            ),
        ],
        cwd=smoke_cwd,
    )

    payload = json.loads(completed.stdout.strip())
    src_path = payload["src_path"].replace("\\", "/")
    cli_path = payload["cli_path"].replace("\\", "/")

    assert payload["version"]
    assert "site-packages" in src_path
    assert "site-packages" in cli_path
    assert payload["src_path"] == payload["dist_src"]
    assert payload["cli_path"] == payload["dist_cli"]


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
