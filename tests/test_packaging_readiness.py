from __future__ import annotations

from importlib import metadata
from pathlib import Path
import tomllib

from src.artifacts.safety import portable_path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"


def test_pyproject_declares_pep517_build_system() -> None:
    pyproject = _load_pyproject()

    assert pyproject["build-system"]["build-backend"] == "setuptools.build_meta"
    requirements = pyproject["build-system"]["requires"]
    assert "wheel" in requirements
    assert any(requirement.startswith("setuptools>=") for requirement in requirements)


def test_pyproject_declares_explicit_package_discovery() -> None:
    pyproject = _load_pyproject()

    package_find = pyproject["tool"]["setuptools"]["packages"]["find"]
    assert package_find["where"] == ["."]
    assert package_find["include"] == ["src*", "cli*"]


def test_installed_project_metadata_is_available() -> None:
    project_metadata = metadata.metadata("stratlake-trade-engine")

    assert project_metadata["Name"] == "stratlake-trade-engine"
    assert project_metadata["Version"] == "0.1.0"


def test_stable_installed_import_smoke() -> None:
    assert portable_path("docs\\manifest.json") == "docs/manifest.json"


def _load_pyproject() -> dict[str, object]:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
