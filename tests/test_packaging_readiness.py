from __future__ import annotations

from importlib import metadata
from pathlib import Path
import re
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
    declared_version = _declared_project_version()
    project_metadata = metadata.metadata("stratlake-trade-engine")

    assert project_metadata["Name"] == "stratlake-trade-engine"
    assert project_metadata["Version"] == declared_version


def test_package_version_is_not_milestone_tag_formatted() -> None:
    declared_version = _declared_project_version()

    # Milestone tags are repository-release identifiers (for example v0.36.0-foo),
    # while package versions stay PEP 440 distribution metadata.
    assert not re.match(r"^v\d+\.\d+\.\d+(?:-|$)", declared_version)


def test_stable_installed_import_smoke() -> None:
    assert portable_path("docs\\manifest.json") == "docs/manifest.json"


def _load_pyproject() -> dict[str, object]:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _declared_project_version() -> str:
    pyproject = _load_pyproject()
    project_table = pyproject["project"]
    assert isinstance(project_table, dict)

    version = project_table["version"]
    assert isinstance(version, str)
    assert version
    return version
