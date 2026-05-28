from __future__ import annotations

from importlib import metadata
from importlib import resources
from pathlib import Path
import re
import tomllib

from src.cli.init_notebook_workspace import _resolve_resource_root
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


def test_m43_release_version_and_tag_metadata_are_consistent() -> None:
    declared_version = _declared_project_version()
    expected_version = "0.43.2"
    expected_tag = "v0.43.2-session-archive-restore-bootstrap"
    version_pattern = re.compile(
        rf"Package/build version:\s*\n`{re.escape(expected_version)}`",
        re.MULTILINE,
    )

    assert declared_version == expected_version
    for path in (
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs" / "m43_release_notes.md",
        REPO_ROOT / "docs" / "m43_release_validation_checklist.md",
    ):
        text = path.read_text(encoding="utf-8")
        assert version_pattern.search(text)
        assert expected_tag in text


def test_stable_installed_import_smoke() -> None:
    assert portable_path("docs\\manifest.json") == "docs/manifest.json"


def test_m42_entry_points_are_declared() -> None:
    scripts = _load_pyproject()["project"]["scripts"]

    assert scripts["stratlake-init-session"] == "src.cli.init_session:main"
    assert scripts["stratlake-session-export"] == "src.cli.session_export:main"
    assert scripts["stratlake-session-import"] == "src.cli.session_import:main"


def test_m42_notebook_workspace_resources_are_packaged() -> None:
    resource_root = _resolve_resource_root()

    assert resource_root.joinpath("configs").joinpath("session.yml").is_file()
    assert resource_root.joinpath("docs").joinpath("notebook_integration.md").is_file()
    assert resource_root.joinpath("docs").joinpath("colab_project_sessions.md").is_file()

    package_files = resources.files("src.resources.notebook_workspace")
    assert package_files.joinpath("configs").joinpath("session.yml").is_file()
    assert package_files.joinpath("docs").joinpath("colab_project_sessions.md").is_file()


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
