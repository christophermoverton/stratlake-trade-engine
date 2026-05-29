from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
COLAB_DOCS = (
    REPO_ROOT / "docs" / "colab_project_sessions.md",
    REPO_ROOT / "src" / "resources" / "notebook_workspace" / "docs" / "colab_project_sessions.md",
)


def test_colab_docs_define_unified_session_profile_variables() -> None:
    expected = (
        'FINTECH_ROOT = Path("/content/fintech-market-ingestion-demo").resolve()',
        'STRATLAKE_ROOT = Path("/content/stratlake-workspace").resolve()',
        'MARKETLAKE_ROOT = FINTECH_ROOT / "data" / "curated"',
        'DRIVE_ROOT = Path("/content/drive/MyDrive/stratlake-fintech-colab").resolve()',
        'START = "2024-10-01"',
        'END = "2025-04-15"',
        'UNIVERSE_CONFIG = STRATLAKE_ROOT / "configs" / "universe.yml"',
        'PATHS_CONFIG = STRATLAKE_ROOT / "configs" / "paths.yml"',
    )
    for path in COLAB_DOCS:
        source = path.read_text(encoding="utf-8")
        for token in expected:
            assert token in source, f"{path} is missing profile token: {token}"


def test_colab_docs_reuse_profile_values_for_init_session_notebook_configs() -> None:
    for path in COLAB_DOCS:
        source = path.read_text(encoding="utf-8")
        assert '--root "{STRATLAKE_ROOT}"' in source
        assert '--marketlake-root "{MARKETLAKE_ROOT}"' in source
        assert '--drive-root "{DRIVE_ROOT}"' in source
        assert "--notebook-configs" in source


def test_colab_docs_use_normalized_drive_mount_path_examples() -> None:
    for path in COLAB_DOCS:
        source = path.read_text(encoding="utf-8")
        assert "/content/drive/MyDrive/" in source
        assert "/content/gdrive/" not in source
        assert "drive/MyDrive/" not in source.replace("/content/drive/MyDrive/", "")


def test_colab_docs_explain_drive_and_session_metadata_boundaries() -> None:
    expected_phrases = (
        "not canonical working data",
        "The profile is an explicit notebook convenience layer only.",
    )
    for path in COLAB_DOCS:
        source = path.read_text(encoding="utf-8")
        for phrase in expected_phrases:
            assert phrase in source, f"{path} is missing boundary phrase: {phrase}"
        assert "diagnostic" in source
        assert "session metadata" in source
        assert "Drive archive-pack directories" in source
        assert "canonical MarketLake roots" in source
