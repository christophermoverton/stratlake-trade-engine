from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GUIDE_PATH = REPO_ROOT / "docs" / "colab_persistence_guide.md"
COLAB_DOCS = (
    REPO_ROOT / "docs" / "colab_project_sessions.md",
    REPO_ROOT / "src" / "resources" / "notebook_workspace" / "docs" / "colab_project_sessions.md",
)


def test_fintech_restore_to_local_handoff_guide_tokens() -> None:
    text = GUIDE_PATH.read_text(encoding="utf-8")

    expected_tokens = (
        "Fintech Restore-to-StratLake Handoff",
        "fintech-backup-data validate",
        "fintech-backup-data inspect",
        "fintech-backup-data restore",
        "--backup-pack-dir",
        "--restore-root",
        "--overwrite-policy fail",
        "/content/fintech-market-ingestion-demo/data/curated",
        "MARKETLAKE_ROOT",
        "stratlake-notebook-doctor",
        "stratlake-validate-marketlake-handoff",
    )
    for token in expected_tokens:
        assert token in text, f"guide missing #490 token: {token}"


def test_fintech_handoff_guide_boundaries_and_dataset_notes() -> None:
    text = GUIDE_PATH.read_text(encoding="utf-8")

    expected_tokens = (
        "daily curated bars",
        "1-minute curated bars",
        "sharded backup packs",
        "non-canonical transport artifacts",
        "local restored partitioned Parquet remains the working dataset",
        "Do not point StratLake at a backup-pack directory",
        "mounted filesystem",
        "no hidden sync",
        "no Google API calls or OAuth behavior",
        "no automatic persistence or restore-on-import",
    )
    for token in expected_tokens:
        assert token in text, f"guide missing boundary/dataset token: {token}"

    assert "/content/drive/MyDrive" in text
    assert "/content/gdrive" not in text


def test_fintech_handoff_guide_distinguishes_session_vs_backup_pack() -> None:
    text = GUIDE_PATH.read_text(encoding="utf-8")

    expected_tokens = (
        "Session Save/Restore vs Archive Backup Packs (Fintech)",
        "fintech-save-session",
        "fintech-restore-session",
        "config continuity",
        "large curated OHLCV datasets",
        "restore-to-local-first handoff into StratLake",
    )
    for token in expected_tokens:
        assert token in text, f"guide missing session/backup distinction token: {token}"


def test_colab_docs_include_fintech_handoff_section_and_drive_path_shape() -> None:
    for path in COLAB_DOCS:
        text = path.read_text(encoding="utf-8")
        assert "### Fintech Restore-To-Local Handoff" in text
        assert "fintech-backup-data validate" in text
        assert "fintech-backup-data inspect" in text
        assert "fintech-backup-data restore" in text
        assert "--backup-pack-dir" in text
        assert "--restore-root" in text
        assert "--overwrite-policy fail" in text
        assert "/content/fintech-market-ingestion-demo/data/curated" in text
        assert 'MARKETLAKE_ROOT = FINTECH_ROOT / "data" / "curated"' in text
        assert "stratlake-notebook-doctor" in text
        assert "stratlake-validate-marketlake-handoff" in text
        assert "/content/drive/MyDrive/" in text
        assert "/content/gdrive/" not in text
        assert "backup-pack directories themselves" in text
