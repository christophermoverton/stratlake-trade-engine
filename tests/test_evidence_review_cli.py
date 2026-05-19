from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from src.catalog import build_evidence_review_for_workflow, write_evidence_review_pack
from src.cli.build_evidence_review import main, run_cli
from tests.catalog_scale_fixtures import build_catalog_scale_tree


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_cli_build_validate_and_html_use_shared_review_pack_surfaces(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifacts_root = tmp_path / "artifacts"
    build_catalog_scale_tree(artifacts_root)
    api_model = build_evidence_review_for_workflow(
        artifacts_root,
        repo_root=tmp_path,
        selected_run_id="strategy_000",
        review_id="cli_review",
    )
    api_result = write_evidence_review_pack(api_model, repo_root=tmp_path, include_html=True)
    api_pack_root = tmp_path / api_result["output_root"]
    api_manifest = (api_pack_root / "manifest.json").read_text(encoding="utf-8")

    cli_payload = run_cli(
        [
            "build",
            "--artifacts-root",
            "artifacts",
            "--repo-root",
            str(tmp_path),
            "--selected-run-id",
            "strategy_000",
            "--review-id",
            "cli_review",
            "--include-html",
            "--overwrite",
        ]
    )
    output = capsys.readouterr().out
    pack_root = tmp_path / cli_payload["output_root"]
    assert cli_payload["review_id"] == "cli_review"
    assert cli_payload["output_root"] == "artifacts/_derived/evidence_review/cli_review"
    assert (pack_root / "report.html").exists()
    assert (pack_root / "manifest.json").read_text(encoding="utf-8") == api_manifest
    _assert_portable_output(output, tmp_path)

    validation = run_cli(
        [
            "validate",
            "--repo-root",
            str(tmp_path),
            "--review-id",
            "cli_review",
        ]
    )
    output = capsys.readouterr().out
    assert validation["status"] in {"pass", "warn"}
    assert validation["missing_files"] == []
    assert validation["invalid_files"] == []
    _assert_portable_output(output, tmp_path)


def test_cli_validate_fails_for_missing_or_invalid_pack(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing_code = main(
        [
            "validate",
            "--repo-root",
            str(tmp_path),
            "--review-id",
            "missing_review",
        ]
    )
    assert missing_code == 2
    capsys.readouterr()

    pack_root = tmp_path / "artifacts" / "_derived" / "evidence_review" / "invalid_review"
    pack_root.mkdir(parents=True)
    (pack_root / "manifest.json").write_text("{}\n", encoding="utf-8")
    invalid_code = main(
        [
            "validate",
            "--repo-root",
            str(tmp_path),
            "--review-id",
            "invalid_review",
        ]
    )
    assert invalid_code == 2


def test_cli_example_script_runs_with_relative_output() -> None:
    completed = subprocess.run(
        [sys.executable, "docs/examples/m38_static_evidence_review_pack_example.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "review_id: example_review" in completed.stdout
    assert (
        "output_root: artifacts/_derived/evidence_review/example_review"
        in completed.stdout
    )
    assert "validation_status:" in completed.stdout
    _assert_portable_output(completed.stdout, REPO_ROOT)


def _assert_portable_output(text: str, root: Path) -> None:
    assert "\\" not in text
    assert "file://" not in text
    assert "../" not in text
    assert str(root) not in text
    assert root.as_posix() not in text
