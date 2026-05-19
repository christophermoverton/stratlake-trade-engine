from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import pytest

from src.catalog import (
    EvidenceReviewError,
    REQUIRED_REVIEW_PACK_FILES,
    build_catalog,
    build_evidence_review_for_workflow,
    write_evidence_review_pack,
)
from src.contracts import validate_json
from tests.catalog_scale_fixtures import build_catalog_scale_tree, snapshot_tree


CONTRACTS_ROOT = Path(__file__).resolve().parents[1] / "contracts"


def test_writer_creates_complete_pack_and_valid_contract_payloads(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    model = build_evidence_review_for_workflow(".", repo_root=tmp_path, selected_run_id="strategy_000")

    result = write_evidence_review_pack(model, repo_root=tmp_path)
    pack_root = tmp_path / result["output_root"]

    assert set(REQUIRED_REVIEW_PACK_FILES).issubset({path.name for path in pack_root.iterdir()})
    assert (pack_root / "selected_lineage.openlineage.json").exists()
    assert (pack_root / "selected_lineage.prov.json").exists()
    assert not (pack_root / "report.html").exists()

    schema_map = {
        "manifest.json": "review_pack_manifest.schema.json",
        "review_request.json": "review_pack_review_request.schema.json",
        "review_summary.json": "review_pack_review_summary.schema.json",
        "catalog_health_diagnostics.json": "review_pack_catalog_health_diagnostics.schema.json",
        "resolver_resolution.json": "review_pack_resolver_resolution.schema.json",
        "evidence_index.json": "review_pack_evidence_index.schema.json",
        "validation.json": "review_pack_validation.schema.json",
    }
    for filename, schema_name in schema_map.items():
        validate_json(
            json.loads((pack_root / filename).read_text(encoding="utf-8")),
            CONTRACTS_ROOT / schema_name,
        )

    report = (pack_root / "report.md").read_text(encoding="utf-8")
    assert "derived, disposable, rebuildable, non-authoritative, and write-back-forbidden" in report
    assert "Canonical artifacts remain the source of truth." in report
    assert "[strategies/registry.jsonl](strategies/registry.jsonl)" in report


def test_writer_is_deterministic_portable_and_report_matches_model(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    model = build_evidence_review_for_workflow(".", repo_root=tmp_path, selected_run_id="strategy_000")

    first = write_evidence_review_pack(model, repo_root=tmp_path)
    pack_root = tmp_path / first["output_root"]
    first_snapshot = snapshot_tree(pack_root)
    second = write_evidence_review_pack(model, repo_root=tmp_path, overwrite=True)
    second_snapshot = snapshot_tree(pack_root)

    assert first["manifest"] == second["manifest"]
    assert first_snapshot == second_snapshot

    serialized_files = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(pack_root.iterdir())
        if path.suffix in {".json", ".csv", ".md"}
    )
    assert "\\" not in serialized_files
    assert "file://" not in serialized_files
    assert str(tmp_path) not in serialized_files
    assert tmp_path.as_posix() not in serialized_files
    assert "../" not in serialized_files

    summary = json.loads((pack_root / "review_summary.json").read_text(encoding="utf-8"))
    report = (pack_root / "report.md").read_text(encoding="utf-8")
    diagnostics = model["catalog_health_diagnostics"]["summary"]
    assert summary["summary"]["diagnostics_finding_count"] == diagnostics["finding_count"]
    assert f"- Findings: `{diagnostics['finding_count']}`" in report
    assert f"- Warnings: `{diagnostics['counts_by_status']['WARN']}`" in report

    inventory_rows = list(csv.DictReader((pack_root / "artifact_inventory.csv").read_text(encoding="utf-8").splitlines()))
    assert [row["path"] for row in inventory_rows] == sorted(row["path"] for row in inventory_rows)
    assert all(row["path"].startswith(first["output_root"] + "/") for row in inventory_rows)


def test_writer_preserves_canonical_identity_and_html_is_opt_in(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    build_catalog_scale_tree(artifacts_root)
    before_canonical = _canonical_snapshot(artifacts_root)
    before_records = [record.to_dict() for record in build_catalog(artifacts_root, repo_root=tmp_path)]
    model = build_evidence_review_for_workflow(
        artifacts_root,
        repo_root=tmp_path,
        selected_run_id="strategy_000",
    )

    result = write_evidence_review_pack(model, repo_root=tmp_path, include_html=True)
    pack_root = tmp_path / result["output_root"]
    assert (pack_root / "report.html").exists()
    assert _canonical_snapshot(artifacts_root) == before_canonical
    assert [record.to_dict() for record in build_catalog(artifacts_root, repo_root=tmp_path)] == before_records

    shutil.rmtree(pack_root)
    assert [record.to_dict() for record in build_catalog(artifacts_root, repo_root=tmp_path)] == before_records


def test_writer_rejects_non_default_or_existing_output_roots(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    model = build_evidence_review_for_workflow(".", repo_root=tmp_path, selected_run_id="strategy_000")
    write_evidence_review_pack(model, repo_root=tmp_path)

    with pytest.raises(EvidenceReviewError, match="overwrite=True"):
        write_evidence_review_pack(model, repo_root=tmp_path)
    with pytest.raises(EvidenceReviewError, match="deterministic derived namespace"):
        write_evidence_review_pack(
            model,
            repo_root=tmp_path,
            output_root="artifacts/_derived/evidence_review/other_review",
        )


def _canonical_snapshot(artifacts_root: Path) -> dict[str, bytes]:
    return {
        path: payload
        for path, payload in snapshot_tree(artifacts_root).items()
        if not path.startswith("_derived/")
    }
