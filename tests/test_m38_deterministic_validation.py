from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
import sqlite3

import pytest

from src.catalog import (
    DerivedIndexError,
    build_catalog,
    build_catalog_health_diagnostics,
    build_derived_index,
    build_evidence_review_for_workflow,
    load_catalog_records,
    validate_evidence_review_pack,
    write_evidence_review_pack,
)
from src.catalog.resolver import CanonicalRecordResolution
from src.cli.build_evidence_review import main as evidence_review_cli_main
from src.cli.build_evidence_review import run_cli as run_evidence_review_cli
from src.contracts import validate_json
from tests.catalog_scale_fixtures import build_catalog_scale_tree, snapshot_tree


CONTRACTS_ROOT = Path(__file__).resolve().parents[1] / "contracts"


def test_m38_repeated_model_pack_and_cli_generation_are_stable(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifacts_root = tmp_path / "artifacts"
    build_catalog_scale_tree(artifacts_root)
    before = _canonical_snapshot(artifacts_root)

    first_model = build_evidence_review_for_workflow(
        artifacts_root,
        repo_root=tmp_path,
        selected_run_id="strategy_000",
        review_id="m38_validation",
    )
    second_model = build_evidence_review_for_workflow(
        artifacts_root,
        repo_root=tmp_path,
        selected_run_id="strategy_000",
        review_id="m38_validation",
    )
    assert first_model == second_model
    _assert_portable_payload(first_model, tmp_path)
    assert _canonical_snapshot(artifacts_root) == before

    first_write = write_evidence_review_pack(first_model, repo_root=tmp_path, include_html=True)
    pack_root = tmp_path / first_write["output_root"]
    first_files = _stable_pack_payloads(pack_root)
    second_write = write_evidence_review_pack(second_model, repo_root=tmp_path, include_html=True, overwrite=True)
    second_files = _stable_pack_payloads(pack_root)

    assert first_write["manifest"] == second_write["manifest"]
    assert first_files == second_files
    _assert_portable_text("\n".join(second_files.values()), tmp_path)
    assert _canonical_snapshot(artifacts_root) == before

    cli_payload = run_evidence_review_cli(
        [
            "build",
            "--artifacts-root",
            "artifacts",
            "--repo-root",
            str(tmp_path),
            "--selected-run-id",
            "strategy_000",
            "--review-id",
            "m38_validation",
            "--include-html",
            "--overwrite",
        ]
    )
    cli_output = capsys.readouterr().out
    assert cli_payload["review_id"] == first_write["review_id"]
    assert _stable_pack_payloads(pack_root) == second_files
    _assert_portable_text(cli_output, tmp_path)

    validation = validate_evidence_review_pack(first_write["output_root"], repo_root=tmp_path)
    assert validation["status"] in {"pass", "warn"}
    assert validation["missing_files"] == []
    assert validation["invalid_files"] == []
    assert _canonical_snapshot(artifacts_root) == before


def test_m38_generated_json_contracts_and_report_parity(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    build_catalog_scale_tree(artifacts_root)
    model = build_evidence_review_for_workflow(
        artifacts_root,
        repo_root=tmp_path,
        selected_run_id="strategy_000",
        review_id="m38_contracts",
    )
    result = write_evidence_review_pack(model, repo_root=tmp_path, include_html=True)
    pack_root = tmp_path / result["output_root"]

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
        validate_json(_read_json(pack_root / filename), CONTRACTS_ROOT / schema_name)

    summary = _read_json(pack_root / "review_summary.json")
    diagnostics = _read_json(pack_root / "catalog_health_diagnostics.json")
    validation = _read_json(pack_root / "validation.json")
    report = (pack_root / "report.md").read_text(encoding="utf-8")
    html = (pack_root / "report.html").read_text(encoding="utf-8")

    assert summary["summary"]["diagnostics_finding_count"] == diagnostics["summary"]["finding_count"]
    assert summary["summary"]["diagnostics_overall_status"] == diagnostics["summary"]["overall_status"]
    assert f"- Findings: `{diagnostics['summary']['finding_count']}`" in report
    assert f"- Overall status: `{diagnostics['summary']['overall_status']}`" in report
    assert validation["status"] in {"pass", "warn", "fail"}
    assert "<pre>" in html
    assert "<script" not in html.lower()
    assert "This evidence review pack is derived" in html
    assert "content" not in json.dumps(_read_json(pack_root / "resolver_resolution.json"), sort_keys=True)


def test_m38_review_packs_are_disposable_and_excluded_from_canonical_scans(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    build_catalog_scale_tree(artifacts_root)
    before_records = _record_payloads(build_catalog(artifacts_root, repo_root=tmp_path))
    before_snapshot = _canonical_snapshot(artifacts_root)
    model = build_evidence_review_for_workflow(
        artifacts_root,
        repo_root=tmp_path,
        selected_run_id="strategy_000",
        review_id="disposable_review",
    )
    result = write_evidence_review_pack(model, repo_root=tmp_path)

    assert _record_payloads(build_catalog(artifacts_root, repo_root=tmp_path)) == before_records
    assert _canonical_snapshot(artifacts_root) == before_snapshot
    assert "disposable_review" not in {
        record["run_id"] for record in _record_payloads(build_catalog(artifacts_root, repo_root=tmp_path))
    }

    shutil.rmtree(tmp_path / result["output_root"])
    assert _record_payloads(build_catalog(artifacts_root, repo_root=tmp_path)) == before_records
    assert _canonical_snapshot(artifacts_root) == before_snapshot


def test_m38_direct_index_auto_identity_parity_and_stale_index_failure(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    build_catalog_scale_tree(artifacts_root)
    index_path = artifacts_root / "_derived" / "catalog_index" / "catalog_index.sqlite"
    build_derived_index(artifacts_root, index_path, repo_root=tmp_path)

    direct = build_evidence_review_for_workflow(
        artifacts_root,
        repo_root=tmp_path,
        selected_run_id="strategy_000",
        index_mode="direct",
        review_id="direct_review",
    )
    indexed = build_evidence_review_for_workflow(
        artifacts_root,
        repo_root=tmp_path,
        index_path=index_path,
        selected_run_id="strategy_000",
        index_mode="index",
        review_id="index_review",
    )
    auto = build_evidence_review_for_workflow(
        artifacts_root,
        repo_root=tmp_path,
        index_path=index_path,
        selected_run_id="strategy_000",
        index_mode="auto",
        review_id="auto_review",
    )

    assert direct["selected_record"] == indexed["selected_record"] == auto["selected_record"]
    assert direct["related_records"] == indexed["related_records"] == auto["related_records"]
    assert direct["lineage_summary"] == indexed["lineage_summary"] == auto["lineage_summary"]
    assert direct["canonical_sources"] == indexed["canonical_sources"] == auto["canonical_sources"]
    assert direct["catalog_health_diagnostics"]["summary"]["overall_status"] in {"PASS", "WARN"}
    assert indexed["catalog_health_diagnostics"]["summary"]["overall_status"] in {"PASS", "WARN"}
    assert auto["catalog_health_diagnostics"]["summary"]["overall_status"] in {"PASS", "WARN"}

    (artifacts_root / "strategies" / "strategy_new").mkdir(parents=True)
    (artifacts_root / "strategies" / "strategy_new" / "_SUCCESS.json").write_text(
        '{"run_id":"strategy_new","status":"completed"}',
        encoding="utf-8",
    )
    with pytest.raises(DerivedIndexError, match="stale"):
        build_evidence_review_for_workflow(
            artifacts_root,
            repo_root=tmp_path,
            index_path=index_path,
            selected_run_id="strategy_000",
            index_mode="auto",
        )

    with sqlite3.connect(index_path) as connection:
        connection.execute(
            "UPDATE metadata SET value_json = ? WHERE key = 'schema_version'",
            (json.dumps(999),),
        )
        connection.commit()
    with pytest.raises(DerivedIndexError, match="schema is incompatible"):
        load_catalog_records(artifacts_root, repo_root=tmp_path, index_path=index_path, mode="index")


def test_m38_resolver_diagnostics_edge_states_and_cli_failure_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifacts_root = tmp_path / "artifacts"
    build_catalog_scale_tree(artifacts_root)
    record = next(record for record in build_catalog(artifacts_root, repo_root=tmp_path) if record.run_id == "strategy_000")
    missing_path = "artifacts/strategies/strategy_000/missing.json"

    from src.catalog.resolver import resolve_canonical_record as real_resolve_canonical_record

    def partial(*args, **kwargs):
        return real_resolve_canonical_record(
            replace(record, source_files=[*record.source_files, missing_path]),
            **kwargs,
        )

    monkeypatch.setattr("src.catalog.review_pack.resolve_canonical_record", partial)
    partial_model = build_evidence_review_for_workflow(
        artifacts_root,
        repo_root=tmp_path,
        selected_run_id="strategy_000",
        review_id="partial_review",
    )
    partial_findings = {
        finding["finding_id"]: finding
        for finding in partial_model["catalog_health_diagnostics"]["findings"]
    }
    assert partial_model["resolver_resolution"]["resolution_status"] == "partial"
    assert partial_findings["resolver_status:selected_record"]["status"] == "WARN"
    assert partial_findings["missing_canonical_sources:selected_record"]["status"] == "WARN"
    _assert_portable_payload(partial_model, tmp_path)

    def unresolved(*args, **kwargs) -> CanonicalRecordResolution:
        return CanonicalRecordResolution(
            record=record,
            source_paths=[],
            resolved_sources=[],
            missing_sources=[],
            source_fingerprint=None,
            resolution_status="unresolved",
            canonicality_status="not_applicable",
            load_source={},
            warnings=["no_declared_sources"],
        )

    monkeypatch.setattr("src.catalog.review_pack.resolve_canonical_record", unresolved)
    unresolved_model = build_evidence_review_for_workflow(
        artifacts_root,
        repo_root=tmp_path,
        selected_run_id="strategy_000",
        review_id="unresolved_review",
    )
    unresolved_findings = {
        finding["finding_id"]: finding
        for finding in unresolved_model["catalog_health_diagnostics"]["findings"]
    }
    assert unresolved_model["resolver_resolution"]["source_fingerprint"] is None
    assert unresolved_findings["resolver_status:selected_record"]["status"] == "FAIL"
    assert unresolved_findings["source_fingerprint_present:selected_record"]["status"] == "WARN"

    unsafe = {
        **unresolved_model,
        "canonicality": {
            **unresolved_model["canonicality"],
            "authority_paths": ["file:///tmp/leak.json", "../outside.json", "artifacts\\bad\\manifest.json"],
        },
    }
    unsafe_diagnostics = build_catalog_health_diagnostics(unsafe)
    unsafe_findings = {finding["finding_id"]: finding for finding in unsafe_diagnostics["findings"]}
    assert unsafe_findings["portable_paths:review_model"]["status"] == "FAIL"

    exit_code = evidence_review_cli_main(
        [
            "validate",
            "--repo-root",
            str(tmp_path),
            "--review-id",
            "missing_review",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    _assert_portable_text(captured.out + captured.err, tmp_path)


def _stable_pack_payloads(pack_root: Path) -> dict[str, str]:
    return {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(pack_root.iterdir())
        if path.suffix in {".json", ".csv", ".md", ".html"}
    }


def _record_payloads(records: list) -> list[dict]:
    return [record.to_dict() for record in records]


def _canonical_snapshot(artifacts_root: Path) -> dict[str, bytes]:
    return {
        path: payload
        for path, payload in snapshot_tree(artifacts_root).items()
        if not path.startswith("_derived/")
    }


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_portable_payload(payload: object, root: Path) -> None:
    _assert_portable_text(json.dumps(payload, sort_keys=True), root)


def _assert_portable_text(text: str, root: Path) -> None:
    assert "\\" not in text
    assert "file://" not in text
    assert "../" not in text
    assert str(root) not in text
    assert root.as_posix() not in text
    assert "C:" not in text
