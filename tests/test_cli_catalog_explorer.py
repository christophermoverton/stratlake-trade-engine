from __future__ import annotations

import json

import src.cli.explore_catalog_evidence as explorer_cli
from src.catalog.explorer import build_evidence_explorer_view, render_evidence_json
from src.catalog.models import CatalogRecord, CatalogValidationStatus
from src.catalog.query import CatalogQuery


def _validation() -> CatalogValidationStatus:
    return CatalogValidationStatus(
        catalog_status="valid",
        marker_status="present",
        manifest_status="missing",
        artifact_status="ok",
        qa_status=None,
        validation_errors=[],
        validation_warnings=[],
    )


def _record(
    run_id: str,
    run_type: str,
    *,
    record_family: str | None = None,
    metadata: dict | None = None,
    robustness_status: str | None = None,
    wfe_status: str | None = None,
    governance_status: str | None = None,
    validation_readiness_present: bool = False,
    release_validation_present: bool = False,
) -> CatalogRecord:
    return CatalogRecord(
        catalog_id=f"catalog_{run_id}",
        run_id=run_id,
        run_type=run_type,
        status="completed",
        artifact_root=f"artifacts/{run_type}/{run_id}",
        source_registry_path=None,
        source_manifest_path=None,
        source_marker_path=None,
        created_at=None,
        timeframe=None,
        start_ts=None,
        end_ts=None,
        strategy_name=None,
        portfolio_name=None,
        allocator_name=None,
        alpha_model_name=None,
        regime_method=None,
        campaign_id=None,
        scenario_id=None,
        metrics_summary=None,
        qa_status=None,
        review_status=None,
        promotion_status=None,
        record_family=record_family,
        robustness_status=robustness_status,
        wfe_status=wfe_status,
        governance_status=governance_status,
        validation_readiness_present=validation_readiness_present,
        release_validation_present=release_validation_present,
        tags=[],
        source_files=[],
        metadata=metadata or {},
        validation=_validation(),
    )


def _records() -> list[CatalogRecord]:
    return [
        _record("strategy_a", "strategy"),
        _record(
            "robustness_a",
            "robustness_bundle",
            record_family="robustness_bundle",
            metadata={"source_run_ids": ["strategy_a"]},
            robustness_status="needs_review",
            wfe_status="weak",
        ),
        _record(
            "governance_a",
            "governance_bundle",
            record_family="governance_bundle",
            metadata={"source_run_ids": ["strategy_a"]},
            governance_status="pass",
        ),
        _record(
            "validation_a",
            "milestone_validation_bundle",
            record_family="milestone_validation_bundle",
            validation_readiness_present=True,
        ),
        _record(
            "release_a",
            "release_validation_artifact",
            record_family="release_validation_artifact",
            release_validation_present=True,
        ),
    ]


def test_cli_json_output_matches_python_rendering(monkeypatch, capsys) -> None:
    records = _records()
    monkeypatch.setattr(explorer_cli, "build_catalog", lambda *args, **kwargs: records)

    code = explorer_cli.main(
        [
            "--record-family",
            "robustness_bundle",
            "--robustness-status",
            "needs_review",
            "--format",
            "json",
        ]
    )
    expected_view = build_evidence_explorer_view(
        records,
        query=CatalogQuery(record_family="robustness_bundle", robustness_status="needs_review"),
        repo_root=".",
    )

    assert code == 0
    assert capsys.readouterr().out == render_evidence_json(expected_view)


def test_cli_markdown_selected_run_includes_lineage(monkeypatch, capsys) -> None:
    monkeypatch.setattr(explorer_cli, "build_catalog", lambda *args, **kwargs: _records())

    code = explorer_cli.main(["--run-id", "strategy_a", "--format", "markdown"])

    assert code == 0
    out = capsys.readouterr().out
    assert "strategy_a" in out
    assert "robustness_a" in out
    assert "run_to_robustness_evidence" in out


def test_cli_table_output(monkeypatch, capsys) -> None:
    monkeypatch.setattr(explorer_cli, "build_catalog", lambda *args, **kwargs: _records())

    code = explorer_cli.main(["--governance-status", "pass", "--format", "table"])

    assert code == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0].startswith("section\trun_id\trun_type")
    assert "governance_a" in out


def test_cli_output_writes_derived_review_file(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(explorer_cli, "build_catalog", lambda *args, **kwargs: _records())
    output_path = tmp_path / "derived" / "evidence.md"

    code = explorer_cli.main(["--run-id", "strategy_a", "--output", str(output_path)])

    assert code == 0
    assert capsys.readouterr().out == ""
    text = output_path.read_text(encoding="utf-8")
    assert text.startswith("# M35 Catalog Evidence Explorer")
    assert "run_to_robustness_evidence" in text


def test_cli_empty_result_json(monkeypatch, capsys) -> None:
    monkeypatch.setattr(explorer_cli, "build_catalog", lambda *args, **kwargs: _records())

    code = explorer_cli.main(["--record-family", "missing", "--format", "json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["total_matching_records"] == 0
