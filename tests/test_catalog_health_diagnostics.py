from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.catalog import (
    build_catalog,
    build_catalog_health_diagnostics,
    build_evidence_review_for_workflow,
)
from src.contracts import validate_json
from tests.catalog_scale_fixtures import build_catalog_scale_tree, snapshot_tree


CONTRACTS_ROOT = Path(__file__).resolve().parents[1] / "contracts"


def test_diagnostics_are_deterministic_schema_valid_and_integrated(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)

    first = build_evidence_review_for_workflow(".", repo_root=tmp_path, selected_run_id="strategy_000")
    second = build_evidence_review_for_workflow(".", repo_root=tmp_path, selected_run_id="strategy_000")
    diagnostics = first["catalog_health_diagnostics"]

    assert diagnostics == second["catalog_health_diagnostics"]
    assert diagnostics["summary"]["overall_status"] in {"PASS", "WARN"}
    assert diagnostics["findings"] == sorted(diagnostics["findings"], key=lambda item: item["finding_id"])
    assert diagnostics["warnings"] == [
        finding for finding in diagnostics["findings"] if finding["status"] in {"WARN", "FAIL"}
    ]
    validate_json(diagnostics, CONTRACTS_ROOT / "review_pack_catalog_health_diagnostics.schema.json")


def test_diagnostics_cover_pass_warn_fail_and_na_states(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    model = build_evidence_review_for_workflow(".", repo_root=tmp_path, selected_run_id="strategy_000")
    diagnostics = model["catalog_health_diagnostics"]
    by_id = {finding["finding_id"]: finding for finding in diagnostics["findings"]}

    assert by_id["resolver_status:selected_record"]["status"] == "PASS"
    assert by_id["derived_index_validation:review_model"]["status"] == "NA"
    assert by_id["governance_evidence_presence:selected_record"]["status"] == "NA"

    broken = {
        **model,
        "canonicality": {},
        "load_source": {},
        "review_root": "outside/review",
        "resolver_resolution": {
            **model["resolver_resolution"],
            "resolution_status": "unresolved",
            "source_fingerprint": None,
        },
    }
    broken_diagnostics = build_catalog_health_diagnostics(broken)
    broken_by_id = {finding["finding_id"]: finding for finding in broken_diagnostics["findings"]}

    assert broken_by_id["canonicality_envelope:review_model"]["status"] == "WARN"
    assert broken_by_id["canonicality_semantics:review_model"]["status"] == "FAIL"
    assert broken_by_id["load_source_envelope:review_model"]["status"] == "FAIL"
    assert broken_by_id["resolver_status:selected_record"]["status"] == "FAIL"
    assert broken_by_id["review_root_namespace:review_model"]["status"] == "FAIL"


def test_path_diagnostics_flag_unsafe_and_derived_authority_paths(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    model = build_evidence_review_for_workflow(".", repo_root=tmp_path, selected_run_id="strategy_000")
    unsafe = {
        **model,
        "canonicality": {
            **model["canonicality"],
            "authority_paths": [
                r"artifacts\bad\manifest.json",
                "/tmp/manifest.json",
                "file:///tmp/manifest.json",
                "../outside.json",
                "artifacts/_derived/evidence_review/review_x/manifest.json",
            ],
        },
    }

    diagnostics = build_catalog_health_diagnostics(unsafe)
    by_id = {finding["finding_id"]: finding for finding in diagnostics["findings"]}

    assert by_id["portable_paths:review_model"]["status"] == "FAIL"
    assert by_id["derived_authority_leakage:review_model"]["status"] == "FAIL"
    assert "../outside.json" in by_id["portable_paths:review_model"]["paths"]


def test_partial_resolver_findings_preserve_missing_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    build_catalog_scale_tree(tmp_path)
    record = next(record for record in build_catalog(tmp_path, repo_root=tmp_path) if record.run_id == "strategy_000")
    missing_path = "strategies/strategy_000/missing.json"

    from src.catalog.resolver import resolve_canonical_record as real_resolve_canonical_record

    def with_missing_source(*args, **kwargs):
        return real_resolve_canonical_record(
            replace(record, source_files=[*record.source_files, missing_path]),
            **kwargs,
        )

    monkeypatch.setattr("src.catalog.review_pack.resolve_canonical_record", with_missing_source)
    model = build_evidence_review_for_workflow(".", repo_root=tmp_path, selected_run_id="strategy_000")
    by_id = {finding["finding_id"]: finding for finding in model["catalog_health_diagnostics"]["findings"]}

    assert by_id["resolver_status:selected_record"]["status"] == "WARN"
    assert by_id["missing_canonical_sources:selected_record"]["status"] == "WARN"
    assert missing_path in by_id["missing_canonical_sources:selected_record"]["paths"]


def test_diagnostics_are_json_safe_portable_and_read_only(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    before = snapshot_tree(tmp_path)
    model = build_evidence_review_for_workflow(".", repo_root=tmp_path, selected_run_id="strategy_000")
    serialized = json.dumps(model["catalog_health_diagnostics"], sort_keys=True)

    assert "\\" not in serialized
    assert "file://" not in serialized
    assert str(tmp_path) not in serialized
    assert snapshot_tree(tmp_path) == before
