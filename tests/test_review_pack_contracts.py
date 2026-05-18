from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.catalog import (
    OPTIONAL_REVIEW_PACK_FILES,
    REQUIRED_REVIEW_PACK_FILES,
    build_review_pack_metadata,
    review_pack_root,
)
from src.contracts import validate_json


CONTRACTS_ROOT = Path(__file__).resolve().parents[1] / "contracts"


def test_review_pack_root_is_m37_derived_namespace_and_portable() -> None:
    assert review_pack_root("review_001") == "artifacts/_derived/evidence_review/review_001"

    with pytest.raises(ValueError, match="Paths must be portable repository-relative paths"):
        review_pack_root("../outside")

    with pytest.raises(ValueError, match="single portable path segments"):
        review_pack_root("nested/review")


def test_review_pack_metadata_is_non_authoritative_and_portable() -> None:
    metadata = build_review_pack_metadata(
        authority_paths=["artifacts\\strategies\\run_001\\manifest.json"],
        fingerprint_payload={"review_id": "review_001"},
    )
    serialized = json.dumps(metadata, sort_keys=True)

    assert metadata["canonicality"]["derived_class"] == "review_pack"
    assert metadata["canonicality"]["non_authoritative"] is True
    assert metadata["canonicality"]["write_back_forbidden"] is True
    assert metadata["load_source"]["schema_version"] == "load_source.v1"
    assert metadata["load_source"]["loaded_from"] == "review_pack"
    assert metadata["load_source"]["non_authoritative"] is True
    assert metadata["canonicality"]["authority_paths"] == [
        "artifacts/strategies/run_001/manifest.json"
    ]
    assert "\\" not in serialized
    assert "file://" not in serialized


def test_review_pack_schema_contracts_accept_minimal_payloads() -> None:
    metadata = build_review_pack_metadata(
        authority_paths=["artifacts/strategies/run_001/manifest.json"],
        fingerprint_payload={"review_id": "review_001"},
    )
    payloads = {
        "review_pack_manifest.schema.json": {
            "schema_version": "review_pack_manifest.v1",
            "review_id": "review_001",
            "artifact_family": "evidence_review_pack",
            "output_root": review_pack_root("review_001"),
            "required_files": list(REQUIRED_REVIEW_PACK_FILES),
            "optional_files": list(OPTIONAL_REVIEW_PACK_FILES),
            **metadata,
        },
        "review_pack_review_request.schema.json": {
            "schema_version": "review_request.v1",
            "review_id": "review_001",
            "selected_run_id": "run_001",
            **metadata,
        },
        "review_pack_review_summary.schema.json": {
            "schema_version": "review_summary.v1",
            "review_id": "review_001",
            "selected_run_id": "run_001",
            "summary": {},
            **metadata,
        },
        "review_pack_catalog_health_diagnostics.schema.json": {
            "schema_version": "catalog_health_diagnostics.v1",
            "review_id": "review_001",
            "summary": {
                "overall_status": "PASS",
                "finding_count": 0,
                "counts_by_status": {"PASS": 0, "WARN": 0, "FAIL": 0, "NA": 0},
                "counts_by_category": {},
                "counts_by_scope": {},
                "selected_catalog_id": None,
                "selected_run_id": None,
            },
            "findings": [],
            "warnings": [],
            **metadata,
        },
        "review_pack_resolver_resolution.schema.json": {
            "schema_version": "resolver_resolution.v1",
            "review_id": "review_001",
            "selected_run_id": "run_001",
            "resolution_status": "resolved",
            **metadata,
        },
        "review_pack_evidence_index.schema.json": {
            "schema_version": "evidence_index.v1",
            "review_id": "review_001",
            "entries": [
                {
                    "path": "artifacts/strategies/run_001/manifest.json",
                    "kind": "canonical_manifest",
                }
            ],
            **metadata,
        },
        "review_pack_validation.schema.json": {
            "schema_version": "review_pack_validation.v1",
            "review_id": "review_001",
            "status": "pass",
            "checks": [],
            **metadata,
        },
    }

    for schema_name, payload in payloads.items():
        validate_json(payload, CONTRACTS_ROOT / schema_name)


def test_review_pack_manifest_rejects_non_portable_paths() -> None:
    metadata = build_review_pack_metadata(
        authority_paths=["artifacts/strategies/run_001/manifest.json"],
        fingerprint_payload={"review_id": "review_001"},
    )
    payload = {
        "schema_version": "review_pack_manifest.v1",
        "review_id": "review_001",
        "artifact_family": "evidence_review_pack",
        "output_root": r"artifacts\_derived\evidence_review\review_001",
        "required_files": list(REQUIRED_REVIEW_PACK_FILES),
        "optional_files": list(OPTIONAL_REVIEW_PACK_FILES),
        **metadata,
    }

    with pytest.raises(ValueError, match="output_root"):
        validate_json(payload, CONTRACTS_ROOT / "review_pack_manifest.schema.json")


def test_review_pack_evidence_index_rejects_non_portable_entry_paths() -> None:
    metadata = build_review_pack_metadata(
        authority_paths=["artifacts/strategies/run_001/manifest.json"],
        fingerprint_payload={"review_id": "review_001"},
    )
    payload = {
        "schema_version": "evidence_index.v1",
        "review_id": "review_001",
        "entries": [{"path": "/tmp/manifest.json", "kind": "canonical_manifest"}],
        **metadata,
    }

    with pytest.raises(ValueError, match="entries\\[0\\].path"):
        validate_json(payload, CONTRACTS_ROOT / "review_pack_evidence_index.schema.json")
