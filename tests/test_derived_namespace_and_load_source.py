from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.catalog import (
    DEFAULT_DERIVED_INDEX_PATH,
    build_catalog,
    build_derived_index,
    build_evidence_view_for_workflow,
    build_lineage_export_for_workflow,
    build_load_source,
    load_catalog_records_with_source,
)
from src.cli.catalog_index import run_cli as run_catalog_index_cli
from tests.catalog_scale_fixtures import build_catalog_scale_tree


def test_default_derived_index_path_uses_m37_namespace(tmp_path: Path, capsys) -> None:
    build_catalog_scale_tree(tmp_path / "artifacts")

    payload = run_catalog_index_cli(
        ["build", "--artifacts-root", "artifacts", "--repo-root", str(tmp_path)]
    )
    capsys.readouterr()

    assert DEFAULT_DERIVED_INDEX_PATH == "artifacts/_derived/catalog_index/catalog_index.sqlite"
    assert (tmp_path / DEFAULT_DERIVED_INDEX_PATH).exists()
    assert payload["canonicality"]["authority_root"] == "artifacts"


@pytest.mark.parametrize(
    "index_path",
    [
        r"C:\Users\christopher\catalog_index.sqlite",
        "/tmp/catalog_index.sqlite",
        "file:///tmp/catalog_index.sqlite",
        "https://example.com/catalog_index.sqlite",
        "s3://bucket/catalog_index.sqlite",
        "../outside/catalog_index.sqlite",
    ],
)
def test_build_load_source_rejects_non_portable_index_paths(index_path: str) -> None:
    with pytest.raises(ValueError, match="Paths must be portable repository-relative paths"):
        build_load_source(loaded_from="derived_index", index_path=index_path)


@pytest.mark.parametrize(
    ("index_path", "expected"),
    [
        (
            "./artifacts/_derived/catalog_index/catalog_index.sqlite",
            "artifacts/_derived/catalog_index/catalog_index.sqlite",
        ),
        (
            r"artifacts\_derived\catalog_index\catalog_index.sqlite",
            "artifacts/_derived/catalog_index/catalog_index.sqlite",
        ),
        ("catalog_index/catalog_index.sqlite", "catalog_index/catalog_index.sqlite"),
    ],
)
def test_build_load_source_normalizes_valid_relative_index_paths(index_path: str, expected: str) -> None:
    payload = build_load_source(loaded_from="derived_index", index_path=index_path)

    assert payload["index_path"] == expected


def test_explicit_legacy_index_path_remains_usable(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    legacy = tmp_path / "catalog_index" / "catalog_index.sqlite"
    build_derived_index(tmp_path, legacy, repo_root=tmp_path)

    result = load_catalog_records_with_source(
        tmp_path,
        repo_root=tmp_path,
        index_path=legacy,
        mode="index",
    )

    assert len(result.records) == 48
    assert result.load_source["loaded_from"] == "derived_index"
    assert result.load_source["index_path"] == "catalog_index/catalog_index.sqlite"


def test_direct_index_and_auto_modes_expose_distinct_load_sources(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    index_path = tmp_path / "_derived" / "catalog_index" / "catalog_index.sqlite"
    build_derived_index(tmp_path, index_path, repo_root=tmp_path)

    direct = load_catalog_records_with_source(tmp_path, repo_root=tmp_path, mode="direct")
    indexed = load_catalog_records_with_source(
        tmp_path, repo_root=tmp_path, index_path=index_path, mode="index"
    )
    auto_index = load_catalog_records_with_source(
        tmp_path, repo_root=tmp_path, index_path=index_path, mode="auto"
    )
    auto_direct = load_catalog_records_with_source(
        tmp_path, repo_root=tmp_path, index_path=tmp_path / "missing.sqlite", mode="auto"
    )

    assert direct.load_source == {
        "schema_version": "load_source.v1",
        "loaded_from": "direct_scan",
        "canonical_source": "artifacts",
        "non_authoritative": False,
        "resolver_hint": "reopen canonical manifests/registries before decision-sensitive use",
        "requested_mode": "direct",
        "resolved_mode": "direct",
        "index_validated": False,
    }
    assert indexed.load_source["loaded_from"] == "derived_index"
    assert indexed.load_source["requested_mode"] == "index"
    assert indexed.load_source["resolved_mode"] == "index"
    assert indexed.load_source["index_validated"] is True
    assert auto_index.load_source["requested_mode"] == "auto"
    assert auto_index.load_source["resolved_mode"] == "index"
    assert auto_direct.load_source["loaded_from"] == "direct_scan"
    assert auto_direct.load_source["requested_mode"] == "auto"
    assert auto_direct.load_source["resolved_mode"] == "direct"
    assert auto_direct.load_source["index_validated"] is False


def test_derived_namespace_is_not_scanned_as_canonical_artifacts(tmp_path: Path) -> None:
    (tmp_path / "_derived" / "evidence" / "view").mkdir(parents=True)
    (tmp_path / "_derived" / "evidence" / "view" / "manifest.json").write_text(
        '{"run_id":"derived_only"}',
        encoding="utf-8",
    )

    assert build_catalog(tmp_path, repo_root=tmp_path) == []


def test_derived_views_include_load_source_and_portable_paths(tmp_path: Path) -> None:
    build_catalog_scale_tree(tmp_path)
    index_path = tmp_path / "_derived" / "catalog_index" / "catalog_index.sqlite"
    build_derived_index(tmp_path, index_path, repo_root=tmp_path)

    evidence = build_evidence_view_for_workflow(
        ".",
        repo_root=tmp_path,
        index_path=index_path,
        index_mode="auto",
        selected_run_id="strategy_000",
    )
    lineage = build_lineage_export_for_workflow(
        ".",
        repo_root=tmp_path,
        index_path=index_path,
        index_mode="auto",
        selected_run_id="strategy_000",
    )
    serialized = json.dumps({"evidence": evidence, "lineage": lineage}, sort_keys=True)

    assert evidence["canonicality"]["derived_class"] == "evidence_view"
    assert evidence["load_source"]["loaded_from"] == "evidence_view"
    assert evidence["load_source"]["resolved_mode"] == "index"
    assert lineage["canonicality"]["derived_class"] == "lineage_export"
    assert lineage["load_source"]["loaded_from"] == "lineage_export"
    assert lineage["load_source"]["non_authoritative"] is True
    assert "\\" not in serialized
    assert "file://" not in serialized
    assert str(tmp_path) not in serialized
