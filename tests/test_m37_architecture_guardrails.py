from __future__ import annotations

import ast
from pathlib import Path
import shutil

from src.catalog import build_catalog, build_derived_index
from tests.catalog_scale_fixtures import build_catalog_scale_tree, snapshot_tree


REPO_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_DERIVED_IMPORT_PREFIXES = (
    "src.catalog.derived_index",
    "src.catalog.lineage_export",
    "src.catalog.explorer",
    "src.catalog.review_pack",
    "src.catalog.workflows",
    "src.cli.build_evidence_review",
)
FORBIDDEN_CATALOG_FACADE_SYMBOLS = frozenset(
    {
        "DEFAULT_DERIVED_INDEX_PATH",
        "build_derived_index",
        "validate_derived_index",
        "load_catalog_records_with_source",
        "export_lineage",
        "validate_lineage_export",
        "build_evidence_explorer_view",
        "render_evidence_json",
        "render_evidence_markdown",
        "render_evidence_table",
        "build_lineage_export_for_workflow",
        "build_evidence_view_for_workflow",
        "build_load_source",
        "derive_view_load_source",
        "build_review_pack_metadata",
        "build_catalog_health_diagnostics",
        "build_evidence_review_for_workflow",
        "review_pack_root",
        "validate_evidence_review_pack",
        "write_evidence_review_pack",
    }
)
ALLOWED_RESOLVER_FACADE_SYMBOLS = frozenset(
    {
        "CanonicalRecordResolution",
        "ResolvedSource",
        "resolve_canonical_record",
        "resolve_canonical_sources",
        "resolve_canonical_record_by_id",
    }
)
DECISION_AUTHORITY_PATHS = (
    "src/research/experiment_tracker.py",
    "src/research/registry.py",
    "src/research/promotion.py",
    "src/research/governance/writer.py",
    "src/research/governance/loader.py",
    "src/research/governance/aggregator.py",
    "src/research/governance/validator.py",
    "src/execution/regime_promotion_gates.py",
    "src/validation/milestone_bundle.py",
    "src/cli/run_milestone_validation.py",
    "src/cli/run_promotion_governance_report.py",
    "src/cli/run_regime_promotion_gates.py",
)


def test_decision_authority_modules_do_not_import_derived_read_models() -> None:
    offenders: list[str] = []
    for relative_path in DECISION_AUTHORITY_PATHS:
        path = REPO_ROOT / relative_path
        for imported_module, imported_symbol in _forbidden_derived_imports_for(path):
            if imported_symbol is None:
                offenders.append(f"{relative_path}: {imported_module}")
            else:
                offenders.append(f"{relative_path}: {imported_module}.{imported_symbol}")

    assert offenders == [], (
        "Decision-authority modules must reopen canonical artifacts through resolver-first APIs, "
        "not import derived read models: "
        + ", ".join(offenders)
    )


def test_resolver_imports_remain_allowed_for_decision_authority_paths() -> None:
    assert "src.catalog.resolver" not in FORBIDDEN_DERIVED_IMPORT_PREFIXES
    assert ALLOWED_RESOLVER_FACADE_SYMBOLS.isdisjoint(FORBIDDEN_CATALOG_FACADE_SYMBOLS)


def test_facade_imports_of_derived_helpers_are_detected() -> None:
    assert _forbidden_derived_imports_from_source("from src.catalog import build_derived_index") == [
        ("src.catalog", "build_derived_index")
    ]
    assert _forbidden_derived_imports_from_source("from src.catalog import export_lineage") == [
        ("src.catalog", "export_lineage")
    ]
    assert _forbidden_derived_imports_from_source(
        "from src.catalog import build_evidence_explorer_view"
    ) == [("src.catalog", "build_evidence_explorer_view")]


def test_resolver_facade_imports_are_allowed() -> None:
    assert _forbidden_derived_imports_from_source(
        "from src.catalog import resolve_canonical_record"
    ) == []


def test_direct_derived_module_imports_are_still_detected() -> None:
    assert _forbidden_derived_imports_from_source(
        "from src.catalog.derived_index import build_derived_index"
    ) == [("src.catalog.derived_index", None)]
    assert _forbidden_derived_imports_from_source("import src.catalog.derived_index") == [
        ("src.catalog.derived_index", None)
    ]


def test_derived_namespace_is_excluded_from_canonical_scans(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    build_catalog_scale_tree(artifacts_root)
    before = _record_payloads(build_catalog(artifacts_root, repo_root=tmp_path))

    derived_root = artifacts_root / "_derived" / "evidence" / "derived_only"
    derived_root.mkdir(parents=True)
    (derived_root / "manifest.json").write_text('{"run_id":"derived_only"}\n', encoding="utf-8")
    (derived_root / "_SUCCESS.json").write_text('{"status":"completed"}\n', encoding="utf-8")

    after = _record_payloads(build_catalog(artifacts_root, repo_root=tmp_path))

    assert after == before
    assert "derived_only" not in {record["run_id"] for record in after}


def test_direct_scan_identity_ignores_derived_index_creation_and_deletion(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    build_catalog_scale_tree(artifacts_root)
    baseline = _record_payloads(build_catalog(artifacts_root, repo_root=tmp_path))
    index_path = artifacts_root / "_derived" / "catalog_index" / "catalog_index.sqlite"

    build_derived_index(artifacts_root, index_path, repo_root=tmp_path)
    with_index = _record_payloads(build_catalog(artifacts_root, repo_root=tmp_path))
    shutil.rmtree(artifacts_root / "_derived")
    after_delete = _record_payloads(build_catalog(artifacts_root, repo_root=tmp_path))

    assert with_index == baseline
    assert after_delete == baseline


def test_rebuilding_derived_index_does_not_mutate_canonical_sources(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    build_catalog_scale_tree(artifacts_root)
    before = snapshot_tree(artifacts_root)
    index_path = artifacts_root / "_derived" / "catalog_index" / "catalog_index.sqlite"

    first = build_derived_index(artifacts_root, index_path, repo_root=tmp_path)
    canonical_after_first = _canonical_snapshot(artifacts_root)
    index_path.unlink()
    second = build_derived_index(artifacts_root, index_path, repo_root=tmp_path)
    canonical_after_second = _canonical_snapshot(artifacts_root)

    assert first == second
    assert canonical_after_first == before
    assert canonical_after_second == before


def _forbidden_derived_imports_for(path: Path) -> list[tuple[str, str | None]]:
    return _forbidden_derived_imports_from_tree(
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    )


def _forbidden_derived_imports_from_source(source: str) -> list[tuple[str, str | None]]:
    return _forbidden_derived_imports_from_tree(ast.parse(source))


def _forbidden_derived_imports_from_tree(tree: ast.AST) -> list[tuple[str, str | None]]:
    imports: list[tuple[str, str | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(
                (alias.name, None)
                for alias in node.names
                if alias.name.startswith(FORBIDDEN_DERIVED_IMPORT_PREFIXES)
            )
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module.startswith(FORBIDDEN_DERIVED_IMPORT_PREFIXES):
                imports.append((node.module, None))
            elif node.module == "src.catalog":
                imports.extend(
                    (node.module, alias.name)
                    for alias in node.names
                    if alias.name in FORBIDDEN_CATALOG_FACADE_SYMBOLS
                )
    return imports


def _record_payloads(records: list) -> list[dict]:
    return [record.to_dict() for record in records]


def _canonical_snapshot(artifacts_root: Path) -> dict[str, bytes]:
    return {
        path: payload
        for path, payload in snapshot_tree(artifacts_root).items()
        if not path.startswith("_derived/")
    }
