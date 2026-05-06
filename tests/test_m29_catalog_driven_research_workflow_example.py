from __future__ import annotations

import ast
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "docs" / "examples" / "m29_catalog_driven_research_workflow.py"
DOC_PATH = REPO_ROOT / "docs" / "catalog_notebook_workflows.md"
NOTEBOOK_PATH = (
    REPO_ROOT
    / "docs"
    / "examples"
    / "notebooks"
    / "m29_catalog_driven_research_workflow.ipynb"
)
ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"(?<![A-Za-z])(?:[A-Za-z]:[\\/]|/[A-Za-z]:/)"),
    re.compile(r"(?<![A-Za-z])/(?:Users|home)/"),
    re.compile(r"file://"),
)
FORBIDDEN_EXAMPLE_TOKENS = (
    "subprocess",
    "os.system",
    "!python",
    "run_strategy(",
    "run_alpha(",
    "run_alpha_evaluation(",
    "run_portfolio(",
    "run_pipeline(",
    "run_research_campaign(",
    "run_campaign(",
    "run_benchmark_pack(",
    "write_text",
    "to_csv",
    "to_parquet",
    "mkdir",
    "touch",
    "unlink",
)


def _load_example_module():
    spec = spec_from_file_location(EXAMPLE_PATH.stem, EXAMPLE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_m29_catalog_workflow_example_imports_and_parses() -> None:
    source = EXAMPLE_PATH.read_text(encoding="utf-8")

    ast.parse(source, filename=str(EXAMPLE_PATH))
    module = _load_example_module()

    assert hasattr(module, "main")
    for expected in (
        "build_catalog",
        "query_catalog",
        "CatalogQuery",
        "records_to_rows",
        "records_to_dicts",
        "get_upstream_records",
        "get_downstream_records",
        "validate_catalog",
    ):
        assert expected in source


def test_m29_catalog_workflow_example_runs_empty_repo(tmp_path, monkeypatch, capsys) -> None:
    module = _load_example_module()
    before = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))

    monkeypatch.chdir(tmp_path)
    code = module.main()
    captured = capsys.readouterr()

    after = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    assert code == 0
    assert before == after
    assert "M29 Catalog-Driven Research Workflow" in captured.out
    assert "catalog_records: 0" in captured.out
    assert "completed_strategy_runs: 0 shown" in captured.out
    assert "validation: records=0 artifacts=0 errors=0 warnings=0" in captured.out
    assert "artifact_paths: none available" in captured.out


def test_m29_catalog_workflow_docs_and_examples_are_read_only_and_path_safe() -> None:
    for path in (EXAMPLE_PATH, DOC_PATH, NOTEBOOK_PATH):
        source = path.read_text(encoding="utf-8")
        for pattern in ABSOLUTE_PATH_PATTERNS:
            assert pattern.search(source) is None, f"{path} contains {pattern.pattern!r}"
        for token in FORBIDDEN_EXAMPLE_TOKENS:
            assert token not in source, f"{path} should not contain {token!r}"


def test_m29_catalog_workflow_notebook_is_valid_clean_and_uses_catalog_api() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] >= 5
    assert isinstance(notebook["cells"], list)
    assert notebook["cells"]
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"] if cell["cell_type"] == "code")

    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    for expected in (
        "# M29 Catalog-Driven Research Workflow",
        "## 1. Build the Catalog",
        "## 2. Query Completed Research Runs",
        "## 3. Filter by Metrics and Run Type",
        "## 4. Inspect Lineage",
        "## 5. Validate Catalog Integrity",
        "## 6. Load Artifact Paths for Follow-Up Analysis",
        "## 7. Notes on Read-Only Workflow",
        "from src.catalog import",
        "build_catalog",
        "query_catalog",
        "CatalogQuery",
        "records_to_rows",
        "records_to_dicts",
        "get_upstream_records",
        "get_downstream_records",
        "validate_catalog",
    ):
        assert expected in source
