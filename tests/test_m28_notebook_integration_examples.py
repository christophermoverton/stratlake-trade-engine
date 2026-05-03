from __future__ import annotations

import ast
import json
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_EXAMPLE_DIR = REPO_ROOT / "docs" / "examples" / "notebooks"
EXAMPLE_FILES = (
    NOTEBOOK_EXAMPLE_DIR / "m28_strategy_execution_api.py",
    NOTEBOOK_EXAMPLE_DIR / "m28_benchmark_pack_execution_api.py",
    NOTEBOOK_EXAMPLE_DIR / "m28_regime_research_inspection.py",
)
IPYNB_EXAMPLE_FILES = (
    NOTEBOOK_EXAMPLE_DIR / "m28_benchmark_pack_execution_api.ipynb",
)
ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"(?<![A-Za-z])(?:[A-Za-z]:[\\/]|/[A-Za-z]:/)"),
    re.compile(r"(?<![A-Za-z])/(?:Users|home)/"),
    re.compile(r"file://"),
)
FORBIDDEN_WORKFLOW_REIMPLEMENTATION_TOKENS = (
    "subprocess",
    "os.system",
    "!python",
    "PipelineRunner(",
    "PipelineSpec.from_yaml",
    "run_strategy_experiment(",
    "run_resolved_config(",
    "write_portfolio_artifacts(",
    "run_regime_policy_stress_tests as _run",
)


def test_notebook_integration_doc_exists_and_links_core_guides() -> None:
    path = REPO_ROOT / "docs" / "notebook_integration.md"
    source = path.read_text(encoding="utf-8")

    assert "StratLake has one execution system with multiple entry points" in source
    assert "docs/notebook_execution_api.md" in source
    assert "docs/concurrency_and_idempotency.md" in source
    assert "docs/pipeline_integration.md" in source
    assert "M28.5" in source
    assert "M28.6" in source
    assert "run_benchmark_pack" in source
    assert "run_pipeline" in source
    assert "run_research_campaign" in source
    assert "run_regime_benchmark_pack" in source


def test_script_backed_notebook_examples_exist_and_parse() -> None:
    assert NOTEBOOK_EXAMPLE_DIR.is_dir()
    for path in EXAMPLE_FILES:
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path))
        assert "from __future__ import annotations" in source


def test_ipynb_notebook_examples_are_valid_clean_and_use_execution_api() -> None:
    for path in IPYNB_EXAMPLE_FILES:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert notebook["nbformat_minor"] >= 5
        assert isinstance(notebook["cells"], list)
        assert notebook["cells"]
        assert all(cell.get("outputs", []) == [] for cell in notebook["cells"] if cell["cell_type"] == "code")

        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        assert "from src.execution import run_benchmark_pack" in source
        assert "run_benchmark_pack(" in source
        assert "output_root=" in source
        assert "artifacts/notebooks/m28_benchmark_pack_execution_api/attempt_001" in source
        assert "notebook_summary(" in source
        assert "load_manifest(" in source
        assert "load_summary_json(" in source
        assert "load_output_json(" in source
        assert "output_path(" in source
        assert "subprocess" not in source
        assert "!python" not in source


def test_notebook_examples_use_execution_surfaces_and_artifact_inspection() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in EXAMPLE_FILES)

    for expected in (
        "from src.execution import run_strategy",
        "from src.execution import run_benchmark_pack",
        "from src.execution.regime_benchmark import run_regime_benchmark_pack",
        "from src.execution.regime_policy_stress_tests import run_regime_policy_stress_tests",
    ):
        assert expected in combined

    for helper in (
        "notebook_summary(",
        "output_keys(",
        "output_path(",
        "load_manifest(",
        "load_metrics_json(",
        "load_summary_json(",
        "load_output_json(",
    ):
        assert helper in combined

    assert "artifacts/notebooks/" in combined
    assert "attempt_001" in combined
    assert "configs/benchmark_packs/m22_scale_repro.yml" in combined


def test_notebook_examples_do_not_reimplement_workflow_logic() -> None:
    for path in EXAMPLE_FILES:
        source = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_WORKFLOW_REIMPLEMENTATION_TOKENS:
            assert forbidden not in source, f"{path} should not contain {forbidden!r}"


def test_notebook_docs_and_examples_do_not_contain_absolute_paths() -> None:
    paths = [REPO_ROOT / "docs" / "notebook_integration.md", *EXAMPLE_FILES, *IPYNB_EXAMPLE_FILES]
    for path in paths:
        source = path.read_text(encoding="utf-8")
        for pattern in ABSOLUTE_PATH_PATTERNS:
            assert pattern.search(source) is None, f"{path} contains {pattern.pattern!r}"


def test_existing_ipynb_examples_are_valid_and_path_safe() -> None:
    for path in (REPO_ROOT / "docs" / "examples").glob("*.ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        source = json.dumps(notebook, sort_keys=True)
        for pattern in ABSOLUTE_PATH_PATTERNS:
            assert pattern.search(source) is None, f"{path} contains {pattern.pattern!r}"
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "code":
                assert cell.get("outputs", []) == []


def test_readme_links_m28_notebook_integration_guidance() -> None:
    source = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "[docs/notebook_integration.md](docs/notebook_integration.md)" in source
    assert "[docs/notebook_execution_api.md](docs/notebook_execution_api.md)" in source
    assert "[docs/concurrency_and_idempotency.md](docs/concurrency_and_idempotency.md)" in source
    assert "[docs/pipeline_integration.md](docs/pipeline_integration.md)" in source
