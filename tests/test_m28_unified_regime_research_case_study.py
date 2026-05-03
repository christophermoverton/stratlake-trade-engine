from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import re
import runpy
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_STUDY_SCRIPT = REPO_ROOT / "docs" / "examples" / "m28_unified_regime_research_case_study.py"
NOTEBOOK_WRAPPER_SCRIPT = (
    REPO_ROOT / "docs" / "examples" / "notebooks" / "m28_unified_regime_research_case_study.py"
)
NOTEBOOK_IPYNB = (
    REPO_ROOT / "docs" / "examples" / "notebooks" / "m28_unified_regime_research_case_study.ipynb"
)
PIPELINE_EXAMPLES = {
    "prefect": REPO_ROOT / "docs" / "examples" / "pipelines" / "m28_prefect_regime_research_flow.py",
    "airflow": REPO_ROOT / "docs" / "examples" / "pipelines" / "m28_airflow_regime_research_dag.py",
    "dagster": REPO_ROOT / "docs" / "examples" / "pipelines" / "m28_dagster_regime_research_job.py",
}
CAPSTONE_CALLABLES = {
    "prefect": "run_m28_capstone_prefect_example",
    "airflow": "run_m28_capstone_airflow_example",
    "dagster": "run_m28_dagster_capstone_example",
}
CAPSTONE_BUILDERS = {
    "prefect": "build_m28_capstone_prefect_output_root",
    "airflow": "build_m28_capstone_airflow_output_root",
    "dagster": "build_m28_dagster_capstone_output_root",
}
ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"(?<![A-Za-z])(?:[A-Za-z]:[\\/]|/[A-Za-z]:/)"),
    re.compile(r"(?<![A-Za-z])/(?:Users|home)/"),
    re.compile(r"file://"),
)
REQUIRED_CASE_STUDY_OUTPUT_KEYS = (
    "manifest_json",
    "summary_json",
    "validation_report_json",
    "cross_layer_comparison_json",
    "artifact_index_json",
)
REQUIRED_SUMMARY_KEYS = (
    "case_study_name",
    "milestone",
    "workflow_stages",
    "execution_surfaces_called",
    "regime_benchmark",
    "cross_layer_validation",
    "limitations",
)
REQUIRED_VALIDATION_REPORT_KEYS = (
    "run_type",
    "schema_version",
    "status",
    "regime_benchmark_checks",
    "cross_layer_validation_checks",
    "case_study_artifact_schema",
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


# ---------------------------------------------------------------------------
# 1. Canonical case study script — existence and parse
# ---------------------------------------------------------------------------


def test_canonical_case_study_script_exists_and_is_import_safe() -> None:
    assert CASE_STUDY_SCRIPT.exists(), f"Canonical case study script not found: {CASE_STUDY_SCRIPT}"
    source = CASE_STUDY_SCRIPT.read_text(encoding="utf-8")
    ast.parse(source, filename=str(CASE_STUDY_SCRIPT))
    assert "from __future__ import annotations" in source


def test_canonical_case_study_exposes_public_callable() -> None:
    namespace = runpy.run_path(str(CASE_STUDY_SCRIPT))
    assert callable(namespace["run_m28_unified_regime_research_case_study"])
    assert callable(namespace["main"])
    assert callable(namespace["parse_args"])
    assert callable(namespace["_dry_run_structural_check"])


def test_canonical_case_study_exposes_module_level_constants() -> None:
    namespace = runpy.run_path(str(CASE_STUDY_SCRIPT))
    assert namespace["CASE_STUDY_NAME"] == "m28_unified_regime_research_case_study"
    assert namespace["MILESTONE"] == "M28.6"
    case_study_outputs = namespace["CASE_STUDY_OUTPUTS"]
    assert isinstance(case_study_outputs, dict)
    for key in REQUIRED_CASE_STUDY_OUTPUT_KEYS:
        assert key in case_study_outputs, f"CASE_STUDY_OUTPUTS missing key: {key}"
    workflow_stages = namespace["WORKFLOW_STAGES"]
    assert isinstance(workflow_stages, list)
    assert len(workflow_stages) >= 2
    execution_surfaces = namespace["EXECUTION_SURFACES_CALLED"]
    assert isinstance(execution_surfaces, list)
    assert any("run_regime_benchmark_pack" in s for s in execution_surfaces)
    assert any("run_cross_layer_validation" in s for s in execution_surfaces)


def test_canonical_case_study_uses_relative_output_root() -> None:
    namespace = runpy.run_path(str(CASE_STUDY_SCRIPT))
    default_root: Path = namespace["DEFAULT_OUTPUT_ROOT"]
    assert isinstance(default_root, Path)
    # Must be under docs/examples/output/
    assert "docs" in default_root.parts
    assert "examples" in default_root.parts
    assert "output" in default_root.parts
    assert "m28_unified_regime_research_case_study" in default_root.parts


def test_canonical_case_study_calls_existing_execution_surfaces() -> None:
    source = CASE_STUDY_SCRIPT.read_text(encoding="utf-8")
    assert "from src.execution.regime_benchmark import run_regime_benchmark_pack" in source
    assert "run_regime_benchmark_pack(" in source
    assert "from src.execution import run_cross_layer_validation" in source
    assert "run_cross_layer_validation(" in source


def test_canonical_case_study_does_not_reimplement_workflow_logic() -> None:
    source = CASE_STUDY_SCRIPT.read_text(encoding="utf-8")
    for forbidden in FORBIDDEN_WORKFLOW_REIMPLEMENTATION_TOKENS:
        assert forbidden not in source, (
            f"Canonical case study should not contain {forbidden!r}"
        )


def test_canonical_case_study_uses_relative_paths_only() -> None:
    source = CASE_STUDY_SCRIPT.read_text(encoding="utf-8")
    for pattern in ABSOLUTE_PATH_PATTERNS:
        match = pattern.search(source)
        assert match is None, (
            f"Canonical case study contains absolute path pattern: {match.group()!r}"
        )


def test_canonical_case_study_regime_config_exists() -> None:
    namespace = runpy.run_path(str(CASE_STUDY_SCRIPT))
    config_relative = namespace["REGIME_BENCHMARK_CONFIG"]
    config_path = REPO_ROOT / config_relative
    assert config_path.exists(), f"Regime benchmark config not found: {config_relative}"


# ---------------------------------------------------------------------------
# 2. Notebook wrapper
# ---------------------------------------------------------------------------


def test_notebook_wrapper_script_exists_and_parses() -> None:
    assert NOTEBOOK_WRAPPER_SCRIPT.exists()
    source = NOTEBOOK_WRAPPER_SCRIPT.read_text(encoding="utf-8")
    ast.parse(source, filename=str(NOTEBOOK_WRAPPER_SCRIPT))
    assert "from __future__ import annotations" in source


def test_notebook_wrapper_exposes_public_callable() -> None:
    namespace = runpy.run_path(str(NOTEBOOK_WRAPPER_SCRIPT))
    assert callable(namespace["run_m28_case_study_notebook_cell"])
    assert callable(namespace["inspect_case_study_artifacts"])


def test_notebook_wrapper_delegates_to_canonical_callable() -> None:
    source = NOTEBOOK_WRAPPER_SCRIPT.read_text(encoding="utf-8")
    assert "from docs.examples.m28_unified_regime_research_case_study import" in source
    assert "run_m28_unified_regime_research_case_study" in source
    assert "run_m28_case_study_notebook_cell" in source


def test_notebook_wrapper_does_not_reimplement_workflow_logic() -> None:
    source = NOTEBOOK_WRAPPER_SCRIPT.read_text(encoding="utf-8")
    for forbidden in FORBIDDEN_WORKFLOW_REIMPLEMENTATION_TOKENS:
        assert forbidden not in source, (
            f"Notebook wrapper should not contain {forbidden!r}"
        )


def test_notebook_wrapper_uses_relative_paths_only() -> None:
    source = NOTEBOOK_WRAPPER_SCRIPT.read_text(encoding="utf-8")
    for pattern in ABSOLUTE_PATH_PATTERNS:
        match = pattern.search(source)
        assert match is None, (
            f"Notebook wrapper contains absolute path pattern: {match.group()!r}"
        )


# ---------------------------------------------------------------------------
# 3. IPYNB notebook
# ---------------------------------------------------------------------------


def test_ipynb_notebook_exists_and_is_valid() -> None:
    assert NOTEBOOK_IPYNB.exists()
    notebook = json.loads(NOTEBOOK_IPYNB.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] >= 5
    assert isinstance(notebook["cells"], list)
    assert notebook["cells"]


def test_ipynb_notebook_has_no_outputs() -> None:
    notebook = json.loads(NOTEBOOK_IPYNB.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("outputs", []) == [], (
                f"Notebook cell {cell.get('id', '?')} should have empty outputs"
            )


def test_ipynb_notebook_calls_canonical_case_study() -> None:
    notebook = json.loads(NOTEBOOK_IPYNB.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "run_m28_unified_regime_research_case_study" in source
    assert "output_root=" in source
    assert "artifacts/notebooks/m28_unified_regime_research_case_study" in source


def test_ipynb_notebook_does_not_contain_absolute_paths() -> None:
    source = NOTEBOOK_IPYNB.read_text(encoding="utf-8")
    for pattern in ABSOLUTE_PATH_PATTERNS:
        match = pattern.search(source)
        assert match is None, (
            f"IPYNB contains absolute path pattern: {match.group()!r}"
        )


# ---------------------------------------------------------------------------
# 4. Pipeline wrappers — capstone callables
# ---------------------------------------------------------------------------


def test_capstone_pipeline_wrappers_are_import_safe_without_optional_dependencies() -> None:
    for orchestrator, path in PIPELINE_EXAMPLES.items():
        namespace = runpy.run_path(str(path))

        # Capstone callable must be present.
        capstone_callable = CAPSTONE_CALLABLES[orchestrator]
        assert callable(namespace[capstone_callable]), (
            f"{orchestrator}: {capstone_callable} must be callable"
        )
        capstone_builder = CAPSTONE_BUILDERS[orchestrator]
        assert callable(namespace[capstone_builder]), (
            f"{orchestrator}: {capstone_builder} must be callable"
        )

        # If the optional package is not installed, scheduler objects should be None.
        package_map = {"prefect": "prefect", "airflow": "airflow", "dagster": "dagster"}
        if importlib.util.find_spec(package_map[orchestrator]) is None:
            for global_name in ("flow", "task", "DAG", "PythonOperator", "job", "op"):
                if global_name in namespace:
                    assert namespace[global_name] is None or callable(namespace[global_name])


def test_capstone_pipeline_callables_delegate_to_canonical_function(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_case_study(output_root, **kwargs: Any) -> dict[str, Any]:
        calls.append({"output_root": output_root})
        return {
            "case_study_name": "m28_unified_regime_research_case_study",
            "milestone": "M28.6",
            "output_root": str(output_root),
            "summary": {"workflow_stages": [], "execution_surfaces_called": [], "cross_layer_validation": {}},
            "validation_report": {"status": "passed"},
            "regime_benchmark_result": {},
            "manifest_path": str(output_root),
            "summary_path": str(output_root),
            "validation_report_path": str(output_root),
            "cross_layer_comparison_path": str(output_root),
            "artifact_index_path": str(output_root),
        }

    monkeypatch.setattr(
        "docs.examples.m28_unified_regime_research_case_study.run_m28_unified_regime_research_case_study",
        fake_case_study,
    )

    for orchestrator, path in PIPELINE_EXAMPLES.items():
        namespace = runpy.run_path(str(path))
        capstone_fn = namespace[CAPSTONE_CALLABLES[orchestrator]]
        result = capstone_fn(f"{orchestrator}-capstone-run", 1)
        assert isinstance(result, dict)
        assert result.get("case_study_name") == "m28_unified_regime_research_case_study"

    assert len(calls) == 3
    for orchestrator, call in zip(PIPELINE_EXAMPLES, calls, strict=True):
        output_root = Path(call["output_root"])
        assert not output_root.is_absolute()
        assert f"orchestrator_examples/{orchestrator}/" in output_root.as_posix()
        assert "capstone_attempt_" in output_root.as_posix()


def test_capstone_pipeline_builders_return_relative_paths() -> None:
    for orchestrator, path in PIPELINE_EXAMPLES.items():
        namespace = runpy.run_path(str(path))
        builder = namespace[CAPSTONE_BUILDERS[orchestrator]]
        built = builder(f"{orchestrator}-test-run", 3)
        assert isinstance(built, Path)
        assert not built.is_absolute()
        assert f"orchestrator_examples/{orchestrator}/" in built.as_posix()
        assert "capstone_attempt_3" in built.as_posix()


def test_capstone_pipeline_wrappers_use_relative_paths_only() -> None:
    forbidden_fragments = ("C:/", "C:\\", "file://", "/Users/", "/home/")
    for path in PIPELINE_EXAMPLES.values():
        source = path.read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            assert fragment not in source, (
                f"{path.name} contains forbidden absolute path fragment: {fragment!r}"
            )


def test_capstone_pipeline_wrappers_are_thin_wrappers() -> None:
    for path in PIPELINE_EXAMPLES.values():
        source = path.read_text(encoding="utf-8")
        assert "run_m28_unified_regime_research_case_study" in source
        assert "subprocess" not in source
        assert "shell=True" not in source


# ---------------------------------------------------------------------------
# 5. Output artifact schema
# ---------------------------------------------------------------------------


def test_case_study_output_schema_keys_are_defined() -> None:
    namespace = runpy.run_path(str(CASE_STUDY_SCRIPT))
    case_study_outputs: dict[str, str] = namespace["CASE_STUDY_OUTPUTS"]
    for key in REQUIRED_CASE_STUDY_OUTPUT_KEYS:
        assert key in case_study_outputs, f"Missing CASE_STUDY_OUTPUTS key: {key}"
    for key, filename in case_study_outputs.items():
        assert isinstance(filename, str)
        assert filename.endswith(".json"), f"Expected .json filename for {key}"
        assert "/" not in filename, f"CASE_STUDY_OUTPUTS filename should be a bare filename: {filename}"


def test_case_study_summary_schema_keys_are_documented() -> None:
    source = CASE_STUDY_SCRIPT.read_text(encoding="utf-8")
    for key in REQUIRED_SUMMARY_KEYS:
        assert f'"{key}"' in source or f"'{key}'" in source, (
            f"summary key {key!r} not found in case study script"
        )


def test_case_study_validation_report_schema_keys_are_documented() -> None:
    source = CASE_STUDY_SCRIPT.read_text(encoding="utf-8")
    for key in REQUIRED_VALIDATION_REPORT_KEYS:
        assert f'"{key}"' in source or f"'{key}'" in source, (
            f"validation report key {key!r} not found in case study script"
        )


# ---------------------------------------------------------------------------
# 6. Documentation page
# ---------------------------------------------------------------------------


def test_case_study_documentation_page_exists_and_links_m28_components() -> None:
    doc_path = REPO_ROOT / "docs" / "milestone_28_unified_regime_research_case_study.md"
    assert doc_path.exists()
    source = doc_path.read_text(encoding="utf-8")
    assert "M28.6" in source
    assert "one execution system" in source.lower() or "one execution system" in source
    assert "run_m28_unified_regime_research_case_study" in source
    assert "cross_layer_validation.md" in source
    assert "notebook_integration.md" in source
    assert "pipeline_integration.md" in source
    assert "concurrency_and_idempotency.md" in source
    assert "research" in source.lower()
    assert "live trading" in source.lower()


def test_case_study_documentation_does_not_contain_absolute_paths() -> None:
    doc_path = REPO_ROOT / "docs" / "milestone_28_unified_regime_research_case_study.md"
    source = doc_path.read_text(encoding="utf-8")
    for pattern in ABSOLUTE_PATH_PATTERNS:
        match = pattern.search(source)
        assert match is None, (
            f"Documentation contains absolute path pattern: {match.group()!r}"
        )


# ---------------------------------------------------------------------------
# 7. Cross-layer validation M28.6 scenario registration
# ---------------------------------------------------------------------------


def test_m28_6_cross_layer_scenario_is_registered_in_all_scenarios() -> None:
    from src.validation.cross_layer import ALL_SCENARIOS, M28_6_SCENARIO, DEFAULT_SCENARIOS

    assert M28_6_SCENARIO in ALL_SCENARIOS
    assert M28_6_SCENARIO not in DEFAULT_SCENARIOS, (
        "M28.6 scenario must remain optional and not appear in DEFAULT_SCENARIOS"
    )


def test_cross_layer_validation_rejects_unknown_scenarios() -> None:
    import pytest
    from src.validation.cross_layer import run_cross_layer_validation

    with pytest.raises(ValueError, match="Unknown cross-layer validation scenario"):
        run_cross_layer_validation(scenarios=["unknown_scenario_xyz"])


def test_cross_layer_validation_all_scenarios_constant_is_superset_of_defaults() -> None:
    from src.validation.cross_layer import ALL_SCENARIOS, DEFAULT_SCENARIOS

    assert set(DEFAULT_SCENARIOS).issubset(set(ALL_SCENARIOS))
    assert len(ALL_SCENARIOS) > len(DEFAULT_SCENARIOS)


# ---------------------------------------------------------------------------
# 8. Dry-run structural check
# ---------------------------------------------------------------------------


def test_dry_run_structural_check_passes() -> None:
    """Validate structure and imports without executing any benchmark workflow."""
    namespace = runpy.run_path(str(CASE_STUDY_SCRIPT))
    dry_run_fn = namespace["_dry_run_structural_check"]
    # Should not raise.
    dry_run_fn()


def test_parse_args_supports_dry_run_flag() -> None:
    namespace = runpy.run_path(str(CASE_STUDY_SCRIPT))
    parse_args = namespace["parse_args"]
    args = parse_args(["--dry-run"])
    assert args.dry_run is True


def test_parse_args_supports_skip_cross_layer_validation_flag() -> None:
    namespace = runpy.run_path(str(CASE_STUDY_SCRIPT))
    parse_args = namespace["parse_args"]
    args = parse_args(["--skip-cross-layer-validation"])
    assert args.skip_cross_layer_validation is True


# ---------------------------------------------------------------------------
# 9. Smoke test (calls the canonical callable with a temp output root)
# ---------------------------------------------------------------------------


def test_smoke_run_m28_unified_regime_research_case_study(tmp_path, monkeypatch) -> None:
    """Lightweight smoke test: runs the canonical case study with a temp output root.

    This test verifies the full artifact assembly path without requiring a
    real benchmark pack execution. It monkeypatches the execution surfaces.
    """
    fake_regime_result = _make_fake_regime_result(tmp_path)
    fake_cl_result = _make_fake_cl_result(tmp_path)

    monkeypatch.setattr(
        "src.execution.regime_benchmark.run_regime_benchmark_pack",
        lambda *a, **kw: fake_regime_result,
    )
    monkeypatch.setattr(
        "src.execution.run_cross_layer_validation",
        lambda **kw: fake_cl_result,
    )

    # Import and call after monkeypatching.
    namespace = runpy.run_path(str(CASE_STUDY_SCRIPT))
    run_fn = namespace["run_m28_unified_regime_research_case_study"]

    output_root = tmp_path / "case_study_output"
    result = run_fn(output_root, include_cross_layer_validation=True)

    # Check returned dict structure.
    assert result["case_study_name"] == "m28_unified_regime_research_case_study"
    assert result["milestone"] == "M28.6"
    assert "output_root" in result
    assert "manifest_path" in result
    assert "summary_path" in result
    assert "validation_report_path" in result
    assert "cross_layer_comparison_path" in result
    assert "artifact_index_path" in result
    assert "summary" in result
    assert "validation_report" in result

    summary = result["summary"]
    for key in REQUIRED_SUMMARY_KEYS:
        assert key in summary, f"summary missing key: {key}"

    validation_report = result["validation_report"]
    for key in REQUIRED_VALIDATION_REPORT_KEYS:
        assert key in validation_report, f"validation_report missing key: {key}"

    # Artifact files should be present on disk.
    for key in REQUIRED_CASE_STUDY_OUTPUT_KEYS:
        artifact_path = output_root / namespace["CASE_STUDY_OUTPUTS"][key]
        assert artifact_path.exists(), f"Expected artifact {key} not found: {artifact_path}"
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)

    # Relative paths only in returned dict.
    output_root_str = str(output_root)
    for path_key in ("manifest_path", "summary_path", "validation_report_path"):
        path_value = result[path_key]
        assert output_root_str not in path_value, (
            f"{path_key} should be relative but contains absolute root: {path_value}"
        )


def test_smoke_run_without_cross_layer_validation(tmp_path, monkeypatch) -> None:
    """Smoke test with cross-layer validation disabled."""
    import src.execution.regime_benchmark as regime_benchmark_mod

    fake_regime_result = _make_fake_regime_result(tmp_path)
    monkeypatch.setattr(regime_benchmark_mod, "run_regime_benchmark_pack", lambda *a, **kw: fake_regime_result)

    namespace = runpy.run_path(str(CASE_STUDY_SCRIPT))
    run_fn = namespace["run_m28_unified_regime_research_case_study"]

    output_root = tmp_path / "case_study_no_cl"
    result = run_fn(output_root, include_cross_layer_validation=False)

    assert result["summary"]["cross_layer_validation"]["skipped"] is True
    cl_path = output_root / namespace["CASE_STUDY_OUTPUTS"]["cross_layer_comparison_json"]
    assert cl_path.exists()
    cl_payload = json.loads(cl_path.read_text(encoding="utf-8"))
    assert cl_payload.get("skipped") is True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_regime_result(tmp_path: Path) -> Any:
    """Build a minimal fake ExecutionResult for the regime benchmark pack."""
    from src.execution.result import ExecutionResult

    artifact_dir = tmp_path / "regime_benchmark_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = artifact_dir / "manifest.json"
    benchmark_summary_path = artifact_dir / "benchmark_summary.json"
    benchmark_matrix_csv_path = artifact_dir / "benchmark_matrix.csv"
    benchmark_matrix_json_path = artifact_dir / "benchmark_matrix.json"

    manifest_payload = {
        "run_id": "fake_regime_run_000",
        "benchmark_name": "m26_regime_policy_benchmark",
        "run_type": "regime_benchmark_pack",
    }
    benchmark_summary_payload = {
        "benchmark_name": "m26_regime_policy_benchmark",
        "regime_sources": ["static", "taxonomy", "gmm"],
    }

    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
    benchmark_summary_path.write_text(json.dumps(benchmark_summary_payload), encoding="utf-8")
    benchmark_matrix_csv_path.write_text("variant,metric\nstatic_baseline,0.1\n", encoding="utf-8")
    benchmark_matrix_json_path.write_text(json.dumps({"rows": []}), encoding="utf-8")

    return ExecutionResult(
        workflow="regime_benchmark_pack",
        run_id="fake_regime_run_000",
        name="m26_regime_policy_benchmark",
        artifact_dir=artifact_dir,
        manifest_path=manifest_path,
        metrics={"variant_count": 3},
        output_paths={
            "manifest_json": manifest_path,
            "benchmark_summary_json": benchmark_summary_path,
            "benchmark_matrix_csv": benchmark_matrix_csv_path,
            "benchmark_matrix_json": benchmark_matrix_json_path,
            "policy_comparison_csv": benchmark_matrix_csv_path,
            "calibration_comparison_csv": benchmark_matrix_csv_path,
        },
        extra={"variant_count": 3},
    )


def _make_fake_cl_result(tmp_path: Path) -> Any:
    """Build a minimal fake ExecutionResult for cross-layer validation."""
    from src.execution.result import ExecutionResult

    cl_dir = tmp_path / "cross_layer_workdir"
    cl_dir.mkdir(parents=True, exist_ok=True)
    report_path = cl_dir / "cross_layer_validation_report.json"
    report_payload = {
        "run_type": "cross_layer_validation",
        "schema_version": 1,
        "status": "passed",
        "scenario_count": 3,
        "pass_count": 3,
        "scenarios": [],
        "comparison_contract": {},
        "limitations": [],
    }
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")

    return ExecutionResult(
        workflow="cross_layer_validation",
        run_id="cross_layer_validation",
        name="cross_layer_validation",
        artifact_dir=cl_dir,
        metrics=report_payload,
        output_paths={"report_json": report_path},
        extra={"scenario_count": 3, "pass_count": 3},
        raw_result=report_payload,
    )
