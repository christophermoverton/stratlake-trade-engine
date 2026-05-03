from __future__ import annotations

import importlib.util
import runpy
from pathlib import Path
from types import SimpleNamespace


EXAMPLES = {
    "airflow": {
        "path": Path("docs/examples/pipelines/m28_airflow_regime_research_dag.py"),
        "callable": "run_m28_airflow_example",
        "builder": "build_m28_airflow_output_root",
        "package": "airflow",
        "optional_globals": ("DAG", "PythonOperator"),
    },
    "prefect": {
        "path": Path("docs/examples/pipelines/m28_prefect_regime_research_flow.py"),
        "callable": "run_m28_prefect_example",
        "builder": "build_m28_prefect_output_root",
        "package": "prefect",
        "optional_globals": ("flow", "task"),
    },
    "dagster": {
        "path": Path("docs/examples/pipelines/m28_dagster_regime_research_job.py"),
        "callable": "run_m28_dagster_example",
        "builder": "build_m28_dagster_output_root",
        "package": "dagster",
        "optional_globals": ("job", "op"),
    },
}


def test_orchestrator_examples_import_without_optional_dependencies() -> None:
    for example in EXAMPLES.values():
        namespace = runpy.run_path(example["path"])

        assert callable(namespace[example["callable"]])
        assert callable(namespace[example["builder"]])

        if importlib.util.find_spec(example["package"]) is None:
            for global_name in example["optional_globals"]:
                assert namespace[global_name] is None


def test_orchestrator_example_callables_delegate_to_execution_api(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_benchmark_pack(config_path: str, *, output_root: Path):
        calls.append({"config_path": config_path, "output_root": output_root})
        return SimpleNamespace(to_dict=lambda: {"workflow": "benchmark_pack"})

    monkeypatch.setattr("src.execution.run_benchmark_pack", fake_run_benchmark_pack)

    for orchestrator, example in EXAMPLES.items():
        namespace = runpy.run_path(example["path"])
        result = namespace[example["callable"]](f"{orchestrator}-run", 2)

        assert result.to_dict() == {"workflow": "benchmark_pack"}

    assert len(calls) == 3
    for orchestrator, call in zip(EXAMPLES, calls, strict=True):
        assert call["config_path"] == "configs/benchmark_packs/m22_scale_repro.yml"
        output_root = call["output_root"]
        assert isinstance(output_root, Path)
        assert not output_root.is_absolute()
        assert output_root.as_posix().startswith(f"artifacts/orchestrator_examples/{orchestrator}/")
        assert output_root.as_posix().endswith("/attempt_2")


def test_orchestrator_examples_are_thin_wrappers_over_existing_surfaces() -> None:
    for example in EXAMPLES.values():
        source = example["path"].read_text(encoding="utf-8")

        assert "from src.execution import run_benchmark_pack" in source
        assert "run_benchmark_pack(" in source
        assert "subprocess" not in source
        assert "shell=True" not in source
        assert "configs/benchmark_packs/m22_scale_repro.yml" in source


def test_orchestrator_examples_use_relative_paths_only() -> None:
    forbidden_fragments = ("C:/", "C:\\", "file://", "/Users/", "/home/")

    for example in EXAMPLES.values():
        source = example["path"].read_text(encoding="utf-8")

        assert all(fragment not in source for fragment in forbidden_fragments)
        assert "Path(\"artifacts\")" in source
