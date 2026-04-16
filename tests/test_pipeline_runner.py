from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from src.cli.run_pipeline import load_pipeline_spec, run_cli
from src.pipeline.pipeline_runner import PipelineRunner, PipelineSpec


def test_pipeline_run_id_is_deterministic_for_equivalent_yaml(tmp_path: Path) -> None:
    first = tmp_path / "first.yml"
    second = tmp_path / "second.yml"

    first.write_text(
        """
id: deterministic_pipeline
steps:
  - id: alpha
    adapter: python_module
    module: src.pipeline.testing
    argv: ["--flag", "one"]
  - id: beta
    depends_on: [alpha]
    adapter: python_module
    module: src.pipeline.testing
    argv: ["--flag", "two"]
""".strip(),
        encoding="utf-8",
    )
    second.write_text(
        """
steps:
  - module: src.pipeline.testing
    argv: ["--flag", "one"]
    adapter: python_module
    id: alpha
  - argv: ["--flag", "two"]
    adapter: python_module
    depends_on: [alpha]
    id: beta
    module: src.pipeline.testing
id: deterministic_pipeline
""".strip(),
        encoding="utf-8",
    )

    first_runner = PipelineRunner(load_pipeline_spec(first))
    second_runner = PipelineRunner(load_pipeline_spec(second))

    assert first_runner.pipeline_run_id() == second_runner.pipeline_run_id()


def test_pipeline_runner_executes_steps_in_dependency_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "pipeline_test_module"
    module_path = tmp_path / f"{module_name}.py"
    tracker_path = tmp_path / "tracker.txt"
    module_path.write_text(
        """
from pathlib import Path
import os


def run_cli(argv=None):
    tracker = Path(os.environ["PIPELINE_TRACKER_PATH"])
    existing = tracker.read_text(encoding="utf-8").splitlines() if tracker.exists() else []
    existing.append(" ".join(argv or []))
    tracker.write_text("\\n".join(existing) + "\\n", encoding="utf-8")
    return {"argv": list(argv or [])}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("PIPELINE_TRACKER_PATH", tracker_path.as_posix())
    monkeypatch.syspath_prepend(tmp_path.as_posix())
    sys.modules.pop(module_name, None)

    spec = PipelineSpec.from_mapping(
        {
            "id": "execution_pipeline",
            "steps": [
                {
                    "id": "prepare",
                    "adapter": "python_module",
                    "module": module_name,
                    "argv": ["--step", "prepare"],
                },
                {
                    "id": "train",
                    "depends_on": ["prepare"],
                    "adapter": "python_module",
                    "module": module_name,
                    "argv": ["--step", "train"],
                },
                {
                    "id": "report",
                    "depends_on": ["train"],
                    "adapter": "python_module",
                    "module": module_name,
                    "argv": ["--step", "report"],
                },
            ],
        }
    )

    result = PipelineRunner(spec).run()

    assert result.execution_order == ("prepare", "train", "report")
    assert [step.step_id for step in result.step_results] == ["prepare", "train", "report"]
    assert tracker_path.read_text(encoding="utf-8").splitlines() == [
        "--step prepare",
        "--step train",
        "--step report",
    ]


def test_pipeline_runner_writes_pipeline_artifacts_with_step_references(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "pipeline_artifact_module"
    module_path = tmp_path / f"{module_name}.py"
    step_root = tmp_path / "step_outputs"
    prepare_dir = step_root / "prepare"
    evaluate_dir = step_root / "evaluate"
    prepare_dir.mkdir(parents=True)
    evaluate_dir.mkdir(parents=True)
    (prepare_dir / "manifest.json").write_text('{"run_id":"prepare"}\n', encoding="utf-8")
    (evaluate_dir / "manifest.json").write_text('{"run_id":"evaluate"}\n', encoding="utf-8")
    module_path.write_text(
        """
from pathlib import Path
import os


def run_cli(argv=None):
    args = list(argv or [])
    stage = args[1]
    root = Path(os.environ["PIPELINE_STEP_OUTPUT_ROOT"])
    artifact_dir = root / stage
    return {
        "artifact_dir": artifact_dir,
        "manifest_path": artifact_dir / "manifest.json",
        "stage": stage,
        "argv": args,
    }
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PIPELINE_STEP_OUTPUT_ROOT", step_root.as_posix())
    monkeypatch.syspath_prepend(tmp_path.as_posix())
    sys.modules.pop(module_name, None)

    spec = PipelineSpec.from_mapping(
        {
            "id": "artifact_pipeline",
            "steps": [
                {
                    "id": "prepare",
                    "adapter": "python_module",
                    "module": module_name,
                    "argv": ["--stage", "prepare"],
                },
                {
                    "id": "evaluate",
                    "depends_on": ["prepare"],
                    "adapter": "python_module",
                    "module": module_name,
                    "argv": ["--stage", "evaluate"],
                },
            ],
        }
    )

    result = PipelineRunner(spec).run()
    pipeline_dir = tmp_path / "artifacts" / "pipelines" / result.pipeline_run_id

    assert result.artifact_dir == pipeline_dir
    assert result.manifest_path == pipeline_dir / "manifest.json"
    assert result.pipeline_metrics_path == pipeline_dir / "pipeline_metrics.json"
    assert result.lineage_path == pipeline_dir / "lineage.json"
    assert pipeline_dir.exists()
    assert result.manifest_path.exists()
    assert result.pipeline_metrics_path.exists()
    assert result.lineage_path.exists()
    assert not (tmp_path / "artifacts" / "research_campaigns").exists()

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    metrics = json.loads(result.pipeline_metrics_path.read_text(encoding="utf-8"))
    lineage = json.loads(result.lineage_path.read_text(encoding="utf-8"))

    assert manifest == {
        "pipeline_name": "artifact_pipeline",
        "pipeline_run_id": result.pipeline_run_id,
        "status": "completed",
        "steps": [
            {
                "depends_on": [],
                "module": module_name,
                "outputs": {
                    "argv": ["--stage", "prepare"],
                    "artifact_dir": prepare_dir.as_posix(),
                    "manifest_path": (prepare_dir / "manifest.json").as_posix(),
                    "stage": "prepare",
                },
                "status": "completed",
                "step_artifact_dir": prepare_dir.as_posix(),
                "step_id": "prepare",
                "step_manifest_path": (prepare_dir / "manifest.json").as_posix(),
            },
            {
                "depends_on": ["prepare"],
                "module": module_name,
                "outputs": {
                    "argv": ["--stage", "evaluate"],
                    "artifact_dir": evaluate_dir.as_posix(),
                    "manifest_path": (evaluate_dir / "manifest.json").as_posix(),
                    "stage": "evaluate",
                },
                "status": "completed",
                "step_artifact_dir": evaluate_dir.as_posix(),
                "step_id": "evaluate",
                "step_manifest_path": (evaluate_dir / "manifest.json").as_posix(),
            },
        ],
    }
    assert metrics == {
        "completed_step_count": 2,
        "metrics": {},
        "pipeline_name": "artifact_pipeline",
        "pipeline_run_id": result.pipeline_run_id,
        "status": "completed",
        "step_count": 2,
    }
    assert lineage == {
        "pipeline_name": "artifact_pipeline",
        "pipeline_run_id": result.pipeline_run_id,
        "status": "completed",
        "steps": [
            {
                "depends_on": [],
                "module": module_name,
                "step_artifact_dir": prepare_dir.as_posix(),
                "step_id": "prepare",
                "step_manifest_path": (prepare_dir / "manifest.json").as_posix(),
            },
            {
                "depends_on": ["prepare"],
                "module": module_name,
                "step_artifact_dir": evaluate_dir.as_posix(),
                "step_id": "evaluate",
                "step_manifest_path": (evaluate_dir / "manifest.json").as_posix(),
            },
        ],
    }


def test_pipeline_runner_artifacts_are_deterministic_across_repeated_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "pipeline_stable_module"
    module_path = tmp_path / f"{module_name}.py"
    step_root = tmp_path / "stable_outputs"
    stable_dir = step_root / "single"
    stable_dir.mkdir(parents=True)
    (stable_dir / "manifest.json").write_text('{"run_id":"single"}\n', encoding="utf-8")
    module_path.write_text(
        """
from pathlib import Path
import os


def run_cli(argv=None):
    root = Path(os.environ["PIPELINE_STABLE_OUTPUT_ROOT"])
    artifact_dir = root / "single"
    return {
        "artifact_dir": artifact_dir,
        "manifest_json": artifact_dir / "manifest.json",
        "payload": {"argv": list(argv or [])},
    }
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PIPELINE_STABLE_OUTPUT_ROOT", step_root.as_posix())
    monkeypatch.syspath_prepend(tmp_path.as_posix())
    sys.modules.pop(module_name, None)

    spec = PipelineSpec.from_mapping(
        {
            "id": "stable_pipeline",
            "steps": [
                {
                    "id": "only",
                    "adapter": "python_module",
                    "module": module_name,
                    "argv": ["--flag", "stable"],
                }
            ],
        }
    )

    first = PipelineRunner(spec).run()
    first_snapshot = {
        "manifest": first.manifest_path.read_text(encoding="utf-8"),
        "metrics": first.pipeline_metrics_path.read_text(encoding="utf-8"),
        "lineage": first.lineage_path.read_text(encoding="utf-8"),
    }

    second = PipelineRunner(spec).run()
    second_snapshot = {
        "manifest": second.manifest_path.read_text(encoding="utf-8"),
        "metrics": second.pipeline_metrics_path.read_text(encoding="utf-8"),
        "lineage": second.lineage_path.read_text(encoding="utf-8"),
    }

    assert first.pipeline_run_id == second.pipeline_run_id
    assert first_snapshot == second_snapshot
    assert first.artifact_dir == second.artifact_dir


def test_run_pipeline_cli_loads_yaml_config(capsys: pytest.CaptureFixture[str]) -> None:
    result = run_cli(["--config", "configs/test_pipeline.yml"])

    captured = capsys.readouterr()

    assert result.pipeline_id == "test_pipeline"
    assert result.execution_order == ("prepare", "evaluate")
    assert "pipeline_run_id:" in captured.out
