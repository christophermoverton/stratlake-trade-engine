from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from src.cli.run_pipeline import load_pipeline_spec, run_cli
from src.pipeline.pipeline_runner import PipelineRunner, PipelineSpec


def _time_sequence(*values: float):
    iterator = iter(values)

    def _fake_time() -> float:
        return next(iterator)

    return _fake_time


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
    monkeypatch.setattr(
        "src.pipeline.pipeline_runner.time.time",
        _time_sequence(100.0, 101.0, 102.5, 104.0, 107.0, 110.0),
    )
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
        "duration_seconds": 10.0,
        "ended_at_unix": 110.0,
        "pipeline_name": "artifact_pipeline",
        "pipeline_run_id": result.pipeline_run_id,
        "run_type": "pipeline_metrics",
        "schema_version": 1,
        "started_at_unix": 100.0,
        "status_counts": {
            "completed": 2,
            "failed": 0,
            "reused": 0,
            "skipped": 0,
        },
        "status": "completed",
        "step_durations_seconds": {
            "evaluate": 3.0,
            "prepare": 1.5,
        },
        "steps": [
            {
                "duration_seconds": 1.5,
                "ended_at_unix": 102.5,
                "started_at_unix": 101.0,
                "status": "completed",
                "step_id": "prepare",
            },
            {
                "duration_seconds": 3.0,
                "ended_at_unix": 107.0,
                "started_at_unix": 104.0,
                "status": "completed",
                "step_id": "evaluate",
            },
        ],
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

    monkeypatch.setattr(
        "src.pipeline.pipeline_runner.time.time",
        _time_sequence(200.0, 201.0, 202.0, 203.0),
    )
    first = PipelineRunner(spec).run()
    first_snapshot = {
        "manifest": first.manifest_path.read_text(encoding="utf-8"),
        "metrics": first.pipeline_metrics_path.read_text(encoding="utf-8"),
        "lineage": first.lineage_path.read_text(encoding="utf-8"),
    }

    monkeypatch.setattr(
        "src.pipeline.pipeline_runner.time.time",
        _time_sequence(200.0, 201.0, 202.0, 203.0),
    )
    second = PipelineRunner(spec).run()
    second_snapshot = {
        "manifest": second.manifest_path.read_text(encoding="utf-8"),
        "metrics": second.pipeline_metrics_path.read_text(encoding="utf-8"),
        "lineage": second.lineage_path.read_text(encoding="utf-8"),
    }

    assert first.pipeline_run_id == second.pipeline_run_id
    assert first_snapshot == second_snapshot
    assert first.artifact_dir == second.artifact_dir


def test_pipeline_runner_writes_metrics_for_failed_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "pipeline_failure_module"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        """
def run_cli(argv=None):
    args = list(argv or [])
    if "--fail" in args:
        raise RuntimeError("boom")
    return {"argv": args}
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(tmp_path.as_posix())
    monkeypatch.setattr(
        "src.pipeline.pipeline_runner.time.time",
        _time_sequence(300.0, 301.0, 302.0, 303.0, 304.5, 306.0),
    )
    sys.modules.pop(module_name, None)

    spec = PipelineSpec.from_mapping(
        {
            "id": "failure_pipeline",
            "steps": [
                {
                    "id": "prepare",
                    "adapter": "python_module",
                    "module": module_name,
                    "argv": ["--ok"],
                },
                {
                    "id": "train",
                    "depends_on": ["prepare"],
                    "adapter": "python_module",
                    "module": module_name,
                    "argv": ["--fail"],
                },
                {
                    "id": "report",
                    "depends_on": ["train"],
                    "adapter": "python_module",
                    "module": module_name,
                    "argv": ["--report"],
                },
            ],
        }
    )

    with pytest.raises(RuntimeError, match="boom"):
        PipelineRunner(spec).run()

    pipeline_dir = tmp_path / "artifacts" / "pipelines" / PipelineRunner(spec).pipeline_run_id()
    metrics = json.loads((pipeline_dir / "pipeline_metrics.json").read_text(encoding="utf-8"))

    assert metrics == {
        "duration_seconds": 6.0,
        "ended_at_unix": 306.0,
        "pipeline_name": "failure_pipeline",
        "pipeline_run_id": PipelineRunner(spec).pipeline_run_id(),
        "run_type": "pipeline_metrics",
        "schema_version": 1,
        "started_at_unix": 300.0,
        "status": "failed",
        "status_counts": {
            "completed": 1,
            "failed": 1,
            "reused": 0,
            "skipped": 1,
        },
        "step_durations_seconds": {
            "prepare": 1.0,
            "report": 0.0,
            "train": 1.5,
        },
        "steps": [
            {
                "duration_seconds": 1.0,
                "ended_at_unix": 302.0,
                "started_at_unix": 301.0,
                "status": "completed",
                "step_id": "prepare",
            },
            {
                "duration_seconds": 1.5,
                "ended_at_unix": 304.5,
                "started_at_unix": 303.0,
                "status": "failed",
                "step_id": "train",
            },
            {
                "duration_seconds": 0.0,
                "ended_at_unix": None,
                "started_at_unix": None,
                "status": "skipped",
                "step_id": "report",
            },
        ],
    }


def test_pipeline_runner_collects_optional_row_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "pipeline_row_count_module"
    module_path = tmp_path / f"{module_name}.py"
    csv_path = tmp_path / "rows.csv"
    csv_path.write_text("symbol\nAAPL\nMSFT\nNVDA\n", encoding="utf-8")
    module_path.write_text(
        """
from pathlib import Path
import os


def run_cli(argv=None):
    return {"csv_path": Path(os.environ["PIPELINE_ROW_COUNT_CSV"])}
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PIPELINE_ROW_COUNT_CSV", csv_path.as_posix())
    monkeypatch.syspath_prepend(tmp_path.as_posix())
    monkeypatch.setattr(
        "src.pipeline.pipeline_runner.time.time",
        _time_sequence(400.0, 401.0, 402.0, 403.0),
    )
    sys.modules.pop(module_name, None)

    spec = PipelineSpec.from_mapping(
        {
            "id": "row_count_pipeline",
            "steps": [
                {
                    "id": "only",
                    "adapter": "python_module",
                    "module": module_name,
                    "argv": [],
                }
            ],
        }
    )

    result = PipelineRunner(spec).run()
    metrics = json.loads(result.pipeline_metrics_path.read_text(encoding="utf-8"))

    assert metrics["row_counts"] == {"only": 3}


def test_run_pipeline_cli_loads_yaml_config(capsys: pytest.CaptureFixture[str]) -> None:
    result = run_cli(["--config", "configs/test_pipeline.yml"])

    captured = capsys.readouterr()

    assert result.pipeline_id == "test_pipeline"
    assert result.execution_order == ("prepare", "evaluate")
    assert "pipeline_run_id:" in captured.out
