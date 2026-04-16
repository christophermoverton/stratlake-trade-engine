from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from src.contracts.validate import ContractValidationError, validate_json
from src.cli.run_pipeline import load_pipeline_spec, run_cli
from src.pipeline.registry import (
    build_pipeline_registry_entry,
    pipeline_registry_path,
    register_pipeline_run,
)
from src.pipeline.pipeline_runner import PipelineRunner, PipelineSpec
from src.research.registry import RegistryError


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
    assert result.state_path == pipeline_dir / "state.json"
    assert pipeline_dir.exists()
    assert result.manifest_path.exists()
    assert result.pipeline_metrics_path.exists()
    assert result.lineage_path.exists()
    assert result.state_path.exists()
    assert not (tmp_path / "artifacts" / "research_campaigns").exists()

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    metrics = json.loads(result.pipeline_metrics_path.read_text(encoding="utf-8"))
    lineage = json.loads(result.lineage_path.read_text(encoding="utf-8"))
    state_payload = json.loads(result.state_path.read_text(encoding="utf-8"))

    assert manifest == {
        "pipeline_name": "artifact_pipeline",
        "pipeline_run_id": result.pipeline_run_id,
        "run_type": "pipeline_manifest",
        "schema_version": 1,
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
        "pipeline_run_id": result.pipeline_run_id,
        "run_type": "pipeline_lineage",
        "schema_version": 1,
        "edges": [
            {
                "from": "step:evaluate",
                "to": f"artifact:{evaluate_dir.as_posix()}",
                "type": "produces",
            },
            {
                "from": "step:prepare",
                "to": f"artifact:{prepare_dir.as_posix()}",
                "type": "produces",
            },
            {
                "from": "step:prepare",
                "to": "step:evaluate",
                "type": "depends_on",
            },
        ],
        "nodes": [
            {
                "id": f"artifact:{evaluate_dir.as_posix()}",
                "metadata": {
                    "manifest_path": (evaluate_dir / "manifest.json").as_posix(),
                    "path": evaluate_dir.as_posix(),
                },
                "type": "artifact",
            },
            {
                "id": f"artifact:{prepare_dir.as_posix()}",
                "metadata": {
                    "manifest_path": (prepare_dir / "manifest.json").as_posix(),
                    "path": prepare_dir.as_posix(),
                },
                "type": "artifact",
            },
            {
                "id": "step:evaluate",
                "metadata": {
                    "module": module_name,
                    "status": "completed",
                },
                "outputs": {
                    "argv": ["--stage", "evaluate"],
                    "artifact_dir": evaluate_dir.as_posix(),
                    "manifest_path": (evaluate_dir / "manifest.json").as_posix(),
                    "stage": "evaluate",
                },
                "type": "step",
            },
            {
                "id": "step:prepare",
                "metadata": {
                    "module": module_name,
                    "status": "completed",
                },
                "outputs": {
                    "argv": ["--stage", "prepare"],
                    "artifact_dir": prepare_dir.as_posix(),
                    "manifest_path": (prepare_dir / "manifest.json").as_posix(),
                    "stage": "prepare",
                },
                "type": "step",
            },
        ],
    }
    assert state_payload == {
        "pipeline_run_id": result.pipeline_run_id,
        "schema_version": 1,
        "state": {},
    }
    assert json.loads(pipeline_registry_path(tmp_path).read_text(encoding="utf-8")) == {
        "artifact_dir": f"artifacts/pipelines/{result.pipeline_run_id}",
        "pipeline_name": "artifact_pipeline",
        "pipeline_run_id": result.pipeline_run_id,
        "status": "completed",
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
        "state": first.state_path.read_text(encoding="utf-8"),
        "registry": pipeline_registry_path(tmp_path).read_text(encoding="utf-8"),
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
        "state": second.state_path.read_text(encoding="utf-8"),
        "registry": pipeline_registry_path(tmp_path).read_text(encoding="utf-8"),
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
    lineage = json.loads((pipeline_dir / "lineage.json").read_text(encoding="utf-8"))
    state_payload = json.loads((pipeline_dir / "state.json").read_text(encoding="utf-8"))
    registry_entry = json.loads(pipeline_registry_path(tmp_path).read_text(encoding="utf-8"))

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
    assert lineage == {
        "pipeline_run_id": PipelineRunner(spec).pipeline_run_id(),
        "run_type": "pipeline_lineage",
        "schema_version": 1,
        "edges": [
            {
                "from": "step:prepare",
                "to": "step:train",
                "type": "depends_on",
            },
            {
                "from": "step:train",
                "to": "step:report",
                "type": "depends_on",
            },
        ],
        "nodes": [
            {
                "id": "step:prepare",
                "metadata": {
                    "module": module_name,
                    "status": "completed",
                },
                "outputs": {
                    "argv": ["--ok"],
                },
                "type": "step",
            },
            {
                "id": "step:report",
                "metadata": {
                    "module": module_name,
                    "status": "skipped",
                },
                "type": "step",
            },
            {
                "id": "step:train",
                "metadata": {
                    "module": module_name,
                    "status": "failed",
                },
                "type": "step",
            },
        ],
    }
    assert state_payload == {
        "pipeline_run_id": PipelineRunner(spec).pipeline_run_id(),
        "schema_version": 1,
        "state": {},
    }
    assert registry_entry == {
        "artifact_dir": f"artifacts/pipelines/{PipelineRunner(spec).pipeline_run_id()}",
        "pipeline_name": "failure_pipeline",
        "pipeline_run_id": PipelineRunner(spec).pipeline_run_id(),
        "status": "failed",
    }


def test_register_pipeline_run_creates_registry_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    entry = register_pipeline_run(
        pipeline_run_id="demo_pipeline_pipeline_123456789abc",
        pipeline_name="demo_pipeline",
        status="completed",
        artifact_dir=tmp_path / "artifacts" / "pipelines" / "demo_pipeline_pipeline_123456789abc",
    )

    registry_path = pipeline_registry_path(tmp_path)

    assert registry_path.exists()
    assert registry_path.read_text(encoding="utf-8") == (
        '{"artifact_dir":"artifacts/pipelines/demo_pipeline_pipeline_123456789abc",'
        '"pipeline_name":"demo_pipeline","pipeline_run_id":"demo_pipeline_pipeline_123456789abc",'
        '"status":"completed"}\n'
    )
    assert entry == {
        "artifact_dir": "artifacts/pipelines/demo_pipeline_pipeline_123456789abc",
        "pipeline_name": "demo_pipeline",
        "pipeline_run_id": "demo_pipeline_pipeline_123456789abc",
        "status": "completed",
    }


def test_register_pipeline_run_appends_new_entries_in_sequence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    register_pipeline_run(
        pipeline_run_id="alpha_pipeline_pipeline_111111111111",
        pipeline_name="alpha_pipeline",
        status="completed",
        artifact_dir=tmp_path / "artifacts" / "pipelines" / "alpha_pipeline_pipeline_111111111111",
    )
    register_pipeline_run(
        pipeline_run_id="beta_pipeline_pipeline_222222222222",
        pipeline_name="beta_pipeline",
        status="failed",
        artifact_dir=tmp_path / "artifacts" / "pipelines" / "beta_pipeline_pipeline_222222222222",
    )

    lines = pipeline_registry_path(tmp_path).read_text(encoding="utf-8").splitlines()

    assert [json.loads(line)["pipeline_run_id"] for line in lines] == [
        "alpha_pipeline_pipeline_111111111111",
        "beta_pipeline_pipeline_222222222222",
    ]


def test_register_pipeline_run_skips_identical_duplicate_run_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    payload = {
        "pipeline_run_id": "stable_pipeline_pipeline_abcdef123456",
        "pipeline_name": "stable_pipeline",
        "status": "completed",
        "artifact_dir": tmp_path / "artifacts" / "pipelines" / "stable_pipeline_pipeline_abcdef123456",
    }

    register_pipeline_run(**payload)
    register_pipeline_run(**payload)

    assert pipeline_registry_path(tmp_path).read_text(encoding="utf-8").splitlines() == [
        '{"artifact_dir":"artifacts/pipelines/stable_pipeline_pipeline_abcdef123456","pipeline_name":"stable_pipeline","pipeline_run_id":"stable_pipeline_pipeline_abcdef123456","status":"completed"}'
    ]


def test_register_pipeline_run_rejects_conflicting_duplicate_run_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    register_pipeline_run(
        pipeline_run_id="stable_pipeline_pipeline_abcdef123456",
        pipeline_name="stable_pipeline",
        status="completed",
        artifact_dir=tmp_path / "artifacts" / "pipelines" / "stable_pipeline_pipeline_abcdef123456",
    )

    with pytest.raises(
        RegistryError,
        match="conflicting entry for pipeline_run_id 'stable_pipeline_pipeline_abcdef123456'",
    ):
        register_pipeline_run(
            pipeline_run_id="stable_pipeline_pipeline_abcdef123456",
            pipeline_name="stable_pipeline",
            status="failed",
            artifact_dir=tmp_path / "artifacts" / "pipelines" / "stable_pipeline_pipeline_abcdef123456",
        )


def test_build_pipeline_registry_entry_is_deterministic_for_equivalent_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    absolute_entry = build_pipeline_registry_entry(
        pipeline_run_id="stable_pipeline_pipeline_abcdef123456",
        pipeline_name="stable_pipeline",
        status="completed",
        artifact_dir=(tmp_path / "artifacts" / "pipelines" / "stable_pipeline_pipeline_abcdef123456").resolve(),
    )
    relative_entry = build_pipeline_registry_entry(
        pipeline_run_id="stable_pipeline_pipeline_abcdef123456",
        pipeline_name="stable_pipeline",
        status="completed",
        artifact_dir=Path("artifacts") / "pipelines" / "stable_pipeline_pipeline_abcdef123456",
    )

    assert absolute_entry == relative_entry


def test_pipeline_runner_injects_immutable_state_and_merges_updates_deterministically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "pipeline_state_module"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        """
from copy import deepcopy


SEEN_STATES = []


def run_cli(argv=None, *, state=None, pipeline_context=None):
    args = list(argv or [])
    stage = args[1]
    observed = deepcopy(state or {})
    SEEN_STATES.append(
        {
            "stage": stage,
            "state": deepcopy(observed),
            "pipeline_context": dict(pipeline_context or {}),
        }
    )
    observed["mutated_in_step"] = stage
    if stage == "prepare":
        return {
            "stage": stage,
            "state_updates": {
                "nested": {"value": "prepare"},
                "watermark_end": "2025-01-10",
            },
        }
    return {
        "stage": stage,
        "state_updates": {
            "nested": {"value": "train"},
        },
    }
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(tmp_path.as_posix())
    monkeypatch.setattr(
        "src.pipeline.pipeline_runner.time.time",
        _time_sequence(700.0, 701.0, 702.0, 703.0, 704.0, 705.0),
    )
    sys.modules.pop(module_name, None)

    spec = PipelineSpec.from_mapping(
        {
            "id": "state_pipeline",
            "steps": [
                {
                    "id": "prepare",
                    "adapter": "python_module",
                    "module": module_name,
                    "argv": ["--stage", "prepare"],
                },
                {
                    "id": "train",
                    "depends_on": ["prepare"],
                    "adapter": "python_module",
                    "module": module_name,
                    "argv": ["--stage", "train"],
                },
            ],
        }
    )

    result = PipelineRunner(spec).run()
    state_payload = json.loads(result.state_path.read_text(encoding="utf-8"))
    module = sys.modules[module_name]

    assert module.SEEN_STATES == [
        {
            "stage": "prepare",
            "state": {},
            "pipeline_context": {
                "pipeline_id": "state_pipeline",
                "pipeline_run_id": result.pipeline_run_id,
                "step_depends_on": [],
                "step_id": "prepare",
            },
        },
        {
            "stage": "train",
            "state": {
                "nested": {"value": "prepare"},
                "watermark_end": "2025-01-10",
            },
            "pipeline_context": {
                "pipeline_id": "state_pipeline",
                "pipeline_run_id": result.pipeline_run_id,
                "step_depends_on": ["prepare"],
                "step_id": "train",
            },
        },
    ]
    assert state_payload == {
        "pipeline_run_id": result.pipeline_run_id,
        "schema_version": 1,
        "state": {
            "nested": {"value": "train"},
            "watermark_end": "2025-01-10",
        },
    }


def test_pipeline_runner_loads_prior_state_from_previous_pipeline_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "pipeline_prior_state_module"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        """
SEEN_STATES = []


def run_cli(argv=None, *, state=None):
    SEEN_STATES.append(dict(state or {}))
    return {"state_updates": {"watermark_end": "2025-01-10"}}
""".strip(),
        encoding="utf-8",
    )

    previous_run_id = "prior_pipeline_run"
    previous_state_path = tmp_path / "artifacts" / "pipelines" / previous_run_id / "state.json"
    previous_state_path.parent.mkdir(parents=True)
    previous_state_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pipeline_run_id": previous_run_id,
                "state": {
                    "nested": {"value": "old"},
                    "watermark_start": "2025-01-01",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(tmp_path.as_posix())
    monkeypatch.setattr(
        "src.pipeline.pipeline_runner.time.time",
        _time_sequence(710.0, 711.0, 712.0, 713.0),
    )
    sys.modules.pop(module_name, None)

    spec = PipelineSpec.from_mapping(
        {
            "id": "load_prior_state_pipeline",
            "parameters": {
                "state": {
                    "previous_pipeline_run_id": previous_run_id,
                }
            },
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
    state_payload = json.loads(result.state_path.read_text(encoding="utf-8"))
    module = sys.modules[module_name]

    assert module.SEEN_STATES == [
        {
            "nested": {"value": "old"},
            "watermark_start": "2025-01-01",
        }
    ]
    assert state_payload == {
        "pipeline_run_id": result.pipeline_run_id,
        "schema_version": 1,
        "state": {
            "nested": {"value": "old"},
            "watermark_end": "2025-01-10",
            "watermark_start": "2025-01-01",
        },
    }


def test_pipeline_runner_does_not_apply_failed_step_state_updates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "pipeline_failed_state_module"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        """
def run_cli(argv=None, *, state=None):
    args = list(argv or [])
    stage = args[1]
    if stage == "prepare":
        return {"state_updates": {"watermark_end": "2025-01-10"}}
    raise RuntimeError("boom")
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(tmp_path.as_posix())
    monkeypatch.setattr(
        "src.pipeline.pipeline_runner.time.time",
        _time_sequence(720.0, 721.0, 722.0, 723.0, 724.0, 725.0),
    )
    sys.modules.pop(module_name, None)

    spec = PipelineSpec.from_mapping(
        {
            "id": "failed_state_pipeline",
            "steps": [
                {
                    "id": "prepare",
                    "adapter": "python_module",
                    "module": module_name,
                    "argv": ["--stage", "prepare"],
                },
                {
                    "id": "train",
                    "depends_on": ["prepare"],
                    "adapter": "python_module",
                    "module": module_name,
                    "argv": ["--stage", "train"],
                },
            ],
        }
    )

    with pytest.raises(RuntimeError, match="boom"):
        PipelineRunner(spec).run()

    state_payload = json.loads(
        (
            tmp_path
            / "artifacts"
            / "pipelines"
            / PipelineRunner(spec).pipeline_run_id()
            / "state.json"
        ).read_text(encoding="utf-8")
    )

    assert state_payload == {
        "pipeline_run_id": PipelineRunner(spec).pipeline_run_id(),
        "schema_version": 1,
        "state": {
            "watermark_end": "2025-01-10",
        },
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


def test_pipeline_runner_builds_dataset_nodes_when_parameters_include_datasets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "pipeline_dataset_module"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        """
def run_cli(argv=None):
    return {"argv": list(argv or [])}
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(tmp_path.as_posix())
    monkeypatch.setattr(
        "src.pipeline.pipeline_runner.time.time",
        _time_sequence(500.0, 501.0, 502.0, 503.0, 504.0, 505.0),
    )
    sys.modules.pop(module_name, None)

    spec = PipelineSpec.from_mapping(
        {
            "id": "dataset_pipeline",
            "parameters": {
                "datasets": {
                    "prices": {
                        "consumers": ["prepare"],
                    }
                }
            },
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
            ],
        }
    )

    result = PipelineRunner(spec).run()
    lineage = json.loads(result.lineage_path.read_text(encoding="utf-8"))

    assert lineage["nodes"][0] == {
        "id": "dataset:prices",
        "metadata": {"name": "prices"},
        "type": "dataset",
    }
    assert {
        "from": "dataset:prices",
        "to": "step:prepare",
        "type": "depends_on",
    } in lineage["edges"]


def test_validate_json_accepts_valid_pipeline_artifacts(tmp_path: Path) -> None:
    contracts_dir = Path.cwd() / "contracts"
    manifest_payload = {
        "pipeline_name": "artifact_pipeline",
        "pipeline_run_id": "artifact_pipeline_pipeline_123456789abc",
        "run_type": "pipeline_manifest",
        "schema_version": 1,
        "status": "completed",
        "steps": [
            {
                "depends_on": [],
                "module": "src.pipeline.testing",
                "outputs": {"stage": "prepare"},
                "status": "completed",
                "step_artifact_dir": "artifacts/pipelines/run/prepare",
                "step_id": "prepare",
                "step_manifest_path": "artifacts/pipelines/run/prepare/manifest.json",
            }
        ],
    }
    metrics_payload = {
        "duration_seconds": 1.5,
        "ended_at_unix": 101.5,
        "pipeline_name": "artifact_pipeline",
        "pipeline_run_id": "artifact_pipeline_pipeline_123456789abc",
        "run_type": "pipeline_metrics",
        "schema_version": 1,
        "started_at_unix": 100.0,
        "status": "completed",
        "status_counts": {"completed": 1, "failed": 0, "reused": 0, "skipped": 0},
        "step_durations_seconds": {"prepare": 1.5},
        "steps": [
            {
                "duration_seconds": 1.5,
                "ended_at_unix": 101.5,
                "started_at_unix": 100.0,
                "status": "completed",
                "step_id": "prepare",
            }
        ],
    }
    lineage_payload = {
        "pipeline_run_id": "artifact_pipeline_pipeline_123456789abc",
        "run_type": "pipeline_lineage",
        "schema_version": 1,
        "edges": [
            {"from": "step:prepare", "to": "artifact:artifacts/pipelines/run/prepare", "type": "produces"}
        ],
        "nodes": [
            {
                "id": "artifact:artifacts/pipelines/run/prepare",
                "metadata": {
                    "manifest_path": "artifacts/pipelines/run/prepare/manifest.json",
                    "path": "artifacts/pipelines/run/prepare",
                },
                "type": "artifact",
            },
            {
                "id": "step:prepare",
                "metadata": {"module": "src.pipeline.testing", "status": "completed"},
                "outputs": {"stage": "prepare"},
                "type": "step",
            },
        ],
    }

    validate_json(manifest_payload, contracts_dir / "pipeline_manifest.schema.json")
    validate_json(metrics_payload, contracts_dir / "pipeline_metrics.schema.json")
    validate_json(lineage_payload, contracts_dir / "lineage.schema.json")


@pytest.mark.parametrize(
    ("schema_name", "payload", "expected_message"),
    [
        (
            "pipeline_manifest.schema.json",
            {
                "pipeline_name": "artifact_pipeline",
                "pipeline_run_id": "artifact_pipeline_pipeline_123456789abc",
                "run_type": "pipeline_manifest",
                "schema_version": 1,
                "status": "completed",
            },
            "JSON contract validation failed for 'pipeline_manifest.schema.json' at '$': 'steps' is a required property",
        ),
        (
            "pipeline_metrics.schema.json",
            {
                "duration_seconds": "1.5",
                "ended_at_unix": 101.5,
                "pipeline_name": "artifact_pipeline",
                "pipeline_run_id": "artifact_pipeline_pipeline_123456789abc",
                "run_type": "pipeline_metrics",
                "schema_version": 1,
                "started_at_unix": 100.0,
                "status": "completed",
                "status_counts": {"completed": 1, "failed": 0, "reused": 0, "skipped": 0},
                "step_durations_seconds": {"prepare": 1.5},
                "steps": [],
            },
            "JSON contract validation failed for 'pipeline_metrics.schema.json' at '$.duration_seconds': '1.5' is not of type 'number'",
        ),
        (
            "pipeline_metrics.schema.json",
            {
                "duration_seconds": 1.5,
                "ended_at_unix": 101.5,
                "pipeline_name": "artifact_pipeline",
                "pipeline_run_id": "artifact_pipeline_pipeline_123456789abc",
                "run_type": "pipeline_metrics",
                "schema_version": 1,
                "started_at_unix": 100.0,
                "status": "unexpected",
                "status_counts": {"completed": 1, "failed": 0, "reused": 0, "skipped": 0},
                "step_durations_seconds": {"prepare": 1.5},
                "steps": [],
            },
            "JSON contract validation failed for 'pipeline_metrics.schema.json' at '$.status': 'unexpected' is not one of ['completed', 'failed']",
        ),
        (
            "lineage.schema.json",
            {
                "pipeline_run_id": "artifact_pipeline_pipeline_123456789abc",
                "run_type": "pipeline_lineage",
                "schema_version": 1,
            },
            "JSON contract validation failed for 'lineage.schema.json' at '$': 'edges' is a required property",
        ),
    ],
)
def test_validate_json_rejects_invalid_pipeline_artifacts(
    schema_name: str,
    payload: dict[str, object],
    expected_message: str,
) -> None:
    schema_path = Path.cwd() / "contracts" / schema_name

    with pytest.raises(ContractValidationError) as exc_info:
        validate_json(payload, schema_path)

    assert str(exc_info.value) == expected_message


def test_pipeline_runner_fails_fast_before_writing_invalid_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "pipeline_invalid_contract_module"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        """
def run_cli(argv=None):
    return {"argv": list(argv or [])}
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(tmp_path.as_posix())
    monkeypatch.setattr(
        "src.pipeline.pipeline_runner.time.time",
        _time_sequence(600.0, 601.0, 602.0, 603.0),
    )
    monkeypatch.setattr(
        "src.pipeline.pipeline_runner._build_pipeline_lineage_payload",
        lambda result: {
            "pipeline_run_id": result.pipeline_run_id,
            "run_type": "pipeline_lineage",
            "schema_version": 1,
            "nodes": [],
        },
    )
    sys.modules.pop(module_name, None)

    spec = PipelineSpec.from_mapping(
        {
            "id": "invalid_pipeline",
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

    with pytest.raises(ContractValidationError) as exc_info:
        PipelineRunner(spec).run()

    pipeline_dir = tmp_path / "artifacts" / "pipelines" / PipelineRunner(spec).pipeline_run_id()

    assert (
        str(exc_info.value)
        == "JSON contract validation failed for 'lineage.schema.json' at '$': 'edges' is a required property"
    )
    assert not pipeline_dir.exists()


def test_run_pipeline_cli_loads_yaml_config(capsys: pytest.CaptureFixture[str]) -> None:
    result = run_cli(["--config", "configs/test_pipeline.yml"])

    captured = capsys.readouterr()

    assert result.pipeline_id == "test_pipeline"
    assert result.execution_order == ("prepare", "evaluate")
    assert "pipeline_run_id:" in captured.out
