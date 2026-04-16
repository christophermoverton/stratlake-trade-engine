from __future__ import annotations

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


def test_run_pipeline_cli_loads_yaml_config(capsys: pytest.CaptureFixture[str]) -> None:
    result = run_cli(["--config", "configs/test_pipeline.yml"])

    captured = capsys.readouterr()

    assert result.pipeline_id == "test_pipeline"
    assert result.execution_order == ("prepare", "evaluate")
    assert "pipeline_run_id:" in captured.out
