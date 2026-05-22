from __future__ import annotations

import json

from docs.examples.m40_dividend_evidence_import_example import (
    EXAMPLE_ROOT,
    run_m40_dividend_evidence_import_example,
)
from docs.examples.m40_dividend_pipeline_step_example import run_m40_dividend_pipeline_step_example


def test_m40_dividend_notebook_style_example_runs_and_writes_ignored_output() -> None:
    summary = run_m40_dividend_evidence_import_example()

    assert summary["qa_status"] == "pass"
    assert summary["written_row_count"] == 2
    assert summary["loaded_row_count"] == 2
    assert str(summary["artifact_path"]).startswith("docs/examples/output/m40_dividend_events/")
    assert (EXAMPLE_ROOT / "data").exists()
    assert (EXAMPLE_ROOT / "artifacts").exists()


def test_m40_dividend_pipeline_step_example_runs_and_uses_api() -> None:
    summary = run_m40_dividend_pipeline_step_example()

    assert summary["pipeline_step"] == "import_dividend_events"
    assert summary["written_row_count"] == 2
    assert summary["loaded_row_count"] == 2
    assert str(summary["dataset_root"]).startswith("docs/examples/output/m40_dividend_events/")
    assert "credential" not in json.dumps(summary, sort_keys=True).lower()


def test_m40_dividend_examples_keep_outputs_under_generated_root() -> None:
    run_m40_dividend_evidence_import_example()
    run_m40_dividend_pipeline_step_example()

    for path in EXAMPLE_ROOT.rglob("*"):
        if path.is_file():
            relative = path.as_posix()
            assert "docs/examples/output/m40_dividend_events/" in relative
