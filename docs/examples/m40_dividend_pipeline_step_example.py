"""CI-safe M40 dividend evidence pipeline-step example.

The example shows a scheduler-free pipeline-style wrapper around the public
Python API. It creates synthetic local inputs and writes generated output under
docs/examples/output/m40_dividend_events/.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from docs.examples.m40_dividend_evidence_import_example import (
    EXAMPLE_ROOT,
    write_synthetic_dividend_fixture,
)
from src.corporate_actions import import_dividend_events, load_dividend_events


def run_dividend_import_pipeline_step(
    *,
    source_data: Path,
    source_metadata: Path,
    output_root: Path,
    artifact_root: Path,
    start: str,
    end: str,
) -> dict[str, Any]:
    result = import_dividend_events(
        source_data_path=source_data,
        source_metadata_path=source_metadata,
        output_root=output_root,
        artifact_root=artifact_root,
        start=start,
        end=end,
        strict=True,
    )
    loaded = load_dividend_events(output_root)
    return {
        "artifact_path": result.artifact_path,
        "dataset_root": result.output_root,
        "pipeline_step": "import_dividend_events",
        "run_id": result.run_id,
        "written_row_count": result.written_row_count,
        "loaded_row_count": int(len(loaded)),
    }


def run_m40_dividend_pipeline_step_example() -> dict[str, Any]:
    fixture_root = EXAMPLE_ROOT / "pipeline_fixtures" / "corporate_actions"
    source_data, source_metadata = write_synthetic_dividend_fixture(fixture_root)
    return run_dividend_import_pipeline_step(
        source_data=source_data,
        source_metadata=source_metadata,
        output_root=EXAMPLE_ROOT / "pipeline_data",
        artifact_root=EXAMPLE_ROOT / "pipeline_artifacts",
        start="2024-01-01",
        end="2025-01-01",
    )


if __name__ == "__main__":
    print(json.dumps(run_m40_dividend_pipeline_step_example(), indent=2, sort_keys=True))
