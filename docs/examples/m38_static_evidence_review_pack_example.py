from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.catalog import (
    build_evidence_review_for_workflow,
    validate_evidence_review_pack,
    write_evidence_review_pack,
)


OUTPUT_ROOT = REPO_ROOT / "docs" / "examples" / "output" / "m38_static_evidence_review_pack_example"


def main() -> None:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    artifacts_root = OUTPUT_ROOT / "artifacts"
    strategy_root = artifacts_root / "strategies" / "example_strategy"
    strategy_root.mkdir(parents=True, exist_ok=True)
    (artifacts_root / "strategies" / "registry.jsonl").write_text(
        json.dumps(
            {
                "run_id": "example_strategy",
                "run_type": "strategy",
                "artifact_dir": "artifacts/strategies/example_strategy",
                "timestamp": "2026-01-01T00:00:00Z",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (strategy_root / "_SUCCESS.json").write_text(
        '{"run_id":"example_strategy","status":"completed"}\n',
        encoding="utf-8",
    )
    (strategy_root / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "example_strategy",
                "run_type": "strategy",
                "artifacts": ["summary.json", "metrics.json", "_SUCCESS.json"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (strategy_root / "summary.json").write_text(
        '{"run_id":"example_strategy","run_type":"strategy","strategy_name":"example"}\n',
        encoding="utf-8",
    )
    (strategy_root / "metrics.json").write_text('{"sharpe_ratio":1.0}\n', encoding="utf-8")

    model = build_evidence_review_for_workflow(
        artifacts_root,
        repo_root=OUTPUT_ROOT,
        selected_run_id="example_strategy",
        review_id="example_review",
    )
    write_result = write_evidence_review_pack(model, repo_root=OUTPUT_ROOT)
    validation = validate_evidence_review_pack(write_result["output_root"], repo_root=OUTPUT_ROOT)

    print(f"review_id: {write_result['review_id']}")
    print(f"output_root: {write_result['output_root']}")
    print(f"validation_status: {validation['status']}")
    print(f"report_path: {write_result['output_root']}/report.md")


if __name__ == "__main__":
    main()
