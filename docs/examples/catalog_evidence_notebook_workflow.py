"""CI-safe notebook-style M35 catalog evidence workflow.

This example builds a tiny synthetic artifact tree in a temporary directory and
uses the public catalog helpers to inspect robustness, governance, validation,
release, and selected-run evidence. It does not download data, require
credentials, or mutate repository artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from src.catalog import (
    build_catalog,
    evidence_for_run,
    evidence_lineage_rows,
    find_governance_evidence,
    find_release_evidence,
    find_robustness_evidence,
    find_validation_evidence,
    render_notebook_markdown,
)


def run_catalog_evidence_notebook_workflow() -> dict[str, Any]:
    """Return the notebook-friendly evidence payload for a synthetic catalog."""

    with TemporaryDirectory(prefix="stratlake_catalog_evidence_") as tmp:
        repo_root = Path(tmp)
        _write_fixture_artifacts(repo_root)
        records = build_catalog(repo_root, repo_root=repo_root)
        return {
            "governance_rows": find_governance_evidence(records, governance_status="pass"),
            "lineage_rows": evidence_lineage_rows(records, run_id="strategy_a", repo_root=repo_root),
            "release_rows": find_release_evidence(records),
            "robustness_rows": find_robustness_evidence(records, robustness_status="needs_review"),
            "selected_run_view": evidence_for_run(records, "strategy_a", repo_root=repo_root),
            "validation_rows": find_validation_evidence(records),
            "markdown_preview": render_notebook_markdown(records, run_id="strategy_a", repo_root=repo_root).splitlines()[:8],
        }


def _write_fixture_artifacts(root: Path) -> None:
    strategy_root = root / "strategies" / "strategy_a"
    _write_json(strategy_root / "_SUCCESS.json", {"run_id": "strategy_a", "status": "completed"})
    _write_json(strategy_root / "summary.json", {"strategy_name": "demo_strategy"})

    robustness_root = root / "robustness" / "robustness_a"
    _write_json(
        robustness_root / "robustness_summary.json",
        {
            "report_id": "robustness_a",
            "robustness_status": "needs_review",
            "source_run_ids": ["strategy_a"],
        },
    )
    (robustness_root / "walk_forward_efficiency.csv").write_text(
        "run_id,status\nstrategy_a,weak\n",
        encoding="utf-8",
    )

    governance_root = root / "promotion_governance" / "governance_a"
    _write_json(
        governance_root / "promotion_governance_summary.json",
        {"row_count": 1, "review_status_counts": {"needs_review": 1}},
    )
    _write_json(governance_root / "consistency_validation.json", {"status": "pass", "finding_count": 0})
    _write_json(governance_root / "manifest.json", {"run_type": "promotion_governance", "validation_status": "pass"})
    (governance_root / "promotion_outcome_matrix.csv").write_text(
        "workflow_type,run_id,review_status\nstrategy,strategy_a,needs_review\n",
        encoding="utf-8",
    )

    validation_root = root / "qa" / "validation_a"
    _write_json(
        validation_root / "summary.json",
        {
            "run_type": "milestone_validation_bundle",
            "status": "passed",
            "source_run_ids": ["strategy_a"],
            "checks": {},
        },
    )
    _write_json(validation_root / "_SUCCESS.json", {"status": "completed"})

    release_root = root / "release_validation" / "release_a"
    _write_json(
        release_root / "release_validation.json",
        {
            "release_id": "release_a",
            "status": "pass",
            "validation_bundle_run_id": "validation_a",
        },
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    print(json.dumps(run_catalog_evidence_notebook_workflow(), indent=2, sort_keys=True))
