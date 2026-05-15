from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SCALE_CONFIG = {
    "strategy_runs": 12,
    "alpha_runs": 8,
    "portfolio_runs": 6,
    "campaign_runs": 3,
    "scenario_runs": 3,
    "robustness_bundles": 6,
    "governance_bundles": 4,
    "milestone_validation_bundles": 3,
    "release_validation_artifacts": 2,
    "registry_only_runs": 1,
}


def build_catalog_scale_tree(root: Path) -> None:
    strategy_registry: list[dict[str, Any]] = []
    alpha_registry: list[dict[str, Any]] = []
    portfolio_registry: list[dict[str, Any]] = []

    for index in range(SCALE_CONFIG["strategy_runs"]):
        run_id = f"strategy_{index:03d}"
        run_root = root / "strategies" / run_id
        _write_json(run_root / "_SUCCESS.json", _marker(run_id))
        _write_json(
            run_root / "summary.json",
            {
                "run_id": run_id,
                "run_type": "strategy",
                "strategy_name": f"scale_strategy_{index % 3}",
                "timeframe": "1D",
            },
        )
        _write_json(run_root / "metrics.json", {"sharpe_ratio": round(1.0 + index / 10, 2)})
        _write_json(
            run_root / "manifest.json",
            {
                "run_id": run_id,
                "run_type": "strategy",
                "artifacts": ["summary.json", "metrics.json", "_SUCCESS.json"],
            },
        )
        strategy_registry.append(_registry_entry(run_id, "strategy", run_root.as_posix()))

    strategy_registry.append(
        _registry_entry(
            "strategy_registry_only",
            "strategy",
            (root / "strategies" / "strategy_registry_only").as_posix(),
        )
    )
    _write_jsonl(root / "strategies" / "registry.jsonl", strategy_registry)

    for index in range(SCALE_CONFIG["alpha_runs"]):
        run_id = f"alpha_{index:03d}"
        run_root = root / "alpha" / run_id
        _write_json(run_root / "_SUCCESS.json", _marker(run_id))
        _write_json(run_root / "alpha_metrics.json", {"mean_ic": round(0.01 * (index + 1), 3)})
        _write_json(
            run_root / "manifest.json",
            {
                "run_id": run_id,
                "run_type": "alpha_evaluation",
                "artifacts": ["alpha_metrics.json", "_SUCCESS.json"],
            },
        )
        alpha_registry.append(_registry_entry(run_id, "alpha_evaluation", run_root.as_posix()))
    _write_jsonl(root / "alpha" / "registry.jsonl", alpha_registry)

    for index in range(SCALE_CONFIG["portfolio_runs"]):
        run_id = f"portfolio_{index:03d}"
        component_ids = [f"strategy_{index:03d}", f"strategy_{(index + 1) % 12:03d}"]
        run_root = root / "portfolios" / run_id
        _write_json(run_root / "_SUCCESS.json", _marker(run_id))
        _write_json(
            run_root / "summary.json",
            {
                "run_id": run_id,
                "run_type": "portfolio",
                "portfolio_name": f"scale_portfolio_{index}",
                "component_run_ids": component_ids,
            },
        )
        _write_json(
            run_root / "manifest.json",
            {
                "run_id": run_id,
                "run_type": "portfolio",
                "component_run_ids": component_ids,
                "artifacts": ["summary.json", "_SUCCESS.json"],
            },
        )
        portfolio_registry.append(_registry_entry(run_id, "portfolio", run_root.as_posix()))
    _write_jsonl(root / "portfolios" / "registry.jsonl", portfolio_registry)

    for index in range(SCALE_CONFIG["campaign_runs"]):
        run_id = f"campaign_{index:03d}"
        run_root = root / "benchmark_pack_scale" / run_id
        _write_json(run_root / "_SUCCESS.json", _marker(run_id))
        _write_json(
            run_root / "manifest.json",
            {
                "run_id": run_id,
                "run_type": "campaign",
                "campaign_id": run_id,
                "artifacts": ["_SUCCESS.json"],
            },
        )

    for index in range(SCALE_CONFIG["scenario_runs"]):
        run_id = f"scenario_{index:03d}"
        run_root = root / "benchmark_pack_scale" / run_id
        _write_json(run_root / "_SUCCESS.json", _marker(run_id))
        _write_json(
            run_root / "manifest.json",
            {
                "run_id": run_id,
                "run_type": "campaign",
                "scenario_id": run_id,
                "scenario_parent_run_id": f"campaign_{index:03d}",
                "artifacts": ["_SUCCESS.json"],
            },
        )

    for index in range(4):
        run_id = f"robustness_{index:03d}"
        source_run_id = f"strategy_{index:03d}"
        run_root = root / "robustness" / run_id
        _write_json(
            run_root / "robustness_summary.json",
            {
                "report_id": run_id,
                "robustness_status": "needs_review" if index % 2 == 0 else "pass",
                "source_run_ids": [source_run_id],
                "checks_present": ["walk_forward_efficiency", "sample_size"],
            },
        )
        _write_csv(run_root / "walk_forward_efficiency.csv", f"run_id,status\n{source_run_id},weak\n")
        _write_json(
            run_root / "sample_size_validation.json",
            {"checks": [{"check_id": "sample_size.minimum_total_samples", "status": "pass"}]},
        )
    _write_json(
        root / "robustness" / "robustness_sparse" / "robustness_summary.json",
        {"report_id": "robustness_sparse"},
    )
    _write_json(
        root / "robustness" / "robustness_orphan" / "robustness_summary.json",
        {"report_id": "robustness_orphan", "source_run_ids": ["missing_run"]},
    )

    for index in range(3):
        run_id = f"governance_{index:03d}"
        source_run_id = f"strategy_{index:03d}"
        run_root = root / "promotion_governance" / run_id
        _write_json(
            run_root / "promotion_governance_summary.json",
            {"review_status_counts": {"needs_review": 1}, "row_count": 1},
        )
        _write_json(run_root / "consistency_validation.json", {"status": "pass"})
        _write_json(run_root / "manifest.json", {"run_type": "promotion_governance"})
        _write_csv(
            run_root / "promotion_outcome_matrix.csv",
            f"workflow_type,run_id,review_status\nstrategy,{source_run_id},needs_review\n",
        )
    aggregate_root = root / "promotion_governance" / "governance_aggregate"
    _write_json(
        aggregate_root / "promotion_governance_summary.json",
        {"review_status_counts": {"pass": 3}, "row_count": 3},
    )
    _write_json(aggregate_root / "consistency_validation.json", {"status": "pass"})

    for index in range(SCALE_CONFIG["milestone_validation_bundles"]):
        run_id = f"validation_{index:03d}"
        run_root = root / "milestone_validation" / run_id
        _write_json(
            run_root / "summary.json",
            {
                "run_id": run_id,
                "run_type": "milestone_validation_bundle",
                "status": "passed",
                "checks": {"catalog": {"status": "passed"}},
                "source_run_ids": [f"strategy_{index:03d}"],
            },
        )
        _write_json(run_root / "_SUCCESS.json", _marker(run_id))

    for index in range(SCALE_CONFIG["release_validation_artifacts"]):
        run_id = f"release_{index:03d}"
        run_root = root / "release_validation" / run_id
        _write_json(
            run_root / "release_validation.json",
            {
                "release_id": run_id,
                "status": "pass",
                "validation_bundle_run_id": f"validation_{index:03d}",
            },
        )


def snapshot_tree(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _marker(run_id: str) -> dict[str, str]:
    return {
        "run_id": run_id,
        "recorded_at_utc": "2026-01-01T00:00:00Z",
        "status": "completed",
    }


def _registry_entry(run_id: str, run_type: str, artifact_dir: str) -> dict[str, str]:
    return {
        "run_id": run_id,
        "run_type": run_type,
        "artifact_dir": artifact_dir,
        "timestamp": "2026-01-01T00:00:00Z",
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _write_jsonl(path: Path, payloads: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(payload, sort_keys=True) + "\n" for payload in payloads),
        encoding="utf-8",
        newline="\n",
    )


def _write_csv(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
