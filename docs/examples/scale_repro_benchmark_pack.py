from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cli.run_research_campaign import (
    _SCENARIO_RANKING_METRIC_PRIORITY,
    _scenario_matrix_fieldnames,
    _scenario_matrix_selection_rule,
    _scenario_matrix_sort_key,
)
from src.config.benchmark_pack import load_benchmark_pack_config
from src.config.research_campaign import expand_research_campaign_scenarios
from src.research.benchmark_pack import run_benchmark_pack
from src.research.registry import canonicalize_value


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "scale_repro_benchmark_pack"
PACK_CONFIG_PATH = REPO_ROOT / "configs" / "benchmark_packs" / "m22_scale_repro.yml"


def run_example(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    reset_output: bool = True,
) -> dict[str, Any]:
    if reset_output and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    config = load_benchmark_pack_config(PACK_CONFIG_PATH)
    partial = run_benchmark_pack(
        config,
        output_root=output_root,
        stop_after_batches=1,
        campaign_runner=_fake_campaign_runner,
    )
    _snapshot_run("partial", partial, output_root=output_root)

    resumed = run_benchmark_pack(
        config,
        output_root=output_root,
        campaign_runner=_fake_campaign_runner,
    )
    _snapshot_run("resumed", resumed, output_root=output_root)

    stable = run_benchmark_pack(
        config,
        output_root=output_root,
        compare_to=resumed.inventory_path,
        campaign_runner=_fake_campaign_runner,
    )
    _snapshot_run("stable", stable, output_root=output_root)

    comparison = (
        None if stable.comparison_path is None else _read_json(stable.comparison_path)
    )
    summary = {
        "pack_id": config.pack_id,
        "pack_run_id": stable.pack_run_id,
        "workflow": [
            "run_partial_benchmark_pack",
            "resume_pending_batches",
            "rerun_stable_pack_against_reference_inventory",
        ],
        "paths": {
            "pack_config": PACK_CONFIG_PATH.as_posix(),
            "output_root": output_root.as_posix(),
            "partial_summary": partial.summary_path.as_posix(),
            "resumed_summary": resumed.summary_path.as_posix(),
            "stable_summary": stable.summary_path.as_posix(),
            "stable_inventory": stable.inventory_path.as_posix(),
            "comparison": None if stable.comparison_path is None else stable.comparison_path.as_posix(),
        },
        "partial": {
            "status": partial.status,
            "batch_status_counts": partial.summary.get("batch_status_counts", {}),
            "pending_batch_ids": partial.summary.get("resume", {}).get("pending_batch_ids", []),
        },
        "resumed": {
            "status": resumed.status,
            "batch_status_counts": resumed.summary.get("batch_status_counts", {}),
            "reused_batch_ids": resumed.summary.get("resume", {}).get("reused_batch_ids", []),
            "completed_batch_ids": resumed.summary.get("resume", {}).get("completed_batch_ids", []),
            "scenario_count": resumed.summary.get("scenario_count"),
        },
        "stable": {
            "status": stable.status,
            "batch_status_counts": stable.summary.get("batch_status_counts", {}),
            "reused_batch_ids": stable.summary.get("resume", {}).get("reused_batch_ids", []),
            "benchmark_matrix_row_count": stable.summary.get("benchmark_matrix", {}).get("row_count"),
            "inventory_digest": stable.inventory.get("aggregate_digest"),
        },
        "comparison": comparison,
    }
    _write_json(output_root / "summary.json", canonicalize_value(summary))
    return summary


def _snapshot_run(label: str, result: Any, *, output_root: Path) -> None:
    snapshot_dir = output_root / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    _write_json(snapshot_dir / f"{label}_checkpoint.json", result.checkpoint)
    _write_json(snapshot_dir / f"{label}_manifest.json", result.manifest)
    _write_json(snapshot_dir / f"{label}_summary.json", result.summary)
    _write_json(snapshot_dir / f"{label}_inventory.json", result.inventory)
    if result.comparison is not None:
        _write_json(snapshot_dir / f"{label}_comparison.json", result.comparison)


def _fake_campaign_runner(config) -> SimpleNamespace:
    scenarios = expand_research_campaign_scenarios(config)
    payload = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    orchestration_run_id = f"research_campaign_orchestration_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]}"
    orchestration_dir = Path(config.outputs.campaign_artifacts_root) / orchestration_run_id
    scenario_root = orchestration_dir / "scenarios"
    scenario_root.mkdir(parents=True, exist_ok=True)

    leaderboard = []
    sweep_keys: list[str] = []
    for scenario in scenarios:
        sweep_values = dict(scenario.sweep_values)
        for key in sweep_values:
            if key not in sweep_keys:
                sweep_keys.append(key)
        row = _scenario_row(scenario_id=scenario.scenario_id, sweep_values=sweep_values)
        leaderboard.append(row)
        scenario_dir = scenario_root / scenario.scenario_id
        scenario_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            scenario_dir / "summary.json",
            {
                "campaign_run_id": f"research_campaign_{hashlib.sha256(scenario.scenario_id.encode('utf-8')).hexdigest()[:12]}",
                "status": "completed",
                "scenario": {
                    "scenario_id": scenario.scenario_id,
                    "fingerprint": scenario.fingerprint,
                    "sweep_values": sweep_values,
                },
            },
        )
        _write_json(
            scenario_dir / "manifest.json",
            {
                "run_type": "research_campaign",
                "status": "completed",
                "scenario_id": scenario.scenario_id,
                "summary_path": "summary.json",
            },
        )
        _write_json(
            scenario_dir / "checkpoint.json",
            {
                "run_type": "research_campaign_checkpoint",
                "status": "completed",
                "scenario_id": scenario.scenario_id,
            },
        )

    ordered = sorted(leaderboard, key=_scenario_matrix_sort_key)
    ranked_rows = [canonicalize_value({"rank": index, **row}) for index, row in enumerate(ordered, start=1)]
    csv_path = orchestration_dir / "scenario_matrix.csv"
    summary_path = orchestration_dir / "scenario_matrix.json"
    fieldnames = _scenario_matrix_fieldnames(sweep_keys)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in ranked_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    scenario_matrix_summary = {
        "scenario_matrix_id": orchestration_run_id,
        "orchestration_run_id": orchestration_run_id,
        "row_count": len(ranked_rows),
        "metric_priority": list(_SCENARIO_RANKING_METRIC_PRIORITY),
        "selection_rule": _scenario_matrix_selection_rule(),
        "sweep_keys": sweep_keys,
        "leaderboard": ranked_rows,
        "csv_path": csv_path.as_posix(),
        "summary_path": summary_path.as_posix(),
    }
    _write_json(summary_path, canonicalize_value(scenario_matrix_summary))

    orchestration_summary_path = orchestration_dir / "summary.json"
    orchestration_manifest_path = orchestration_dir / "manifest.json"
    orchestration_summary = {
        "run_type": "research_campaign_orchestration",
        "orchestration_run_id": orchestration_run_id,
        "status": "completed",
        "scenario_count": len(ranked_rows),
        "scenario_status_counts": {"completed": len(ranked_rows)},
        "scenario_matrix": scenario_matrix_summary,
        "scenarios": [
            {
                "scenario_id": scenario.scenario_id,
                "status": "completed",
                "campaign_summary_path": (scenario_root / scenario.scenario_id / "summary.json").as_posix(),
            }
            for scenario in scenarios
        ],
    }
    orchestration_manifest = {
        "run_type": "research_campaign_orchestration",
        "orchestration_run_id": orchestration_run_id,
        "status": "completed",
        "scenario_count": len(ranked_rows),
        "summary_path": orchestration_summary_path.as_posix(),
        "artifact_files": ["manifest.json", "summary.json", "scenario_matrix.csv", "scenario_matrix.json"],
    }
    _write_json(orchestration_summary_path, canonicalize_value(orchestration_summary))
    _write_json(orchestration_manifest_path, canonicalize_value(orchestration_manifest))
    return SimpleNamespace(
        orchestration_run_id=orchestration_run_id,
        orchestration_artifact_dir=orchestration_dir,
        orchestration_summary_path=orchestration_summary_path,
        orchestration_manifest_path=orchestration_manifest_path,
        scenario_matrix_csv_path=csv_path,
        scenario_matrix_summary_path=summary_path,
        orchestration_summary=canonicalize_value(orchestration_summary),
        orchestration_manifest=canonicalize_value(orchestration_manifest),
    )


def _scenario_row(*, scenario_id: str, sweep_values: dict[str, Any]) -> dict[str, Any]:
    digest = int(hashlib.sha256(scenario_id.encode("utf-8")).hexdigest()[:8], 16)
    alpha_ic_ir_max = round(0.6 + ((digest % 13) / 20.0), 6)
    strategy_sharpe_ratio_max = round(0.8 + (((digest // 7) % 19) / 15.0), 6)
    portfolio_sharpe_ratio = round(0.7 + (((digest // 17) % 23) / 18.0), 6)
    row = {
        "scenario_id": scenario_id,
        "description": None,
        "source": "include",
        "status": "completed",
        "preflight_status": "passed",
        "campaign_run_id": f"research_campaign_{hashlib.sha256((scenario_id + ':campaign').encode('utf-8')).hexdigest()[:12]}",
        "campaign_summary_path": f"scenarios/{scenario_id}/summary.json",
        "fingerprint": hashlib.sha256((scenario_id + ':fingerprint').encode("utf-8")).hexdigest()[:16],
        "sweep_summary": " | ".join(f"{key}={value}" for key, value in sweep_values.items()),
        "alpha_run_count": 0,
        "strategy_run_count": 3,
        "alpha_mean_ic_max": round(alpha_ic_ir_max * 0.72, 6),
        "alpha_ic_ir_max": alpha_ic_ir_max,
        "strategy_cumulative_return_max": round(strategy_sharpe_ratio_max * 0.11, 6),
        "strategy_sharpe_ratio_max": strategy_sharpe_ratio_max,
        "candidate_selected_count": 0,
        "portfolio_total_return": round(portfolio_sharpe_ratio * 0.14, 6),
        "portfolio_sharpe_ratio": portfolio_sharpe_ratio,
        "review_entry_count": 3,
        "review_promotion_status": "reviewed",
        "review_promotion_gate_status": "pass",
    }
    ranking_metric, ranking_value = _scenario_ranking_metric(row)
    row["ranking_metric"] = ranking_metric
    row["ranking_value"] = ranking_value
    for key, value in sweep_values.items():
        row[f"sweep_{key}"] = value
    return canonicalize_value(row)


def _scenario_ranking_metric(row: dict[str, Any]) -> tuple[str | None, float | int | None]:
    for metric_name in _SCENARIO_RANKING_METRIC_PRIORITY:
        value = row.get(metric_name)
        if value is None:
            continue
        return metric_name, value
    return None, None


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the canonical M22.7 benchmark-pack example workflow.")
    parser.add_argument("--output-root", help="Optional output directory override.")
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Preserve an existing output directory instead of resetting it first.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_example(
        output_root=DEFAULT_OUTPUT_ROOT if args.output_root is None else Path(args.output_root),
        reset_output=not args.no_reset,
    )
    print(
        f"Benchmark example completed: status={summary['stable']['status']} | "
        f"comparison_matches={summary['comparison']['matches'] if summary['comparison'] else 'n/a'} | "
        f"output={DEFAULT_OUTPUT_ROOT.as_posix() if args.output_root is None else Path(args.output_root).as_posix()}"
    )


if __name__ == "__main__":
    main()
