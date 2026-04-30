from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.market_simulation import MarketSimulationConfig, load_market_simulation_config
from src.execution.market_simulation import run_market_simulation_scenarios
from src.research.registry import canonicalize_value, stable_timestamp_from_run_id


CASE_STUDY_NAME = "m27_market_simulation_case_study"
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "regime_stress_tests" / f"{CASE_STUDY_NAME}.yml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "docs" / "examples" / "output" / CASE_STUDY_NAME
SOURCE_ARTIFACT_DIR_NAME = "source_simulation_artifacts"
CASE_STUDY_OUTPUTS = {
    "simulation_summary_json": "simulation_summary.json",
    "leaderboard_csv": "leaderboard.csv",
    "case_study_report_md": "case_study_report.md",
    "manifest_json": "manifest.json",
}


def run_example(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    reset_output: bool = True,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    if reset_output and output_root.exists():
        _reset_output_root(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    source_output_root = output_root / SOURCE_ARTIFACT_DIR_NAME
    config = _case_study_config(config_path=config_path, output_root=source_output_root)
    result = run_market_simulation_scenarios(
        config_path=_write_runtime_config(source_output_root, config)
    ).raw_result
    metrics = result.simulation_stress_metrics_result
    if metrics is None:
        raise RuntimeError("The M27 case study requires stress_metrics.enabled=true.")

    leaderboard_rows = _read_csv(metrics.leaderboard_path)
    summary_rows = _read_csv(metrics.summary_path)
    path_metric_rows = _read_csv(metrics.path_metrics_path)

    leaderboard_path = output_root / CASE_STUDY_OUTPUTS["leaderboard_csv"]
    _write_csv(leaderboard_path, leaderboard_rows)

    simulation_summary = _simulation_summary(
        result=result,
        summary_rows=summary_rows,
        leaderboard_rows=leaderboard_rows,
        path_metric_rows=path_metric_rows,
    )
    _write_json(output_root / CASE_STUDY_OUTPUTS["simulation_summary_json"], simulation_summary)

    manifest = _manifest(
        config_path=config_path,
        output_root=output_root,
        result=result,
        simulation_summary=simulation_summary,
    )
    _write_json(output_root / CASE_STUDY_OUTPUTS["manifest_json"], manifest)

    report = _report(
        result=result,
        simulation_summary=simulation_summary,
        leaderboard_rows=leaderboard_rows,
        summary_rows=summary_rows,
        path_metric_rows=path_metric_rows,
        manifest=manifest,
    )
    (output_root / CASE_STUDY_OUTPUTS["case_study_report_md"]).write_text(
        report,
        encoding="utf-8",
        newline="\n",
    )
    return simulation_summary


def _case_study_config(*, config_path: Path, output_root: Path) -> MarketSimulationConfig:
    loaded = load_market_simulation_config(config_path)
    payload = loaded.to_dict()
    payload["output_root"] = _relative_path(output_root)
    return MarketSimulationConfig.from_mapping(payload)


def _write_runtime_config(output_root: Path, config: MarketSimulationConfig) -> Path:
    path = output_root / "_runtime_market_simulation_config.json"
    _write_json(path, config.to_dict())
    return path


def _simulation_summary(
    *,
    result: Any,
    summary_rows: list[dict[str, str]],
    leaderboard_rows: list[dict[str, str]],
    path_metric_rows: list[dict[str, str]],
) -> dict[str, Any]:
    best = leaderboard_rows[0] if leaderboard_rows else {}
    worst = leaderboard_rows[-1] if leaderboard_rows else {}
    metrics = result.simulation_stress_metrics_result
    simulation_types = sorted({row["simulation_type"] for row in summary_rows if row.get("simulation_type")})
    return canonicalize_value(
        {
            "case_study_name": CASE_STUDY_NAME,
            "milestone": "M27",
            "source_simulation_run_id": result.simulation_run_id,
            "scenario_count": int(result.simulation_manifest.get("scenario_count", 0)),
            "path_metric_row_count": metrics.path_metric_row_count,
            "summary_row_count": metrics.summary_row_count,
            "leaderboard_row_count": metrics.leaderboard_row_count,
            "simulation_types": simulation_types,
            "best_ranked_scenario": _ranked_scenario(best),
            "worst_ranked_scenario": _ranked_scenario(worst),
            "policy_failure_rate": metrics.policy_failure_rate,
            "regime_only_monte_carlo_note": (
                "Monte Carlo paths are regime-only and do not fabricate return or policy metrics."
            ),
            "metric_availability": {
                "return_metric_path_rows": _true_count(path_metric_rows, "has_return_metrics"),
                "policy_metric_path_rows": _true_count(path_metric_rows, "has_policy_metrics"),
                "regime_metric_path_rows": _true_count(path_metric_rows, "has_regime_metrics"),
            },
            "limitations": [
                "Fixture-backed examples validate deterministic stress-test workflows only.",
                "Scenario metrics are generated from checked-in fixtures and configured overlays, not live data.",
                "The leaderboard ranks configured stress artifacts and does not forecast future returns.",
                "Regime-transition Monte Carlo contributes regime metrics only; returns and policy outcomes are not inferred.",
            ],
        }
    )


def _ranked_scenario(row: Mapping[str, str]) -> dict[str, Any]:
    return {
        "scenario_name": row.get("scenario_name", ""),
        "simulation_type": row.get("simulation_type", ""),
        "ranking_metric": row.get("ranking_metric", ""),
        "ranking_value": _float(row.get("ranking_value"), default=0.0),
    }


def _manifest(
    *,
    config_path: Path,
    output_root: Path,
    result: Any,
    simulation_summary: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = result.simulation_stress_metrics_result
    source_paths = {
        "scenario_catalog_csv": result.scenario_catalog_csv_path,
        "scenario_catalog_json": result.scenario_catalog_json_path,
        "simulation_config_json": result.simulation_config_path,
        "input_inventory_json": result.input_inventory_path,
        "simulation_manifest_json": result.simulation_manifest_path,
        "simulation_path_metrics_csv": metrics.path_metrics_path,
        "simulation_summary_csv": metrics.summary_path,
        "simulation_leaderboard_csv": metrics.leaderboard_path,
        "simulation_metrics_manifest_json": metrics.manifest_path,
    }
    source_paths.update(
        {
            f"historical_replay_{index}_manifest_json": replay.manifest_path
            for index, replay in enumerate(result.historical_episode_replay_results, start=1)
        }
    )
    source_paths.update(
        {
            f"block_bootstrap_{index}_manifest_json": bootstrap.manifest_path
            for index, bootstrap in enumerate(result.block_bootstrap_results, start=1)
        }
    )
    source_paths.update(
        {
            f"monte_carlo_{index}_manifest_json": monte_carlo.manifest_path
            for index, monte_carlo in enumerate(result.monte_carlo_results, start=1)
        }
    )
    source_paths.update(
        {
            f"shock_overlay_{index}_manifest_json": overlay.manifest_path
            for index, overlay in enumerate(result.shock_overlay_results, start=1)
        }
    )
    return canonicalize_value(
        {
            "artifact_type": "market_simulation_case_study",
            "case_study_name": CASE_STUDY_NAME,
            "schema_version": "1.0",
            "generated_timestamp": stable_timestamp_from_run_id(result.simulation_run_id),
            "source_config_path": _relative_path(config_path),
            "source_simulation_run_id": result.simulation_run_id,
            "source_simulation_artifact_paths": {
                key: _relative_path(path) for key, path in sorted(source_paths.items())
            },
            "output_artifact_paths": {
                key: _relative_path(output_root / filename)
                for key, filename in CASE_STUDY_OUTPUTS.items()
            },
            "row_counts": {
                "scenario_catalog": simulation_summary["scenario_count"],
                "simulation_path_metrics": simulation_summary["path_metric_row_count"],
                "simulation_summary": simulation_summary["summary_row_count"],
                "leaderboard": simulation_summary["leaderboard_row_count"],
            },
            "limitations": simulation_summary["limitations"],
        }
    )


def _report(
    *,
    result: Any,
    simulation_summary: Mapping[str, Any],
    leaderboard_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
    path_metric_rows: list[dict[str, str]],
    manifest: Mapping[str, Any],
) -> str:
    best = simulation_summary["best_ranked_scenario"]
    worst = simulation_summary["worst_ranked_scenario"]
    types = ", ".join(simulation_summary["simulation_types"])
    leaderboard_preview = _leaderboard_preview(leaderboard_rows)
    scenario_lines = "\n".join(
        f"- `{row['scenario_name']}` ({row['simulation_type']}): "
        f"{row['path_count']} path row(s), policy failure rate {row['policy_failure_rate'] or '0.0'}."
        for row in summary_rows
    )
    metric_table = _scenario_metric_table(summary_rows)
    path_table = _path_metric_table(path_metric_rows)
    failure_table = _failure_table(path_metric_rows)
    return (
        "# M27 Market Simulation Stress Testing Case Study\n\n"
        "## Research Question\n\n"
        "How does one adaptive policy evaluation workflow behave when the same fixture-backed "
        "policy evidence is passed through historical replay, empirical bootstrap paths, "
        "regime-only Monte Carlo paths, deterministic shock overlays, and simulation-aware metrics?\n\n"
        "## Scenario Methods\n\n"
        f"The case study ran `{result.simulation_run_id}` with {simulation_summary['scenario_count']} "
        f"configured scenarios covering: {types}.\n\n"
        f"{scenario_lines}\n\n"
        "## Artifact Flow\n\n"
        "The script calls the existing market simulation framework, then stitches the framework "
        "artifacts into a compact case-study output directory. Source artifacts remain under "
        f"`{SOURCE_ARTIFACT_DIR_NAME}/`; the root directory keeps only the summary, leaderboard, "
        "report, and manifest.\n\n"
        "## Generated Outputs\n\n"
        "- `simulation_summary.json`\n"
        "- `leaderboard.csv`\n"
        "- `case_study_report.md`\n"
        "- `manifest.json`\n\n"
        "## Leaderboard Interpretation\n\n"
        f"The leaderboard ranks scenarios by `{best['ranking_metric']}`. The best-ranked scenario is "
        f"`{best['scenario_name']}` ({best['simulation_type']}) with value "
        f"{best['ranking_value']:.6f}. The worst-ranked scenario is `{worst['scenario_name']}` "
        f"({worst['simulation_type']}) with value {worst['ranking_value']:.6f}.\n\n"
        f"{leaderboard_preview}\n\n"
        "## Scenario Metric Summary\n\n"
        "This table shows the scenario-level metrics used by the leaderboard and diagnostics. "
        "Blank values indicate that the source artifact does not support that metric family.\n\n"
        f"{metric_table}\n\n"
        "## Path-Level Diagnostics\n\n"
        "The metrics layer emits one row per replay episode, overlay episode, bootstrap path, "
        "or Monte Carlo regime path. This compact view keeps the most actionable columns.\n\n"
        f"{path_table}\n\n"
        "## Failure Diagnostics\n\n"
        f"{failure_table}\n\n"
        "## Historical Replay Interpretation\n\n"
        "Historical replay uses the configured fixture episode rows as-is and compares adaptive "
        "policy returns with the static baseline where source columns are available. This is a "
        "deterministic replay of checked-in fixture data, not evidence of future performance.\n\n"
        "## Block Bootstrap Interpretation\n\n"
        "The regime-aware block bootstrap resamples observed contiguous fixture blocks into "
        "deterministic empirical return paths. These paths preserve provenance to source blocks "
        "and support return, drawdown, tail, and regime-transition metrics.\n\n"
        "## Monte Carlo Interpretation\n\n"
        "The regime-transition Monte Carlo scenario generates regime-label paths only. It contributes "
        "transition and stress-regime-share metrics, and intentionally leaves return and policy "
        "metrics unavailable instead of fabricating them.\n\n"
        "## Shock Overlay Interpretation\n\n"
        "The shock overlay applies configured deterministic return and confidence transformations "
        "to the historical replay output. It shows how overlay artifacts feed the same metrics layer "
        "without introducing live data or forecasting assumptions.\n\n"
        "## Simulation-Aware Metrics Interpretation\n\n"
        f"The metrics layer produced {simulation_summary['path_metric_row_count']} path metric rows, "
        f"{simulation_summary['summary_row_count']} scenario summary rows, and "
        f"{simulation_summary['leaderboard_row_count']} leaderboard rows. The aggregate policy "
        f"failure rate is {simulation_summary['policy_failure_rate']:.6f} under the configured "
        "thresholds.\n\n"
        "## Limitations\n\n"
        "- Fixture-backed examples validate workflow plumbing and artifact contracts only.\n"
        "- Simulated paths do not forecast future returns or recommend trades.\n"
        "- Monte Carlo outputs are regime-only and do not contain synthetic returns.\n"
        "- Policy metrics depend on source artifacts that include adaptive and static policy columns.\n\n"
        "## Recommended Follow-Ups\n\n"
        "- Add larger fixture packs that exercise more regime transitions while staying CI-safe.\n"
        "- Extend overlays to consume file-backed or bootstrap outputs when those framework contracts land.\n"
        "- Compare multiple candidate policies once source review-pack integration is available.\n\n"
        "## Manifest\n\n"
        f"See `{manifest['output_artifact_paths']['manifest_json']}` for relative source and output paths.\n"
    )


def _leaderboard_preview(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "No leaderboard rows were generated."
    lines = ["| Rank | Scenario | Type | Ranking Value | Decision |", "| --- | --- | --- | ---: | --- |"]
    for row in rows:
        lines.append(
            f"| {row['rank']} | `{row['scenario_name']}` | `{row['simulation_type']}` | "
            f"{_float(row.get('ranking_value'), default=0.0):.6f} | `{row['decision_label']}` |"
        )
    return "\n".join(lines)


def _scenario_metric_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "No scenario summary rows were generated."
    lines = [
        "| Scenario | Return rows | Policy rows | Regime rows | Mean total return | "
        "Worst drawdown | Mean stress share | Mean stress score | Notes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['scenario_name']}` | {row['paths_with_return_metrics']} | "
            f"{row['paths_with_policy_metrics']} | {row['paths_with_regime_metrics']} | "
            f"{_fmt(row.get('mean_total_return'))} | {_fmt(row.get('worst_max_drawdown'))} | "
            f"{_fmt(row.get('mean_stress_regime_share'))} | {_fmt(row.get('mean_stress_score'))} | "
            f"`{row.get('notes') or 'available'}` |"
        )
    return "\n".join(lines)


def _path_metric_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "No path metric rows were generated."
    lines = [
        "| Scenario | Path/Episode | Total return | Max drawdown | Regime transitions | "
        "Stress share | Failure | Reason | Stress score |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |",
    ]
    for row in rows:
        path_label = row.get("episode_id") or row.get("path_id") or f"path_{row.get('path_index', '')}"
        lines.append(
            f"| `{row['scenario_name']}` | `{path_label}` | {_fmt(row.get('total_return'))} | "
            f"{_fmt(row.get('max_drawdown'))} | {_fmt(row.get('regime_transition_count'), decimals=0)} | "
            f"{_fmt(row.get('stress_regime_share'))} | `{row.get('policy_failure')}` | "
            f"`{row.get('failure_reason') or 'none'}` | {_fmt(row.get('stress_score'))} |"
        )
    return "\n".join(lines)


def _failure_table(rows: list[dict[str, str]]) -> str:
    reason_counts: dict[str, int] = {}
    failed = 0
    for row in rows:
        if str(row.get("policy_failure", "")).lower() == "true":
            failed += 1
        for reason in str(row.get("failure_reason") or "").split(";"):
            if reason:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
    if not reason_counts:
        return "No configured policy failure threshold was breached."
    lines = [
        f"{failed} of {len(rows)} path metric rows breached at least one configured threshold.",
        "",
        "| Failure reason | Count |",
        "| --- | ---: |",
    ]
    for reason, count in sorted(reason_counts.items()):
        lines.append(f"| `{reason}` | {count} |")
    return "\n".join(lines)


def _fmt(value: Any, *, decimals: int = 6) -> str:
    if value in {None, ""}:
        return ""
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(canonicalize_value(payload), indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )


def _true_count(rows: Iterable[Mapping[str, str]], key: str) -> int:
    return sum(str(row.get(key, "")).strip().lower() == "true" for row in rows)


def _float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _relative_path(path: str | Path) -> str:
    resolved = Path(path)
    if not resolved.is_absolute():
        return resolved.as_posix()
    try:
        return resolved.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.name


def _reset_output_root(path: Path) -> None:
    resolved = path.resolve()
    allowed_root = (REPO_ROOT / "docs" / "examples" / "output").resolve()
    if allowed_root not in [resolved, *resolved.parents]:
        raise ValueError(f"Refusing to reset output outside docs/examples/output: {_relative_path(resolved)}")
    shutil.rmtree(resolved)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the M27 market simulation case study.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH.as_posix(), help="Case-study config path.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT.as_posix(), help="Output directory.")
    parser.add_argument("--no-reset", action="store_true", help="Preserve the existing output directory.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_example(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        reset_output=not args.no_reset,
    )
    print(
        "M27 market simulation case study completed: "
        f"run_id={summary['source_simulation_run_id']} | "
        f"scenarios={summary['scenario_count']} | "
        f"leaderboard_rows={summary['leaderboard_row_count']} | "
        f"output={_relative_path(Path(args.output_root))}"
    )


if __name__ == "__main__":
    main()
