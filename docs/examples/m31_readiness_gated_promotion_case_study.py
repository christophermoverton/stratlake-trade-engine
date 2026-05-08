from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.research.promotion import evaluate_promotion_gates, write_promotion_gate_artifact
from src.research.registry import build_review_metadata, canonicalize_value


OUTPUT_DIR = Path("docs/examples/output/m31_readiness_gated_promotion_case_study")
GATE_CONFIG_PATH = REPO_ROOT / "configs" / "statistical_readiness_promotion_gates_example.yml"


SCENARIO_METRICS: dict[str, dict[str, float]] = {
    "eligible": {
        "effective_n": 120.0,
        "p_value": 0.01,
        "hit_rate_p_value": 0.02,
        "autocorr_lag1": 0.05,
        "split_mean_diff_p": 0.40,
        "sharpe_stability_ratio": 1.35,
    },
    "warn": {
        "effective_n": 120.0,
        "p_value": 0.01,
        "hit_rate_p_value": 0.02,
        "autocorr_lag1": 0.05,
        "split_mean_diff_p": 0.01,
        "sharpe_stability_ratio": 0.70,
    },
    "needs_review": {
        "effective_n": 120.0,
        "p_value": 0.12,
        "hit_rate_p_value": 0.20,
        "autocorr_lag1": 0.05,
        "split_mean_diff_p": 0.40,
        "sharpe_stability_ratio": 1.35,
    },
    "blocked": {
        "effective_n": 12.0,
        "p_value": 0.12,
        "hit_rate_p_value": 0.20,
        "autocorr_lag1": 0.05,
        "split_mean_diff_p": 0.01,
        "sharpe_stability_ratio": 0.70,
    },
}


def main() -> int:
    gate_config = _load_gate_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    registry_entries: list[dict[str, Any]] = []
    scenario_summaries: dict[str, dict[str, Any]] = {}
    for scenario_name, metrics in sorted(SCENARIO_METRICS.items()):
        run_id = f"m31_{scenario_name}_strategy"
        run_dir = OUTPUT_DIR / "runs" / scenario_name
        evaluation = evaluate_promotion_gates(
            run_type="strategy",
            config=gate_config,
            sources={"metrics": metrics},
        )
        if evaluation is None:
            raise AssertionError("Expected configured promotion gate evaluation.")

        promotion_gate_path = write_promotion_gate_artifact(run_dir, evaluation)
        if promotion_gate_path is None:
            raise AssertionError("Expected promotion_gates.json to be written.")

        metrics_path = run_dir / "metrics.json"
        _write_json(metrics_path, metrics)

        promotion_summary = evaluation.summary()
        review_metadata = build_review_metadata(
            promotion_status=evaluation.promotion_status,
            promotion_gate_summary=promotion_summary,
        )
        entry = canonicalize_value(
            {
                "run_id": run_id,
                "run_type": "strategy",
                "artifact_path": _relative_to_output(run_dir),
                "metrics_path": _relative_to_output(metrics_path),
                "promotion_gates_path": _relative_to_output(promotion_gate_path),
                "promotion_status": evaluation.promotion_status,
                "review_status": review_metadata["status"],
                "review_metadata": review_metadata,
                "promotion_gate_summary": promotion_summary,
            }
        )
        registry_entries.append(entry)
        scenario_summaries[scenario_name] = entry

    _write_registry_jsonl(OUTPUT_DIR / "registry.jsonl", registry_entries)
    _write_json(OUTPUT_DIR / "review_summary.json", _build_review_summary(registry_entries))
    _write_json(OUTPUT_DIR / "campaign_summary.json", _build_campaign_summary(scenario_summaries))
    _write_json(OUTPUT_DIR / "candidate_review_summary.json", _build_candidate_review_summary(scenario_summaries))
    _write_json(OUTPUT_DIR / "manifest.json", _build_manifest(scenario_summaries))

    print("M31 readiness-gated promotion case study")
    print(f"output_dir: {OUTPUT_DIR.as_posix()}")
    for scenario_name in sorted(scenario_summaries):
        summary = scenario_summaries[scenario_name]["promotion_gate_summary"]
        print(
            f"{scenario_name}: promotion_status={summary['promotion_status']} "
            f"highest_severity={summary['highest_severity']} "
            f"reason_codes={','.join(summary['decision_reason_codes']) or 'none'}"
        )
    return 0


def _load_gate_config() -> dict[str, Any]:
    payload = yaml.safe_load(GATE_CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("promotion_gates"), dict):
        raise AssertionError("Expected promotion_gates mapping in canonical M31 example config.")
    return dict(payload["promotion_gates"])


def _build_review_summary(registry_entries: list[dict[str, Any]]) -> dict[str, Any]:
    return canonicalize_value(
        {
            "run_type": "m31_readiness_review_example",
            "entry_count": len(registry_entries),
            "promotion_status_counts": _count_values(entry["promotion_status"] for entry in registry_entries),
            "review_status_counts": _count_values(entry["review_status"] for entry in registry_entries),
            "entries": registry_entries,
        }
    )


def _build_campaign_summary(scenario_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    review_summary = scenario_summaries["blocked"]["promotion_gate_summary"]
    scenario_matrix = [
        {
            "scenario_id": scenario_name,
            "promotion_status": entry["promotion_status"],
            "review_status": entry["review_status"],
            "highest_severity": entry["promotion_gate_summary"]["highest_severity"],
            "decision_reason_codes": entry["promotion_gate_summary"]["decision_reason_codes"],
        }
        for scenario_name, entry in sorted(scenario_summaries.items())
    ]
    return canonicalize_value(
        {
            "run_type": "research_campaign",
            "campaign_run_id": "m31_readiness_gated_promotion_case_study",
            "status": "completed",
            "scenario_matrix": scenario_matrix,
            "final_outcomes": {
                "review_promotion_status": review_summary["promotion_status"],
                "review_promotion_gate_status": review_summary["evaluation_status"],
                "review_promotion_highest_severity": review_summary["highest_severity"],
                "review_promotion_severity_counts": review_summary["severity_counts"],
                "review_promotion_decision_reason_codes": review_summary["decision_reason_codes"],
                "review_promotion_gate_summary": review_summary,
            },
        }
    )


def _build_candidate_review_summary(scenario_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return canonicalize_value(
        {
            "run_type": "candidate_selection_review",
            "candidate_selection_run_id": "m31_candidate_selection_example",
            "portfolio_run_id": "m31_portfolio_example",
            "promotion_context": {
                "candidate_promotion_status_counts": _count_values(
                    entry["promotion_status"] for entry in scenario_summaries.values()
                ),
                "candidate_review_status_counts": _count_values(
                    entry["review_status"] for entry in scenario_summaries.values()
                ),
                "portfolio_promotion_gate_summary": scenario_summaries["warn"]["promotion_gate_summary"],
            },
        }
    )


def _build_manifest(scenario_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    run_files = [
        f"runs/{scenario_name}/metrics.json"
        for scenario_name in sorted(scenario_summaries)
    ] + [
        f"runs/{scenario_name}/promotion_gates.json"
        for scenario_name in sorted(scenario_summaries)
    ]
    return canonicalize_value(
        {
            "run_type": "m31_readiness_gated_promotion_case_study",
            "artifact_files": sorted(
                [
                    "candidate_review_summary.json",
                    "campaign_summary.json",
                    "manifest.json",
                    "registry.jsonl",
                    "review_summary.json",
                    *run_files,
                ]
            ),
            "example_config": "configs/statistical_readiness_promotion_gates_example.yml",
            "output_dir": OUTPUT_DIR.as_posix(),
            "scenario_count": len(scenario_summaries),
        }
    )


def _count_values(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        text = str(value).strip()
        if text:
            counts[text] = counts.get(text, 0) + 1
    return dict(sorted(counts.items()))


def _relative_to_output(path: Path) -> str:
    return path.relative_to(OUTPUT_DIR).as_posix()


def _write_registry_jsonl(path: Path, entries: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(entry, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for entry in entries),
        encoding="utf-8",
        newline="\n",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(canonicalize_value(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
