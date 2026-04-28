from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli.run_regime_promotion_gates import run_cli
from src.config.regime_promotion_gates import load_regime_promotion_gate_config
from src.research.regime_promotion_gates import run_regime_promotion_gates


def _write_benchmark(
    tmp_path: Path,
    rows: list[dict[str, object]],
    *,
    transition_rows: list[dict[str, object]] | None = None,
    policy_rows: list[dict[str, object]] | None = None,
) -> Path:
    benchmark_path = tmp_path / "artifacts" / "regime_benchmarks" / "test_run"
    benchmark_path.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(benchmark_path / "benchmark_matrix.csv", index=False)
    (benchmark_path / "benchmark_summary.json").write_text(
        json.dumps(
            {
                "benchmark_name": "test_benchmark",
                "benchmark_run_id": "test_run",
                "matrix_row_count": len(rows),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
        newline="\n",
    )
    (benchmark_path / "artifact_provenance.json").write_text(
        json.dumps({"benchmark_run_id": "test_run"}, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    if transition_rows is not None:
        (benchmark_path / "transition_summary.json").write_text(
            json.dumps({"benchmark_run_id": "test_run", "variants": transition_rows}, indent=2),
            encoding="utf-8",
            newline="\n",
        )
    if policy_rows is not None:
        pd.DataFrame(policy_rows).to_csv(benchmark_path / "policy_comparison.csv", index=False)
    return benchmark_path


def _base_row(name: str, variant_type: str, regime_source: str) -> dict[str, object]:
    return {
        "benchmark_run_id": "test_run",
        "variant_name": name,
        "variant_type": variant_type,
        "regime_source": regime_source,
        "calibration_profile": None,
        "classifier_model": None,
        "policy_name": None,
        "is_static_baseline": variant_type == "static_baseline",
        "observation_count": 100,
        "avg_regime_duration": None,
        "dominant_regime_share": None,
        "transition_rate": None,
        "mean_confidence": None,
        "median_confidence": None,
        "high_entropy_observation_count": None,
        "low_confidence_observation_count": None,
        "mean_entropy": None,
        "policy_turnover": None,
        "policy_state_change_count": None,
        "adaptive_vs_static_return_delta": None,
        "adaptive_vs_static_sharpe_delta": None,
        "adaptive_vs_static_max_drawdown_delta": None,
        "max_drawdown": -0.10,
        "source_artifact_path": f"variants/{name}/manifest.json",
        "warnings": "",
    }


def _gmm_row(name: str = "gmm_classifier", *, mean_confidence: float = 0.70) -> dict[str, object]:
    row = _base_row(name, "gmm_classifier", "gmm")
    row.update(
        {
            "classifier_model": "gaussian_mixture_model",
            "avg_regime_duration": 8.0,
            "dominant_regime_share": 0.60,
            "transition_rate": 0.10,
            "mean_confidence": mean_confidence,
            "median_confidence": 0.72,
            "mean_entropy": 0.55,
            "high_entropy_observation_count": 5,
            "low_confidence_observation_count": 10,
        }
    )
    return row


def _policy_row(
    *,
    return_delta: float = 0.03,
    sharpe_delta: float = 0.10,
    drawdown_delta: float = 0.01,
    turnover: float | None = 0.20,
) -> dict[str, object]:
    row = _base_row("policy_optimized", "policy_optimized", "policy")
    row.update(
        {
            "calibration_profile": "baseline",
            "classifier_model": "gaussian_mixture_model",
            "policy_name": "m25_policy_default",
            "avg_regime_duration": 8.0,
            "dominant_regime_share": 0.60,
            "transition_rate": 0.10,
            "mean_confidence": 0.70,
            "median_confidence": 0.72,
            "mean_entropy": 0.55,
            "high_entropy_observation_count": 5,
            "low_confidence_observation_count": 10,
            "policy_turnover": turnover,
            "policy_state_change_count": 6,
            "adaptive_vs_static_return_delta": return_delta,
            "adaptive_vs_static_sharpe_delta": sharpe_delta,
            "adaptive_vs_static_max_drawdown_delta": drawdown_delta,
        }
    )
    return row


def _write_config(
    tmp_path: Path,
    *,
    optional_missing: str = "warn",
    transition_failure_impact: str = "fail",
    include_turnover: bool = True,
) -> Path:
    payload = {
        "gate_name": "test_regime_policy_governance",
        "decision_policy": "strict_with_warnings",
        "missing_metric_policy": {
            "required_metric_missing": "fail",
            "optional_metric_missing": optional_missing,
            "not_applicable_metric": "ignore",
        },
        "confidence": {
            "enabled": True,
            "required_for_variants": ["gmm_classifier", "policy_optimized"],
            "min_mean_confidence": 0.55,
            "min_median_confidence": 0.60,
            "max_low_confidence_pct": 0.25,
        },
        "entropy": {
            "enabled": True,
            "required_for_variants": ["gmm_classifier", "policy_optimized"],
            "max_mean_entropy": 1.10,
            "max_high_entropy_pct": 0.20,
        },
        "stability": {
            "enabled": True,
            "min_avg_regime_duration": 3,
            "max_dominant_regime_share": 0.85,
            "max_transition_rate": 0.35,
        },
        "transition_behavior": {
            "enabled": True,
            "failure_impact": transition_failure_impact,
            "max_transition_rate": 0.35,
            "max_transition_instability_score": 0.40,
            "max_transition_concentration": 0.75,
        },
        "adaptive_uplift": {
            "enabled": True,
            "required_for_variants": ["policy_optimized"],
            "min_return_delta": 0.00,
            "min_sharpe_delta": 0.00,
            "min_drawdown_delta": -0.02,
        },
        "drawdown": {
            "enabled": True,
            "max_allowed_drawdown": -0.25,
            "max_high_vol_drawdown": -0.20,
        },
        "policy_turnover": {
            "enabled": include_turnover,
            "required_for_variants": ["policy_optimized"],
            "max_policy_turnover": 0.50,
            "max_policy_state_changes": 30,
        },
    }
    path = tmp_path / "gates.yml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8", newline="\n")
    return path


def _transition_rows(*, instability: float = 0.10) -> list[dict[str, object]]:
    return [
        {
            "variant_name": "gmm_classifier",
            "transition_instability_score": instability,
            "transition_concentration": 0.30,
        },
        {
            "variant_name": "policy_optimized",
            "transition_instability_score": 0.10,
            "transition_concentration": 0.30,
        },
    ]


def _policy_comparison_rows(*, high_vol_drawdown: float = -0.10) -> list[dict[str, object]]:
    return [
        {
            "variant_name": "policy_optimized",
            "policy_name": "m25_policy_default",
            "high_volatility_regime_drawdown": high_vol_drawdown,
        }
    ]


def test_regime_promotion_gates_accepts_clean_variants(tmp_path: Path) -> None:
    benchmark_path = _write_benchmark(
        tmp_path,
        [_gmm_row(), _policy_row()],
        transition_rows=_transition_rows(),
        policy_rows=_policy_comparison_rows(),
    )
    config = load_regime_promotion_gate_config(_write_config(tmp_path))

    result = run_regime_promotion_gates(benchmark_path, config)

    assert result.decision_summary["decision_counts"]["accepted"] == 2
    assert result.decision_summary["decision_counts"]["rejected"] == 0
    assert result.gate_results_csv_path.exists()
    assert result.gate_results_json_path.exists()


def test_regime_promotion_gates_warns_for_optional_missing_metric(tmp_path: Path) -> None:
    benchmark_path = _write_benchmark(
        tmp_path,
        [_gmm_row()],
    )
    config = load_regime_promotion_gate_config(_write_config(tmp_path))

    result = run_regime_promotion_gates(benchmark_path, config)

    assert result.decision_summary["decision_counts"]["accepted_with_warnings"] == 1
    warning_rows = pd.read_csv(result.warning_gates_csv_path)
    assert "transition_instability_score" in warning_rows["metric_name"].tolist()


def test_regime_promotion_gates_can_emit_needs_review(tmp_path: Path) -> None:
    benchmark_path = _write_benchmark(
        tmp_path,
        [_gmm_row()],
        transition_rows=_transition_rows(instability=0.60),
        policy_rows=_policy_comparison_rows(),
    )
    config = load_regime_promotion_gate_config(
        _write_config(tmp_path, optional_missing="ignore", transition_failure_impact="review")
    )

    result = run_regime_promotion_gates(benchmark_path, config)

    assert result.decision_summary["decision_counts"]["needs_review"] == 1
    assert result.decision_summary["variant_decisions"][0]["review_gates"] == 1


def test_regime_promotion_gates_rejects_failed_threshold(tmp_path: Path) -> None:
    benchmark_path = _write_benchmark(
        tmp_path,
        [_gmm_row(mean_confidence=0.40)],
        transition_rows=_transition_rows(),
        policy_rows=_policy_comparison_rows(),
    )
    config = load_regime_promotion_gate_config(_write_config(tmp_path, optional_missing="ignore"))

    result = run_regime_promotion_gates(benchmark_path, config)

    assert result.decision_summary["decision_counts"]["rejected"] == 1
    failed_rows = pd.read_csv(result.failed_gates_csv_path)
    assert "mean_confidence" in failed_rows["metric_name"].tolist()


def test_regime_promotion_gates_rejects_missing_required_metric(tmp_path: Path) -> None:
    policy = _policy_row(turnover=None)
    benchmark_path = _write_benchmark(
        tmp_path,
        [policy],
        transition_rows=_transition_rows(),
        policy_rows=_policy_comparison_rows(),
    )
    config = load_regime_promotion_gate_config(_write_config(tmp_path))

    result = run_regime_promotion_gates(benchmark_path, config)

    assert result.decision_summary["decision_counts"]["rejected"] == 1
    missing_rows = pd.read_csv(result.gate_results_csv_path)
    assert (
        missing_rows.loc[missing_rows["metric_name"].eq("policy_turnover"), "status"].iloc[0]
        == "metric_missing"
    )


def test_regime_promotion_gates_preserves_not_applicable_metrics(tmp_path: Path) -> None:
    static = _base_row("static_baseline", "static_baseline", "static")
    benchmark_path = _write_benchmark(tmp_path, [static])
    config = load_regime_promotion_gate_config(_write_config(tmp_path))

    result = run_regime_promotion_gates(benchmark_path, config)

    gate_rows = pd.read_csv(result.gate_results_csv_path)
    assert "not_applicable" in gate_rows["status"].tolist()
    assert result.decision_summary["decision_counts"]["accepted"] == 1


def test_regime_promotion_gates_preserves_deterministic_variant_order(tmp_path: Path) -> None:
    benchmark_path = _write_benchmark(
        tmp_path,
        [_policy_row(), _gmm_row()],
        transition_rows=list(reversed(_transition_rows())),
        policy_rows=_policy_comparison_rows(),
    )
    config = load_regime_promotion_gate_config(_write_config(tmp_path))

    first = run_regime_promotion_gates(benchmark_path, config)
    second = run_regime_promotion_gates(benchmark_path, config)

    assert [
        item["variant_name"]
        for item in first.decision_summary["variant_decisions"]
    ] == ["policy_optimized", "gmm_classifier"]
    assert first.gate_results_csv_path.read_text(encoding="utf-8") == second.gate_results_csv_path.read_text(
        encoding="utf-8"
    )


def test_regime_promotion_gates_manifest_and_summary_counts(tmp_path: Path) -> None:
    benchmark_path = _write_benchmark(
        tmp_path,
        [_gmm_row(mean_confidence=0.40), _policy_row()],
        transition_rows=_transition_rows(),
        policy_rows=_policy_comparison_rows(),
    )
    config = load_regime_promotion_gate_config(_write_config(tmp_path, optional_missing="ignore"))

    result = run_regime_promotion_gates(benchmark_path, config)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(result.decision_summary_path.read_text(encoding="utf-8"))

    assert manifest["row_counts"]["gate_results"] > 0
    assert manifest["row_counts"]["failed_gates"] == 1
    assert summary["decision_counts"]["accepted"] == 1
    assert summary["decision_counts"]["rejected"] == 1


def test_regime_promotion_gate_cli_smoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    benchmark_path = _write_benchmark(
        tmp_path,
        [_gmm_row()],
        transition_rows=_transition_rows(),
        policy_rows=_policy_comparison_rows(),
    )
    config_path = _write_config(tmp_path, optional_missing="ignore")

    result = run_cli(["--benchmark-path", benchmark_path.as_posix(), "--config", config_path.as_posix()])
    captured = capsys.readouterr()

    assert "Regime Promotion Gate Summary" in captured.out
    assert "decision_summary" in captured.out
    assert result.decision_summary_path.exists()


def test_regime_promotion_gate_cli_missing_benchmark_matrix_failure(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "missing_matrix"
    benchmark_path.mkdir()
    config_path = _write_config(tmp_path)

    with pytest.raises(FileNotFoundError, match="benchmark_matrix"):
        run_cli(["--benchmark-path", benchmark_path.as_posix(), "--config", config_path.as_posix()])


def test_regime_promotion_gate_cli_missing_config_failure(tmp_path: Path) -> None:
    benchmark_path = _write_benchmark(tmp_path, [_gmm_row()])

    with pytest.raises(Exception, match="does not exist"):
        run_cli(
            [
                "--benchmark-path",
                benchmark_path.as_posix(),
                "--config",
                (tmp_path / "missing.yml").as_posix(),
            ]
        )
