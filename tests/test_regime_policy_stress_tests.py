from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.cli.run_regime_policy_stress_tests import run_cli
from src.config.regime_policy_stress_tests import RegimePolicyStressTestConfigError
from src.config.regime_policy_stress_tests import RegimePolicyStressTestConfig
from src.research.regime_policy_stress_tests import run_regime_policy_stress_tests


def _write_review_pack(tmp_path: Path) -> Path:
    review = tmp_path / "review_pack"
    review.mkdir()
    (review / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_type": "regime_review_pack",
                "review_run_id": "review_001",
                "source_benchmark_run_id": "benchmark_001",
            },
            indent=2,
        ),
        encoding="utf-8",
        newline="\n",
    )
    (review / "review_summary.json").write_text(
        json.dumps(
            {
                "review_run_id": "review_001",
                "source_benchmark_run_id": "benchmark_001",
                "decision_counts": {"accepted": 2},
            },
            indent=2,
        ),
        encoding="utf-8",
        newline="\n",
    )
    return review


def _policy_rows() -> list[dict[str, object]]:
    return [
        {
            "policy_name": "static_baseline",
            "policy_type": "static",
            "is_baseline": True,
            "regime_source": "calibrated_taxonomy",
            "classifier_model": "gmm_classifier",
            "calibration_profile": "m26",
            "observation_count": 120,
            "base_return": 0.05,
            "base_sharpe": 0.9,
            "base_max_drawdown": -0.12,
            "base_policy_turnover": 0.18,
            "base_state_change_count": 12,
            "base_fallback_activation_count": 3,
            "base_confidence": 0.67,
            "base_entropy": 0.88,
        },
        {
            "policy_name": "policy_optimized",
            "policy_type": "adaptive",
            "is_baseline": False,
            "regime_source": "calibrated_taxonomy",
            "classifier_model": "gmm_classifier",
            "calibration_profile": "m26",
            "observation_count": 120,
            "base_return": 0.08,
            "base_sharpe": 1.2,
            "base_max_drawdown": -0.10,
            "base_policy_turnover": 0.24,
            "base_state_change_count": 16,
            "base_fallback_activation_count": 5,
            "base_confidence": 0.73,
            "base_entropy": 0.82,
            "scenario_name": "high_vol_persistence",
            "scenario_type": "high_vol_persistence",
            "stress_return": 0.05,
            "stress_sharpe": 1.0,
            "stress_max_drawdown": -0.13,
            "policy_turnover": 0.31,
            "policy_state_change_count": 21,
            "fallback_activation_count": 8,
            "mean_confidence_under_stress": 0.65,
            "mean_entropy_under_stress": 0.98,
        },
        {
            "policy_name": "gmm_calibrated_overlay",
            "policy_type": "adaptive",
            "is_baseline": False,
            "regime_source": "calibrated_taxonomy",
            "classifier_model": "gmm_classifier",
            "calibration_profile": "m26",
            "observation_count": 120,
            "base_return": 0.074,
            "base_sharpe": 1.16,
            "base_max_drawdown": -0.098,
            "base_policy_turnover": 0.21,
            "base_state_change_count": 14,
            "base_fallback_activation_count": 4,
            "base_confidence": 0.75,
            "base_entropy": 0.8,
            "scenario_name": "classifier_uncertainty",
            "scenario_type": "classifier_uncertainty",
            "stress_return": 0.046,
            "stress_sharpe": 0.94,
            "stress_max_drawdown": -0.121,
            "policy_turnover": 0.29,
            "policy_state_change_count": 19,
            "fallback_activation_count": 8,
            "mean_confidence_under_stress": 0.47,
            "mean_entropy_under_stress": 1.19,
        },
    ]


def _write_policy_csv(tmp_path: Path, rows: list[dict[str, object]] | None = None) -> Path:
    path = tmp_path / "policies.csv"
    pd.DataFrame(rows or _policy_rows()).to_csv(path, index=False)
    return path


def _write_policy_json(tmp_path: Path, rows: list[dict[str, object]] | None = None) -> Path:
    path = tmp_path / "policies.json"
    path.write_text(json.dumps({"rows": rows or _policy_rows()}, indent=2), encoding="utf-8", newline="\n")
    return path


def _config(tmp_path: Path, review: Path, policies: Path) -> RegimePolicyStressTestConfig:
    return RegimePolicyStressTestConfig.from_mapping(
        {
            "stress_test_name": "test_stress",
            "source_review_pack": review.as_posix(),
            "source_policy_candidates": {
                "policy_metrics_path": policies.as_posix(),
                "baseline_policy": "static_baseline",
                "candidate_policy_names": ["policy_optimized", "gmm_calibrated_overlay"],
            },
            "source_candidate_selection": "artifacts/candidate_selection/sample_selection_run",
            "regime_context": {
                "preferred_regime_source": "calibrated_taxonomy",
                "ml_overlay": "gmm_classifier",
                "transition_window_bars": 5,
                "confidence_floor": 0.55,
                "entropy_ceiling": 1.10,
            },
            "scenarios": [
                {
                    "name": "transition_shock",
                    "type": "transition_shock",
                    "from_regime": "low_vol",
                    "to_regime": "high_vol",
                    "shock_start": "2026-03-01",
                    "shock_length_bars": 10,
                },
                {
                    "name": "regime_whipsaw",
                    "type": "regime_whipsaw",
                    "regimes": ["low_vol", "high_vol"],
                    "cycle_length_bars": 3,
                    "cycles": 6,
                },
                {
                    "name": "high_vol_persistence",
                    "type": "high_vol_persistence",
                    "target_regime": "high_vol",
                    "persistence_length_bars": 40,
                },
                {
                    "name": "classifier_uncertainty",
                    "type": "classifier_uncertainty",
                    "confidence_multiplier": 0.65,
                    "entropy_multiplier": 1.35,
                },
                {
                    "name": "taxonomy_ml_disagreement",
                    "type": "taxonomy_ml_disagreement",
                    "disagreement_rate": 0.30,
                    "conflict_resolution_modes": ["taxonomy_precedence", "ml_precedence", "fallback"],
                },
                {
                    "name": "confidence_collapse",
                    "type": "confidence_collapse",
                    "confidence_floor_override": 0.25,
                    "collapse_length_bars": 20,
                },
            ],
            "stress_gates": {
                "max_policy_turnover": 0.60,
                "max_stress_drawdown": -0.18,
                "min_adaptive_vs_static_drawdown_delta": -0.03,
                "max_fallback_activation_rate": 0.75,
                "max_state_change_count": 40,
            },
            "output_root": (tmp_path / "stress_outputs").as_posix(),
        }
    )


def test_regime_policy_stress_tests_generates_artifacts(tmp_path: Path) -> None:
    result = run_regime_policy_stress_tests(
        _config(tmp_path, _write_review_pack(tmp_path), _write_policy_csv(tmp_path))
    )

    assert result.stress_matrix_csv_path.exists()
    assert result.stress_matrix_json_path.exists()
    assert result.stress_leaderboard_csv_path.exists()
    assert result.scenario_summary_path.exists()
    assert result.policy_stress_summary_path.exists()
    assert result.scenario_catalog_path.exists()
    assert result.scenario_results_csv_path.exists()
    assert result.fallback_usage_csv_path.exists()
    assert result.policy_turnover_csv_path.exists()
    assert result.adaptive_vs_static_comparison_csv_path.exists()
    assert result.config_path.exists()
    assert result.manifest_path.exists()


def test_regime_policy_stress_tests_leaderboard_includes_static_baseline(tmp_path: Path) -> None:
    result = run_regime_policy_stress_tests(
        _config(tmp_path, _write_review_pack(tmp_path), _write_policy_csv(tmp_path))
    )

    leaderboard = pd.read_csv(result.stress_leaderboard_csv_path)
    assert "static_baseline" in set(leaderboard["policy_name"])


def test_regime_policy_stress_tests_summary_has_baseline_and_adaptive_metadata(tmp_path: Path) -> None:
    result = run_regime_policy_stress_tests(
        _config(tmp_path, _write_review_pack(tmp_path), _write_policy_csv(tmp_path))
    )

    assert result.policy_stress_summary["baseline_policy"] == "static_baseline"
    assert result.policy_stress_summary["baseline_included_in_leaderboard"] is True
    assert result.policy_stress_summary["adaptive_policy_count"] == 2
    assert result.policy_stress_summary["most_resilient_adaptive_policy"] in {
        "policy_optimized",
        "gmm_calibrated_overlay",
    }
    assert result.policy_stress_summary["baseline_rank"] is not None
    assert result.scenario_summary["baseline_included_in_leaderboard"] is True
    assert result.scenario_summary["adaptive_policy_count"] == 2
    assert result.scenario_summary["baseline_rank"] is not None


def test_regime_policy_stress_tests_records_source_candidate_selection_as_provenance(tmp_path: Path) -> None:
    result = run_regime_policy_stress_tests(
        _config(tmp_path, _write_review_pack(tmp_path), _write_policy_csv(tmp_path))
    )

    assert result.manifest["source_paths"]["source_candidate_selection"] == "artifacts/candidate_selection/sample_selection_run"
    assert result.scenario_summary["provenance"]["source_candidate_selection"] == "artifacts/candidate_selection/sample_selection_run"
    assert result.policy_stress_summary["provenance"]["source_candidate_selection"] == "artifacts/candidate_selection/sample_selection_run"


def test_regime_policy_stress_tests_includes_deterministic_transform_limitation_text(tmp_path: Path) -> None:
    result = run_regime_policy_stress_tests(
        _config(tmp_path, _write_review_pack(tmp_path), _write_policy_csv(tmp_path))
    )

    expected = "Derived stress transforms are deterministic approximations, not market simulations."
    assert expected in result.scenario_summary["limitations"]
    assert expected in result.manifest["limitations"]


def test_regime_policy_stress_tests_supports_json_policy_input(tmp_path: Path) -> None:
    result = run_regime_policy_stress_tests(
        _config(tmp_path, _write_review_pack(tmp_path), _write_policy_json(tmp_path))
    )

    assert result.scenario_summary["policy_count"] == 3


def test_regime_policy_stress_tests_missing_required_policy_columns_fails(tmp_path: Path) -> None:
    path = _write_policy_csv(tmp_path, [{"policy_name": "only_name"}])

    with pytest.raises(ValueError, match="missing required fields"):
        run_regime_policy_stress_tests(_config(tmp_path, _write_review_pack(tmp_path), path))


def test_regime_policy_stress_tests_missing_optional_metrics_warns(tmp_path: Path) -> None:
    rows = _policy_rows()
    for row in rows:
        row.pop("base_entropy", None)
    result = run_regime_policy_stress_tests(
        _config(tmp_path, _write_review_pack(tmp_path), _write_policy_csv(tmp_path, rows))
    )

    assert result.manifest["warning_count"] > 0


def test_regime_policy_stress_tests_missing_baseline_fails(tmp_path: Path) -> None:
    cfg = _config(tmp_path, _write_review_pack(tmp_path), _write_policy_csv(tmp_path))
    cfg = RegimePolicyStressTestConfig.from_mapping(
        {
            **cfg.to_dict(),
            "source_policy_candidates": {
                **cfg.to_dict()["source_policy_candidates"],
                "baseline_policy": "missing_baseline",
            },
        }
    )

    with pytest.raises(ValueError, match="Baseline policy"):
        run_regime_policy_stress_tests(cfg)


def test_regime_policy_stress_tests_evaluates_all_scenario_types(tmp_path: Path) -> None:
    result = run_regime_policy_stress_tests(
        _config(tmp_path, _write_review_pack(tmp_path), _write_policy_csv(tmp_path))
    )

    matrix = pd.read_csv(result.stress_matrix_csv_path)
    assert set(matrix["scenario_type"]) == {
        "transition_shock",
        "regime_whipsaw",
        "high_vol_persistence",
        "classifier_uncertainty",
        "taxonomy_ml_disagreement",
        "confidence_collapse",
    }


def test_regime_policy_stress_tests_is_deterministic(tmp_path: Path) -> None:
    config = _config(tmp_path, _write_review_pack(tmp_path), _write_policy_csv(tmp_path))

    first = run_regime_policy_stress_tests(config)
    second = run_regime_policy_stress_tests(config)

    assert first.stress_run_id == second.stress_run_id
    assert first.stress_matrix_csv_path.read_text(encoding="utf-8") == second.stress_matrix_csv_path.read_text(encoding="utf-8")


def test_regime_policy_stress_tests_precomputed_metrics_pass_through(tmp_path: Path) -> None:
    result = run_regime_policy_stress_tests(
        _config(tmp_path, _write_review_pack(tmp_path), _write_policy_csv(tmp_path))
    )

    matrix = pd.read_csv(result.stress_matrix_csv_path)
    row = matrix[
        (matrix["policy_name"] == "policy_optimized")
        & (matrix["scenario_name"] == "high_vol_persistence")
    ].iloc[0]
    assert row["stress_return"] == pytest.approx(0.05)


def test_regime_policy_stress_tests_gate_failures_set_stress_failed_and_reason(tmp_path: Path) -> None:
    cfg = _config(tmp_path, _write_review_pack(tmp_path), _write_policy_csv(tmp_path))
    cfg = RegimePolicyStressTestConfig.from_mapping(
        {
            **cfg.to_dict(),
            "stress_gates": {
                "max_policy_turnover": 0.10,
                "max_stress_drawdown": -0.10,
                "min_adaptive_vs_static_drawdown_delta": 0.20,
                "max_fallback_activation_rate": 0.01,
                "max_state_change_count": 3,
            },
        }
    )

    result = run_regime_policy_stress_tests(cfg)
    matrix = pd.read_csv(result.stress_matrix_csv_path)

    assert (~matrix["stress_passed"]).any()
    assert matrix["primary_failure_reason"].astype(str).str.len().gt(0).any()


def test_regime_policy_stress_tests_cli_smoke(tmp_path: Path) -> None:
    review = _write_review_pack(tmp_path)
    policy = _write_policy_csv(tmp_path)
    config = _config(tmp_path, review, policy)
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8", newline="\n")

    result = run_cli(["--config", config_path.as_posix()])

    assert result.stress_matrix_csv_path.exists()


def test_regime_policy_stress_tests_cli_override_smoke(tmp_path: Path) -> None:
    review = _write_review_pack(tmp_path)
    policy = _write_policy_csv(tmp_path)
    config = _config(tmp_path, review, policy)
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8", newline="\n")

    alt_policy = _write_policy_json(tmp_path)
    output_root = tmp_path / "override_outputs"
    result = run_cli(
        [
            "--config",
            config_path.as_posix(),
            "--policy-metrics-path",
            alt_policy.as_posix(),
            "--output-root",
            output_root.as_posix(),
        ]
    )

    assert str(result.output_dir).startswith(str(output_root))


def test_regime_policy_stress_tests_cli_missing_config_fails() -> None:
    with pytest.raises(RegimePolicyStressTestConfigError):
        run_cli(["--config", "missing_config.yml"])
