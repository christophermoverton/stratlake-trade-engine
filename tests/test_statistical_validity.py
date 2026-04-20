from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli.run_strategy import run_cli as run_strategy_cli
from src.config.robustness import RobustnessConfig, load_robustness_config
from src.research.statistical_validity import apply_statistical_validity_controls


def _write_cross_section_dataset(root: Path, periods: int = 60) -> None:
    timestamps = pd.date_range("2025-01-01", periods=periods, freq="D", tz="UTC")
    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    for symbol_index, symbol in enumerate(symbols):
        closes = []
        level = 100.0 + float(symbol_index * 7)
        for index in range(periods):
            drift = (symbol_index - 2) * 0.6 + (0.4 if index % 5 else -0.3)
            level += drift
            closes.append(level)
        close_series = pd.Series(closes, dtype="float64")
        frame = pd.DataFrame(
            {
                "symbol": pd.Series([symbol] * periods, dtype="string"),
                "ts_utc": timestamps,
                "timeframe": pd.Series(["1D"] * periods, dtype="string"),
                "date": pd.Series(timestamps.strftime("%Y-%m-%d"), dtype="string"),
                "close": close_series,
                "feature_ret_1d": close_series.div(close_series.shift(1)).sub(1.0).fillna(0.0),
                "high": close_series + 1.0,
                "low": close_series - 1.0,
                "market_return": pd.Series([0.001] * periods, dtype="float64"),
            }
        )
        dataset_dir = root / "data" / "curated" / "features_daily" / f"symbol={symbol}" / "year=2025"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(dataset_dir / "part-0.parquet", index=False)


def _validity_config(*, validity_ranking_method: str = "adjusted_q_value", batching: bool = False) -> RobustnessConfig:
    payload: dict[str, object] = {
        "sweep": {
            "strategy": {"name": ["cross_section_momentum"]},
        },
        "ranking": {
            "primary_metric": "sharpe_ratio",
            "tie_breakers": ["total_return"],
        },
        "statistical_controls": {
            "multiple_testing_awareness": True,
            "deflated_sharpe_ratio": True,
            "primary_metric": "sharpe_ratio",
            "validity_ranking_method": validity_ranking_method,
            "min_splits_for_inference": 4,
            "dsr_min_observations": 20,
            "low_neighbor_gap_threshold": 0.05,
        },
    }
    if batching:
        payload["sweep"] = {
            "strategy": {"name": ["cross_section_momentum"]},
            "batching": {"batch_size": 2, "batch_index": 0},
        }
    return RobustnessConfig.from_mapping(payload)


def _variant_metrics_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variant_id": "variant_a",
                "variant_order": 0,
                "rank": 1,
                "is_best_variant": True,
                "sharpe_ratio": 1.50,
                "total_return": 0.20,
            },
            {
                "variant_id": "variant_b",
                "variant_order": 1,
                "rank": 2,
                "is_best_variant": False,
                "sharpe_ratio": 1.45,
                "total_return": 0.22,
            },
            {
                "variant_id": "variant_c",
                "variant_order": 2,
                "rank": 3,
                "is_best_variant": False,
                "sharpe_ratio": 1.10,
                "total_return": 0.18,
            },
        ]
    )


def _stability_metrics_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    split_values = {
        "variant_a": [0.60, 0.58, 0.55, 0.62],
        "variant_b": [0.08, -0.02, 0.01, -0.01],
        "variant_c": [0.22, 0.20, 0.18, 0.25],
    }
    for variant_id, values in split_values.items():
        for index, value in enumerate(values):
            rows.append(
                {
                    "variant_id": variant_id,
                    "period_id": f"period_{index:04d}",
                    "sharpe_ratio": value,
                }
            )
    return pd.DataFrame(rows)


def _neighbor_metrics_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "left_variant_id": "variant_a",
                "right_variant_id": "variant_b",
                "metric_gap": 0.03,
            },
            {
                "left_variant_id": "variant_b",
                "right_variant_id": "variant_c",
                "metric_gap": 0.04,
            },
        ]
    )


def _summary_payload() -> dict[str, object]:
    return {
        "threshold_pass_rate": 0.40,
        "rank_consistency_mean": 0.25,
    }


def _child_payloads_for_dsr() -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    base_dates = pd.date_range("2025-01-01", periods=30, freq="D", tz="UTC")
    return_map = {
        "variant_a": [0.012] * 30,
        "variant_b": [0.03, -0.03] * 15,
        "variant_c": [0.009] * 30,
    }
    for variant_id, returns in return_map.items():
        payloads.append(
            {
                "variant": type("Variant", (), {"variant_id": variant_id})(),
                "executed": {
                    "artifact_kind": "strategy",
                    "results_df": pd.DataFrame(
                        {
                            "ts_utc": base_dates,
                            "timeframe": pd.Series(["1D"] * len(returns), dtype="string"),
                            "strategy_return": pd.Series(returns, dtype="float64"),
                        }
                    ),
                },
            }
        )
    return payloads


def test_adjusted_q_values_drive_validity_ranking_and_warning_codes() -> None:
    config = _validity_config(validity_ranking_method="adjusted_q_value")

    result = apply_statistical_validity_controls(
        variant_metrics=_variant_metrics_frame(),
        stability_metrics=_stability_metrics_frame(),
        neighbor_metrics=_neighbor_metrics_frame(),
        child_payloads=[],
        robustness_config=config,
        higher_is_better=True,
        summary=_summary_payload(),
    )

    frame = result.variant_metrics.set_index("variant_id")
    assert result.summary["correction_method_used"] == "fdr_bh"
    assert result.summary["validity_ranking_method_used"] == "adjusted_q_value"
    assert frame.loc["variant_a", "validity_rank"] == 1
    assert frame.loc["variant_c", "validity_rank"] == 2
    assert frame.loc["variant_b", "validity_rank"] == 3
    assert frame.loc["variant_b", "adjusted_q_value"] > config.statistical_controls.fdr_alpha
    assert "fdr_nonpass" in str(frame.loc["variant_b", "validity_warning_codes"])
    assert "low_neighbor_gap" in str(frame.loc["variant_a", "validity_warning_codes"])
    assert result.summary["warning_counts_by_code"]["rank_instability"] == 3
    assert result.summary["warning_counts_by_code"]["low_threshold_pass_rate"] == 3


def test_batching_marks_correction_not_applicable_without_hidden_fallback() -> None:
    config = _validity_config(validity_ranking_method="adjusted_q_value", batching=True)

    result = apply_statistical_validity_controls(
        variant_metrics=_variant_metrics_frame(),
        stability_metrics=_stability_metrics_frame(),
        neighbor_metrics=pd.DataFrame(),
        child_payloads=[],
        robustness_config=config,
        higher_is_better=True,
        summary=_summary_payload(),
    )

    frame = result.variant_metrics
    assert result.summary["correction_method_used"] == "not_applicable"
    assert result.summary["validity_ranking_available"] is False
    assert "batch reassembly" in str(result.summary["validity_ranking_unavailable_reason"])
    assert frame["validity_rank"].isna().all()
    assert frame["validity_warning_codes"].astype("string").str.contains("correction_not_applicable").all()


def test_deflated_sharpe_ratio_can_drive_validity_ranking() -> None:
    config = RobustnessConfig.from_mapping(
        {
            "sweep": {"strategy": {"name": ["cross_section_momentum"]}},
            "ranking": {"primary_metric": "sharpe_ratio", "tie_breakers": ["total_return"]},
            "statistical_controls": {
                "multiple_testing_awareness": False,
                "deflated_sharpe_ratio": True,
                "primary_metric": "sharpe_ratio",
                "validity_ranking_method": "deflated_sharpe_ratio",
                "dsr_min_observations": 20,
            },
        }
    )

    result = apply_statistical_validity_controls(
        variant_metrics=_variant_metrics_frame(),
        stability_metrics=pd.DataFrame(),
        neighbor_metrics=pd.DataFrame(),
        child_payloads=_child_payloads_for_dsr(),
        robustness_config=config,
        higher_is_better=True,
        summary={"threshold_pass_rate": None, "rank_consistency_mean": None},
    )

    frame = result.variant_metrics.set_index("variant_id")
    assert result.summary["validity_ranking_method_used"] == "deflated_sharpe_ratio"
    assert frame["validity_rank"].notna().all()
    ordered = (
        result.variant_metrics.sort_values("validity_rank", kind="stable")
        .loc[:, ["variant_id", "deflated_sharpe_ratio"]]
        .reset_index(drop=True)
    )
    assert ordered["deflated_sharpe_ratio"].tolist() == sorted(
        ordered["deflated_sharpe_ratio"].tolist(),
        reverse=True,
    )


def test_extended_example_config_persists_statistical_validity_artifacts_and_is_stable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_cross_section_dataset(tmp_path, periods=60)
    monkeypatch.chdir(tmp_path)

    example_config = Path(__file__).resolve().parents[1] / "configs" / "robustness_extended_example.yml"
    config = load_robustness_config(example_config)
    assert config.statistical_controls.validity_ranking_method == "adjusted_q_value"
    assert config.statistical_controls.deflated_sharpe_ratio is True

    first = run_strategy_cli(
        [
            "--strategy",
            "cross_section_momentum",
            "--robustness",
            str(example_config),
        ]
    )
    second = run_strategy_cli(
        [
            "--strategy",
            "cross_section_momentum",
            "--robustness",
            str(example_config),
        ]
    )

    assert first.run_id == second.run_id
    first_snapshot = {
        path.relative_to(first.experiment_dir).as_posix(): path.read_bytes()
        for path in sorted(first.experiment_dir.rglob("*"))
        if path.is_file()
    }
    second_snapshot = {
        path.relative_to(second.experiment_dir).as_posix(): path.read_bytes()
        for path in sorted(second.experiment_dir.rglob("*"))
        if path.is_file()
    }
    assert first_snapshot == second_snapshot

    metrics_by_config = pd.read_csv(first.experiment_dir / "metrics_by_config.csv")
    ranked_configs = pd.read_csv(first.experiment_dir / "ranked_configs.csv")
    summary = json.loads((first.experiment_dir / "summary.json").read_text(encoding="utf-8"))
    aggregate = json.loads((first.experiment_dir / "aggregate_metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((first.experiment_dir / "manifest.json").read_text(encoding="utf-8"))

    required_columns = {
        "raw_primary_metric",
        "correction_method",
        "correction_eligible",
        "raw_p_value",
        "adjusted_q_value",
        "deflated_sharpe_ratio",
        "validity_warning_codes",
        "validity_rank",
        "validity_notes",
    }
    assert required_columns.issubset(metrics_by_config.columns)
    assert required_columns.issubset(ranked_configs.columns)
    assert "statistical_validity" in summary
    assert summary["statistical_validity"]["enabled"] is True
    assert summary["statistical_validity"]["family_scope"] == "full_evaluated_sweep"
    assert "statistical_validity" in aggregate
    assert "statistical_validity" in manifest
