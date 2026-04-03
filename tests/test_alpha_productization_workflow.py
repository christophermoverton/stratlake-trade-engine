from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.cli.run_alpha import AlphaRunResult, run_cli as run_alpha_cli
from src.cli.run_alpha_evaluation import AlphaEvaluationRunResult, run_cli as run_alpha_evaluation_cli
from src.cli.run_portfolio import PortfolioRunResult, run_cli as run_portfolio_cli
from src.research.review import compare_research_runs


def _write_features_daily_alpha_dataset(root: Path) -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", periods=8, freq="D", tz="UTC")
    rows: list[dict[str, object]] = []
    symbols = ["AAA", "BBB", "CCC"]
    for symbol_index, symbol in enumerate(symbols):
        base = float(symbol_index + 1)
        close = 90.0 + base * 10.0
        for ts_index, ts_utc in enumerate(timestamps):
            close += 1.5 + base * 0.4 + ts_index * 0.2
            rows.append(
                {
                    "symbol": symbol,
                    "ts_utc": ts_utc,
                    "timeframe": "1D",
                    "date": ts_utc.strftime("%Y-%m-%d"),
                    "feature_ret_1d": base * 0.04 + ts_index * 0.002,
                    "feature_ret_5d": base * 0.06 + ts_index * 0.003,
                    "feature_ret_20d": base * 0.08 + ts_index * 0.004,
                    "feature_vol_20d": base * 0.01 + ts_index * 0.0005,
                    "feature_close_to_sma20": base * 0.03 + ts_index * 0.001,
                    "target_ret_1d": base * 0.01 + ts_index * 0.001,
                    "target_ret_5d": base * 0.02 + ts_index * 0.0015,
                    "close": close,
                }
            )

    frame = pd.DataFrame(rows).sort_values(["symbol", "ts_utc"], kind="stable").reset_index(drop=True)
    for column in ("symbol", "timeframe", "date"):
        frame[column] = frame[column].astype("string")
    warmup_index = frame.groupby("symbol", sort=False).head(3).index
    frame.loc[warmup_index, "feature_vol_20d"] = None
    frame.loc[warmup_index, "feature_close_to_sma20"] = None
    numeric_columns = [
        "feature_ret_1d",
        "feature_ret_5d",
        "feature_ret_20d",
        "feature_vol_20d",
        "feature_close_to_sma20",
        "target_ret_1d",
        "target_ret_5d",
        "close",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")

    for symbol in symbols:
        symbol_frame = frame.loc[frame["symbol"].eq(symbol)].copy(deep=True)
        dataset_dir = root / "data" / "curated" / "features_daily" / f"symbol={symbol}" / "year=2025"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        symbol_frame.to_parquet(dataset_dir / "part-0.parquet", index=False)
    return frame


def test_config_driven_builtin_alpha_run_uses_catalog_resolution_and_realized_return_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _write_features_daily_alpha_dataset(tmp_path)
    config_path = tmp_path / "alpha_eval.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "alpha_evaluation": {
                    "alpha_name": "cs_linear_ret_1d",
                    "price_column": "",
                    "realized_return_column": "target_ret_1d",
                    "signal_mapping": {
                        "policy": "top_bottom_quantile",
                        "quantile": 0.34,
                        "metadata": {"name": "top_q34"},
                    },
                    "start": "2025-01-01",
                    "end": "2025-01-09",
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    result = run_alpha_evaluation_cli(["--config", str(config_path)])

    assert isinstance(result, AlphaEvaluationRunResult)
    assert result.alpha_name == "cs_linear_ret_1d"
    assert result.resolved_config["dataset"] == "features_daily"
    assert result.resolved_config["target_column"] == "target_ret_1d"
    assert result.resolved_config["feature_columns"] == [
        "feature_ret_1d",
        "feature_ret_5d",
        "feature_ret_20d",
    ]
    assert result.resolved_config["realized_return_column"] == "target_ret_1d"
    assert result.resolved_config["signal_mapping"]["metadata"]["name"] == "top_q34"

    predictions = pd.read_parquet(result.artifact_dir / "predictions.parquet")
    signals = pd.read_parquet(result.artifact_dir / "signals.parquet")
    manifest = json.loads((result.artifact_dir / "manifest.json").read_text(encoding="utf-8"))

    assert not predictions.empty
    assert list(predictions.columns) == ["symbol", "ts_utc", "timeframe", "prediction_score"]
    assert signals["signal"].isin([-1.0, 0.0, 1.0]).all()
    assert manifest["artifact_paths"]["predictions"] == "predictions.parquet"
    assert manifest["artifact_paths"]["signals"] == "signals.parquet"
    assert manifest["artifact_paths"]["signal_mapping"] == "signal_mapping.json"

    ordered_dataset = dataset.sort_values(["symbol", "ts_utc"], kind="stable").reset_index(drop=True)
    last_row_mask = ordered_dataset.groupby("symbol", sort=False).cumcount().eq(
        ordered_dataset.groupby("symbol", sort=False)["symbol"].transform("size").sub(1)
    )
    expected_forward = ordered_dataset.loc[~last_row_mask, "target_ret_1d"].reset_index(drop=True)
    actual_forward = result.aligned_frame["forward_return"].reset_index(drop=True)
    pd.testing.assert_series_equal(
        actual_forward,
        expected_forward,
        check_names=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_full_alpha_workflow_persists_sleeves_loads_portfolio_and_integrates_with_unified_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_features_daily_alpha_dataset(tmp_path)
    portfolio_root = tmp_path / "artifacts" / "portfolios"
    portfolio_config_path = tmp_path / "portfolio.yml"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.cli.run_portfolio.DEFAULT_PORTFOLIO_ARTIFACTS_ROOT", portfolio_root)

    alpha_result = run_alpha_cli(
        [
            "--alpha-name",
            "cs_linear_ret_1d",
            "--signal-policy",
            "top_bottom_quantile",
            "--signal-quantile",
            "0.34",
            "--start",
            "2025-01-01",
            "--end",
            "2025-01-09",
        ]
    )

    assert isinstance(alpha_result, AlphaRunResult)
    assert (alpha_result.artifact_dir / "signals.parquet").exists()
    assert (alpha_result.artifact_dir / "sleeve_returns.csv").exists()
    assert (alpha_result.artifact_dir / "sleeve_equity_curve.csv").exists()
    assert (alpha_result.artifact_dir / "sleeve_metrics.json").exists()
    assert alpha_result.scaffold_path is not None

    portfolio_config_path.write_text(
        yaml.safe_dump(
            {
                "portfolio_name": "alpha_sleeve_portfolio",
                "allocator": "equal_weight",
                "components": [
                    {
                        "run_id": alpha_result.run_id,
                        "artifact_type": "alpha_sleeve",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    portfolio_result = run_portfolio_cli(
        [
            "--portfolio-config",
            str(portfolio_config_path),
            "--timeframe",
            "1D",
        ]
    )

    assert isinstance(portfolio_result, PortfolioRunResult)
    assert [component["artifact_type"] for component in portfolio_result.components] == ["alpha_sleeve"]
    assert portfolio_result.components[0]["run_id"] == alpha_result.run_id
    assert portfolio_result.components[0]["strategy_name"] == "cs_linear_ret_1d"

    review = compare_research_runs(
        run_types=["alpha_evaluation", "portfolio"],
        alpha_artifacts_root=tmp_path / "artifacts" / "alpha",
        portfolio_artifacts_root=portfolio_root,
        output_path=tmp_path / "review",
        emit_plots=False,
    )

    alpha_entry = next(entry for entry in review.entries if entry.run_id == alpha_result.run_id)
    portfolio_entry = next(entry for entry in review.entries if entry.run_id == portfolio_result.run_id)

    assert alpha_entry.mapping_name == "top_bottom_quantile[q=0.34]"
    assert alpha_entry.sleeve_metric_name == "sharpe_ratio"
    assert alpha_entry.linked_portfolio_count == 1
    assert alpha_entry.linked_portfolio_names == "alpha_sleeve_portfolio"
    assert portfolio_entry.entity_name == "alpha_sleeve_portfolio"

    review_payload = json.loads(review.json_path.read_text(encoding="utf-8"))
    alpha_row = next(row for row in review_payload["entries"] if row["run_id"] == alpha_result.run_id)
    assert alpha_row["mapping_name"] == "top_bottom_quantile[q=0.34]"
    assert alpha_row["linked_portfolio_count"] == 1
    assert alpha_row["linked_portfolio_names"] == "alpha_sleeve_portfolio"


def test_run_alpha_accepts_custom_catalog_path_with_custom_alpha_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_features_daily_alpha_dataset(tmp_path)
    custom_catalog_path = tmp_path / "alphas_custom.yml"
    custom_catalog_path.write_text(
        yaml.safe_dump(
            {
                "custom_rank_alpha": {
                    "alpha_name": "custom_rank_alpha",
                    "dataset": "features_daily",
                    "target_column": "target_ret_1d",
                    "feature_columns": [
                        "feature_ret_1d",
                        "feature_ret_5d",
                        "feature_ret_20d",
                    ],
                    "model_type": "rank_composite",
                    "model_params": {
                        "normalize": True,
                        "use_ic_weights": True,
                    },
                    "alpha_horizon": 1,
                    "defaults": {
                        "price_column": "close",
                        "min_cross_section_size": 2,
                    },
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    result = run_alpha_cli(
        [
            "--config",
            str(custom_catalog_path),
            "--alpha-name",
            "custom_rank_alpha",
            "--start",
            "2025-01-01",
            "--end",
            "2025-01-09",
        ]
    )

    assert result.alpha_name == "custom_rank_alpha"
    assert result.resolved_config["alpha_catalog_path"] == str(custom_catalog_path)
    assert result.run_id
    assert (result.artifact_dir / "alpha_metrics.json").exists()
