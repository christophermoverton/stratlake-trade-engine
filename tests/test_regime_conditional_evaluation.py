"""Tests for M24.3 — Regime-conditional evaluation (strategy, alpha, portfolio).

Coverage:
- Unit tests for each surface (strategy, alpha, portfolio)
- Deterministic output tests
- Sparse / empty / undefined / unmatched regime handling
- Artifact persistence and loading (metrics_by_regime.csv, regime_conditional_summary.json)
- Schema and stable-ordering tests
- Backward-compatibility: existing aggregate metrics are unaffected
- Alignment semantics: only matched_defined rows contribute
- evaluate_all_dimensions convenience wrapper
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.regimes import (
    METRICS_BY_REGIME_FILENAME,
    REGIME_CONDITIONAL_SUMMARY_FILENAME,
    RegimeAlignmentConfig,
    RegimeConditionalConfig,
    RegimeConditionalResult,
    align_regime_labels,
    evaluate_all_dimensions,
    evaluate_alpha_metrics_by_regime,
    evaluate_portfolio_metrics_by_regime,
    evaluate_strategy_metrics_by_regime,
    load_regime_conditional_manifest,
    load_regime_conditional_metrics,
    load_regime_conditional_summary,
    resolve_regime_conditional_config,
    write_regime_conditional_artifacts,
    write_regime_conditional_artifacts_multi_dimension,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_NUM_DAYS = 60

# Two distinct regime labels for testing multi-group behavior.
_REGIME_LABELS_POOL = [
    "volatility=low_volatility|trend=uptrend|drawdown_recovery=near_peak|stress=low_stress",
    "volatility=high_volatility|trend=downtrend|drawdown_recovery=underwater|stress=high_stress",
]
_UNDEF_LABEL = (
    "volatility=undefined|trend=undefined|drawdown_recovery=undefined|stress=undefined"
)


def _build_aligned_frame(
    data_columns: dict,
    *,
    n: int = _NUM_DAYS,
    n_undefined: int = 4,
    n_unmatched: int = 2,
    surface: str = "strategy",
) -> pd.DataFrame:
    """Build an aligned frame with explicit regime columns.

    Rows 0..n_undefined-1               -> matched_undefined
    Rows n_undefined..+n_unmatched-1    -> unmatched_timestamp
    Remaining rows                      -> matched_defined (alternating between two labels)
    """
    ts = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    frame = pd.DataFrame({"ts_utc": ts})
    for col, values in data_columns.items():
        frame[col] = values

    statuses = []
    regime_labels_col = []
    volatility_state_col = []
    trend_state_col = []
    drawdown_recovery_state_col = []
    stress_state_col = []
    is_defined_col = []
    has_match_col = []

    for i in range(n):
        if i < n_undefined:
            statuses.append("matched_undefined")
            regime_labels_col.append(_UNDEF_LABEL)
            volatility_state_col.append("undefined")
            trend_state_col.append("undefined")
            drawdown_recovery_state_col.append("undefined")
            stress_state_col.append("undefined")
            is_defined_col.append(False)
            has_match_col.append(True)
        elif i < n_undefined + n_unmatched:
            statuses.append("unmatched_timestamp")
            regime_labels_col.append(_UNDEF_LABEL)
            volatility_state_col.append("undefined")
            trend_state_col.append("undefined")
            drawdown_recovery_state_col.append("undefined")
            stress_state_col.append("undefined")
            is_defined_col.append(False)
            has_match_col.append(False)
        else:
            label_idx = (i - n_undefined - n_unmatched) % len(_REGIME_LABELS_POOL)
            label = _REGIME_LABELS_POOL[label_idx]
            statuses.append("matched_defined")
            regime_labels_col.append(label)
            volatility_state_col.append("low_volatility" if label_idx == 0 else "high_volatility")
            trend_state_col.append("uptrend" if label_idx == 0 else "downtrend")
            drawdown_recovery_state_col.append("near_peak" if label_idx == 0 else "underwater")
            stress_state_col.append("low_stress" if label_idx == 0 else "high_stress")
            is_defined_col.append(True)
            has_match_col.append(True)

    frame["regime_label"] = regime_labels_col
    frame["regime_volatility_state"] = volatility_state_col
    frame["regime_trend_state"] = trend_state_col
    frame["regime_drawdown_recovery_state"] = drawdown_recovery_state_col
    frame["regime_stress_state"] = stress_state_col
    frame["regime_is_defined"] = is_defined_col
    frame["regime_has_exact_timestamp_match"] = has_match_col
    frame["regime_alignment_status"] = statuses
    frame["regime_surface"] = surface
    frame.attrs = {}
    return frame


def _aligned_strategy() -> pd.DataFrame:
    import numpy as np

    rng = np.random.default_rng(7)
    returns = rng.normal(0.0005, 0.01, size=_NUM_DAYS).tolist()
    return _build_aligned_frame({"strategy_return": returns}, surface="strategy")


def _aligned_alpha() -> pd.DataFrame:
    import numpy as np

    rng = np.random.default_rng(99)
    ic = rng.normal(0.03, 0.12, size=_NUM_DAYS).tolist()
    rank_ic = rng.normal(0.04, 0.10, size=_NUM_DAYS).tolist()
    return _build_aligned_frame(
        {"ic": ic, "rank_ic": rank_ic, "sample_size": [20] * _NUM_DAYS},
        surface="alpha",
    )


def _aligned_portfolio() -> pd.DataFrame:
    import numpy as np

    rng = np.random.default_rng(13)
    returns = rng.normal(0.0006, 0.012, size=_NUM_DAYS).tolist()
    return _build_aligned_frame({"portfolio_return": returns}, surface="portfolio")


def _strategy_frame_raw() -> pd.DataFrame:
    import numpy as np

    rng = np.random.default_rng(7)
    returns = rng.normal(0.0005, 0.01, size=_NUM_DAYS).tolist()
    return pd.DataFrame(
        {
            "ts_utc": pd.date_range("2025-01-01", periods=_NUM_DAYS, freq="D", tz="UTC"),
            "strategy_return": returns,
        }
    )


# ---------------------------------------------------------------------------
# RegimeConditionalConfig tests
# ---------------------------------------------------------------------------


class TestRegimeConditionalConfig:
    def test_defaults(self) -> None:
        config = RegimeConditionalConfig()
        assert config.min_observations == 5
        assert config.regime_prefix == "regime_"
        assert config.periods_per_year == 252

    def test_from_dict(self) -> None:
        config = resolve_regime_conditional_config({"min_observations": 3})
        assert config.min_observations == 3

    def test_invalid_min_observations(self) -> None:
        with pytest.raises(ValueError, match="min_observations"):
            RegimeConditionalConfig(min_observations=0)

    def test_invalid_periods_per_year(self) -> None:
        with pytest.raises(ValueError, match="periods_per_year"):
            RegimeConditionalConfig(periods_per_year=0)

    def test_invalid_regime_prefix(self) -> None:
        with pytest.raises(ValueError, match="regime_prefix"):
            RegimeConditionalConfig(regime_prefix="")

    def test_unknown_dict_keys_raise(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            resolve_regime_conditional_config({"unknown_key": 1})

    def test_none_returns_defaults(self) -> None:
        config = resolve_regime_conditional_config(None)
        assert config.min_observations == 5


# ---------------------------------------------------------------------------
# Strategy conditional evaluation tests
# ---------------------------------------------------------------------------


class TestStrategyConditionalEvaluation:
    def test_returns_regime_conditional_result(self) -> None:
        aligned = _aligned_strategy()
        result = evaluate_strategy_metrics_by_regime(aligned)
        assert isinstance(result, RegimeConditionalResult)
        assert result.surface == "strategy"
        assert result.dimension == "composite"

    def test_metrics_frame_columns_stable(self) -> None:
        aligned = _aligned_strategy()
        result = evaluate_strategy_metrics_by_regime(aligned)
        expected_cols = [
            "regime_label", "dimension", "observation_count", "coverage_status",
            "total_return", "annualized_return", "volatility", "annualized_volatility",
            "sharpe_ratio", "max_drawdown", "win_rate",
        ]
        assert list(result.metrics_by_regime.columns) == expected_cols

    def test_metrics_frame_sorted_by_regime_label(self) -> None:
        aligned = _aligned_strategy()
        result = evaluate_strategy_metrics_by_regime(aligned)
        labels = result.metrics_by_regime["regime_label"].tolist()
        assert labels == sorted(labels)

    def test_metrics_frame_has_two_regime_groups(self) -> None:
        aligned = _aligned_strategy()
        result = evaluate_strategy_metrics_by_regime(aligned)
        assert len(result.metrics_by_regime) == 2

    def test_only_matched_defined_rows_contribute(self) -> None:
        aligned = _aligned_strategy()
        expected_defined = (aligned["regime_alignment_status"] == "matched_defined").sum()
        result = evaluate_strategy_metrics_by_regime(aligned)
        assert result.alignment_summary["matched_defined"] == expected_defined

    def test_alignment_summary_counts_correct(self) -> None:
        aligned = _aligned_strategy()
        result = evaluate_strategy_metrics_by_regime(aligned)
        summary = result.alignment_summary
        assert summary["total_rows"] == _NUM_DAYS
        assert summary["matched_undefined"] == 4
        assert summary["unmatched_timestamp"] == 2
        assert summary["matched_defined"] == _NUM_DAYS - 4 - 2

    def test_alignment_summary_keys(self) -> None:
        aligned = _aligned_strategy()
        result = evaluate_strategy_metrics_by_regime(aligned)
        summary = result.alignment_summary
        for key in ("total_rows", "matched_defined", "matched_undefined", "unmatched_timestamp"):
            assert key in summary

    def test_coverage_status_values_are_valid(self) -> None:
        aligned = _aligned_strategy()
        result = evaluate_strategy_metrics_by_regime(aligned)
        valid_statuses = {"sufficient", "sparse", "empty"}
        for val in result.metrics_by_regime["coverage_status"]:
            assert val in valid_statuses

    def test_sufficient_rows_have_float_metrics(self) -> None:
        aligned = _aligned_strategy()
        result = evaluate_strategy_metrics_by_regime(aligned, config={"min_observations": 1})
        sufficient = result.metrics_by_regime[result.metrics_by_regime["coverage_status"] == "sufficient"]
        assert not sufficient.empty
        assert sufficient["total_return"].notna().all()
        assert sufficient["sharpe_ratio"].notna().all()

    def test_sparse_rows_have_null_metrics(self) -> None:
        aligned = _aligned_strategy()
        result = evaluate_strategy_metrics_by_regime(aligned, config={"min_observations": 10000})
        sparse = result.metrics_by_regime[result.metrics_by_regime["coverage_status"].isin(["sparse", "empty"])]
        assert not sparse.empty
        assert sparse["total_return"].isna().all()
        assert sparse["sharpe_ratio"].isna().all()

    def test_dimension_volatility_groups_by_volatility_state(self) -> None:
        aligned = _aligned_strategy()
        result = evaluate_strategy_metrics_by_regime(aligned, dimension="volatility")
        assert result.dimension == "volatility"
        labels = result.metrics_by_regime["regime_label"].tolist()
        for label in labels:
            assert "|" not in label, f"Composite label found in 'volatility': {label!r}"
        assert "low_volatility" in labels
        assert "high_volatility" in labels

    def test_dimension_trend_groups_by_trend_state(self) -> None:
        aligned = _aligned_strategy()
        result = evaluate_strategy_metrics_by_regime(aligned, dimension="trend")
        assert result.dimension == "trend"
        labels = result.metrics_by_regime["regime_label"].tolist()
        assert "uptrend" in labels
        assert "downtrend" in labels

    def test_invalid_dimension_raises(self) -> None:
        aligned = _aligned_strategy()
        with pytest.raises(ValueError, match="Unknown regime dimension"):
            evaluate_strategy_metrics_by_regime(aligned, dimension="bad_dimension")

    def test_missing_return_column_raises(self) -> None:
        aligned = _aligned_strategy()
        with pytest.raises(ValueError, match="nonexistent_col"):
            evaluate_strategy_metrics_by_regime(aligned, return_column="nonexistent_col")

    def test_missing_alignment_status_column_raises(self) -> None:
        frame = _strategy_frame_raw()
        with pytest.raises(ValueError, match="regime_alignment_status"):
            evaluate_strategy_metrics_by_regime(frame)

    def test_empty_frame_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            evaluate_strategy_metrics_by_regime(pd.DataFrame())

    def test_deterministic_repeated_calls(self) -> None:
        aligned = _aligned_strategy()
        result_a = evaluate_strategy_metrics_by_regime(aligned)
        result_b = evaluate_strategy_metrics_by_regime(aligned)
        pd.testing.assert_frame_equal(result_a.metrics_by_regime, result_b.metrics_by_regime)

    def test_metadata_includes_surface_and_dimension(self) -> None:
        aligned = _aligned_strategy()
        result = evaluate_strategy_metrics_by_regime(aligned)
        assert result.metadata["surface"] == "strategy"
        assert result.metadata["dimension"] == "composite"

    def test_observation_counts_sum_to_matched_defined(self) -> None:
        aligned = _aligned_strategy()
        result = evaluate_strategy_metrics_by_regime(aligned)
        total_obs = result.metrics_by_regime["observation_count"].sum()
        assert total_obs == result.alignment_summary["matched_defined"]


# ---------------------------------------------------------------------------
# Alpha conditional evaluation tests
# ---------------------------------------------------------------------------


class TestAlphaConditionalEvaluation:
    def test_returns_regime_conditional_result(self) -> None:
        aligned = _aligned_alpha()
        result = evaluate_alpha_metrics_by_regime(aligned)
        assert isinstance(result, RegimeConditionalResult)
        assert result.surface == "alpha"

    def test_metrics_frame_columns_stable(self) -> None:
        aligned = _aligned_alpha()
        result = evaluate_alpha_metrics_by_regime(aligned)
        expected_cols = [
            "regime_label", "dimension", "observation_count", "coverage_status",
            "mean_ic", "mean_rank_ic", "ic_std", "rank_ic_std", "ic_ir", "rank_ic_ir",
        ]
        assert list(result.metrics_by_regime.columns) == expected_cols

    def test_metrics_frame_has_two_regime_groups(self) -> None:
        aligned = _aligned_alpha()
        result = evaluate_alpha_metrics_by_regime(aligned)
        assert len(result.metrics_by_regime) == 2

    def test_sufficient_rows_have_ic_values(self) -> None:
        aligned = _aligned_alpha()
        result = evaluate_alpha_metrics_by_regime(aligned, config={"min_observations": 1})
        sufficient = result.metrics_by_regime[result.metrics_by_regime["coverage_status"] == "sufficient"]
        assert not sufficient.empty
        assert sufficient["mean_ic"].notna().all()
        assert sufficient["mean_rank_ic"].notna().all()

    def test_sparse_rows_have_null_ic(self) -> None:
        aligned = _aligned_alpha()
        result = evaluate_alpha_metrics_by_regime(aligned, config={"min_observations": 10000})
        sparse = result.metrics_by_regime[result.metrics_by_regime["coverage_status"].isin(["sparse", "empty"])]
        assert not sparse.empty
        assert sparse["mean_ic"].isna().all()

    def test_ic_ir_computed_correctly(self) -> None:
        aligned = _aligned_alpha()
        result = evaluate_alpha_metrics_by_regime(aligned, config={"min_observations": 1})
        sufficient = result.metrics_by_regime[result.metrics_by_regime["coverage_status"] == "sufficient"]
        assert not sufficient.empty
        row = sufficient.iloc[0]
        if (
            row["ic_std"] is not None
            and row["ic_std"] != 0.0
            and row["mean_ic"] is not None
            and row["ic_ir"] is not None
        ):
            expected_ir = row["mean_ic"] / row["ic_std"]
            assert abs(row["ic_ir"] - expected_ir) < 1e-10

    def test_missing_ic_column_raises(self) -> None:
        aligned = _aligned_alpha()
        with pytest.raises(ValueError, match="nonexistent_ic"):
            evaluate_alpha_metrics_by_regime(aligned, ic_column="nonexistent_ic")

    def test_deterministic_repeated_calls(self) -> None:
        aligned = _aligned_alpha()
        result_a = evaluate_alpha_metrics_by_regime(aligned)
        result_b = evaluate_alpha_metrics_by_regime(aligned)
        pd.testing.assert_frame_equal(result_a.metrics_by_regime, result_b.metrics_by_regime)

    def test_dimension_trend_groups_by_trend_state(self) -> None:
        aligned = _aligned_alpha()
        result = evaluate_alpha_metrics_by_regime(aligned, dimension="trend")
        assert result.dimension == "trend"
        labels = result.metrics_by_regime["regime_label"].tolist()
        for label in labels:
            assert "|" not in label
        assert "uptrend" in labels
        assert "downtrend" in labels

    def test_alignment_summary_correct(self) -> None:
        aligned = _aligned_alpha()
        result = evaluate_alpha_metrics_by_regime(aligned)
        assert result.alignment_summary["total_rows"] == _NUM_DAYS
        assert result.alignment_summary["matched_defined"] == _NUM_DAYS - 4 - 2


# ---------------------------------------------------------------------------
# Portfolio conditional evaluation tests
# ---------------------------------------------------------------------------


class TestPortfolioConditionalEvaluation:
    def test_returns_regime_conditional_result(self) -> None:
        aligned = _aligned_portfolio()
        result = evaluate_portfolio_metrics_by_regime(aligned)
        assert isinstance(result, RegimeConditionalResult)
        assert result.surface == "portfolio"

    def test_metrics_frame_columns_stable(self) -> None:
        aligned = _aligned_portfolio()
        result = evaluate_portfolio_metrics_by_regime(aligned)
        expected_cols = [
            "regime_label", "dimension", "observation_count", "coverage_status",
            "total_return", "annualized_return", "volatility", "annualized_volatility",
            "sharpe_ratio", "max_drawdown", "win_rate",
        ]
        assert list(result.metrics_by_regime.columns) == expected_cols

    def test_metrics_frame_has_two_regime_groups(self) -> None:
        aligned = _aligned_portfolio()
        result = evaluate_portfolio_metrics_by_regime(aligned)
        assert len(result.metrics_by_regime) == 2

    def test_sufficient_rows_have_float_metrics(self) -> None:
        aligned = _aligned_portfolio()
        result = evaluate_portfolio_metrics_by_regime(aligned, config={"min_observations": 1})
        sufficient = result.metrics_by_regime[result.metrics_by_regime["coverage_status"] == "sufficient"]
        assert not sufficient.empty
        assert sufficient["total_return"].notna().all()
        assert sufficient["sharpe_ratio"].notna().all()

    def test_deterministic_repeated_calls(self) -> None:
        aligned = _aligned_portfolio()
        result_a = evaluate_portfolio_metrics_by_regime(aligned)
        result_b = evaluate_portfolio_metrics_by_regime(aligned)
        pd.testing.assert_frame_equal(result_a.metrics_by_regime, result_b.metrics_by_regime)

    def test_custom_return_column(self) -> None:
        aligned = _aligned_portfolio()
        aligned = aligned.rename(columns={"portfolio_return": "custom_return"})
        result = evaluate_portfolio_metrics_by_regime(aligned, return_column="custom_return")
        assert not result.metrics_by_regime.empty
        assert "regime_label" in result.metrics_by_regime.columns

    def test_alignment_summary_correct(self) -> None:
        aligned = _aligned_portfolio()
        result = evaluate_portfolio_metrics_by_regime(aligned)
        assert result.alignment_summary["matched_defined"] == _NUM_DAYS - 4 - 2


# ---------------------------------------------------------------------------
# evaluate_all_dimensions tests
# ---------------------------------------------------------------------------


class TestEvaluateAllDimensions:
    def test_strategy_returns_all_dimensions(self) -> None:
        aligned = _aligned_strategy()
        results = evaluate_all_dimensions(aligned, surface="strategy")
        expected_keys = {"composite", "volatility", "trend", "drawdown_recovery", "stress"}
        assert set(results.keys()) == expected_keys

    def test_alpha_returns_all_dimensions(self) -> None:
        aligned = _aligned_alpha()
        results = evaluate_all_dimensions(aligned, surface="alpha")
        expected_keys = {"composite", "volatility", "trend", "drawdown_recovery", "stress"}
        assert set(results.keys()) == expected_keys

    def test_portfolio_returns_all_dimensions(self) -> None:
        aligned = _aligned_portfolio()
        results = evaluate_all_dimensions(aligned, surface="portfolio")
        expected_keys = {"composite", "volatility", "trend", "drawdown_recovery", "stress"}
        assert set(results.keys()) == expected_keys

    def test_invalid_surface_raises(self) -> None:
        aligned = _aligned_strategy()
        with pytest.raises(ValueError, match="Unsupported surface"):
            evaluate_all_dimensions(aligned, surface="bad_surface")  # type: ignore[arg-type]

    def test_all_results_are_regime_conditional_results(self) -> None:
        aligned = _aligned_strategy()
        results = evaluate_all_dimensions(aligned, surface="strategy")
        for _dim, result in results.items():
            assert isinstance(result, RegimeConditionalResult)

    def test_composite_has_two_groups(self) -> None:
        aligned = _aligned_strategy()
        results = evaluate_all_dimensions(aligned, surface="strategy")
        assert len(results["composite"].metrics_by_regime) == 2

    def test_each_dimension_has_non_empty_metrics(self) -> None:
        aligned = _aligned_strategy()
        results = evaluate_all_dimensions(aligned, surface="strategy")
        for dim, result in results.items():
            assert not result.metrics_by_regime.empty, f"Dimension {dim!r} produced empty metrics"


# ---------------------------------------------------------------------------
# Artifact persistence tests
# ---------------------------------------------------------------------------


class TestRegimeConditionalArtifacts:
    def test_write_artifacts_creates_expected_files(self, tmp_path: Path) -> None:
        aligned = _aligned_strategy()
        result = evaluate_strategy_metrics_by_regime(aligned)
        manifest = write_regime_conditional_artifacts(tmp_path, result, run_id="test_run")

        assert (tmp_path / METRICS_BY_REGIME_FILENAME).exists()
        assert (tmp_path / REGIME_CONDITIONAL_SUMMARY_FILENAME).exists()
        assert (tmp_path / "regime_conditional_manifest.json").exists()
        assert isinstance(manifest, dict)

    def test_manifest_contains_expected_keys(self, tmp_path: Path) -> None:
        aligned = _aligned_strategy()
        result = evaluate_strategy_metrics_by_regime(aligned)
        manifest = write_regime_conditional_artifacts(tmp_path, result)

        assert "surface" in manifest
        assert "artifacts" in manifest
        assert manifest["surface"] == "strategy"
        assert METRICS_BY_REGIME_FILENAME in manifest["artifacts"].values()

    def test_load_metrics_roundtrip(self, tmp_path: Path) -> None:
        aligned = _aligned_strategy()
        result = evaluate_strategy_metrics_by_regime(aligned)
        write_regime_conditional_artifacts(tmp_path, result)

        loaded = load_regime_conditional_metrics(tmp_path)
        assert isinstance(loaded, pd.DataFrame)
        assert "regime_label" in loaded.columns
        assert "observation_count" in loaded.columns
        assert len(loaded) == 2  # Two regime labels in fixture.

    def test_load_summary_roundtrip(self, tmp_path: Path) -> None:
        aligned = _aligned_strategy()
        result = evaluate_strategy_metrics_by_regime(aligned)
        write_regime_conditional_artifacts(tmp_path, result, run_id="abc")

        summary = load_regime_conditional_summary(tmp_path)
        assert summary["surface"] == "strategy"
        assert summary["run_id"] == "abc"
        assert "alignment_summary" in summary
        assert "coverage_breakdown" in summary
        assert summary["regime_label_count"] == 2

    def test_load_manifest_roundtrip(self, tmp_path: Path) -> None:
        aligned = _aligned_strategy()
        result = evaluate_strategy_metrics_by_regime(aligned)
        write_regime_conditional_artifacts(tmp_path, result)

        manifest = load_regime_conditional_manifest(tmp_path)
        assert "artifacts" in manifest

    def test_multi_dimension_write_produces_combined_csv(self, tmp_path: Path) -> None:
        aligned = _aligned_strategy()
        results = evaluate_all_dimensions(aligned, surface="strategy")
        manifest = write_regime_conditional_artifacts_multi_dimension(
            tmp_path, results, run_id="multi_test"
        )

        combined = load_regime_conditional_metrics(tmp_path)
        assert "dimension" in combined.columns
        assert combined["dimension"].nunique() > 1
        assert isinstance(manifest, dict)
        assert "dimensions" in manifest

    def test_multi_dimension_summary_has_all_dimensions(self, tmp_path: Path) -> None:
        aligned = _aligned_strategy()
        results = evaluate_all_dimensions(aligned, surface="strategy")
        write_regime_conditional_artifacts_multi_dimension(tmp_path, results)
        summary = load_regime_conditional_summary(tmp_path)
        assert "dimensions_summary" in summary
        assert "composite" in summary["dimensions_summary"]

    def test_metrics_csv_has_stable_column_order(self, tmp_path: Path) -> None:
        aligned = _aligned_strategy()
        result = evaluate_strategy_metrics_by_regime(aligned)
        write_regime_conditional_artifacts(tmp_path, result)

        loaded = load_regime_conditional_metrics(tmp_path)
        expected_first_cols = ["regime_label", "dimension", "observation_count", "coverage_status"]
        actual_cols = list(loaded.columns)
        for col in expected_first_cols:
            assert col in actual_cols

    def test_alpha_artifacts_roundtrip(self, tmp_path: Path) -> None:
        aligned = _aligned_alpha()
        result = evaluate_alpha_metrics_by_regime(aligned)
        write_regime_conditional_artifacts(tmp_path, result)

        loaded = load_regime_conditional_metrics(tmp_path)
        assert "mean_ic" in loaded.columns
        assert "mean_rank_ic" in loaded.columns

    def test_portfolio_artifacts_roundtrip(self, tmp_path: Path) -> None:
        aligned = _aligned_portfolio()
        result = evaluate_portfolio_metrics_by_regime(aligned)
        write_regime_conditional_artifacts(tmp_path, result)

        loaded = load_regime_conditional_metrics(tmp_path)
        assert "sharpe_ratio" in loaded.columns
        assert "max_drawdown" in loaded.columns

    def test_summary_schema_version_present(self, tmp_path: Path) -> None:
        aligned = _aligned_portfolio()
        result = evaluate_portfolio_metrics_by_regime(aligned)
        write_regime_conditional_artifacts(tmp_path, result)

        summary = load_regime_conditional_summary(tmp_path)
        assert "schema_version" in summary
        assert summary["schema_version"] == 1

    def test_summary_includes_observation_counts(self, tmp_path: Path) -> None:
        aligned = _aligned_strategy()
        result = evaluate_strategy_metrics_by_regime(aligned)
        write_regime_conditional_artifacts(tmp_path, result)
        summary = load_regime_conditional_summary(tmp_path)
        assert "observation_counts" in summary
        assert len(summary["observation_counts"]) == 2

    def test_taxonomy_version_in_manifest(self, tmp_path: Path) -> None:
        aligned = _aligned_strategy()
        result = evaluate_strategy_metrics_by_regime(aligned)
        manifest = write_regime_conditional_artifacts(tmp_path, result)
        assert manifest["taxonomy_version"] == "regime_taxonomy_v1"

    def test_write_artifacts_idempotent(self, tmp_path: Path) -> None:
        aligned = _aligned_strategy()
        result = evaluate_strategy_metrics_by_regime(aligned)
        write_regime_conditional_artifacts(tmp_path, result)
        write_regime_conditional_artifacts(tmp_path, result)

        loaded = load_regime_conditional_metrics(tmp_path)
        assert len(loaded) == 2


# ---------------------------------------------------------------------------
# Undefined and unmatched regime row handling
# ---------------------------------------------------------------------------


class TestUndefinedUnmatchedHandling:
    def test_unmatched_rows_excluded_from_metrics(self) -> None:
        aligned = _aligned_strategy()
        n_unmatched = (aligned["regime_alignment_status"] == "unmatched_timestamp").sum()
        result = evaluate_strategy_metrics_by_regime(aligned)
        assert result.alignment_summary["unmatched_timestamp"] == n_unmatched
        assert n_unmatched == 2

    def test_undefined_rows_excluded_from_metrics(self) -> None:
        aligned = _aligned_strategy()
        n_undefined = (aligned["regime_alignment_status"] == "matched_undefined").sum()
        result = evaluate_strategy_metrics_by_regime(aligned)
        assert result.alignment_summary["matched_undefined"] == n_undefined
        assert n_undefined == 4

    def test_unavailable_policy_mark_unmatched(self) -> None:
        import numpy as np

        rng = np.random.default_rng(7)
        returns = rng.normal(0.0005, 0.01, size=_NUM_DAYS).tolist()
        frame = pd.DataFrame({
            "ts_utc": pd.date_range("2025-01-01", periods=_NUM_DAYS, freq="D", tz="UTC"),
            "strategy_return": returns,
        })
        aligned = align_regime_labels(
            frame,
            None,
            config=RegimeAlignmentConfig(unavailable_policy="mark_unmatched"),
        )
        assert (aligned["regime_alignment_status"] == "regime_labels_unavailable").all()

        result = evaluate_strategy_metrics_by_regime(aligned)
        assert result.alignment_summary["matched_defined"] == 0
        assert result.metrics_by_regime.empty

    def test_all_undefined_produces_empty_metrics(self) -> None:
        aligned = _aligned_strategy().copy()
        aligned["regime_alignment_status"] = "matched_undefined"
        result = evaluate_strategy_metrics_by_regime(aligned)
        assert result.metrics_by_regime.empty
        assert result.alignment_summary["matched_defined"] == 0

    def test_coverage_breakdown_includes_sparse_for_low_obs(self) -> None:
        import numpy as np

        rng = np.random.default_rng(7)
        returns = rng.normal(0.0005, 0.01, size=_NUM_DAYS).tolist()
        aligned = _build_aligned_frame(
            {"strategy_return": returns},
            n_undefined=0,
            n_unmatched=0,
            surface="strategy",
        )
        result = evaluate_strategy_metrics_by_regime(aligned, config={"min_observations": 10000})
        statuses = set(result.metrics_by_regime["coverage_status"].unique())
        assert statuses.issubset({"sparse", "empty"})


# ---------------------------------------------------------------------------
# Backward compatibility: aggregate metrics unchanged
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_existing_metrics_functions_unaffected(self) -> None:
        from src.research.metrics import (
            annualized_return,
            max_drawdown,
            sharpe_ratio,
            total_return,
        )
        import numpy as np

        rng = np.random.default_rng(7)
        returns_list = rng.normal(0.0005, 0.01, size=_NUM_DAYS).tolist()
        raw_returns = pd.Series(returns_list)

        total_before = total_return(raw_returns)
        sharpe_before = sharpe_ratio(raw_returns)
        dd_before = max_drawdown(raw_returns)
        ann_before = annualized_return(raw_returns)

        aligned = _aligned_strategy()
        aligned_returns = aligned["strategy_return"]

        assert total_return(aligned_returns) == pytest.approx(total_before, rel=1e-10)
        assert sharpe_ratio(aligned_returns) == pytest.approx(sharpe_before, rel=1e-10)
        assert max_drawdown(aligned_returns) == pytest.approx(dd_before, rel=1e-10)
        assert annualized_return(aligned_returns) == pytest.approx(ann_before, rel=1e-10)

    def test_regime_columns_added_not_mutated(self) -> None:
        aligned = _aligned_strategy()
        assert "regime_alignment_status" in aligned.columns
        assert "regime_label" in aligned.columns
        assert "strategy_return" in aligned.columns
        assert aligned["strategy_return"].notna().all()


# ---------------------------------------------------------------------------
# Sparse and low-observation edge cases
# ---------------------------------------------------------------------------


class TestSparseRegimeCases:
    def test_single_row_per_regime_produces_sparse_coverage(self) -> None:
        ts = pd.date_range("2025-01-01", periods=2, freq="D", tz="UTC")
        sparse_frame = pd.DataFrame({
            "ts_utc": ts,
            "strategy_return": [0.001, -0.002],
            "regime_label": _REGIME_LABELS_POOL,
            "regime_volatility_state": ["low_volatility", "high_volatility"],
            "regime_trend_state": ["uptrend", "downtrend"],
            "regime_drawdown_recovery_state": ["near_peak", "underwater"],
            "regime_stress_state": ["low_stress", "high_stress"],
            "regime_is_defined": [True, True],
            "regime_has_exact_timestamp_match": [True, True],
            "regime_alignment_status": ["matched_defined", "matched_defined"],
            "regime_surface": ["strategy", "strategy"],
        })
        result = evaluate_strategy_metrics_by_regime(sparse_frame)
        for _, row in result.metrics_by_regime.iterrows():
            assert row["coverage_status"] == "sparse", (
                f"Expected sparse for 1 obs, got {row['coverage_status']!r}"
            )

    def test_zero_rows_after_filtering_empty_metrics_frame(self) -> None:
        aligned = _aligned_strategy().copy()
        aligned["regime_alignment_status"] = "matched_undefined"
        result = evaluate_strategy_metrics_by_regime(aligned)
        assert result.metrics_by_regime.empty
        assert result.alignment_summary["matched_defined"] == 0

    def test_min_observations_one_sufficient_for_single_row(self) -> None:
        ts = pd.date_range("2025-01-01", periods=1, freq="D", tz="UTC")
        single_frame = pd.DataFrame({
            "ts_utc": ts,
            "strategy_return": [0.001],
            "regime_label": [_REGIME_LABELS_POOL[0]],
            "regime_volatility_state": ["low_volatility"],
            "regime_trend_state": ["uptrend"],
            "regime_drawdown_recovery_state": ["near_peak"],
            "regime_stress_state": ["low_stress"],
            "regime_is_defined": [True],
            "regime_has_exact_timestamp_match": [True],
            "regime_alignment_status": ["matched_defined"],
            "regime_surface": ["strategy"],
        })
        result = evaluate_strategy_metrics_by_regime(single_frame, config={"min_observations": 1})
        assert len(result.metrics_by_regime) == 1
        assert result.metrics_by_regime.iloc[0]["coverage_status"] == "sufficient"
