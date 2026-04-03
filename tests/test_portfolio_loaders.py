from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.portfolio import (
    build_aligned_return_matrix,
    load_portfolio_component_runs_returns,
    load_strategy_run_returns,
    load_strategy_runs_returns,
)


def test_load_strategy_run_returns_loads_valid_single_run(tmp_path: Path) -> None:
    run_dir = _write_run(
        tmp_path,
        run_id="20260325T100000Z_alpha",
        strategy_name="alpha_v1",
        rows=[
            {"ts_utc": "2022-01-02T00:00:00Z", "strategy_return": 0.02, "equity": 1.02},
            {"ts_utc": "2022-01-01T00:00:00Z", "strategy_return": 0.01, "equity": 1.00},
        ],
    )

    loaded = load_strategy_run_returns(run_dir)

    assert loaded.columns.tolist() == ["ts_utc", "strategy_name", "strategy_return", "run_id"]
    assert loaded["strategy_name"].tolist() == ["alpha_v1", "alpha_v1"]
    assert loaded["run_id"].tolist() == ["20260325T100000Z_alpha", "20260325T100000Z_alpha"]
    assert loaded["strategy_return"].tolist() == [0.01, 0.02]
    assert loaded["ts_utc"].tolist() == list(
        pd.to_datetime(["2022-01-01T00:00:00Z", "2022-01-02T00:00:00Z"], utc=True)
    )


def test_load_strategy_runs_returns_is_deterministic_independent_of_input_order(tmp_path: Path) -> None:
    alpha_dir = _write_run(
        tmp_path,
        run_id="20260325T100000Z_alpha",
        strategy_name="alpha_v1",
        rows=[
            {"ts_utc": "2022-01-02T00:00:00Z", "strategy_return": 0.02},
            {"ts_utc": "2022-01-01T00:00:00Z", "strategy_return": 0.01},
        ],
    )
    beta_dir = _write_run(
        tmp_path,
        run_id="20260325T100500Z_beta",
        strategy_name="beta_v1",
        rows=[
            {"ts_utc": "2022-01-02T00:00:00Z", "strategy_return": -0.01},
            {"ts_utc": "2022-01-01T00:00:00Z", "strategy_return": 0.03},
        ],
    )

    forward = load_strategy_runs_returns([alpha_dir, beta_dir])
    reversed_order = load_strategy_runs_returns([beta_dir, alpha_dir])

    pd.testing.assert_frame_equal(forward, reversed_order)
    assert forward["strategy_name"].tolist() == ["alpha_v1", "beta_v1", "alpha_v1", "beta_v1"]


def test_build_aligned_return_matrix_uses_intersection_alignment(tmp_path: Path) -> None:
    alpha_dir = _write_run(
        tmp_path,
        run_id="20260325T101000Z_alpha",
        strategy_name="alpha_v1",
        rows=[
            {"ts_utc": "2022-01-01T00:00:00Z", "strategy_return": 0.01},
            {"ts_utc": "2022-01-02T00:00:00Z", "strategy_return": 0.02},
            {"ts_utc": "2022-01-03T00:00:00Z", "strategy_return": 0.03},
        ],
    )
    beta_dir = _write_run(
        tmp_path,
        run_id="20260325T101500Z_beta",
        strategy_name="beta_v1",
        rows=[
            {"ts_utc": "2022-01-02T00:00:00Z", "strategy_return": -0.01},
            {"ts_utc": "2022-01-03T00:00:00Z", "strategy_return": 0.04},
            {"ts_utc": "2022-01-04T00:00:00Z", "strategy_return": 0.05},
        ],
    )

    strategy_returns = load_strategy_runs_returns([beta_dir, alpha_dir])
    matrix = build_aligned_return_matrix(strategy_returns)

    assert matrix.index.tolist() == list(pd.to_datetime(["2022-01-02T00:00:00Z", "2022-01-03T00:00:00Z"], utc=True))
    assert matrix.columns.tolist() == ["alpha_v1", "beta_v1"]
    assert matrix.loc[pd.Timestamp("2022-01-02T00:00:00Z"), "alpha_v1"] == pytest.approx(0.02)
    assert matrix.loc[pd.Timestamp("2022-01-02T00:00:00Z"), "beta_v1"] == pytest.approx(-0.01)


def test_build_aligned_return_matrix_fails_for_non_overlapping_runs(tmp_path: Path) -> None:
    alpha_dir = _write_run(
        tmp_path,
        run_id="20260325T102000Z_alpha",
        strategy_name="alpha_v1",
        rows=[{"ts_utc": "2022-01-01T00:00:00Z", "strategy_return": 0.01}],
    )
    beta_dir = _write_run(
        tmp_path,
        run_id="20260325T102500Z_beta",
        strategy_name="beta_v1",
        rows=[{"ts_utc": "2022-01-03T00:00:00Z", "strategy_return": 0.02}],
    )

    strategy_returns = load_strategy_runs_returns([alpha_dir, beta_dir])

    with pytest.raises(ValueError, match="Aligned return matrix is empty under 'intersection' alignment"):
        build_aligned_return_matrix(strategy_returns)


def test_load_strategy_run_returns_fails_for_missing_artifact(tmp_path: Path) -> None:
    run_dir = tmp_path / "missing_equity_curve"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"strategy_name": "alpha_v1"}), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="equity_curve\\.csv"):
        load_strategy_run_returns(run_dir)


def test_load_strategy_run_returns_fails_for_non_utc_timestamps(tmp_path: Path) -> None:
    run_dir = _write_run(
        tmp_path,
        run_id="20260325T103000Z_alpha",
        strategy_name="alpha_v1",
        rows=[{"ts_utc": "2022-01-01", "strategy_return": 0.01}],
    )

    with pytest.raises(ValueError, match="timezone-aware UTC"):
        load_strategy_run_returns(run_dir)


def test_load_strategy_runs_returns_rejects_duplicate_strategy_identifiers(tmp_path: Path) -> None:
    alpha_one = _write_run(
        tmp_path,
        run_id="20260325T103500Z_alpha_1",
        strategy_name="alpha_v1",
        rows=[{"ts_utc": "2022-01-01T00:00:00Z", "strategy_return": 0.01}],
    )
    alpha_two = _write_run(
        tmp_path,
        run_id="20260325T104000Z_alpha_2",
        strategy_name="alpha_v1",
        rows=[{"ts_utc": "2022-01-02T00:00:00Z", "strategy_return": 0.02}],
    )

    with pytest.raises(ValueError, match="Duplicate strategy identifiers"):
        load_strategy_runs_returns([alpha_one, alpha_two])


def test_load_strategy_run_returns_averages_duplicate_timestamp_rows(tmp_path: Path) -> None:
    run_dir = _write_run(
        tmp_path,
        run_id="20260325T104500Z_alpha",
        strategy_name="alpha_v1",
        rows=[
            {"ts_utc": "2022-01-01T00:00:00Z", "strategy_return": 0.10},
            {"ts_utc": "2022-01-01T00:00:00Z", "strategy_return": -0.05},
            {"ts_utc": "2022-01-02T00:00:00Z", "strategy_return": 0.01},
        ],
    )

    loaded = load_strategy_run_returns(run_dir)

    assert loaded["ts_utc"].tolist() == list(
        pd.to_datetime(["2022-01-01T00:00:00Z", "2022-01-02T00:00:00Z"], utc=True)
    )
    assert loaded["strategy_return"].tolist() == [pytest.approx(0.025), pytest.approx(0.01)]


def test_load_portfolio_component_runs_returns_supports_alpha_sleeves(tmp_path: Path) -> None:
    alpha_dir = _write_alpha_sleeve_run(
        tmp_path,
        run_id="20260325T105000Z_alpha_eval",
        alpha_name="alpha_model_v1",
        rows=[
            {"ts_utc": "2022-01-01T00:00:00Z", "sleeve_return": 0.03},
            {"ts_utc": "2022-01-02T00:00:00Z", "sleeve_return": -0.01},
        ],
    )
    beta_dir = _write_run(
        tmp_path,
        run_id="20260325T105500Z_beta",
        strategy_name="beta_v1",
        rows=[
            {"ts_utc": "2022-01-01T00:00:00Z", "strategy_return": 0.01},
            {"ts_utc": "2022-01-02T00:00:00Z", "strategy_return": 0.02},
        ],
    )

    loaded = load_portfolio_component_runs_returns(
        [
            {
                "source_artifact_path": alpha_dir,
                "artifact_type": "alpha_sleeve",
                "strategy_name": "alpha_sleeve_v1",
            },
            {
                "source_artifact_path": beta_dir,
                "artifact_type": "strategy",
                "strategy_name": "beta_v1",
            },
        ]
    )
    matrix = build_aligned_return_matrix(loaded)

    assert loaded.columns.tolist() == ["ts_utc", "strategy_name", "strategy_return", "run_id", "artifact_type"]
    assert sorted(loaded["artifact_type"].astype("string").unique().tolist()) == ["alpha_sleeve", "strategy"]
    assert loaded["strategy_name"].tolist() == [
        "alpha_sleeve_v1",
        "beta_v1",
        "alpha_sleeve_v1",
        "beta_v1",
    ]
    assert matrix.columns.tolist() == ["alpha_sleeve_v1", "beta_v1"]
    assert matrix.loc[pd.Timestamp("2022-01-01T00:00:00Z"), "alpha_sleeve_v1"] == pytest.approx(0.03)
    assert matrix.loc[pd.Timestamp("2022-01-02T00:00:00Z"), "beta_v1"] == pytest.approx(0.02)


def test_load_portfolio_component_runs_returns_rejects_missing_alpha_sleeve_artifact(tmp_path: Path) -> None:
    run_dir = tmp_path / "alpha_missing_sleeve"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"alpha_name": "alpha_model_v1", "run_id": "alpha-run"}),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="sleeve_returns\\.csv"):
        load_portfolio_component_runs_returns(
            [
                {
                    "source_artifact_path": run_dir,
                    "artifact_type": "alpha_sleeve",
                    "strategy_name": "alpha_sleeve_v1",
                }
            ]
        )


def _write_run(
    root: Path,
    *,
    run_id: str,
    strategy_name: str,
    rows: list[dict[str, object]],
) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(
        json.dumps({"strategy_name": strategy_name}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    frame = pd.DataFrame(rows)
    if "equity" not in frame.columns:
        frame["equity"] = 1.0
    frame["symbol"] = "SPY"
    frame["signal"] = 1.0
    frame["position"] = 1.0
    frame.to_csv(run_dir / "equity_curve.csv", index=False)
    return run_dir


def _write_alpha_sleeve_run(
    root: Path,
    *,
    run_id: str,
    alpha_name: str,
    rows: list[dict[str, object]],
) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(
        json.dumps({"alpha_name": alpha_name}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "alpha_name": alpha_name}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(run_dir / "sleeve_returns.csv", index=False)
    return run_dir
