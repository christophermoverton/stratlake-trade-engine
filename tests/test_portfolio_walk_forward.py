from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.portfolio import EqualWeightAllocator, run_portfolio_walk_forward
from src.portfolio.walk_forward import PortfolioWalkForwardError
from src.research import experiment_tracker
from src.research.registry import default_registry_path, load_registry


def test_run_portfolio_walk_forward_writes_split_and_aggregate_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy_root = tmp_path / "artifacts" / "strategies"
    portfolio_root = tmp_path / "artifacts" / "portfolios"
    _write_strategy_run(
        strategy_root,
        run_id="run-alpha",
        strategy_name="alpha_v1",
        rows=[
            {"ts_utc": "2025-01-01T00:00:00Z", "strategy_return": 0.01},
            {"ts_utc": "2025-01-02T00:00:00Z", "strategy_return": 0.02},
            {"ts_utc": "2025-01-03T00:00:00Z", "strategy_return": 0.03},
            {"ts_utc": "2025-01-04T00:00:00Z", "strategy_return": 0.04},
        ],
    )
    _write_strategy_run(
        strategy_root,
        run_id="run-beta",
        strategy_name="beta_v1",
        rows=[
            {"ts_utc": "2025-01-01T00:00:00Z", "strategy_return": 0.02},
            {"ts_utc": "2025-01-02T00:00:00Z", "strategy_return": 0.00},
            {"ts_utc": "2025-01-03T00:00:00Z", "strategy_return": -0.01},
            {"ts_utc": "2025-01-04T00:00:00Z", "strategy_return": 0.01},
        ],
    )
    evaluation_path = tmp_path / "evaluation.yml"
    _write_evaluation_config(
        evaluation_path,
        {
            "mode": "rolling",
            "timeframe": "1d",
            "start": "2025-01-01",
            "end": "2025-01-05",
            "train_window": "2D",
            "test_window": "1D",
            "step": "1D",
        },
    )

    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", strategy_root)

    result = run_portfolio_walk_forward(
        component_run_ids=["run-beta", "run-alpha"],
        evaluation_config_path=evaluation_path,
        allocator=EqualWeightAllocator(),
        timeframe="1D",
        output_dir=portfolio_root,
        portfolio_name="core_portfolio",
        initial_capital=100.0,
    )

    experiment_dir = Path(result["experiment_dir"])
    assert result["run_id"] == experiment_dir.name
    assert result["split_count"] == 2
    assert (experiment_dir / "config.json").exists()
    assert (experiment_dir / "components.json").exists()
    assert (experiment_dir / "manifest.json").exists()
    assert (experiment_dir / "metrics_by_split.csv").exists()
    assert (experiment_dir / "aggregate_metrics.json").exists()
    assert (experiment_dir / "splits" / "rolling_0000" / "split.json").exists()
    assert (experiment_dir / "splits" / "rolling_0000" / "weights.csv").exists()
    assert (experiment_dir / "splits" / "rolling_0000" / "portfolio_returns.csv").exists()
    assert (experiment_dir / "splits" / "rolling_0000" / "portfolio_equity_curve.csv").exists()
    assert (experiment_dir / "splits" / "rolling_0000" / "metrics.json").exists()
    assert (experiment_dir / "splits" / "rolling_0000" / "qa_summary.json").exists()

    metrics_by_split = pd.read_csv(experiment_dir / "metrics_by_split.csv")
    assert metrics_by_split["split_id"].tolist() == ["rolling_0000", "rolling_0001"]
    assert metrics_by_split["start"].tolist() == ["2025-01-03", "2025-01-04"]
    assert metrics_by_split["end"].tolist() == ["2025-01-04", "2025-01-05"]
    assert metrics_by_split["row_count"].tolist() == [1, 1]
    assert metrics_by_split["total_return"].tolist() == pytest.approx([0.01, 0.025])
    assert "total_turnover" in metrics_by_split.columns
    assert "trade_count" in metrics_by_split.columns

    aggregate_metrics = json.loads((experiment_dir / "aggregate_metrics.json").read_text(encoding="utf-8"))
    assert aggregate_metrics["split_count"] == 2
    assert aggregate_metrics["split_ids"] == ["rolling_0000", "rolling_0001"]
    assert aggregate_metrics["metric_summary"]["total_return"] == pytest.approx(0.0175)
    assert aggregate_metrics["metric_statistics"]["total_return"] == {
        "mean": pytest.approx(0.0175),
        "median": pytest.approx(0.0175),
        "std": pytest.approx(0.0075),
        "min": pytest.approx(0.01),
        "max": pytest.approx(0.025),
    }

    split_metadata = json.loads(
        (experiment_dir / "splits" / "rolling_0000" / "split.json").read_text(encoding="utf-8")
    )
    assert split_metadata == {
        "split_id": "rolling_0000",
        "mode": "rolling",
        "train_start": "2025-01-01",
        "train_end": "2025-01-03",
        "test_start": "2025-01-03",
        "test_end": "2025-01-04",
        "start": "2025-01-03",
        "end": "2025-01-04",
        "row_count": 1,
    }

    manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["split_count"] == 2
    assert manifest["split_artifact_dirs"] == ["splits/rolling_0000", "splits/rolling_0001"]
    assert manifest["aggregate_metric_summary"]["total_return"] == pytest.approx(0.0175)
    assert "aggregate_metrics.json" in manifest["artifact_files"]
    assert "splits/rolling_0000/portfolio_returns.csv" in manifest["artifact_files"]

    registry_entries = load_registry(default_registry_path(portfolio_root))
    assert [entry["run_id"] for entry in registry_entries] == [result["run_id"]]
    assert registry_entries[0]["split_count"] == 2
    assert registry_entries[0]["evaluation_config_path"] == evaluation_path.as_posix()
    assert registry_entries[0]["metrics"]["total_return"] == pytest.approx(0.0175)
    assert registry_entries[0]["metadata"]["aggregate_metrics"]["split_ids"] == [
        "rolling_0000",
        "rolling_0001",
    ]


def test_run_portfolio_walk_forward_is_deterministic_for_identical_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy_root = tmp_path / "artifacts" / "strategies"
    portfolio_root = tmp_path / "artifacts" / "portfolios"
    _write_strategy_run(
        strategy_root,
        run_id="run-alpha",
        strategy_name="alpha_v1",
        rows=[
            {"ts_utc": "2025-01-01T00:00:00Z", "strategy_return": 0.01},
            {"ts_utc": "2025-01-02T00:00:00Z", "strategy_return": 0.02},
            {"ts_utc": "2025-01-03T00:00:00Z", "strategy_return": 0.03},
        ],
    )
    _write_strategy_run(
        strategy_root,
        run_id="run-beta",
        strategy_name="beta_v1",
        rows=[
            {"ts_utc": "2025-01-01T00:00:00Z", "strategy_return": 0.03},
            {"ts_utc": "2025-01-02T00:00:00Z", "strategy_return": 0.01},
            {"ts_utc": "2025-01-03T00:00:00Z", "strategy_return": -0.01},
        ],
    )
    evaluation_path = tmp_path / "evaluation.yml"
    _write_evaluation_config(
        evaluation_path,
        {
            "mode": "fixed",
            "timeframe": "1d",
            "train_start": "2025-01-01",
            "train_end": "2025-01-02",
            "test_start": "2025-01-02",
            "test_end": "2025-01-04",
        },
    )

    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", strategy_root)

    first = run_portfolio_walk_forward(
        component_run_ids=["run-alpha", "run-beta"],
        evaluation_config_path=evaluation_path,
        allocator=EqualWeightAllocator(),
        timeframe="1D",
        output_dir=portfolio_root,
        portfolio_name="core_portfolio",
    )
    second = run_portfolio_walk_forward(
        component_run_ids=["run-alpha", "run-beta"],
        evaluation_config_path=evaluation_path,
        allocator=EqualWeightAllocator(),
        timeframe="1D",
        output_dir=portfolio_root,
        portfolio_name="core_portfolio",
    )

    assert first["run_id"] == second["run_id"]
    assert first["aggregate_metrics"] == second["aggregate_metrics"]
    assert _artifact_bytes(Path(first["experiment_dir"])) == _artifact_bytes(Path(second["experiment_dir"]))


def test_run_portfolio_walk_forward_rejects_missing_split_component_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy_root = tmp_path / "artifacts" / "strategies"
    portfolio_root = tmp_path / "artifacts" / "portfolios"
    _write_strategy_run(
        strategy_root,
        run_id="run-alpha",
        strategy_name="alpha_v1",
        rows=[
            {"ts_utc": "2025-01-01T00:00:00Z", "strategy_return": 0.01},
            {"ts_utc": "2025-01-02T00:00:00Z", "strategy_return": 0.02},
            {"ts_utc": "2025-01-03T00:00:00Z", "strategy_return": 0.03},
        ],
    )
    _write_strategy_run(
        strategy_root,
        run_id="run-beta",
        strategy_name="beta_v1",
        rows=[
            {"ts_utc": "2025-01-01T00:00:00Z", "strategy_return": 0.02},
            {"ts_utc": "2025-01-02T00:00:00Z", "strategy_return": 0.01},
        ],
    )
    evaluation_path = tmp_path / "evaluation.yml"
    _write_evaluation_config(
        evaluation_path,
        {
            "mode": "fixed",
            "timeframe": "1d",
            "train_start": "2025-01-01",
            "train_end": "2025-01-02",
            "test_start": "2025-01-03",
            "test_end": "2025-01-04",
        },
    )

    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", strategy_root)

    with pytest.raises(PortfolioWalkForwardError, match="missing component returns"):
        run_portfolio_walk_forward(
            component_run_ids=["run-alpha", "run-beta"],
            evaluation_config_path=evaluation_path,
            allocator=EqualWeightAllocator(),
            timeframe="1D",
            output_dir=portfolio_root,
            portfolio_name="core_portfolio",
        )


def _write_strategy_run(
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
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "strategy_name": strategy_name}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    frame = pd.DataFrame(rows)
    frame["equity"] = 1.0
    frame["symbol"] = "SPY"
    frame["signal"] = 1.0
    frame["position"] = 1.0
    frame.to_csv(run_dir / "equity_curve.csv", index=False)
    return run_dir


def _write_evaluation_config(path: Path, payload: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump({"evaluation": payload}, sort_keys=False), encoding="utf-8")


def _artifact_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
