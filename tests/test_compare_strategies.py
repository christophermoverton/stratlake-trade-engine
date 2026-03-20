from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli.compare_strategies import parse_args, run_cli
from src.research import compare


def test_parse_args_supports_comparison_flags() -> None:
    args = parse_args(
        [
            "--strategies",
            "momentum_v1,mean_reversion_v1",
            "--metric",
            "total_return",
            "--top_k",
            "2",
            "--from_registry",
            "--output_path",
            "artifacts/custom",
        ]
    )

    assert args.strategies == "momentum_v1,mean_reversion_v1"
    assert args.metric == "total_return"
    assert args.top_k == 2
    assert args.from_registry is True
    assert args.output_path == "artifacts/custom"
    assert args.evaluation is None


def test_compare_strategies_runs_fresh_execution_and_writes_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake_run_strategy_experiment(strategy_name: str, start=None, end=None):
        calls.append(strategy_name)
        return compare.StrategyRunResult(
            strategy_name=strategy_name,
            run_id=f"run-{strategy_name}",
            metrics={
                "total_return": 0.11 if strategy_name == "momentum_v1" else 0.05,
                "cumulative_return": 0.11 if strategy_name == "momentum_v1" else 0.05,
                "sharpe_ratio": 1.4 if strategy_name == "momentum_v1" else 0.8,
                "max_drawdown": 0.07 if strategy_name == "momentum_v1" else 0.03,
            },
            experiment_dir=tmp_path / f"run-{strategy_name}",
            results_df=None,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(compare, "run_strategy_experiment", fake_run_strategy_experiment)

    result = compare.compare_strategies(
        ["momentum_v1", "mean_reversion_v1"],
        output_path=tmp_path,
    )

    assert calls == ["momentum_v1", "mean_reversion_v1"]
    assert [entry.strategy_name for entry in result.leaderboard] == [
        "momentum_v1",
        "mean_reversion_v1",
    ]
    assert result.csv_path == tmp_path / "leaderboard.csv"
    assert result.json_path == tmp_path / "leaderboard.json"
    assert result.csv_path.exists()
    assert result.json_path.exists()

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert payload["selection_mode"] == "fresh"
    assert payload["metric"] == "sharpe_ratio"
    assert payload["leaderboard"][0]["strategy_name"] == "momentum_v1"


def test_compare_strategies_runs_walk_forward_execution_when_evaluation_provided(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    strategy_configs = {
        "momentum_v1": {"dataset": "features_daily", "parameters": {"lookback_short": 5}},
        "mean_reversion_v1": {"dataset": "features_daily", "parameters": {"lookback": 10, "threshold": 1.0}},
    }
    calls: list[tuple[str, Path]] = []

    monkeypatch.setattr(compare, "get_strategy_config", lambda strategy_name: strategy_configs[strategy_name])
    monkeypatch.setattr(compare, "build_strategy", lambda strategy_name, config: type("S", (), {"name": strategy_name, "dataset": config["dataset"]})())

    def fake_run_walk_forward_experiment(strategy_name, strategy, evaluation_path, strategy_config):
        calls.append((strategy_name, evaluation_path))
        return compare.WalkForwardRunResult(
            strategy_name=strategy_name,
            run_id=f"wf-{strategy_name}",
            experiment_dir=tmp_path / f"wf-{strategy_name}",
            metrics={
                "total_return": 0.2 if strategy_name == "momentum_v1" else 0.15,
                "cumulative_return": 0.2 if strategy_name == "momentum_v1" else 0.15,
                "sharpe_ratio": 1.9 if strategy_name == "momentum_v1" else 1.1,
                "max_drawdown": 0.08,
            },
            aggregate_summary={"split_count": 2},
            splits=[],
        )

    monkeypatch.setattr(compare, "run_walk_forward_experiment", fake_run_walk_forward_experiment)

    result = compare.compare_strategies(
        ["momentum_v1", "mean_reversion_v1"],
        evaluation_path=Path("configs/evaluation.yml"),
        output_path=tmp_path / "wf.csv",
    )

    assert calls == [
        ("momentum_v1", Path("configs/evaluation.yml")),
        ("mean_reversion_v1", Path("configs/evaluation.yml")),
    ]
    assert result.evaluation_mode == "walk_forward"
    assert result.csv_path == tmp_path / "wf.csv"
    assert [entry.run_id for entry in result.leaderboard] == ["wf-momentum_v1", "wf-mean_reversion_v1"]


def test_compare_strategies_uses_latest_matching_registry_run_and_applies_metric_tie_breaks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifacts" / "strategies"
    registry_path = artifact_root / "registry.jsonl"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "run_id": "run-old",
                        "timestamp": "2026-03-19T00:00:00Z",
                        "strategy_name": "momentum_v1",
                        "evaluation_mode": "single",
                        "metrics_summary": {"sharpe_ratio": 1.0, "total_return": 0.09},
                    }
                ),
                json.dumps(
                    {
                        "run_id": "run-new",
                        "timestamp": "2026-03-19T00:05:00Z",
                        "strategy_name": "momentum_v1",
                        "evaluation_mode": "single",
                        "metrics_summary": {"sharpe_ratio": 1.1, "total_return": 0.1},
                    }
                ),
                json.dumps(
                    {
                        "run_id": "run-alpha",
                        "timestamp": "2026-03-19T00:01:00Z",
                        "strategy_name": "alpha_v1",
                        "evaluation_mode": "single",
                        "metrics_summary": {"sharpe_ratio": 1.1, "total_return": 0.2},
                    }
                )
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(compare.experiment_tracker, "ARTIFACTS_ROOT", artifact_root)

    result = compare.compare_strategies(
        ["momentum_v1", "alpha_v1"],
        from_registry=True,
        output_path=tmp_path,
    )

    assert [entry.run_id for entry in result.leaderboard] == ["run-alpha", "run-new"]
    assert [entry.strategy_name for entry in result.leaderboard] == ["alpha_v1", "momentum_v1"]
    assert result.selection_mode == "registry"


def test_compare_strategies_handles_missing_metric_values_and_top_k(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run_strategy_experiment(strategy_name: str, start=None, end=None):
        sharpe_ratio = None if strategy_name == "buy_and_hold_v1" else 0.9
        return compare.StrategyRunResult(
            strategy_name=strategy_name,
            run_id=f"run-{strategy_name}",
            metrics={
                "total_return": 0.05,
                "cumulative_return": 0.05,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": 0.02,
            },
            experiment_dir=tmp_path / f"run-{strategy_name}",
            results_df=None,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(compare, "run_strategy_experiment", fake_run_strategy_experiment)

    result = compare.compare_strategies(
        ["buy_and_hold_v1", "momentum_v1"],
        top_k=1,
        output_path=tmp_path / "custom.csv",
    )

    assert len(result.leaderboard) == 1
    assert result.leaderboard[0].strategy_name == "momentum_v1"


def test_compare_strategies_raises_for_missing_registry_strategy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifacts" / "strategies"
    registry_path = artifact_root / "registry.jsonl"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(compare.experiment_tracker, "ARTIFACTS_ROOT", artifact_root)

    with pytest.raises(ValueError, match="No registry runs found for strategy 'missing_v1'"):
        compare.compare_strategies(["missing_v1"], from_registry=True, output_path=tmp_path)


def test_run_cli_prints_leaderboard_summary(monkeypatch, capsys, tmp_path: Path) -> None:
    expected_result = compare.ComparisonResult(
        metric="sharpe_ratio",
        evaluation_mode="single",
        selection_mode="fresh",
        selection_rule="freshly executed run per strategy",
        leaderboard=[
            compare.LeaderboardEntry(
                rank=1,
                strategy_name="momentum_v1",
                run_id="run-momentum_v1",
                evaluation_mode="single",
                selected_metric_name="sharpe_ratio",
                selected_metric_value=1.23,
                cumulative_return=0.12,
                total_return=0.12,
                sharpe_ratio=1.23,
                max_drawdown=0.04,
                annualized_return=None,
                annualized_volatility=None,
                volatility=None,
                win_rate=None,
                hit_rate=None,
                profit_factor=None,
                turnover=None,
                exposure_pct=None,
            )
        ],
        csv_path=tmp_path / "leaderboard.csv",
        json_path=tmp_path / "leaderboard.json",
    )

    monkeypatch.setattr("src.cli.compare_strategies.compare_strategies", lambda *args, **kwargs: expected_result)

    result = run_cli(["--strategies", "momentum_v1"])

    assert result is expected_result
    stdout = capsys.readouterr().out
    assert "metric: sharpe_ratio" in stdout
    assert "strategy" in stdout
    assert "momentum_v1" in stdout
    assert "leaderboard_csv:" in stdout
