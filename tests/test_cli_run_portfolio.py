from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli.run_portfolio import (
    PortfolioRunResult,
    PortfolioWalkForwardRunResult,
    parse_args,
    parse_run_ids,
    run_cli,
)


def test_parse_args_accepts_portfolio_runner_flags() -> None:
    args = parse_args(
        [
            "--portfolio-config",
            "configs/portfolios.yml",
            "--portfolio-name",
            "core_portfolio",
            "--from-registry",
            "--evaluation",
            "configs/evaluation.yml",
            "--output-dir",
            "artifacts/custom-portfolios",
            "--timeframe",
            "1D",
        ]
    )

    assert args.portfolio_config == "configs/portfolios.yml"
    assert args.portfolio_name == "core_portfolio"
    assert args.from_registry is True
    assert args.evaluation == "configs/evaluation.yml"
    assert args.output_dir == "artifacts/custom-portfolios"
    assert args.timeframe == "1D"


def test_parse_run_ids_supports_mixed_cli_formats() -> None:
    assert parse_run_ids(["run-a,run-b", "run-c", " run-d "]) == [
        "run-a",
        "run-b",
        "run-c",
        "run-d",
    ]


def test_run_cli_builds_portfolio_from_explicit_run_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    strategy_root = tmp_path / "artifacts" / "strategies"
    portfolio_root = tmp_path / "artifacts" / "portfolios"
    alpha_run = _write_strategy_run(
        strategy_root,
        run_id="run-alpha",
        strategy_name="alpha_v1",
        rows=[
            {"ts_utc": "2025-01-01T00:00:00Z", "strategy_return": 0.01},
            {"ts_utc": "2025-01-02T00:00:00Z", "strategy_return": 0.02},
        ],
    )
    _write_strategy_run(
        strategy_root,
        run_id="run-beta",
        strategy_name="beta_v1",
        rows=[
            {"ts_utc": "2025-01-01T00:00:00Z", "strategy_return": 0.03},
            {"ts_utc": "2025-01-02T00:00:00Z", "strategy_return": -0.01},
        ],
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.cli.run_portfolio.experiment_tracker.ARTIFACTS_ROOT", strategy_root)
    monkeypatch.setattr("src.cli.run_portfolio.DEFAULT_PORTFOLIO_ARTIFACTS_ROOT", portfolio_root)

    result = run_cli(
        [
            "--portfolio-name",
            "core_portfolio",
            "--run-ids",
            "run-beta",
            "run-alpha",
            "--timeframe",
            "1D",
        ]
    )

    assert isinstance(result, PortfolioRunResult)
    assert result.portfolio_name == "core_portfolio"
    assert result.allocator_name == "equal_weight"
    assert result.component_count == 2
    assert result.timeframe == "1D"
    assert [component["run_id"] for component in result.components] == ["run-alpha", "run-beta"]
    assert result.experiment_dir == portfolio_root / result.run_id
    assert (result.experiment_dir / "manifest.json").exists()
    assert (result.experiment_dir / "qa_summary.json").exists()

    registry_entries = _read_registry(portfolio_root / "registry.jsonl")
    assert [entry["run_id"] for entry in registry_entries] == [result.run_id]
    assert registry_entries[0]["component_run_ids"] == ["run-alpha", "run-beta"]
    assert registry_entries[0]["artifact_path"] == result.experiment_dir.as_posix()

    stdout = capsys.readouterr().out
    assert "Portfolio: core_portfolio" in stdout
    assert f"Run ID: {result.run_id}" in stdout
    assert "Allocator: equal_weight" in stdout
    assert "Components: 2 strategies" in stdout
    assert "Timeframe: 1D" in stdout
    assert "Total Return:" in stdout
    assert "Sharpe Ratio:" in stdout
    assert "Max Drawdown:" in stdout
    assert alpha_run.exists()


def test_run_cli_builds_portfolio_from_config_run_ids(
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
    config_path = tmp_path / "portfolio.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "portfolios": {
                    "core_portfolio": {
                        "allocator": "equal_weight",
                        "initial_capital": 100.0,
                        "alignment_policy": "intersection",
                        "components": [
                            {"strategy_name": "beta_v1", "run_id": "run-beta"},
                            {"strategy_name": "alpha_v1", "run_id": "run-alpha"},
                        ],
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.cli.run_portfolio.experiment_tracker.ARTIFACTS_ROOT", strategy_root)
    monkeypatch.setattr("src.cli.run_portfolio.DEFAULT_PORTFOLIO_ARTIFACTS_ROOT", portfolio_root)

    result = run_cli(
        [
            "--portfolio-config",
            str(config_path),
            "--portfolio-name",
            "core_portfolio",
            "--timeframe",
            "1D",
        ]
    )

    assert result.config["portfolio_name"] == "core_portfolio"
    assert result.config["initial_capital"] == pytest.approx(100.0)
    assert result.config["alignment_policy"] == "intersection"
    assert result.config["execution"] == {
        "enabled": False,
        "execution_delay": 1,
        "transaction_cost_bps": 0.0,
        "slippage_bps": 0.0,
    }
    assert result.config["timeframe"] == "1D"


def test_run_cli_selects_latest_registry_run_per_strategy_deterministically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    strategy_root = tmp_path / "artifacts" / "strategies"
    portfolio_root = tmp_path / "artifacts" / "portfolios"
    _write_strategy_run(
        strategy_root,
        run_id="run-alpha-old",
        strategy_name="alpha_v1",
        rows=[
            {"ts_utc": "2025-01-01T00:00:00Z", "strategy_return": 0.01},
            {"ts_utc": "2025-01-02T00:00:00Z", "strategy_return": 0.01},
        ],
    )
    _write_strategy_run(
        strategy_root,
        run_id="run-alpha-new",
        strategy_name="alpha_v1",
        rows=[
            {"ts_utc": "2025-01-01T00:00:00Z", "strategy_return": 0.02},
            {"ts_utc": "2025-01-02T00:00:00Z", "strategy_return": 0.01},
        ],
    )
    _write_strategy_run(
        strategy_root,
        run_id="run-beta-z",
        strategy_name="beta_v1",
        rows=[
            {"ts_utc": "2025-01-01T00:00:00Z", "strategy_return": 0.01},
            {"ts_utc": "2025-01-02T00:00:00Z", "strategy_return": -0.01},
        ],
    )
    _write_strategy_run(
        strategy_root,
        run_id="run-beta-a",
        strategy_name="beta_v1",
        rows=[
            {"ts_utc": "2025-01-01T00:00:00Z", "strategy_return": 0.03},
            {"ts_utc": "2025-01-02T00:00:00Z", "strategy_return": -0.02},
        ],
    )
    registry_path = strategy_root / "registry.jsonl"
    registry_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "run_id": "run-alpha-old",
                        "run_type": "strategy",
                        "timestamp": "2026-03-24T00:00:00Z",
                        "strategy_name": "alpha_v1",
                        "timeframe": "1D",
                        "artifact_path": (strategy_root / "run-alpha-old").as_posix(),
                    }
                ),
                json.dumps(
                    {
                        "run_id": "run-alpha-new",
                        "run_type": "strategy",
                        "timestamp": "2026-03-25T00:00:00Z",
                        "strategy_name": "alpha_v1",
                        "timeframe": "1D",
                        "artifact_path": (strategy_root / "run-alpha-new").as_posix(),
                    }
                ),
                json.dumps(
                    {
                        "run_id": "run-beta-z",
                        "run_type": "strategy",
                        "timestamp": "2026-03-25T00:05:00Z",
                        "strategy_name": "beta_v1",
                        "timeframe": "1D",
                        "artifact_path": (strategy_root / "run-beta-z").as_posix(),
                    }
                ),
                json.dumps(
                    {
                        "run_id": "run-beta-a",
                        "run_type": "strategy",
                        "timestamp": "2026-03-25T00:05:00Z",
                        "strategy_name": "beta_v1",
                        "timeframe": "1D",
                        "artifact_path": (strategy_root / "run-beta-a").as_posix(),
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "portfolio.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "portfolio_name": "registry_portfolio",
                "allocator": "equal_weight",
                "components": [
                    {"strategy_name": "beta_v1"},
                    {"strategy_name": "alpha_v1"},
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.cli.run_portfolio.experiment_tracker.ARTIFACTS_ROOT", strategy_root)
    monkeypatch.setattr("src.cli.run_portfolio.DEFAULT_PORTFOLIO_ARTIFACTS_ROOT", portfolio_root)

    result = run_cli(
        [
            "--portfolio-config",
            str(config_path),
            "--from-registry",
            "--timeframe",
            "1D",
        ]
    )

    assert [component["run_id"] for component in result.components] == [
        "run-alpha-new",
        "run-beta-z",
    ]


def test_run_cli_is_deterministic_for_identical_inputs(
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
        ],
    )
    _write_strategy_run(
        strategy_root,
        run_id="run-beta",
        strategy_name="beta_v1",
        rows=[
            {"ts_utc": "2025-01-01T00:00:00Z", "strategy_return": 0.01},
            {"ts_utc": "2025-01-02T00:00:00Z", "strategy_return": 0.03},
        ],
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.cli.run_portfolio.experiment_tracker.ARTIFACTS_ROOT", strategy_root)
    monkeypatch.setattr("src.cli.run_portfolio.DEFAULT_PORTFOLIO_ARTIFACTS_ROOT", portfolio_root)

    first = run_cli(
        [
            "--portfolio-name",
            "core_portfolio",
            "--run-ids",
            "run-alpha",
            "run-beta",
            "--timeframe",
            "1D",
        ]
    )
    first_artifacts = _artifact_bytes(first.experiment_dir)

    second = run_cli(
        [
            "--portfolio-name",
            "core_portfolio",
            "--run-ids",
            "run-alpha",
            "run-beta",
            "--timeframe",
            "1D",
        ]
    )
    second_artifacts = _artifact_bytes(second.experiment_dir)

    assert first.run_id == second.run_id
    assert first.metrics == second.metrics
    assert first_artifacts == second_artifacts


def test_run_cli_supports_walk_forward_portfolios(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
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
    evaluation_path.write_text(
        yaml.safe_dump(
            {
                "evaluation": {
                    "mode": "rolling",
                    "timeframe": "1d",
                    "start": "2025-01-01",
                    "end": "2025-01-05",
                    "train_window": "2D",
                    "test_window": "1D",
                    "step": "1D",
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.cli.run_portfolio.experiment_tracker.ARTIFACTS_ROOT", strategy_root)
    monkeypatch.setattr("src.portfolio.walk_forward.experiment_tracker.ARTIFACTS_ROOT", strategy_root)
    monkeypatch.setattr("src.cli.run_portfolio.DEFAULT_PORTFOLIO_ARTIFACTS_ROOT", portfolio_root)

    result = run_cli(
        [
            "--portfolio-name",
            "core_portfolio",
            "--run-ids",
            "run-alpha",
            "run-beta",
            "--evaluation",
            str(evaluation_path),
            "--timeframe",
            "1D",
        ]
    )

    assert isinstance(result, PortfolioWalkForwardRunResult)
    assert result.split_count == 2
    assert result.experiment_dir == portfolio_root / result.run_id
    assert (result.experiment_dir / "aggregate_metrics.json").exists()
    assert (result.experiment_dir / "metrics_by_split.csv").exists()
    assert (result.experiment_dir / "splits" / "rolling_0000" / "qa_summary.json").exists()
    assert result.aggregate_metrics["metric_summary"]["total_return"] == pytest.approx(0.0175)

    registry_entries = _read_registry(portfolio_root / "registry.jsonl")
    assert [entry["run_id"] for entry in registry_entries] == [result.run_id]
    assert registry_entries[0]["split_count"] == 2
    assert registry_entries[0]["evaluation_config_path"] == evaluation_path.as_posix()

    stdout = capsys.readouterr().out
    assert "Portfolio: core_portfolio" in stdout
    assert f"Run ID: {result.run_id}" in stdout
    assert "Splits: 2" in stdout
    assert "Mean Total Return:" in stdout
    assert "Mean Sharpe Ratio:" in stdout
    assert "Worst Max Drawdown:" in stdout


def test_run_cli_rejects_missing_run_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy_root = tmp_path / "artifacts" / "strategies"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.cli.run_portfolio.experiment_tracker.ARTIFACTS_ROOT", strategy_root)

    with pytest.raises(ValueError, match="could not be resolved to an artifact directory"):
        run_cli(
            [
                "--portfolio-name",
                "core_portfolio",
                "--run-ids",
                "missing-run",
                "--timeframe",
                "1D",
            ]
        )


def test_run_cli_rejects_invalid_argument_combinations() -> None:
    with pytest.raises(ValueError, match="exactly one of --portfolio-config or --run-ids"):
        run_cli(["--portfolio-name", "core_portfolio", "--timeframe", "1D"])

    with pytest.raises(ValueError, match="cannot be combined with --from-registry"):
        run_cli(
            [
                "--portfolio-name",
                "core_portfolio",
                "--run-ids",
                "run-a",
                "--from-registry",
                "--timeframe",
                "1D",
            ]
        )


def test_run_cli_rejects_non_overlapping_component_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy_root = tmp_path / "artifacts" / "strategies"
    _write_strategy_run(
        strategy_root,
        run_id="run-alpha",
        strategy_name="alpha_v1",
        rows=[{"ts_utc": "2025-01-01T00:00:00Z", "strategy_return": 0.01}],
    )
    _write_strategy_run(
        strategy_root,
        run_id="run-beta",
        strategy_name="beta_v1",
        rows=[{"ts_utc": "2025-01-02T00:00:00Z", "strategy_return": 0.01}],
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.cli.run_portfolio.experiment_tracker.ARTIFACTS_ROOT", strategy_root)

    with pytest.raises(
        ValueError,
        match="Aligned return matrix is empty under 'intersection' alignment",
    ):
        run_cli(
            [
                "--portfolio-name",
                "core_portfolio",
                "--run-ids",
                "run-alpha",
                "run-beta",
                "--timeframe",
                "1D",
            ]
        )


def test_run_cli_strict_sanity_failure_prevents_portfolio_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy_root = tmp_path / "artifacts" / "strategies"
    _write_strategy_run(
        strategy_root,
        run_id="run-alpha",
        strategy_name="alpha_v1",
        rows=[{"ts_utc": "2025-01-01T00:00:00Z", "strategy_return": 0.01}],
    )
    _write_strategy_run(
        strategy_root,
        run_id="run-beta",
        strategy_name="beta_v1",
        rows=[{"ts_utc": "2025-01-01T00:00:00Z", "strategy_return": 0.01}],
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.cli.run_portfolio.experiment_tracker.ARTIFACTS_ROOT", strategy_root)

    portfolio_output = pd.DataFrame(
        {
            "ts_utc": pd.to_datetime(["2025-01-01T00:00:00Z"], utc=True),
            "strategy_return__alpha_v1": [0.6],
            "strategy_return__beta_v1": [0.6],
            "weight__alpha_v1": [0.5],
            "weight__beta_v1": [0.5],
            "gross_portfolio_return": [0.6],
            "portfolio_weight_change": [1.0],
            "portfolio_abs_weight_change": [1.0],
            "portfolio_turnover": [1.0],
            "portfolio_rebalance_event": [1],
            "portfolio_transaction_cost": [0.0],
            "portfolio_slippage_cost": [0.0],
            "portfolio_execution_friction": [0.0],
            "net_portfolio_return": [0.6],
            "portfolio_return": [0.6],
            "portfolio_equity_curve": [1.6],
        }
    )
    write_calls = {"count": 0}

    monkeypatch.setattr("src.cli.run_portfolio.construct_portfolio", lambda *args, **kwargs: portfolio_output)
    monkeypatch.setattr(
        "src.cli.run_portfolio.load_strategy_runs_returns",
        lambda run_dirs: pd.DataFrame({"ts_utc": pd.to_datetime(["2025-01-01T00:00:00Z"], utc=True)}),
    )
    monkeypatch.setattr(
        "src.cli.run_portfolio.build_aligned_return_matrix",
        lambda strategy_returns: pd.DataFrame({"alpha_v1": [0.6], "beta_v1": [0.6]}, index=pd.DatetimeIndex(pd.to_datetime(["2025-01-01T00:00:00Z"], utc=True), name="ts_utc")),
    )
    monkeypatch.setattr(
        "src.cli.run_portfolio._resolve_portfolio_inputs",
        lambda **kwargs: (
            {
                "portfolio_name": "core_portfolio",
                "allocator": "equal_weight",
                "components": [
                    {"strategy_name": "alpha_v1", "run_id": "run-alpha", "source_artifact_path": "a"},
                    {"strategy_name": "beta_v1", "run_id": "run-beta", "source_artifact_path": "b"},
                ],
                "initial_capital": 1.0,
                "alignment_policy": "intersection",
                "execution": {
                    "enabled": False,
                    "execution_delay": 1,
                    "transaction_cost_bps": 0.0,
                    "slippage_bps": 0.0,
                },
                "validation": None,
                "sanity": {
                    "strict_sanity_checks": True,
                    "max_abs_period_return": 0.1,
                },
                "timeframe": "1D",
                "evaluation_config_path": None,
            },
            [Path("a"), Path("b")],
            [
                {"strategy_name": "alpha_v1", "run_id": "run-alpha", "source_artifact_path": "a"},
                {"strategy_name": "beta_v1", "run_id": "run-beta", "source_artifact_path": "b"},
            ],
        ),
    )
    monkeypatch.setattr(
        "src.portfolio.write_portfolio_artifacts",
        lambda *args, **kwargs: write_calls.__setitem__("count", write_calls["count"] + 1),
    )

    with pytest.raises(ValueError, match="absolute portfolio_return exceeds configured maximum"):
        run_cli(
            [
                "--portfolio-name",
                "core_portfolio",
                "--run-ids",
                "run-alpha",
                "run-beta",
                "--timeframe",
                "1D",
            ]
        )

    assert write_calls["count"] == 0


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


def _read_registry(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _artifact_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
