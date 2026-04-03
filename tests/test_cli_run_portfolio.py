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
    assert args.strict is False


def test_parse_args_accepts_optimizer_and_risk_overrides() -> None:
    args = parse_args(
        [
            "--portfolio-name",
            "core_portfolio",
            "--run-ids",
            "run-a",
            "--timeframe",
            "1D",
            "--optimizer-method",
            "max_sharpe",
            "--enable-volatility-targeting",
            "--volatility-target-volatility",
            "0.10",
            "--volatility-target-lookback",
            "15",
            "--risk-target-volatility",
            "0.12",
            "--risk-volatility-window",
            "30",
            "--risk-var-confidence-level",
            "0.9",
            "--risk-cvar-confidence-level",
            "0.9",
            "--risk-allow-scale-up",
            "--risk-max-volatility-scale",
            "1.5",
        ]
    )

    assert args.optimizer_method == "max_sharpe"
    assert args.enable_volatility_targeting is True
    assert args.volatility_target_volatility == pytest.approx(0.10)
    assert args.volatility_target_lookback == 15
    assert args.risk_target_volatility == pytest.approx(0.12)
    assert args.risk_volatility_window == 30
    assert args.risk_var_confidence_level == pytest.approx(0.9)
    assert args.risk_cvar_confidence_level == pytest.approx(0.9)
    assert args.risk_allow_scale_up is True
    assert args.risk_max_volatility_scale == pytest.approx(1.5)


def test_parse_args_accepts_portfolio_strict_flag() -> None:
    args = parse_args(
        [
            "--portfolio-name",
            "core_portfolio",
            "--run-ids",
            "run-a",
            "--timeframe",
            "1D",
            "--strict",
        ]
    )

    assert args.strict is True


def test_parse_args_accepts_portfolio_simulation_flag() -> None:
    args = parse_args(
        [
            "--portfolio-name",
            "core_portfolio",
            "--run-ids",
            "run-a",
            "--timeframe",
            "1D",
            "--simulation",
            "configs/simulation.yml",
        ]
    )

    assert args.simulation == "configs/simulation.yml"


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
    assert f"Artifact Dir: {result.experiment_dir.as_posix()}" in stdout
    assert "Allocator: equal_weight" in stdout
    assert "Optimizer Method: equal_weight" in stdout
    assert "Components: 2 strategies" in stdout
    assert "Timeframe: 1D" in stdout
    assert "Total Return:" in stdout
    assert "Sharpe Ratio:" in stdout
    assert "Realized Volatility:" in stdout
    assert "Max Drawdown:" in stdout
    assert "VaR:" in stdout
    assert "CVaR:" in stdout
    assert "Simulation: disabled" in stdout
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
    assert result.config["optimizer"]["method"] == "equal_weight"
    assert result.config["optimizer"]["long_only"] is True
    assert result.config["execution"] == {
        "enabled": False,
        "execution_delay": 1,
        "transaction_cost_bps": 0.0,
        "slippage_bps": 0.0,
        "fixed_fee": 0.0,
        "fixed_fee_model": "per_rebalance",
        "slippage_model": "constant",
        "slippage_turnover_scale": 1.0,
        "slippage_volatility_scale": 1.0,
    }
    assert result.config["strict_mode"] == {
        "enabled": False,
        "source": "default",
    }
    assert result.config["validation"]["long_only"] is True
    assert result.config["risk"] == {
        "volatility_window": 20,
        "target_volatility": None,
        "min_volatility_scale": 0.0,
        "max_volatility_scale": 1.0,
        "allow_scale_up": False,
        "var_confidence_level": 0.95,
        "cvar_confidence_level": 0.95,
        "volatility_epsilon": 1e-12,
        "periods_per_year_override": None,
    }
    assert result.config["volatility_targeting"] == {
        "enabled": False,
        "target_volatility": None,
        "lookback_periods": 20,
        "volatility_epsilon": 1e-12,
    }
    assert result.config["timeframe"] == "1D"


def test_run_cli_builds_portfolio_from_config_alpha_sleeve_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy_root = tmp_path / "artifacts" / "strategies"
    alpha_root = tmp_path / "artifacts" / "alpha"
    portfolio_root = tmp_path / "artifacts" / "portfolios"
    _write_alpha_sleeve_run(
        alpha_root,
        run_id="alpha-sleeve-run",
        alpha_name="alpha_model_v1",
        rows=[
            {"ts_utc": "2025-01-01T00:00:00Z", "sleeve_return": 0.03},
            {"ts_utc": "2025-01-02T00:00:00Z", "sleeve_return": -0.01},
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
                "portfolio_name": "alpha_sleeve_portfolio",
                "allocator": "equal_weight",
                "components": [
                    {
                        "strategy_name": "alpha_sleeve_v1",
                        "run_id": "alpha-sleeve-run",
                        "artifact_type": "alpha_sleeve",
                    },
                    {
                        "strategy_name": "beta_v1",
                        "run_id": "run-beta",
                        "artifact_type": "strategy",
                    },
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
            "--timeframe",
            "1D",
        ]
    )

    assert isinstance(result, PortfolioRunResult)
    assert [component["artifact_type"] for component in result.components] == ["alpha_sleeve", "strategy"]
    assert [component["strategy_name"] for component in result.components] == ["alpha_sleeve_v1", "beta_v1"]
    returns_frame = pd.read_csv(result.experiment_dir / "portfolio_returns.csv")
    assert "strategy_return__alpha_sleeve_v1" in returns_frame.columns
    assert "strategy_return__beta_v1" in returns_frame.columns
    components_payload = json.loads((result.experiment_dir / "components.json").read_text(encoding="utf-8"))
    assert components_payload["components"][0]["artifact_type"] == "alpha_sleeve"
    assert components_payload["components"][0]["source_artifact_path"] == "artifacts/alpha/alpha-sleeve-run"


def test_run_cli_applies_optimizer_and_risk_cli_overrides_deterministically(
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
            {"ts_utc": "2025-01-03T00:00:00Z", "strategy_return": 0.01},
        ],
    )
    _write_strategy_run(
        strategy_root,
        run_id="run-beta",
        strategy_name="beta_v1",
        rows=[
            {"ts_utc": "2025-01-01T00:00:00Z", "strategy_return": 0.00},
            {"ts_utc": "2025-01-02T00:00:00Z", "strategy_return": 0.01},
            {"ts_utc": "2025-01-03T00:00:00Z", "strategy_return": 0.03},
        ],
    )
    config_path = tmp_path / "portfolio.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "portfolio_name": "override_portfolio",
                "allocator": "equal_weight",
                "optimizer": {"method": "equal_weight"},
                "risk": {
                    "volatility_window": 20,
                    "target_volatility": None,
                    "allow_scale_up": False,
                    "max_volatility_scale": 1.0,
                    "var_confidence_level": 0.95,
                    "cvar_confidence_level": 0.95,
                },
                "components": [
                    {"strategy_name": "alpha_v1", "run_id": "run-alpha"},
                    {"strategy_name": "beta_v1", "run_id": "run-beta"},
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
            "--timeframe",
            "1D",
            "--optimizer-method",
            "max_sharpe",
            "--enable-volatility-targeting",
            "--volatility-target-volatility",
            "0.10",
            "--volatility-target-lookback",
            "15",
            "--risk-target-volatility",
            "0.12",
            "--risk-volatility-window",
            "30",
            "--risk-var-confidence-level",
            "0.9",
            "--risk-cvar-confidence-level",
            "0.9",
            "--risk-allow-scale-up",
            "--risk-max-volatility-scale",
            "1.5",
        ]
    )

    assert result.allocator_name == "max_sharpe"
    assert result.config["allocator"] == "max_sharpe"
    assert result.config["optimizer"]["method"] == "max_sharpe"
    assert result.config["volatility_targeting"]["enabled"] is True
    assert result.config["volatility_targeting"]["target_volatility"] == pytest.approx(0.10)
    assert result.config["volatility_targeting"]["lookback_periods"] == 15
    assert result.config["risk"]["target_volatility"] == pytest.approx(0.12)
    assert result.config["risk"]["volatility_window"] == 30
    assert result.config["risk"]["allow_scale_up"] is True
    assert result.config["risk"]["max_volatility_scale"] == pytest.approx(1.5)
    assert result.config["risk"]["var_confidence_level"] == pytest.approx(0.9)
    assert result.config["risk"]["cvar_confidence_level"] == pytest.approx(0.9)

    config_payload = json.loads((result.experiment_dir / "config.json").read_text(encoding="utf-8"))
    assert config_payload["allocator"] == "max_sharpe"
    assert config_payload["optimizer"]["method"] == "max_sharpe"
    assert config_payload["volatility_targeting"]["enabled"] is True
    assert config_payload["volatility_targeting"]["target_volatility"] == pytest.approx(0.10)
    assert config_payload["risk"]["target_volatility"] == pytest.approx(0.12)
    assert config_payload["runtime"]["risk"]["volatility_window"] == 30


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
    assert f"Artifact Dir: {result.experiment_dir.as_posix()}" in stdout
    assert "Optimizer Method: equal_weight" in stdout
    assert "Splits: 2" in stdout
    assert "Mean Total Return:" in stdout
    assert "Mean Sharpe Ratio:" in stdout
    assert "Mean Realized Volatility:" in stdout
    assert "Worst Max Drawdown:" in stdout
    assert "Mean VaR:" in stdout
    assert "Mean CVaR:" in stdout
    assert "Simulation: disabled" in stdout


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

    with pytest.raises(ValueError, match="mutually exclusive"):
        run_cli(
            [
                "--portfolio-name",
                "core_portfolio",
                "--run-ids",
                "run-a",
                "--timeframe",
                "1D",
                "--execution-enabled",
                "--disable-execution-model",
            ]
        )

    with pytest.raises(ValueError, match="cannot be combined with --evaluation"):
        run_cli(
            [
                "--portfolio-name",
                "core_portfolio",
                "--run-ids",
                "run-a",
                "--timeframe",
                "1D",
                "--evaluation",
                "configs/evaluation.yml",
                "--simulation",
                "configs/simulation.yml",
            ]
        )

    with pytest.raises(ValueError, match="requires --risk-max-volatility-scale > 1.0"):
        run_cli(
            [
                "--portfolio-name",
                "core_portfolio",
                "--run-ids",
                "run-a",
                "--timeframe",
                "1D",
                "--risk-allow-scale-up",
                "--risk-max-volatility-scale",
                "1.0",
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
    registry_calls = {"count": 0}

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
    monkeypatch.setattr(
        "src.cli.run_portfolio.register_portfolio_run",
        lambda *args, **kwargs: registry_calls.__setitem__("count", registry_calls["count"] + 1),
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
    assert registry_calls["count"] == 0


def test_run_cli_passes_strict_flag_to_portfolio_walk_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.cli.run_portfolio._resolve_portfolio_inputs",
        lambda **kwargs: (
            {
                "portfolio_name": "core_portfolio",
                "allocator": "equal_weight",
                "components": [
                    {"strategy_name": "alpha_v1", "run_id": "run-alpha", "source_artifact_path": "a"},
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
                "sanity": None,
                "timeframe": "1D",
                "evaluation_config_path": "configs/evaluation.yml",
            },
            [Path("a")],
            [
                {"strategy_name": "alpha_v1", "run_id": "run-alpha", "source_artifact_path": "a"},
            ],
        ),
    )
    calls: dict[str, object] = {}

    def fake_run_portfolio_walk_forward(**kwargs):
        calls.update(kwargs)
        return {
            "portfolio_name": "core_portfolio",
            "run_id": "wf-portfolio",
            "allocator_name": "equal_weight",
            "timeframe": "1D",
            "component_count": 1,
            "split_count": 1,
            "metrics": {"total_return": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0},
            "aggregate_metrics": {
                "metric_statistics": {
                    "total_return": {"mean": 0.0},
                    "sharpe_ratio": {"mean": 0.0},
                    "max_drawdown": {"min": 0.0},
                    "conditional_value_at_risk": {"mean": 0.0},
                }
            },
            "experiment_dir": Path("artifacts/portfolios/wf-portfolio"),
            "config": {"strict_mode": {"enabled": True, "source": "cli"}},
            "components": [{"strategy_name": "alpha_v1", "run_id": "run-alpha"}],
        }

    monkeypatch.setattr("src.cli.run_portfolio.run_portfolio_walk_forward", fake_run_portfolio_walk_forward)

    run_cli(
        [
            "--portfolio-name",
            "core_portfolio",
            "--run-ids",
            "run-alpha",
            "--evaluation",
            "configs/evaluation.yml",
            "--timeframe",
            "1D",
            "--strict",
        ]
    )

    assert calls["strict_mode"] is True
    assert calls["validation_config"]["strict_sanity_checks"] is True
    assert calls["risk_config"]["volatility_window"] == 20
    assert calls["sanity_config"]["strict_sanity_checks"] is True


def test_run_cli_strict_portfolio_config_from_registry_succeeds_and_persists_auditable_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy_root = tmp_path / "artifacts" / "strategies"
    portfolio_root = tmp_path / "artifacts" / "portfolios"
    repo_root = Path(__file__).resolve().parents[1]
    portfolio_config_path = repo_root / "configs" / "portfolios.yml"

    _write_registered_strategy_run(
        strategy_root,
        run_id="run-momentum",
        strategy_name="momentum_v1",
        rows=[
            {"ts_utc": "2025-01-01T00:00:00Z", "strategy_return": 0.010},
            {"ts_utc": "2025-01-02T00:00:00Z", "strategy_return": -0.012},
            {"ts_utc": "2025-01-03T00:00:00Z", "strategy_return": 0.015},
            {"ts_utc": "2025-01-04T00:00:00Z", "strategy_return": -0.010},
            {"ts_utc": "2025-01-05T00:00:00Z", "strategy_return": 0.008},
            {"ts_utc": "2025-01-06T00:00:00Z", "strategy_return": -0.009},
            {"ts_utc": "2025-01-07T00:00:00Z", "strategy_return": 0.011},
            {"ts_utc": "2025-01-08T00:00:00Z", "strategy_return": -0.007},
        ],
        timeframe="1D",
        timestamp="2026-03-25T00:00:00Z",
    )
    _write_registered_strategy_run(
        strategy_root,
        run_id="run-meanrev",
        strategy_name="mean_reversion_v1",
        rows=[
            {"ts_utc": "2025-01-01T00:00:00Z", "strategy_return": -0.004},
            {"ts_utc": "2025-01-02T00:00:00Z", "strategy_return": 0.007},
            {"ts_utc": "2025-01-03T00:00:00Z", "strategy_return": -0.006},
            {"ts_utc": "2025-01-04T00:00:00Z", "strategy_return": 0.005},
            {"ts_utc": "2025-01-05T00:00:00Z", "strategy_return": -0.003},
            {"ts_utc": "2025-01-06T00:00:00Z", "strategy_return": 0.004},
            {"ts_utc": "2025-01-07T00:00:00Z", "strategy_return": -0.002},
            {"ts_utc": "2025-01-08T00:00:00Z", "strategy_return": 0.003},
        ],
        timeframe="1D",
        timestamp="2026-03-25T00:05:00Z",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.cli.run_portfolio.experiment_tracker.ARTIFACTS_ROOT", strategy_root)
    monkeypatch.setattr("src.cli.run_portfolio.DEFAULT_PORTFOLIO_ARTIFACTS_ROOT", portfolio_root)

    cli_args = [
        "--portfolio-config",
        str(portfolio_config_path),
        "--portfolio-name",
        "strict_valid_builtin_pair",
        "--from-registry",
        "--timeframe",
        "1D",
        "--strict",
    ]
    first = run_cli(cli_args)
    second = run_cli(cli_args)

    assert isinstance(first, PortfolioRunResult)
    assert first.run_id == second.run_id
    assert first.metrics == second.metrics
    assert _artifact_bytes(first.experiment_dir) == _artifact_bytes(second.experiment_dir)

    assert first.portfolio_name == "strict_valid_builtin_pair"
    assert first.allocator_name == "equal_weight"
    assert first.component_count == 2
    assert [component["strategy_name"] for component in first.components] == [
        "mean_reversion_v1",
        "momentum_v1",
    ]
    assert [component["run_id"] for component in first.components] == [
        "run-meanrev",
        "run-momentum",
    ]

    required_files = {
        "config.json",
        "manifest.json",
        "metrics.json",
        "portfolio_equity_curve.csv",
        "portfolio_returns.csv",
        "qa_summary.json",
        "weights.csv",
    }
    assert required_files.issubset({path.name for path in first.experiment_dir.iterdir() if path.is_file()})

    config_payload = json.loads((first.experiment_dir / "config.json").read_text(encoding="utf-8"))
    metrics_payload = json.loads((first.experiment_dir / "metrics.json").read_text(encoding="utf-8"))
    manifest_payload = json.loads((first.experiment_dir / "manifest.json").read_text(encoding="utf-8"))
    qa_payload = json.loads((first.experiment_dir / "qa_summary.json").read_text(encoding="utf-8"))
    returns_frame = pd.read_csv(first.experiment_dir / "portfolio_returns.csv")

    assert config_payload["strict_mode"] == {
        "enabled": True,
        "source": "cli",
    }
    assert config_payload["runtime"]["strict_mode"] == config_payload["strict_mode"]
    assert config_payload["runtime"]["execution"] == {
        "enabled": False,
        "execution_delay": 1,
        "transaction_cost_bps": 0.0,
        "slippage_bps": 0.0,
        "fixed_fee": 0.0,
        "fixed_fee_model": "per_rebalance",
        "slippage_model": "constant",
        "slippage_turnover_scale": 1.0,
        "slippage_volatility_scale": 1.0,
    }
    assert config_payload["runtime"]["portfolio_validation"]["strict_sanity_checks"] is True
    assert config_payload["runtime"]["sanity"]["strict_sanity_checks"] is True
    assert config_payload["runtime"]["risk"]["volatility_window"] == 20
    assert config_payload["validation"] == config_payload["runtime"]["portfolio_validation"]
    assert config_payload["optimizer"]["method"] == "equal_weight"
    assert config_payload["validation"]["long_only"] is True

    assert metrics_payload["sanity_status"] == "pass"
    assert metrics_payload["sanity_issue_count"] == pytest.approx(0.0)
    assert metrics_payload["sanity_warning_count"] == pytest.approx(0.0)

    assert manifest_payload["portfolio_name"] == "strict_valid_builtin_pair"
    assert manifest_payload["strict_mode"] == {
        "enabled": True,
        "source": "cli",
    }
    assert manifest_payload["qa_summary_status"] == "pass"
    assert manifest_payload["row_counts"]["portfolio_returns"] == len(returns_frame)
    assert manifest_payload["row_counts"]["portfolio_returns"] > 0
    assert "portfolio_returns.csv" in manifest_payload["artifact_files"]
    assert "metrics.json" in manifest_payload["artifact_files"]

    assert qa_payload["validation_status"] == "pass"
    assert qa_payload["optimizer_method"] == "equal_weight"
    assert qa_payload["sanity"]["status"] == "pass"
    assert qa_payload["sanity"]["strict_sanity_checks"] is True
    assert qa_payload["sanity"]["issue_count"] == 0
    assert qa_payload["issues"] == []

    assert len(returns_frame) > 0
    assert "portfolio_return" in returns_frame.columns
    assert "weight__momentum_v1" in returns_frame.columns
    assert "strategy_return__mean_reversion_v1" in returns_frame.columns

    registry_entries = _read_registry(portfolio_root / "registry.jsonl")
    assert [entry["run_id"] for entry in registry_entries] == [first.run_id]
    assert registry_entries[0]["portfolio_name"] == "strict_valid_builtin_pair"
    assert registry_entries[0]["config"]["strict_mode"] == {
        "enabled": True,
        "source": "cli",
    }


def test_run_cli_writes_portfolio_simulation_artifacts(
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
            {"ts_utc": "2025-01-03T00:00:00Z", "strategy_return": -0.01},
        ],
    )
    _write_strategy_run(
        strategy_root,
        run_id="run-beta",
        strategy_name="beta_v1",
        rows=[
            {"ts_utc": "2025-01-01T00:00:00Z", "strategy_return": 0.00},
            {"ts_utc": "2025-01-02T00:00:00Z", "strategy_return": 0.01},
            {"ts_utc": "2025-01-03T00:00:00Z", "strategy_return": 0.02},
        ],
    )
    simulation_path = tmp_path / "simulation.yml"
    simulation_path.write_text(
        yaml.safe_dump(
            {
                "simulation": {
                    "method": "bootstrap",
                    "num_paths": 3,
                    "path_length": 4,
                    "seed": 21,
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
            "--portfolio-name",
            "sim_portfolio",
            "--run-ids",
            "run-alpha",
            "run-beta",
            "--timeframe",
            "1D",
            "--simulation",
            str(simulation_path),
        ]
    )

    assert result.simulation_result is not None
    simulation_dir = result.experiment_dir / "simulation"
    assert (simulation_dir / "summary.json").exists()
    assert (simulation_dir / "simulated_paths.csv").exists()
    manifest_payload = json.loads((result.experiment_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_payload["simulation"]["enabled"] is True
    assert manifest_payload["simulation"]["summary_path"] == "simulation/summary.json"
    assert manifest_payload["simulation"]["config_path"] == "simulation/config.json"
    assert "simulation/path_metrics.csv" in manifest_payload["artifact_files"]


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


def _write_registered_strategy_run(
    root: Path,
    *,
    run_id: str,
    strategy_name: str,
    rows: list[dict[str, object]],
    timeframe: str,
    timestamp: str,
) -> Path:
    run_dir = _write_strategy_run(
        root,
        run_id=run_id,
        strategy_name=strategy_name,
        rows=rows,
    )
    registry_path = root / "registry.jsonl"
    entry = {
        "run_id": run_id,
        "run_type": "strategy",
        "timestamp": timestamp,
        "strategy_name": strategy_name,
        "timeframe": timeframe,
        "artifact_path": run_dir.as_posix(),
    }
    existing_lines = []
    if registry_path.exists():
        existing_lines = [
            line
            for line in registry_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    existing_lines.append(json.dumps(entry))
    registry_path.write_text("\n".join(existing_lines) + "\n", encoding="utf-8")
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
