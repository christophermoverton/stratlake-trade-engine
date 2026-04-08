from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.cli.run_research_campaign import (
    parse_args,
    run_cli,
    run_research_campaign,
)
from src.config.research_campaign import resolve_research_campaign_config


def test_parse_args_accepts_optional_config_path() -> None:
    args = parse_args(["--config", "configs/custom_campaign.yml"])

    assert args.config == "configs/custom_campaign.yml"


def test_run_research_campaign_executes_stages_in_deterministic_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    alpha_catalog = tmp_path / "alphas.yml"
    strategy_config = tmp_path / "strategies.yml"
    portfolio_config = tmp_path / "portfolios.yml"
    for path in (alpha_catalog, strategy_config, portfolio_config):
        path.write_text("{}", encoding="utf-8")

    config = resolve_research_campaign_config(
        {
            "targets": {
                "alpha_names": ["alpha_one"],
                "strategy_names": ["momentum_v1"],
                "portfolio_names": ["candidate_portfolio"],
                "alpha_catalog_path": alpha_catalog.as_posix(),
                "strategy_config_path": strategy_config.as_posix(),
                "portfolio_config_path": portfolio_config.as_posix(),
            },
            "dataset_selection": {
                "dataset": "features_daily",
                "timeframe": "1D",
                "evaluation_horizon": 5,
                "mapping_name": "top_bottom_quantile_q20",
            },
            "time_windows": {
                "start": "2025-01-01",
                "end": "2025-03-01",
                "train_start": "2025-01-01",
                "train_end": "2025-02-01",
                "predict_start": "2025-02-01",
                "predict_end": "2025-03-01",
            },
            "comparison": {
                "enabled": True,
                "top_k": 3,
            },
            "candidate_selection": {
                "enabled": True,
                "execution": {
                    "enable_review": True,
                    "register_run": True,
                },
                "output": {
                    "path": (tmp_path / "candidate_selection").as_posix(),
                    "review_output_path": (tmp_path / "candidate_review").as_posix(),
                },
            },
            "portfolio": {
                "enabled": True,
                "from_candidate_selection": True,
            },
            "review": {
                "output": {
                    "path": (tmp_path / "reviews").as_posix(),
                    "emit_plots": False,
                }
            },
            "outputs": {
                "alpha_artifacts_root": (tmp_path / "alpha_artifacts").as_posix(),
                "comparison_output_path": (tmp_path / "comparisons").as_posix(),
                "portfolio_artifacts_root": (tmp_path / "portfolio_artifacts").as_posix(),
            },
        }
    )

    call_order: list[tuple[str, list[str]]] = []
    candidate_dir = tmp_path / "candidate_selection" / "candidate_run"
    portfolio_dir = tmp_path / "portfolio_artifacts" / "portfolio_run"
    review_dir = tmp_path / "candidate_review"

    monkeypatch.setattr(
        "src.cli.run_research_campaign.run_alpha_cli.run_cli",
        lambda argv: call_order.append(("alpha", list(argv)))
        or SimpleNamespace(alpha_name="alpha_one", run_id="alpha_run", artifact_dir=tmp_path / "alpha"),
    )
    monkeypatch.setattr(
        "src.cli.run_research_campaign.run_strategy_cli.run_cli",
        lambda argv: call_order.append(("strategy", list(argv)))
        or SimpleNamespace(strategy_name="momentum_v1", run_id="strategy_run", experiment_dir=tmp_path / "strategy"),
    )
    monkeypatch.setattr(
        "src.cli.run_research_campaign.compare_alpha_cli.run_cli",
        lambda argv: call_order.append(("alpha_comparison", list(argv)))
        or SimpleNamespace(comparison_id="alpha_cmp"),
    )
    monkeypatch.setattr(
        "src.cli.run_research_campaign.compare_strategies_cli.run_cli",
        lambda argv: call_order.append(("strategy_comparison", list(argv)))
        or SimpleNamespace(comparison_id="strategy_cmp"),
    )
    monkeypatch.setattr(
        "src.cli.run_research_campaign.run_candidate_selection_cli.run_cli",
        lambda argv: call_order.append(("candidate_selection", list(argv)))
        or SimpleNamespace(run_id="candidate_run", artifact_dir=candidate_dir),
    )
    monkeypatch.setattr(
        "src.cli.run_research_campaign.run_portfolio_cli.run_cli",
        lambda argv: call_order.append(("portfolio", list(argv)))
        or SimpleNamespace(run_id="portfolio_run", experiment_dir=portfolio_dir, portfolio_name="candidate_portfolio"),
    )
    monkeypatch.setattr(
        "src.cli.run_research_campaign.review_candidate_selection_cli.run_cli",
        lambda argv: call_order.append(("candidate_review", list(argv)))
        or SimpleNamespace(review_dir=review_dir),
    )
    monkeypatch.setattr(
        "src.cli.run_research_campaign.compare_research_cli.run_cli",
        lambda argv: call_order.append(("review", list(argv)))
        or SimpleNamespace(review_id="review_run"),
    )

    result = run_research_campaign(config)
    from src.cli.run_research_campaign import print_summary

    print_summary(result)
    stdout = capsys.readouterr().out

    assert [name for name, _argv in call_order] == [
        "alpha",
        "strategy",
        "alpha_comparison",
        "strategy_comparison",
        "candidate_selection",
        "portfolio",
        "candidate_review",
        "review",
    ]
    assert result.candidate_selection_result.run_id == "candidate_run"
    assert result.portfolio_result.run_id == "portfolio_run"
    assert result.review_result.review_id == "review_run"

    alpha_argv = dict(enumerate(call_order[0][1]))
    assert "--alpha-name" in alpha_argv.values()
    assert "--config" in alpha_argv.values()

    candidate_argv = call_order[4][1]
    assert "--register-run" in candidate_argv
    assert "--output-path" in candidate_argv

    portfolio_argv = call_order[5][1]
    assert "--from-candidate-selection" in portfolio_argv
    assert candidate_dir.as_posix() in portfolio_argv

    candidate_review_argv = call_order[6][1]
    assert "--candidate-selection-path" in candidate_review_argv
    assert "--portfolio-path" in candidate_review_argv

    review_argv = call_order[7][1]
    assert "--from-registry" in review_argv
    assert "--disable-plots" in review_argv

    assert "Research Campaign Summary" in stdout
    assert "Research: alpha_runs=1 | strategy_runs=1" in stdout
    assert "Comparison: alpha=alpha_cmp | strategy=strategy_cmp" in stdout
    assert "Selection/Portfolio: candidate=candidate_run | portfolio=portfolio_run" in stdout
    assert "Review: candidate_review=" in stdout
    assert "review_id=review_run" in stdout


def test_run_research_campaign_preflight_requires_candidate_alpha_name(tmp_path: Path) -> None:
    alpha_catalog = tmp_path / "alphas.yml"
    strategy_config = tmp_path / "strategies.yml"
    alpha_catalog.write_text("{}", encoding="utf-8")
    strategy_config.write_text("{}", encoding="utf-8")

    config = resolve_research_campaign_config(
        {
            "targets": {
                "alpha_names": ["alpha_one", "alpha_two"],
                "alpha_catalog_path": alpha_catalog.as_posix(),
                "strategy_config_path": strategy_config.as_posix(),
            },
            "candidate_selection": {
                "enabled": True,
            },
        }
    )

    with pytest.raises(ValueError, match="Candidate selection requires one resolved alpha_name"):
        run_research_campaign(config)


def test_run_cli_uses_resolved_config_from_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    alpha_catalog = tmp_path / "alphas.yml"
    strategy_config = tmp_path / "strategies.yml"
    alpha_catalog.write_text("{}", encoding="utf-8")
    strategy_config.write_text("{}", encoding="utf-8")

    config = resolve_research_campaign_config(
        {
            "targets": {
                "alpha_names": ["alpha_one"],
                "alpha_catalog_path": alpha_catalog.as_posix(),
                "strategy_config_path": strategy_config.as_posix(),
            }
        }
    )

    monkeypatch.setattr("src.cli.run_research_campaign.resolve_cli_config", lambda _path=None: config)
    monkeypatch.setattr(
        "src.cli.run_research_campaign.run_alpha_cli.run_cli",
        lambda argv: SimpleNamespace(alpha_name="alpha_one", run_id="alpha_run", artifact_dir=tmp_path / "alpha"),
    )
    monkeypatch.setattr(
        "src.cli.run_research_campaign.compare_research_cli.run_cli",
        lambda argv: SimpleNamespace(review_id="review_run"),
    )

    result = run_cli(["--config", "configs/override.yml"])

    assert result.alpha_results[0].run_id == "alpha_run"
    assert result.review_result.review_id == "review_run"
