from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.cli.run_research_campaign import (
    CampaignPreflightCheck,
    CampaignPreflightResult,
    _resolve_candidate_selection_reference,
    parse_args,
    print_summary,
    run_cli,
    run_research_campaign,
)
from src.config.research_campaign import resolve_research_campaign_config


def _write_feature_fixture(root: Path, dataset: str = "features_daily") -> Path:
    dataset_root = root / "curated" / dataset / "symbol=AAA" / "year=2025"
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "part-0.parquet").write_text("fixture", encoding="utf-8")
    return dataset_root


def _write_candidate_selection_registry_entry(
    registry_path: Path,
    *,
    run_id: str,
    artifact_dir: Path,
    alpha_name: str,
    dataset: str,
    timeframe: str,
    evaluation_horizon: int,
    mapping_names: list[str],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "config_snapshot": {
            "filters": {
                "alpha_name": alpha_name,
            },
            "primary_metric": "ic_ir",
        },
        "provenance": {
            "dataset": dataset,
            "timeframe": timeframe,
            "evaluation_horizon": evaluation_horizon,
            "mapping_names": mapping_names,
            "upstream": {"alpha_run_ids": [f"{alpha_name}_alpha_run"]},
        },
    }
    summary = {"run_id": run_id, "selected_candidates": 2}
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (artifact_dir / "selection_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "run_id": run_id,
        "run_type": "candidate_selection",
        "timestamp": "2026-04-01T00:00:00Z",
        "alpha_name": alpha_name,
        "dataset": dataset,
        "timeframe": timeframe,
        "evaluation_horizon": evaluation_horizon,
        "artifact_path": artifact_dir.as_posix(),
        "manifest_path": (artifact_dir / "manifest.json").as_posix(),
        "summary_path": (artifact_dir / "selection_summary.json").as_posix(),
        "metadata": {
            "mapping_names": mapping_names,
            "primary_metric": "ic_ir",
        },
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


def _write_portfolio_registry_entry(
    registry_path: Path,
    *,
    run_id: str,
    artifact_dir: Path,
    portfolio_name: str,
    timeframe: str,
    candidate_selection_run_id: str | None = None,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "run_type": "portfolio",
        "portfolio_name": portfolio_name,
    }
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    config = {
        "candidate_selection_provenance": {
            "run_id": candidate_selection_run_id,
        }
    }
    entry = {
        "run_id": run_id,
        "run_type": "portfolio",
        "timestamp": "2026-04-01T00:05:00Z",
        "portfolio_name": portfolio_name,
        "timeframe": timeframe,
        "artifact_path": artifact_dir.as_posix(),
        "config": config,
        "component_run_ids": ["strategy_run_a", "strategy_run_b"],
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


def _build_full_campaign_config(
    *,
    tmp_path: Path,
    alpha_catalog: Path,
    strategy_config: Path,
    portfolio_config: Path,
) -> object:
    return resolve_research_campaign_config(
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
                "alpha_name": "alpha_a",
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
                "campaign_artifacts_root": (tmp_path / "campaign_artifacts").as_posix(),
                "comparison_output_path": (tmp_path / "comparisons").as_posix(),
                "portfolio_artifacts_root": (tmp_path / "portfolio_artifacts").as_posix(),
            },
        }
    )


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
    features_root = tmp_path / "features_root"
    alpha_catalog.write_text(
        """
alpha_one:
  alpha_name: alpha_one
  dataset: features_daily
  target_column: target_ret_1d
  feature_columns: [feature_ret_1d]
  model_type: cross_sectional_linear
  model_params: {}
  alpha_horizon: 1
""".strip(),
        encoding="utf-8",
    )
    strategy_config.write_text(
        """
momentum_v1:
  dataset: features_daily
  parameters: {}
""".strip(),
        encoding="utf-8",
    )
    portfolio_config.write_text(
        """
portfolios:
  candidate_portfolio:
    allocator: equal_weight
    components:
      - strategy_name: momentum_v1
""".strip(),
        encoding="utf-8",
    )
    _write_feature_fixture(features_root)
    monkeypatch.setenv("FEATURES_ROOT", features_root.as_posix())

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
                "alpha_name": "alpha_a",
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
                "campaign_artifacts_root": (tmp_path / "campaign_artifacts").as_posix(),
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
    assert result.preflight_summary_path.exists()
    assert result.campaign_artifact_dir.exists()
    assert result.campaign_checkpoint_path.exists()
    assert result.campaign_manifest_path.exists()
    assert result.campaign_summary_path.exists()
    assert result.preflight_summary["status"] == "passed"
    assert result.preflight_summary["check_counts"]["failed"] == 0

    campaign_checkpoint = json.loads(result.campaign_checkpoint_path.read_text(encoding="utf-8"))
    campaign_manifest = json.loads(result.campaign_manifest_path.read_text(encoding="utf-8"))
    campaign_summary = json.loads(result.campaign_summary_path.read_text(encoding="utf-8"))

    alpha_argv = dict(enumerate(call_order[0][1]))
    assert "--alpha-name" in alpha_argv.values()
    assert "--config" in alpha_argv.values()
    assert "--artifacts-root" in alpha_argv.values()

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
    assert "--alpha-artifacts-root" in review_argv
    assert "--portfolio-artifacts-root" in review_argv

    assert "Research Campaign Summary" in stdout
    assert f"Campaign: {result.campaign_run_id}" in stdout
    assert "Preflight: passed" in stdout
    assert "Research: alpha_runs=1 | strategy_runs=1" in stdout
    assert "Comparison: alpha=alpha_cmp | strategy=strategy_cmp" in stdout
    assert "Selection/Portfolio: candidate=candidate_run | portfolio=portfolio_run" in stdout
    assert "Review: candidate_review=" in stdout
    assert "review_id=review_run" in stdout
    assert "Campaign Artifacts: manifest=" in stdout

    assert campaign_checkpoint["run_type"] == "research_campaign_checkpoint"
    assert campaign_checkpoint["campaign_run_id"] == result.campaign_run_id
    assert campaign_checkpoint["status"] == "completed"
    assert set(campaign_checkpoint["stage_input_fingerprints"]) == {
        "preflight",
        "research",
        "comparison",
        "candidate_selection",
        "portfolio",
        "candidate_review",
        "review",
    }
    assert campaign_checkpoint["stage_input_fingerprints"]["candidate_selection"] == campaign_checkpoint["stages"][3]["input_fingerprint"]
    assert campaign_checkpoint["stage_states"]["candidate_review"] == "completed"
    assert campaign_checkpoint["stage_states"]["portfolio"] == "completed"

    assert campaign_manifest["run_type"] == "research_campaign"
    assert campaign_manifest["campaign_run_id"] == result.campaign_run_id
    assert campaign_manifest["status"] == "completed"
    assert campaign_manifest["checkpoint_path"] == "checkpoint.json"
    assert campaign_manifest["summary_path"] == "summary.json"
    assert campaign_manifest["stage_statuses"]["candidate_review"] == "completed"
    assert campaign_manifest["skipped_stage_names"] == []
    assert campaign_manifest["resumable_stage_names"] == []
    assert campaign_manifest["stage_execution"]["candidate_review"]["resume"]["resumable"] is False
    assert campaign_manifest["stage_execution"]["candidate_review"]["retry"]["attempted"] is False
    assert campaign_manifest["stage_execution"]["candidate_review"]["skip"]["skipped"] is False
    assert (
        campaign_manifest["stage_execution"]["candidate_review"]["fingerprint"]["input_fingerprint"]
        == campaign_checkpoint["stage_input_fingerprints"]["candidate_review"]
    )
    assert campaign_manifest["selected_run_ids"]["portfolio_run_id"] == "portfolio_run"

    assert campaign_summary["run_type"] == "research_campaign"
    assert campaign_summary["campaign_run_id"] == result.campaign_run_id
    assert campaign_summary["status"] == "completed"
    assert campaign_summary["checkpoint_path"].endswith("checkpoint.json")
    assert campaign_summary["selected_run_ids"]["alpha_run_ids"] == ["alpha_run"]
    assert campaign_summary["selected_run_ids"]["strategy_run_ids"] == ["strategy_run"]
    assert campaign_summary["selected_run_ids"]["candidate_selection_run_id"] == "candidate_run"
    assert campaign_summary["selected_run_ids"]["portfolio_run_id"] == "portfolio_run"
    assert campaign_summary["selected_run_ids"]["review_id"] == "review_run"
    assert campaign_summary["stage_statuses"]["candidate_review"] == "completed"
    assert campaign_summary["final_outcomes"]["skipped_stage_names"] == []
    assert campaign_summary["final_outcomes"]["resumable_stage_names"] == []
    assert campaign_summary["checkpoint"]["stage_states"]["candidate_review"] == "completed"
    candidate_review_stage = next(
        stage for stage in campaign_summary["stages"] if stage["stage_name"] == "candidate_review"
    )
    assert candidate_review_stage["execution_metadata"] == campaign_summary["stage_execution"]["candidate_review"]
    assert candidate_review_stage["execution_metadata"]["resume"]["checkpoint_state"] == "completed"
    assert candidate_review_stage["execution_metadata"]["reuse"]["reused"] is False
    assert candidate_review_stage["execution_metadata"]["skip"]["skipped"] is False
    assert candidate_review_stage["execution_metadata"]["failure"] is None
    assert campaign_summary["output_paths"]["candidate_review_dir"] == review_dir.as_posix()
    assert campaign_summary["output_paths"]["campaign_checkpoint"] == result.campaign_checkpoint_path.as_posix()
    assert "candidate_review_counts" in campaign_summary["final_outcomes"]
    assert [stage["stage_name"] for stage in campaign_summary["stages"]] == [
        "preflight",
        "research",
        "comparison",
        "candidate_selection",
        "portfolio",
        "candidate_review",
        "review",
    ]


def test_run_research_campaign_reuses_matching_checkpoint_stages_on_resume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    alpha_catalog = tmp_path / "alphas.yml"
    strategy_config = tmp_path / "strategies.yml"
    portfolio_config = tmp_path / "portfolios.yml"
    features_root = tmp_path / "features_root"
    alpha_catalog.write_text(
        """
alpha_one:
  alpha_name: alpha_one
  dataset: features_daily
  target_column: target_ret_1d
  feature_columns: [feature_ret_1d]
  model_type: cross_sectional_linear
  model_params: {}
  alpha_horizon: 1
""".strip(),
        encoding="utf-8",
    )
    strategy_config.write_text("momentum_v1:\n  dataset: features_daily\n  parameters: {}\n", encoding="utf-8")
    portfolio_config.write_text(
        """
portfolios:
  candidate_portfolio:
    allocator: equal_weight
    components:
      - strategy_name: momentum_v1
""".strip(),
        encoding="utf-8",
    )
    _write_feature_fixture(features_root)
    monkeypatch.setenv("FEATURES_ROOT", features_root.as_posix())
    config = _build_full_campaign_config(
        tmp_path=tmp_path,
        alpha_catalog=alpha_catalog,
        strategy_config=strategy_config,
        portfolio_config=portfolio_config,
    )

    def install_stage_stubs(call_order: list[str]) -> None:
        monkeypatch.setattr(
            "src.cli.run_research_campaign.run_alpha_cli.run_cli",
            lambda argv: call_order.append("alpha")
            or SimpleNamespace(alpha_name="alpha_one", run_id="alpha_run", artifact_dir=tmp_path / "alpha"),
        )
        monkeypatch.setattr(
            "src.cli.run_research_campaign.run_strategy_cli.run_cli",
            lambda argv: call_order.append("strategy")
            or SimpleNamespace(
                strategy_name="momentum_v1",
                run_id="strategy_run",
                experiment_dir=tmp_path / "strategy",
                metrics={"cumulative_return": 0.2, "sharpe_ratio": 1.1, "max_drawdown": -0.05},
            ),
        )
        monkeypatch.setattr(
            "src.cli.run_research_campaign.compare_alpha_cli.run_cli",
            lambda argv: call_order.append("alpha_comparison")
            or SimpleNamespace(comparison_id="alpha_cmp"),
        )
        monkeypatch.setattr(
            "src.cli.run_research_campaign.compare_strategies_cli.run_cli",
            lambda argv: call_order.append("strategy_comparison")
            or SimpleNamespace(comparison_id="strategy_cmp"),
        )
        monkeypatch.setattr(
            "src.cli.run_research_campaign.run_candidate_selection_cli.run_cli",
            lambda argv: call_order.append("candidate_selection")
            or SimpleNamespace(
                run_id="candidate_run",
                artifact_dir=tmp_path / "candidate_selection" / "candidate_run",
                primary_metric="ic_ir",
                universe_count=8,
                eligible_count=4,
                selected_count=2,
                rejected_count=2,
                pruned_by_redundancy=1,
            ),
        )
        monkeypatch.setattr(
            "src.cli.run_research_campaign.run_portfolio_cli.run_cli",
            lambda argv: call_order.append("portfolio")
            or SimpleNamespace(
                run_id="portfolio_run",
                experiment_dir=tmp_path / "portfolio_artifacts" / "portfolio_run",
                portfolio_name="candidate_portfolio",
                component_count=2,
                metrics={"total_return": 0.12, "sharpe_ratio": 1.0, "max_drawdown": -0.04},
            ),
        )
        monkeypatch.setattr(
            "src.cli.run_research_campaign.review_candidate_selection_cli.run_cli",
            lambda argv: call_order.append("candidate_review")
            or SimpleNamespace(
                review_dir=tmp_path / "candidate_review" / "candidate_run",
                candidate_selection_run_id="candidate_run",
                portfolio_run_id="portfolio_run",
                total_candidates=4,
                selected_candidates=2,
                rejected_candidates=2,
            ),
        )
        monkeypatch.setattr(
            "src.cli.run_research_campaign.compare_research_cli.run_cli",
            lambda argv: call_order.append("review")
            or SimpleNamespace(review_id="review_run", entries=[]),
        )

    first_call_order: list[str] = []
    install_stage_stubs(first_call_order)
    first_result = run_research_campaign(config)

    second_call_order: list[str] = []
    install_stage_stubs(second_call_order)
    second_result = run_research_campaign(config)
    second_checkpoint = json.loads(second_result.campaign_checkpoint_path.read_text(encoding="utf-8"))

    assert first_call_order == [
        "alpha",
        "strategy",
        "alpha_comparison",
        "strategy_comparison",
        "candidate_selection",
        "portfolio",
        "candidate_review",
        "review",
    ]
    assert second_call_order == []
    assert [record.stage_name for record in second_result.stage_records] == [
        "preflight",
        "research",
        "comparison",
        "candidate_selection",
        "portfolio",
        "candidate_review",
        "review",
    ]
    assert [record.status for record in second_result.stage_records] == [
        "reused",
        "reused",
        "reused",
        "reused",
        "reused",
        "reused",
        "reused",
    ]
    assert [stage["stage_name"] for stage in second_checkpoint["stages"]] == [
        "preflight",
        "research",
        "comparison",
        "candidate_selection",
        "portfolio",
        "candidate_review",
        "review",
    ]
    assert second_checkpoint["stage_states"] == {
        "preflight": "reused",
        "research": "reused",
        "comparison": "reused",
        "candidate_selection": "reused",
        "portfolio": "reused",
        "candidate_review": "reused",
        "review": "reused",
    }
    assert all(stage["source"] == "checkpoint" for stage in second_checkpoint["stages"])
    assert first_result.campaign_checkpoint["stage_input_fingerprints"] == second_checkpoint["stage_input_fingerprints"]


def test_run_research_campaign_reruns_stage_when_checkpoint_fingerprint_is_stale(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    alpha_catalog = tmp_path / "alphas.yml"
    strategy_config = tmp_path / "strategies.yml"
    portfolio_config = tmp_path / "portfolios.yml"
    features_root = tmp_path / "features_root"
    alpha_catalog.write_text(
        """
alpha_one:
  alpha_name: alpha_one
  dataset: features_daily
  target_column: target_ret_1d
  feature_columns: [feature_ret_1d]
  model_type: cross_sectional_linear
  model_params: {}
  alpha_horizon: 1
""".strip(),
        encoding="utf-8",
    )
    strategy_config.write_text("momentum_v1:\n  dataset: features_daily\n  parameters: {}\n", encoding="utf-8")
    portfolio_config.write_text(
        """
portfolios:
  candidate_portfolio:
    allocator: equal_weight
    components:
      - strategy_name: momentum_v1
""".strip(),
        encoding="utf-8",
    )
    _write_feature_fixture(features_root)
    monkeypatch.setenv("FEATURES_ROOT", features_root.as_posix())
    config = _build_full_campaign_config(
        tmp_path=tmp_path,
        alpha_catalog=alpha_catalog,
        strategy_config=strategy_config,
        portfolio_config=portfolio_config,
    )

    def install_stage_stubs(call_order: list[str]) -> None:
        monkeypatch.setattr(
            "src.cli.run_research_campaign.run_alpha_cli.run_cli",
            lambda argv: call_order.append("alpha")
            or SimpleNamespace(alpha_name="alpha_one", run_id="alpha_run", artifact_dir=tmp_path / "alpha"),
        )
        monkeypatch.setattr(
            "src.cli.run_research_campaign.run_strategy_cli.run_cli",
            lambda argv: call_order.append("strategy")
            or SimpleNamespace(
                strategy_name="momentum_v1",
                run_id="strategy_run",
                experiment_dir=tmp_path / "strategy",
                metrics={"cumulative_return": 0.2, "sharpe_ratio": 1.1, "max_drawdown": -0.05},
            ),
        )
        monkeypatch.setattr(
            "src.cli.run_research_campaign.compare_alpha_cli.run_cli",
            lambda argv: call_order.append("alpha_comparison")
            or SimpleNamespace(comparison_id="alpha_cmp"),
        )
        monkeypatch.setattr(
            "src.cli.run_research_campaign.compare_strategies_cli.run_cli",
            lambda argv: call_order.append("strategy_comparison")
            or SimpleNamespace(comparison_id="strategy_cmp"),
        )
        monkeypatch.setattr(
            "src.cli.run_research_campaign.run_candidate_selection_cli.run_cli",
            lambda argv: call_order.append("candidate_selection")
            or SimpleNamespace(run_id="candidate_run", artifact_dir=tmp_path / "candidate_selection" / "candidate_run"),
        )
        monkeypatch.setattr(
            "src.cli.run_research_campaign.run_portfolio_cli.run_cli",
            lambda argv: call_order.append("portfolio")
            or SimpleNamespace(
                run_id="portfolio_run",
                experiment_dir=tmp_path / "portfolio_artifacts" / "portfolio_run",
                portfolio_name="candidate_portfolio",
            ),
        )
        monkeypatch.setattr(
            "src.cli.run_research_campaign.review_candidate_selection_cli.run_cli",
            lambda argv: call_order.append("candidate_review")
            or SimpleNamespace(
                review_dir=tmp_path / "candidate_review" / "candidate_run",
                candidate_selection_run_id="candidate_run",
                portfolio_run_id="portfolio_run",
            ),
        )
        monkeypatch.setattr(
            "src.cli.run_research_campaign.compare_research_cli.run_cli",
            lambda argv: call_order.append("review")
            or SimpleNamespace(review_id="review_run", entries=[]),
        )

    first_call_order: list[str] = []
    install_stage_stubs(first_call_order)
    first_result = run_research_campaign(config)

    checkpoint = json.loads(first_result.campaign_checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["stage_input_fingerprints"]["research"] = "stale-fingerprint"
    checkpoint["stages"][1]["input_fingerprint"] = "stale-fingerprint"
    first_result.campaign_checkpoint_path.write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    second_call_order: list[str] = []
    install_stage_stubs(second_call_order)
    second_result = run_research_campaign(config)
    second_checkpoint = json.loads(second_result.campaign_checkpoint_path.read_text(encoding="utf-8"))

    assert second_call_order == ["alpha", "strategy"]
    assert [record.status for record in second_result.stage_records] == [
        "reused",
        "completed",
        "reused",
        "reused",
        "reused",
        "reused",
        "reused",
    ]
    assert second_checkpoint["stage_states"]["preflight"] == "reused"
    assert second_checkpoint["stage_states"]["research"] == "completed"
    assert second_checkpoint["stage_states"]["comparison"] == "reused"
    assert [stage["stage_name"] for stage in second_checkpoint["stages"]] == [
        "preflight",
        "research",
        "comparison",
        "candidate_selection",
        "portfolio",
        "candidate_review",
        "review",
    ]


def test_run_research_campaign_can_disable_checkpoint_reuse_via_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    alpha_catalog = tmp_path / "alphas.yml"
    strategy_config = tmp_path / "strategies.yml"
    portfolio_config = tmp_path / "portfolios.yml"
    features_root = tmp_path / "features_root"
    alpha_catalog.write_text(
        """
alpha_one:
  alpha_name: alpha_one
  dataset: features_daily
  target_column: target_ret_1d
  feature_columns: [feature_ret_1d]
  model_type: cross_sectional_linear
  model_params: {}
  alpha_horizon: 1
""".strip(),
        encoding="utf-8",
    )
    strategy_config.write_text("momentum_v1:\n  dataset: features_daily\n  parameters: {}\n", encoding="utf-8")
    portfolio_config.write_text(
        """
portfolios:
  candidate_portfolio:
    allocator: equal_weight
    components:
      - strategy_name: momentum_v1
""".strip(),
        encoding="utf-8",
    )
    _write_feature_fixture(features_root)
    monkeypatch.setenv("FEATURES_ROOT", features_root.as_posix())
    config = resolve_research_campaign_config(
        _build_full_campaign_config(
            tmp_path=tmp_path,
            alpha_catalog=alpha_catalog,
            strategy_config=strategy_config,
            portfolio_config=portfolio_config,
        ).to_dict(),
        {
            "reuse_policy": {
                "enable_checkpoint_reuse": False,
            }
        },
    )

    def install_stage_stubs(call_order: list[str]) -> None:
        monkeypatch.setattr(
            "src.cli.run_research_campaign.run_alpha_cli.run_cli",
            lambda argv: call_order.append("alpha")
            or SimpleNamespace(alpha_name="alpha_one", run_id="alpha_run", artifact_dir=tmp_path / "alpha"),
        )
        monkeypatch.setattr(
            "src.cli.run_research_campaign.run_strategy_cli.run_cli",
            lambda argv: call_order.append("strategy")
            or SimpleNamespace(
                strategy_name="momentum_v1",
                run_id="strategy_run",
                experiment_dir=tmp_path / "strategy",
                metrics={"cumulative_return": 0.2, "sharpe_ratio": 1.1, "max_drawdown": -0.05},
            ),
        )
        monkeypatch.setattr(
            "src.cli.run_research_campaign.compare_alpha_cli.run_cli",
            lambda argv: call_order.append("alpha_comparison")
            or SimpleNamespace(comparison_id="alpha_cmp"),
        )
        monkeypatch.setattr(
            "src.cli.run_research_campaign.compare_strategies_cli.run_cli",
            lambda argv: call_order.append("strategy_comparison")
            or SimpleNamespace(comparison_id="strategy_cmp"),
        )
        monkeypatch.setattr(
            "src.cli.run_research_campaign.run_candidate_selection_cli.run_cli",
            lambda argv: call_order.append("candidate_selection")
            or SimpleNamespace(
                run_id="candidate_run",
                artifact_dir=tmp_path / "candidate_selection" / "candidate_run",
            ),
        )
        monkeypatch.setattr(
            "src.cli.run_research_campaign.run_portfolio_cli.run_cli",
            lambda argv: call_order.append("portfolio")
            or SimpleNamespace(
                run_id="portfolio_run",
                experiment_dir=tmp_path / "portfolio_artifacts" / "portfolio_run",
                portfolio_name="candidate_portfolio",
            ),
        )
        monkeypatch.setattr(
            "src.cli.run_research_campaign.review_candidate_selection_cli.run_cli",
            lambda argv: call_order.append("candidate_review")
            or SimpleNamespace(
                review_dir=tmp_path / "candidate_review" / "candidate_run",
                candidate_selection_run_id="candidate_run",
                portfolio_run_id="portfolio_run",
            ),
        )
        monkeypatch.setattr(
            "src.cli.run_research_campaign.compare_research_cli.run_cli",
            lambda argv: call_order.append("review")
            or SimpleNamespace(review_id="review_run", entries=[]),
        )

    first_call_order: list[str] = []
    install_stage_stubs(first_call_order)
    first_result = run_research_campaign(config)

    second_call_order: list[str] = []
    install_stage_stubs(second_call_order)
    second_result = run_research_campaign(config)
    second_summary = json.loads(second_result.campaign_summary_path.read_text(encoding="utf-8"))

    assert first_call_order == [
        "alpha",
        "strategy",
        "alpha_comparison",
        "strategy_comparison",
        "candidate_selection",
        "portfolio",
        "candidate_review",
        "review",
    ]
    assert second_call_order == first_call_order
    assert second_result.campaign_checkpoint["reused_stage_count"] == 0
    assert all(record.status == "completed" for record in second_result.stage_records)
    assert (
        next(stage for stage in second_summary["stages"] if stage["stage_name"] == "research")["details"]["reuse_policy"]["reason"]
        == "Checkpoint reuse disabled by reuse_policy.enable_checkpoint_reuse=false."
    )
    assert first_result.campaign_run_id == second_result.campaign_run_id


def test_run_research_campaign_can_force_rerun_specific_stage_without_invalidating_downstream(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    alpha_catalog = tmp_path / "alphas.yml"
    strategy_config = tmp_path / "strategies.yml"
    portfolio_config = tmp_path / "portfolios.yml"
    features_root = tmp_path / "features_root"
    alpha_catalog.write_text(
        """
alpha_one:
  alpha_name: alpha_one
  dataset: features_daily
  target_column: target_ret_1d
  feature_columns: [feature_ret_1d]
  model_type: cross_sectional_linear
  model_params: {}
  alpha_horizon: 1
""".strip(),
        encoding="utf-8",
    )
    strategy_config.write_text("momentum_v1:\n  dataset: features_daily\n  parameters: {}\n", encoding="utf-8")
    portfolio_config.write_text(
        """
portfolios:
  candidate_portfolio:
    allocator: equal_weight
    components:
      - strategy_name: momentum_v1
""".strip(),
        encoding="utf-8",
    )
    _write_feature_fixture(features_root)
    monkeypatch.setenv("FEATURES_ROOT", features_root.as_posix())
    config = resolve_research_campaign_config(
        _build_full_campaign_config(
            tmp_path=tmp_path,
            alpha_catalog=alpha_catalog,
            strategy_config=strategy_config,
            portfolio_config=portfolio_config,
        ).to_dict(),
        {"reuse_policy": {"force_rerun_stages": ["research"]}},
    )

    def install_stage_stubs(call_order: list[str]) -> None:
        monkeypatch.setattr(
            "src.cli.run_research_campaign.run_alpha_cli.run_cli",
            lambda argv: call_order.append("alpha")
            or SimpleNamespace(alpha_name="alpha_one", run_id="alpha_run", artifact_dir=tmp_path / "alpha"),
        )
        monkeypatch.setattr(
            "src.cli.run_research_campaign.run_strategy_cli.run_cli",
            lambda argv: call_order.append("strategy")
            or SimpleNamespace(
                strategy_name="momentum_v1",
                run_id="strategy_run",
                experiment_dir=tmp_path / "strategy",
                metrics={"cumulative_return": 0.2, "sharpe_ratio": 1.1, "max_drawdown": -0.05},
            ),
        )
        monkeypatch.setattr(
            "src.cli.run_research_campaign.compare_alpha_cli.run_cli",
            lambda argv: call_order.append("alpha_comparison")
            or SimpleNamespace(comparison_id="alpha_cmp"),
        )
        monkeypatch.setattr(
            "src.cli.run_research_campaign.compare_strategies_cli.run_cli",
            lambda argv: call_order.append("strategy_comparison")
            or SimpleNamespace(comparison_id="strategy_cmp"),
        )
        monkeypatch.setattr(
            "src.cli.run_research_campaign.run_candidate_selection_cli.run_cli",
            lambda argv: call_order.append("candidate_selection")
            or SimpleNamespace(run_id="candidate_run", artifact_dir=tmp_path / "candidate_selection" / "candidate_run"),
        )
        monkeypatch.setattr(
            "src.cli.run_research_campaign.run_portfolio_cli.run_cli",
            lambda argv: call_order.append("portfolio")
            or SimpleNamespace(
                run_id="portfolio_run",
                experiment_dir=tmp_path / "portfolio_artifacts" / "portfolio_run",
                portfolio_name="candidate_portfolio",
            ),
        )
        monkeypatch.setattr(
            "src.cli.run_research_campaign.review_candidate_selection_cli.run_cli",
            lambda argv: call_order.append("candidate_review")
            or SimpleNamespace(
                review_dir=tmp_path / "candidate_review" / "candidate_run",
                candidate_selection_run_id="candidate_run",
                portfolio_run_id="portfolio_run",
            ),
        )
        monkeypatch.setattr(
            "src.cli.run_research_campaign.compare_research_cli.run_cli",
            lambda argv: call_order.append("review")
            or SimpleNamespace(review_id="review_run", entries=[]),
        )

    install_stage_stubs([])
    run_research_campaign(config)

    second_call_order: list[str] = []
    install_stage_stubs(second_call_order)
    second_result = run_research_campaign(config)
    second_summary = json.loads(second_result.campaign_summary_path.read_text(encoding="utf-8"))

    assert second_call_order == ["alpha", "strategy"]
    assert [record.status for record in second_result.stage_records] == [
        "reused",
        "completed",
        "reused",
        "reused",
        "reused",
        "reused",
        "reused",
    ]
    assert (
        next(stage for stage in second_summary["stages"] if stage["stage_name"] == "research")["details"]["reuse_policy"]["reason"]
        == "Rerun required by reuse_policy.force_rerun_stages."
    )


def test_run_research_campaign_can_invalidate_downstream_after_forced_rerun(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    alpha_catalog = tmp_path / "alphas.yml"
    strategy_config = tmp_path / "strategies.yml"
    portfolio_config = tmp_path / "portfolios.yml"
    features_root = tmp_path / "features_root"
    alpha_catalog.write_text(
        """
alpha_one:
  alpha_name: alpha_one
  dataset: features_daily
  target_column: target_ret_1d
  feature_columns: [feature_ret_1d]
  model_type: cross_sectional_linear
  model_params: {}
  alpha_horizon: 1
""".strip(),
        encoding="utf-8",
    )
    strategy_config.write_text("momentum_v1:\n  dataset: features_daily\n  parameters: {}\n", encoding="utf-8")
    portfolio_config.write_text(
        """
portfolios:
  candidate_portfolio:
    allocator: equal_weight
    components:
      - strategy_name: momentum_v1
""".strip(),
        encoding="utf-8",
    )
    _write_feature_fixture(features_root)
    monkeypatch.setenv("FEATURES_ROOT", features_root.as_posix())
    config = resolve_research_campaign_config(
        _build_full_campaign_config(
            tmp_path=tmp_path,
            alpha_catalog=alpha_catalog,
            strategy_config=strategy_config,
            portfolio_config=portfolio_config,
        ).to_dict(),
        {
            "reuse_policy": {
                "force_rerun_stages": ["research"],
                "invalidate_downstream_after_stages": ["research"],
            }
        },
    )

    def install_stage_stubs(call_order: list[str]) -> None:
        monkeypatch.setattr(
            "src.cli.run_research_campaign.run_alpha_cli.run_cli",
            lambda argv: call_order.append("alpha")
            or SimpleNamespace(alpha_name="alpha_one", run_id="alpha_run", artifact_dir=tmp_path / "alpha"),
        )
        monkeypatch.setattr(
            "src.cli.run_research_campaign.run_strategy_cli.run_cli",
            lambda argv: call_order.append("strategy")
            or SimpleNamespace(
                strategy_name="momentum_v1",
                run_id="strategy_run",
                experiment_dir=tmp_path / "strategy",
                metrics={"cumulative_return": 0.2, "sharpe_ratio": 1.1, "max_drawdown": -0.05},
            ),
        )
        monkeypatch.setattr(
            "src.cli.run_research_campaign.compare_alpha_cli.run_cli",
            lambda argv: call_order.append("alpha_comparison")
            or SimpleNamespace(comparison_id="alpha_cmp"),
        )
        monkeypatch.setattr(
            "src.cli.run_research_campaign.compare_strategies_cli.run_cli",
            lambda argv: call_order.append("strategy_comparison")
            or SimpleNamespace(comparison_id="strategy_cmp"),
        )
        monkeypatch.setattr(
            "src.cli.run_research_campaign.run_candidate_selection_cli.run_cli",
            lambda argv: call_order.append("candidate_selection")
            or SimpleNamespace(run_id="candidate_run", artifact_dir=tmp_path / "candidate_selection" / "candidate_run"),
        )
        monkeypatch.setattr(
            "src.cli.run_research_campaign.run_portfolio_cli.run_cli",
            lambda argv: call_order.append("portfolio")
            or SimpleNamespace(
                run_id="portfolio_run",
                experiment_dir=tmp_path / "portfolio_artifacts" / "portfolio_run",
                portfolio_name="candidate_portfolio",
            ),
        )
        monkeypatch.setattr(
            "src.cli.run_research_campaign.review_candidate_selection_cli.run_cli",
            lambda argv: call_order.append("candidate_review")
            or SimpleNamespace(
                review_dir=tmp_path / "candidate_review" / "candidate_run",
                candidate_selection_run_id="candidate_run",
                portfolio_run_id="portfolio_run",
            ),
        )
        monkeypatch.setattr(
            "src.cli.run_research_campaign.compare_research_cli.run_cli",
            lambda argv: call_order.append("review")
            or SimpleNamespace(review_id="review_run", entries=[]),
        )

    install_stage_stubs([])
    run_research_campaign(config)

    second_call_order: list[str] = []
    install_stage_stubs(second_call_order)
    second_result = run_research_campaign(config)
    second_summary = json.loads(second_result.campaign_summary_path.read_text(encoding="utf-8"))

    assert second_call_order == [
        "alpha",
        "strategy",
        "alpha_comparison",
        "strategy_comparison",
        "candidate_selection",
        "portfolio",
        "candidate_review",
        "review",
    ]
    assert [record.status for record in second_result.stage_records] == [
        "reused",
        "completed",
        "completed",
        "completed",
        "completed",
        "completed",
        "completed",
    ]
    comparison_stage = next(stage for stage in second_summary["stages"] if stage["stage_name"] == "comparison")
    assert comparison_stage["details"]["reuse_policy"]["invalidated_by_stage"] == "research"
    assert comparison_stage["details"]["reuse_policy"]["reason"] == (
        "Rerun required because upstream stage 'research' invalidated downstream reuse."
    )


def test_run_research_campaign_allows_multi_alpha_candidate_selection_without_explicit_alpha_filter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alpha_catalog = tmp_path / "alphas.yml"
    strategy_config = tmp_path / "strategies.yml"
    features_root = tmp_path / "features_root"
    alpha_catalog.write_text(
        """
alpha_one:
  alpha_name: alpha_one
  dataset: features_daily
  target_column: target_ret_1d
  feature_columns: [feature_ret_1d]
  model_type: cross_sectional_linear
  model_params: {}
  alpha_horizon: 1
alpha_two:
  alpha_name: alpha_two
  dataset: features_daily
  target_column: target_ret_1d
  feature_columns: [feature_ret_1d]
  model_type: cross_sectional_linear
  model_params: {}
  alpha_horizon: 1
""".strip(),
        encoding="utf-8",
    )
    strategy_config.write_text("momentum_v1:\n  dataset: features_daily\n  parameters: {}\n", encoding="utf-8")
    _write_feature_fixture(features_root)
    monkeypatch.setenv("FEATURES_ROOT", features_root.as_posix())

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
            "outputs": {
                "campaign_artifacts_root": (tmp_path / "campaign_artifacts").as_posix(),
            },
        }
    )

    candidate_selection_calls: list[list[str]] = []

    monkeypatch.setattr(
        "src.cli.run_research_campaign.run_alpha_cli.run_cli",
        lambda argv: SimpleNamespace(
            alpha_name=argv[argv.index("--alpha-name") + 1],
            run_id=f"{argv[argv.index('--alpha-name') + 1]}_run",
            artifact_dir=tmp_path / "alpha",
        ),
    )
    monkeypatch.setattr(
        "src.cli.run_research_campaign.run_candidate_selection_cli.run_cli",
        lambda argv: candidate_selection_calls.append(list(argv))
        or SimpleNamespace(
            run_id="candidate_run",
            artifact_dir=tmp_path / "candidate_selection" / "candidate_run",
        ),
    )
    monkeypatch.setattr(
        "src.cli.run_research_campaign.compare_research_cli.run_cli",
        lambda argv: SimpleNamespace(review_id="review_run"),
    )

    result = run_research_campaign(config)

    assert result.preflight_summary["status"] == "passed"
    assert candidate_selection_calls
    assert "--alpha-name" not in candidate_selection_calls[0]
    check = next(
        item
        for item in result.preflight_summary["checks"]
        if item["check_id"] == "candidate_selection.alpha_name"
    )
    assert check["status"] == "passed"
    assert "full campaign alpha universe" in check["message"]


def test_run_cli_uses_resolved_config_from_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    alpha_catalog = tmp_path / "alphas.yml"
    strategy_config = tmp_path / "strategies.yml"
    features_root = tmp_path / "features_root"
    alpha_catalog.write_text(
        """
alpha_one:
  alpha_name: alpha_one
  dataset: features_daily
  target_column: target_ret_1d
  feature_columns: [feature_ret_1d]
  model_type: cross_sectional_linear
  model_params: {}
  alpha_horizon: 1
""".strip(),
        encoding="utf-8",
    )
    strategy_config.write_text("momentum_v1:\n  dataset: features_daily\n  parameters: {}\n", encoding="utf-8")
    _write_feature_fixture(features_root)
    monkeypatch.setenv("FEATURES_ROOT", features_root.as_posix())

    config = resolve_research_campaign_config(
        {
            "targets": {
                "alpha_names": ["alpha_one"],
                "alpha_catalog_path": alpha_catalog.as_posix(),
                "strategy_config_path": strategy_config.as_posix(),
            },
            "outputs": {
                "campaign_artifacts_root": (tmp_path / "campaign_artifacts").as_posix(),
            },
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
    assert result.preflight_summary["status"] == "passed"
    assert result.campaign_manifest_path.exists()
    assert result.campaign_summary_path.exists()
    assert result.campaign_checkpoint_path.exists()


def test_run_research_campaign_resolves_candidate_selection_registry_for_portfolio_chaining(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    alpha_catalog = tmp_path / "alphas.yml"
    strategy_config = tmp_path / "strategies.yml"
    portfolio_config = tmp_path / "portfolios.yml"
    features_root = tmp_path / "features_root"
    candidate_root = tmp_path / "candidate_selection"
    candidate_registry = candidate_root / "registry.jsonl"
    candidate_artifact_dir = candidate_root / "candidate_run_registry"

    alpha_catalog.write_text(
        """
alpha_one:
  alpha_name: alpha_one
  dataset: features_daily
  target_column: target_ret_1d
  feature_columns: [feature_ret_1d]
  model_type: cross_sectional_linear
  model_params: {}
  alpha_horizon: 5
""".strip(),
        encoding="utf-8",
    )
    strategy_config.write_text("momentum_v1:\n  dataset: features_daily\n  parameters: {}\n", encoding="utf-8")
    portfolio_config.write_text(
        """
portfolios:
  candidate_portfolio:
    allocator: equal_weight
    components:
      - strategy_name: momentum_v1
""".strip(),
        encoding="utf-8",
    )
    _write_feature_fixture(features_root)
    _write_candidate_selection_registry_entry(
        candidate_registry,
        run_id="candidate_run_registry",
        artifact_dir=candidate_artifact_dir,
        alpha_name="alpha_one",
        dataset="features_daily",
        timeframe="1D",
        evaluation_horizon=5,
        mapping_names=["top_bottom_quantile_q20"],
    )
    monkeypatch.setenv("FEATURES_ROOT", features_root.as_posix())

    config = resolve_research_campaign_config(
        {
            "targets": {
                "alpha_names": ["alpha_one"],
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
            "portfolio": {
                "enabled": True,
                "from_candidate_selection": True,
            },
            "candidate_selection": {
                "enabled": False,
                "alpha_name": "alpha_one",
                "dataset": "features_daily",
                "timeframe": "1D",
                "evaluation_horizon": 5,
                "mapping_name": "top_bottom_quantile_q20",
                "output": {
                    "path": candidate_root.as_posix(),
                    "registry_path": candidate_registry.as_posix(),
                },
            },
            "review": {
                "output": {
                    "path": (tmp_path / "reviews").as_posix(),
                    "emit_plots": False,
                }
            },
            "outputs": {
                "alpha_artifacts_root": (tmp_path / "alpha_artifacts").as_posix(),
                "campaign_artifacts_root": (tmp_path / "campaign_artifacts").as_posix(),
                "portfolio_artifacts_root": (tmp_path / "portfolio_artifacts").as_posix(),
            },
        }
    )

    portfolio_calls: list[list[str]] = []
    monkeypatch.setattr(
        "src.cli.run_research_campaign.run_alpha_cli.run_cli",
        lambda argv: SimpleNamespace(alpha_name="alpha_one", run_id="alpha_run", artifact_dir=tmp_path / "alpha"),
    )
    monkeypatch.setattr(
        "src.cli.run_research_campaign.run_portfolio_cli.run_cli",
        lambda argv: portfolio_calls.append(list(argv))
        or SimpleNamespace(
            run_id="portfolio_run",
            experiment_dir=tmp_path / "portfolio_artifacts" / "portfolio_run",
            portfolio_name="candidate_portfolio",
        ),
    )
    monkeypatch.setattr(
        "src.cli.run_research_campaign.compare_research_cli.run_cli",
        lambda argv: SimpleNamespace(review_id="review_run"),
    )

    result = run_research_campaign(config)

    assert len(portfolio_calls) == 1
    assert "--from-candidate-selection" in portfolio_calls[0]
    assert candidate_artifact_dir.as_posix() in portfolio_calls[0]
    portfolio_record = next(record for record in result.stage_records if record.stage_name == "portfolio")
    assert portfolio_record.details["input_candidate_selection"]["source"] == "registry"
    assert portfolio_record.details["input_candidate_selection"]["run_id"] == "candidate_run_registry"


def test_run_research_campaign_resolves_candidate_selection_stage_from_registry_with_explicit_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    alpha_catalog = tmp_path / "alphas.yml"
    strategy_config = tmp_path / "strategies.yml"
    features_root = tmp_path / "features_root"
    candidate_root = tmp_path / "candidate_selection"
    candidate_registry = candidate_root / "registry.jsonl"
    candidate_artifact_dir = candidate_root / "candidate_run_registry"

    alpha_catalog.write_text(
        """
alpha_one:
  alpha_name: alpha_one
  dataset: features_daily
  target_column: target_ret_1d
  feature_columns: [feature_ret_1d]
  model_type: cross_sectional_linear
  model_params: {}
  alpha_horizon: 5
""".strip(),
        encoding="utf-8",
    )
    strategy_config.write_text("momentum_v1:\n  dataset: features_daily\n  parameters: {}\n", encoding="utf-8")
    _write_feature_fixture(features_root)
    _write_candidate_selection_registry_entry(
        candidate_registry,
        run_id="candidate_run_registry",
        artifact_dir=candidate_artifact_dir,
        alpha_name="alpha_one",
        dataset="features_daily",
        timeframe="1D",
        evaluation_horizon=5,
        mapping_names=["top_bottom_quantile_q20"],
    )
    monkeypatch.setenv("FEATURES_ROOT", features_root.as_posix())

    config = resolve_research_campaign_config(
        {
            "targets": {
                "alpha_names": ["alpha_one"],
                "alpha_catalog_path": alpha_catalog.as_posix(),
                "strategy_config_path": strategy_config.as_posix(),
            },
            "dataset_selection": {
                "dataset": "features_daily",
                "timeframe": "1D",
                "evaluation_horizon": 5,
                "mapping_name": "top_bottom_quantile_q20",
            },
            "candidate_selection": {
                "enabled": True,
                "execution": {
                    "from_registry": True,
                },
                "output": {
                    "path": candidate_root.as_posix(),
                    "registry_path": candidate_registry.as_posix(),
                },
            },
            "review": {
                "output": {
                    "path": (tmp_path / "reviews").as_posix(),
                    "emit_plots": False,
                }
            },
            "outputs": {
                "alpha_artifacts_root": (tmp_path / "alpha_artifacts").as_posix(),
                "campaign_artifacts_root": (tmp_path / "campaign_artifacts").as_posix(),
            },
        }
    )

    candidate_calls: list[list[str]] = []
    monkeypatch.setattr(
        "src.cli.run_research_campaign.run_alpha_cli.run_cli",
        lambda argv: SimpleNamespace(alpha_name="alpha_one", run_id="alpha_run", artifact_dir=tmp_path / "alpha"),
    )
    monkeypatch.setattr(
        "src.cli.run_research_campaign.run_candidate_selection_cli.run_cli",
        lambda argv: candidate_calls.append(list(argv))
        or SimpleNamespace(run_id="candidate_run_registry", artifact_dir=candidate_artifact_dir),
    )
    monkeypatch.setattr(
        "src.cli.run_research_campaign.compare_research_cli.run_cli",
        lambda argv: SimpleNamespace(review_id="review_run"),
    )

    result = run_research_campaign(config)

    assert len(candidate_calls) == 1
    assert "--candidate-selection-path" in candidate_calls[0]
    assert candidate_artifact_dir.as_posix() in candidate_calls[0]
    assert "--from-registry" not in candidate_calls[0]
    candidate_record = next(record for record in result.stage_records if record.stage_name == "candidate_selection")
    assert candidate_record.details["input_reference"]["source"] == "registry"
    assert candidate_record.details["input_reference"]["run_id"] == "candidate_run_registry"


def test_run_research_campaign_fails_when_candidate_selection_registry_match_is_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    alpha_catalog = tmp_path / "alphas.yml"
    strategy_config = tmp_path / "strategies.yml"
    portfolio_config = tmp_path / "portfolios.yml"
    features_root = tmp_path / "features_root"
    candidate_root = tmp_path / "candidate_selection"
    candidate_registry = candidate_root / "registry.jsonl"

    alpha_catalog.write_text(
        """
alpha_one:
  alpha_name: alpha_one
  dataset: features_daily
  target_column: target_ret_1d
  feature_columns: [feature_ret_1d]
  model_type: cross_sectional_linear
  model_params: {}
  alpha_horizon: 5
""".strip(),
        encoding="utf-8",
    )
    strategy_config.write_text("momentum_v1:\n  dataset: features_daily\n  parameters: {}\n", encoding="utf-8")
    portfolio_config.write_text(
        """
portfolios:
  candidate_portfolio:
    allocator: equal_weight
    components:
      - strategy_name: momentum_v1
""".strip(),
        encoding="utf-8",
    )
    _write_feature_fixture(features_root)
    _write_candidate_selection_registry_entry(
        candidate_registry,
        run_id="candidate_run_a",
        artifact_dir=candidate_root / "candidate_run_a",
        alpha_name="alpha_one",
        dataset="features_daily",
        timeframe="1D",
        evaluation_horizon=5,
        mapping_names=["top_bottom_quantile_q20"],
    )
    _write_candidate_selection_registry_entry(
        candidate_registry,
        run_id="candidate_run_b",
        artifact_dir=candidate_root / "candidate_run_b",
        alpha_name="alpha_one",
        dataset="features_daily",
        timeframe="1D",
        evaluation_horizon=5,
        mapping_names=["top_bottom_quantile_q20"],
    )
    monkeypatch.setenv("FEATURES_ROOT", features_root.as_posix())

    config = resolve_research_campaign_config(
        {
            "targets": {
                "alpha_names": ["alpha_one"],
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
            "portfolio": {
                "enabled": True,
                "from_candidate_selection": True,
            },
            "candidate_selection": {
                "enabled": False,
                "output": {
                    "path": candidate_root.as_posix(),
                    "registry_path": candidate_registry.as_posix(),
                },
            },
            "outputs": {
                "campaign_artifacts_root": (tmp_path / "campaign_artifacts").as_posix(),
            },
        }
    )

    with pytest.raises(ValueError, match="Ambiguous candidate_selection registry state"):
        run_research_campaign(config)


def test_resolve_candidate_selection_reference_reports_incomplete_registry_state(
    tmp_path: Path,
) -> None:
    alpha_catalog = tmp_path / "alphas.yml"
    strategy_config = tmp_path / "strategies.yml"
    candidate_root = tmp_path / "candidate_selection"
    candidate_registry = candidate_root / "registry.jsonl"
    candidate_artifact_dir = candidate_root / "candidate_run_registry"
    candidate_artifact_dir.mkdir(parents=True, exist_ok=True)

    alpha_catalog.write_text(
        """
alpha_one:
  alpha_name: alpha_one
  dataset: features_daily
  target_column: target_ret_1d
  feature_columns: [feature_ret_1d]
  model_type: cross_sectional_linear
  model_params: {}
  alpha_horizon: 5
""".strip(),
        encoding="utf-8",
    )
    strategy_config.write_text("momentum_v1:\n  dataset: features_daily\n  parameters: {}\n", encoding="utf-8")
    candidate_registry.write_text(
        json.dumps(
            {
                "run_id": "candidate_run_registry",
                "run_type": "candidate_selection",
                "timestamp": "2026-04-01T00:00:00Z",
                "alpha_name": "alpha_one",
                "dataset": "features_daily",
                "timeframe": "1D",
                "evaluation_horizon": 5,
                "artifact_path": candidate_artifact_dir.as_posix(),
                "manifest_path": (candidate_artifact_dir / "missing_manifest.json").as_posix(),
                "metadata": {
                    "mapping_names": ["top_bottom_quantile_q20"],
                    "primary_metric": "ic_ir",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    config = resolve_research_campaign_config(
        {
            "targets": {
                "alpha_names": ["alpha_one"],
                "alpha_catalog_path": alpha_catalog.as_posix(),
                "strategy_config_path": strategy_config.as_posix(),
            },
            "dataset_selection": {
                "dataset": "features_daily",
                "timeframe": "1D",
                "evaluation_horizon": 5,
                "mapping_name": "top_bottom_quantile_q20",
            },
            "candidate_selection": {
                "enabled": False,
                "alpha_name": "alpha_one",
                "dataset": "features_daily",
                "timeframe": "1D",
                "evaluation_horizon": 5,
                "mapping_name": "top_bottom_quantile_q20",
                "output": {
                    "path": candidate_root.as_posix(),
                    "registry_path": candidate_registry.as_posix(),
                },
            },
        }
    )

    with pytest.raises(ValueError, match="Candidate-selection registry state is incomplete") as exc_info:
        _resolve_candidate_selection_reference(config)

    assert "manifest missing" in str(exc_info.value)
    assert "candidate_run_registry" in str(exc_info.value)


def test_run_research_campaign_chains_portfolio_registry_into_candidate_review(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    alpha_catalog = tmp_path / "alphas.yml"
    strategy_config = tmp_path / "strategies.yml"
    features_root = tmp_path / "features_root"
    portfolio_artifacts_root = tmp_path / "portfolio_artifacts"
    portfolio_registry = portfolio_artifacts_root / "registry.jsonl"
    portfolio_artifact_dir = portfolio_artifacts_root / "portfolio_run_registry"
    candidate_artifact_dir = tmp_path / "candidate_selection" / "candidate_run"

    alpha_catalog.write_text(
        """
alpha_one:
  alpha_name: alpha_one
  dataset: features_daily
  target_column: target_ret_1d
  feature_columns: [feature_ret_1d]
  model_type: cross_sectional_linear
  model_params: {}
  alpha_horizon: 5
""".strip(),
        encoding="utf-8",
    )
    strategy_config.write_text("momentum_v1:\n  dataset: features_daily\n  parameters: {}\n", encoding="utf-8")
    _write_feature_fixture(features_root)
    _write_portfolio_registry_entry(
        portfolio_registry,
        run_id="portfolio_run_registry",
        artifact_dir=portfolio_artifact_dir,
        portfolio_name="candidate_portfolio",
        timeframe="1D",
    )
    monkeypatch.setenv("FEATURES_ROOT", features_root.as_posix())

    config = resolve_research_campaign_config(
        {
            "targets": {
                "alpha_names": ["alpha_one"],
                "alpha_catalog_path": alpha_catalog.as_posix(),
                "strategy_config_path": strategy_config.as_posix(),
            },
            "dataset_selection": {
                "dataset": "features_daily",
                "timeframe": "1D",
                "evaluation_horizon": 5,
                "mapping_name": "top_bottom_quantile_q20",
            },
            "candidate_selection": {
                "enabled": True,
                "execution": {
                    "enable_review": True,
                },
                "output": {
                    "path": (tmp_path / "candidate_selection").as_posix(),
                    "review_output_path": (tmp_path / "candidate_review").as_posix(),
                },
            },
            "portfolio": {
                "enabled": False,
                "portfolio_name": "candidate_portfolio",
                "timeframe": "1D",
            },
            "review": {
                "output": {
                    "path": (tmp_path / "reviews").as_posix(),
                    "emit_plots": False,
                }
            },
            "outputs": {
                "alpha_artifacts_root": (tmp_path / "alpha_artifacts").as_posix(),
                "campaign_artifacts_root": (tmp_path / "campaign_artifacts").as_posix(),
                "portfolio_artifacts_root": portfolio_artifacts_root.as_posix(),
            },
        }
    )

    candidate_review_calls: list[list[str]] = []
    monkeypatch.setattr(
        "src.cli.run_research_campaign.run_alpha_cli.run_cli",
        lambda argv: SimpleNamespace(alpha_name="alpha_one", run_id="alpha_run", artifact_dir=tmp_path / "alpha"),
    )
    monkeypatch.setattr(
        "src.cli.run_research_campaign.run_candidate_selection_cli.run_cli",
        lambda argv: SimpleNamespace(run_id="candidate_run", artifact_dir=candidate_artifact_dir),
    )
    monkeypatch.setattr(
        "src.cli.run_research_campaign.review_candidate_selection_cli.run_cli",
        lambda argv: candidate_review_calls.append(list(argv))
        or SimpleNamespace(
            review_dir=tmp_path / "candidate_review",
            candidate_selection_run_id="candidate_run",
            portfolio_run_id="portfolio_run_registry",
        ),
    )
    monkeypatch.setattr(
        "src.cli.run_research_campaign.compare_research_cli.run_cli",
        lambda argv: SimpleNamespace(review_id="review_run"),
    )

    result = run_research_campaign(config)
    checkpoint = json.loads(result.campaign_checkpoint_path.read_text(encoding="utf-8"))

    assert len(candidate_review_calls) == 1
    assert "--candidate-selection-path" in candidate_review_calls[0]
    assert candidate_artifact_dir.as_posix() in candidate_review_calls[0]
    assert "--portfolio-path" in candidate_review_calls[0]
    assert portfolio_artifact_dir.as_posix() in candidate_review_calls[0]
    assert result.candidate_review_result.portfolio_run_id == "portfolio_run_registry"
    assert checkpoint["stage_states"]["portfolio"] == "reused"
    registry_check = next(
        check
        for check in result.preflight_summary["checks"]
        if check["check_id"] == "registry.portfolio.candidate_review"
    )
    assert registry_check["status"] == "passed"


def test_run_research_campaign_marks_reused_candidate_selection_stage_in_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    alpha_catalog = tmp_path / "alphas.yml"
    strategy_config = tmp_path / "strategies.yml"
    features_root = tmp_path / "features_root"
    portfolio_artifacts_root = tmp_path / "portfolio_artifacts"
    portfolio_registry = portfolio_artifacts_root / "registry.jsonl"
    portfolio_artifact_dir = portfolio_artifacts_root / "portfolio_run_registry"
    candidate_root = tmp_path / "candidate_selection"
    candidate_registry = candidate_root / "registry.jsonl"
    candidate_artifact_dir = candidate_root / "candidate_run_registry"

    alpha_catalog.write_text(
        """
alpha_one:
  alpha_name: alpha_one
  dataset: features_daily
  target_column: target_ret_1d
  feature_columns: [feature_ret_1d]
  model_type: cross_sectional_linear
  model_params: {}
  alpha_horizon: 5
""".strip(),
        encoding="utf-8",
    )
    strategy_config.write_text("momentum_v1:\n  dataset: features_daily\n  parameters: {}\n", encoding="utf-8")
    _write_feature_fixture(features_root)
    _write_candidate_selection_registry_entry(
        candidate_registry,
        run_id="candidate_run_registry",
        artifact_dir=candidate_artifact_dir,
        alpha_name="alpha_one",
        dataset="features_daily",
        timeframe="1D",
        evaluation_horizon=5,
        mapping_names=["top_bottom_quantile_q20"],
    )
    _write_portfolio_registry_entry(
        portfolio_registry,
        run_id="portfolio_run_registry",
        artifact_dir=portfolio_artifact_dir,
        portfolio_name="candidate_portfolio",
        timeframe="1D",
        candidate_selection_run_id="candidate_run_registry",
    )
    monkeypatch.setenv("FEATURES_ROOT", features_root.as_posix())

    config = resolve_research_campaign_config(
        {
            "targets": {
                "alpha_names": ["alpha_one"],
                "alpha_catalog_path": alpha_catalog.as_posix(),
                "strategy_config_path": strategy_config.as_posix(),
            },
            "dataset_selection": {
                "dataset": "features_daily",
                "timeframe": "1D",
                "evaluation_horizon": 5,
                "mapping_name": "top_bottom_quantile_q20",
            },
            "candidate_selection": {
                "enabled": False,
                "alpha_name": "alpha_one",
                "output": {
                    "path": candidate_root.as_posix(),
                    "registry_path": candidate_registry.as_posix(),
                },
            },
            "portfolio": {
                "enabled": True,
                "from_candidate_selection": True,
                "portfolio_name": "candidate_portfolio",
                "timeframe": "1D",
            },
            "review": {
                "output": {
                    "path": (tmp_path / "reviews").as_posix(),
                    "emit_plots": False,
                }
            },
            "outputs": {
                "campaign_artifacts_root": (tmp_path / "campaign_artifacts").as_posix(),
                "portfolio_artifacts_root": portfolio_artifacts_root.as_posix(),
            },
        }
    )

    monkeypatch.setattr(
        "src.cli.run_research_campaign.run_alpha_cli.run_cli",
        lambda argv: SimpleNamespace(alpha_name="alpha_one", run_id="alpha_run", artifact_dir=tmp_path / "alpha"),
    )
    monkeypatch.setattr(
        "src.cli.run_research_campaign.run_portfolio_cli.run_cli",
        lambda argv: SimpleNamespace(
            run_id="portfolio_run",
            experiment_dir=tmp_path / "portfolio_artifacts" / "portfolio_run",
            portfolio_name="candidate_portfolio",
        ),
    )
    monkeypatch.setattr(
        "src.cli.run_research_campaign.compare_research_cli.run_cli",
        lambda argv: SimpleNamespace(review_id="review_run"),
    )

    result = run_research_campaign(config)
    checkpoint = json.loads(result.campaign_checkpoint_path.read_text(encoding="utf-8"))

    assert checkpoint["stage_states"]["candidate_selection"] == "reused"
    assert checkpoint["stage_states"]["portfolio"] == "completed"
    assert checkpoint["stages"][3]["source"] == "registry"



def test_run_research_campaign_writes_pending_checkpoint_states_after_preflight_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    alpha_catalog = tmp_path / "alphas.yml"
    strategy_config = tmp_path / "strategies.yml"
    alpha_catalog.write_text(
        """
alpha_one:
  alpha_name: alpha_one
  dataset: features_daily
  target_column: target_ret_1d
  feature_columns: [feature_ret_1d]
  model_type: cross_sectional_linear
  model_params: {}
  alpha_horizon: 5
""".strip(),
        encoding="utf-8",
    )
    strategy_config.write_text("momentum_v1:\n  dataset: features_daily\n  parameters: {}\n", encoding="utf-8")

    config = resolve_research_campaign_config(
        {
            "targets": {
                "alpha_names": ["alpha_one"],
                "alpha_catalog_path": alpha_catalog.as_posix(),
                "strategy_config_path": strategy_config.as_posix(),
            },
            "dataset_selection": {
                "dataset": "features_daily",
                "timeframe": "1D",
                "evaluation_horizon": 5,
            },
            "outputs": {
                "campaign_artifacts_root": (tmp_path / "campaign_artifacts").as_posix(),
            },
        }
    )

    def fake_preflight(*_args: object, **_kwargs: object) -> CampaignPreflightResult:
        summary_path = tmp_path / "campaign_artifacts" / "research_campaign_demo" / "preflight_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "status": "failed",
            "check_counts": {"failed": 1, "passed": 0, "warning": 0},
            "checks": [
                {
                    "check_id": "dataset.features_daily",
                    "status": "failed",
                    "message": "Dataset root unavailable.",
                    "details": {},
                }
            ],
            "failed_checks": ["Dataset root unavailable."],
        }
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        return CampaignPreflightResult(
            status="failed",
            summary_path=summary_path,
            summary=summary,
            checks=(
                CampaignPreflightCheck(
                    check_id="dataset.features_daily",
                    status="failed",
                    message="Dataset root unavailable.",
                ),
            ),
        )

    monkeypatch.setattr("src.cli.run_research_campaign._run_preflight", fake_preflight)

    with pytest.raises(ValueError, match="Campaign preflight failed"):
        run_research_campaign(config)

    checkpoint_path = next((tmp_path / "campaign_artifacts").rglob("checkpoint.json"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))

    assert checkpoint["status"] == "failed"
    assert checkpoint["stage_states"]["preflight"] == "failed"
    assert checkpoint["stage_states"]["research"] == "pending"
    assert checkpoint["stage_states"]["review"] == "pending"


def test_run_research_campaign_persists_failed_stage_metadata_and_retries_on_resume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    alpha_catalog = tmp_path / "alphas.yml"
    strategy_config = tmp_path / "strategies.yml"
    portfolio_config = tmp_path / "portfolios.yml"
    features_root = tmp_path / "features_root"
    _write_feature_fixture(features_root)
    monkeypatch.setenv("FEATURES_ROOT", features_root.as_posix())

    alpha_catalog.write_text(
        """
alpha_one:
  alpha_name: alpha_one
  dataset: features_daily
  target_column: target_ret_1d
  feature_columns: [feature_ret_1d]
  model_type: cross_sectional_linear
  model_params: {}
  alpha_horizon: 1
""".strip(),
        encoding="utf-8",
    )
    strategy_config.write_text("momentum_v1:\n  dataset: features_daily\n  parameters: {}\n", encoding="utf-8")
    portfolio_config.write_text(
        """
portfolios:
  candidate_portfolio:
    allocator: equal_weight
    components:
      - strategy_name: momentum_v1
""".strip(),
        encoding="utf-8",
    )
    config = _build_full_campaign_config(
        tmp_path=tmp_path,
        alpha_catalog=alpha_catalog,
        strategy_config=strategy_config,
        portfolio_config=portfolio_config,
    )

    def _write_json(path: Path, payload: dict[str, object]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return path

    attempts = {"candidate_selection": 0}

    monkeypatch.setattr(
        "src.cli.run_research_campaign.run_alpha_cli.run_cli",
        lambda argv: SimpleNamespace(alpha_name="alpha_one", run_id="alpha_run", artifact_dir=tmp_path / "alpha"),
    )
    monkeypatch.setattr(
        "src.cli.run_research_campaign.run_strategy_cli.run_cli",
        lambda argv: SimpleNamespace(
            strategy_name="momentum_v1",
            run_id="strategy_run",
            experiment_dir=tmp_path / "strategy",
            metrics={"cumulative_return": 0.2, "sharpe_ratio": 1.1, "max_drawdown": -0.05},
        ),
    )
    monkeypatch.setattr(
        "src.cli.run_research_campaign.compare_alpha_cli.run_cli",
        lambda argv: SimpleNamespace(comparison_id="alpha_cmp"),
    )
    monkeypatch.setattr(
        "src.cli.run_research_campaign.compare_strategies_cli.run_cli",
        lambda argv: SimpleNamespace(comparison_id="strategy_cmp"),
    )

    def _candidate_selection_cli(argv: list[str]) -> SimpleNamespace:
        attempts["candidate_selection"] += 1
        if attempts["candidate_selection"] == 1:
            raise RuntimeError("candidate selection exploded")
        artifact_dir = tmp_path / "candidate_selection" / "candidate_run"
        return SimpleNamespace(
            run_id="candidate_run",
            artifact_dir=artifact_dir,
            summary_json=_write_json(artifact_dir / "selection_summary.json", {"selected_candidates": 2}),
            manifest_json=_write_json(artifact_dir / "manifest.json", {"run_id": "candidate_run"}),
            primary_metric="ic_ir",
            universe_count=8,
            eligible_count=4,
            selected_count=2,
            rejected_count=2,
            pruned_by_redundancy=1,
        )

    monkeypatch.setattr("src.cli.run_research_campaign.run_candidate_selection_cli.run_cli", _candidate_selection_cli)
    monkeypatch.setattr(
        "src.cli.run_research_campaign.run_portfolio_cli.run_cli",
        lambda argv: SimpleNamespace(
            run_id="portfolio_run",
            experiment_dir=tmp_path / "portfolio_artifacts" / "portfolio_run",
            portfolio_name="candidate_portfolio",
            component_count=2,
            metrics={"total_return": 0.16, "sharpe_ratio": 1.2, "max_drawdown": -0.04},
        ),
    )
    monkeypatch.setattr(
        "src.cli.run_research_campaign.review_candidate_selection_cli.run_cli",
        lambda argv: SimpleNamespace(
            review_dir=tmp_path / "candidate_review" / "candidate_run",
            candidate_review_summary_json=_write_json(
                tmp_path / "candidate_review" / "candidate_run" / "candidate_review_summary.json",
                {"selected_candidates": 2, "rejected_candidates": 2},
            ),
            manifest_json=_write_json(
                tmp_path / "candidate_review" / "candidate_run" / "manifest.json",
                {"run_id": "candidate_review"},
            ),
            candidate_selection_run_id="candidate_run",
            portfolio_run_id="portfolio_run",
            total_candidates=4,
            selected_candidates=2,
            rejected_candidates=2,
        ),
    )
    monkeypatch.setattr(
        "src.cli.run_research_campaign.compare_research_cli.run_cli",
        lambda argv: SimpleNamespace(review_id="review_run", entries=[]),
    )

    with pytest.raises(RuntimeError, match="candidate selection exploded"):
        run_research_campaign(config)

    failed_checkpoint_path = next((tmp_path / "campaign_artifacts").rglob("checkpoint.json"))
    failed_summary_path = next((tmp_path / "campaign_artifacts").rglob("summary.json"))
    failed_checkpoint = json.loads(failed_checkpoint_path.read_text(encoding="utf-8"))
    failed_summary = json.loads(failed_summary_path.read_text(encoding="utf-8"))
    failed_stage = next(stage for stage in failed_summary["stages"] if stage["stage_name"] == "candidate_selection")

    assert failed_checkpoint["status"] == "failed"
    assert failed_checkpoint["stage_states"]["candidate_selection"] == "failed"
    assert failed_checkpoint["failed_stage_count"] == 1
    assert failed_stage["details"]["failure"] == {
        "exception_type": "RuntimeError",
        "kind": "exception",
        "message": "candidate selection exploded",
        "retryable": True,
        "stage_name": "candidate_selection",
    }
    assert failed_summary["final_outcomes"]["failed_stage_names"] == ["candidate_selection"]
    assert failed_summary["final_outcomes"]["retry_stage_names"] == []
    assert failed_summary["final_outcomes"]["resumable_stage_names"] == [
        "candidate_selection",
        "portfolio",
        "candidate_review",
        "review",
    ]
    assert failed_stage["execution_metadata"]["resume"]["resumable"] is True
    assert failed_stage["execution_metadata"]["retry"]["attempted"] is False
    assert failed_stage["execution_metadata"]["failure"]["message"] == "candidate selection exploded"
    assert failed_stage["execution_metadata"]["fingerprint"]["input_fingerprint"] == (
        failed_checkpoint["stage_input_fingerprints"]["candidate_selection"]
    )

    resumed_result = run_research_campaign(config)
    resumed_summary = json.loads(resumed_result.campaign_summary_path.read_text(encoding="utf-8"))
    resumed_manifest = json.loads(resumed_result.campaign_manifest_path.read_text(encoding="utf-8"))
    resumed_checkpoint = json.loads(resumed_result.campaign_checkpoint_path.read_text(encoding="utf-8"))
    resumed_stage = next(stage for stage in resumed_summary["stages"] if stage["stage_name"] == "candidate_selection")

    assert resumed_checkpoint["stage_states"]["preflight"] == "reused"
    assert resumed_checkpoint["stage_states"]["research"] == "reused"
    assert resumed_checkpoint["stage_states"]["comparison"] == "reused"
    assert resumed_checkpoint["stage_states"]["candidate_selection"] == "completed"
    assert resumed_stage["details"]["retry"]["attempted"] is True
    assert resumed_stage["details"]["retry"]["previous_state"] == "failed"
    assert resumed_stage["details"]["retry"]["previous_failure"]["message"] == "candidate selection exploded"
    assert resumed_summary["final_outcomes"]["retry_stage_names"] == ["candidate_selection"]
    assert resumed_summary["stage_state_counts"]["reused"] == 3
    assert resumed_stage["execution_metadata"]["retry"]["attempted"] is True
    assert resumed_stage["execution_metadata"]["retry"]["previous_state"] == "failed"
    assert resumed_stage["execution_metadata"]["reuse"]["reused"] is False
    assert resumed_manifest["stage_execution"]["candidate_selection"]["retry"]["attempted"] is True


def test_run_research_campaign_persists_partial_stage_metadata_for_interrupts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    alpha_catalog = tmp_path / "alphas.yml"
    strategy_config = tmp_path / "strategies.yml"
    portfolio_config = tmp_path / "portfolios.yml"
    features_root = tmp_path / "features_root"
    _write_feature_fixture(features_root)
    monkeypatch.setenv("FEATURES_ROOT", features_root.as_posix())

    alpha_catalog.write_text(
        """
alpha_one:
  alpha_name: alpha_one
  dataset: features_daily
  target_column: target_ret_1d
  feature_columns: [feature_ret_1d]
  model_type: cross_sectional_linear
  model_params: {}
  alpha_horizon: 1
""".strip(),
        encoding="utf-8",
    )
    strategy_config.write_text("momentum_v1:\n  dataset: features_daily\n  parameters: {}\n", encoding="utf-8")
    portfolio_config.write_text(
        """
portfolios:
  candidate_portfolio:
    allocator: equal_weight
    components:
      - strategy_name: momentum_v1
""".strip(),
        encoding="utf-8",
    )
    config = _build_full_campaign_config(
        tmp_path=tmp_path,
        alpha_catalog=alpha_catalog,
        strategy_config=strategy_config,
        portfolio_config=portfolio_config,
    )

    attempts = {"comparison": 0}

    monkeypatch.setattr(
        "src.cli.run_research_campaign.run_alpha_cli.run_cli",
        lambda argv: SimpleNamespace(alpha_name="alpha_one", run_id="alpha_run", artifact_dir=tmp_path / "alpha"),
    )
    monkeypatch.setattr(
        "src.cli.run_research_campaign.run_strategy_cli.run_cli",
        lambda argv: SimpleNamespace(
            strategy_name="momentum_v1",
            run_id="strategy_run",
            experiment_dir=tmp_path / "strategy",
            metrics={"cumulative_return": 0.2, "sharpe_ratio": 1.1, "max_drawdown": -0.05},
        ),
    )

    def _alpha_compare(argv: list[str]) -> SimpleNamespace:
        attempts["comparison"] += 1
        if attempts["comparison"] == 1:
            raise KeyboardInterrupt("manual stop")
        return SimpleNamespace(comparison_id="alpha_cmp")

    monkeypatch.setattr("src.cli.run_research_campaign.compare_alpha_cli.run_cli", _alpha_compare)
    monkeypatch.setattr(
        "src.cli.run_research_campaign.compare_strategies_cli.run_cli",
        lambda argv: SimpleNamespace(comparison_id="strategy_cmp"),
    )
    monkeypatch.setattr(
        "src.cli.run_research_campaign.run_candidate_selection_cli.run_cli",
        lambda argv: SimpleNamespace(
            run_id="candidate_run",
            artifact_dir=tmp_path / "candidate_selection" / "candidate_run",
            summary_json=tmp_path / "candidate_selection" / "candidate_run" / "selection_summary.json",
            manifest_json=tmp_path / "candidate_selection" / "candidate_run" / "manifest.json",
            primary_metric="ic_ir",
            universe_count=8,
            eligible_count=4,
            selected_count=2,
            rejected_count=2,
            pruned_by_redundancy=1,
        ),
    )
    monkeypatch.setattr(
        "src.cli.run_research_campaign.run_portfolio_cli.run_cli",
        lambda argv: SimpleNamespace(
            run_id="portfolio_run",
            experiment_dir=tmp_path / "portfolio_artifacts" / "portfolio_run",
            portfolio_name="candidate_portfolio",
            component_count=2,
            metrics={"total_return": 0.16, "sharpe_ratio": 1.2, "max_drawdown": -0.04},
        ),
    )
    monkeypatch.setattr(
        "src.cli.run_research_campaign.review_candidate_selection_cli.run_cli",
        lambda argv: SimpleNamespace(
            review_dir=tmp_path / "candidate_review" / "candidate_run",
            candidate_review_summary_json=tmp_path / "candidate_review" / "candidate_run" / "candidate_review_summary.json",
            manifest_json=tmp_path / "candidate_review" / "candidate_run" / "manifest.json",
            candidate_selection_run_id="candidate_run",
            portfolio_run_id="portfolio_run",
            total_candidates=4,
            selected_candidates=2,
            rejected_candidates=2,
        ),
    )
    monkeypatch.setattr(
        "src.cli.run_research_campaign.compare_research_cli.run_cli",
        lambda argv: SimpleNamespace(review_id="review_run", entries=[]),
    )

    with pytest.raises(KeyboardInterrupt, match="manual stop"):
        run_research_campaign(config)

    partial_checkpoint_path = next((tmp_path / "campaign_artifacts").rglob("checkpoint.json"))
    partial_summary_path = next((tmp_path / "campaign_artifacts").rglob("summary.json"))
    partial_checkpoint = json.loads(partial_checkpoint_path.read_text(encoding="utf-8"))
    partial_summary = json.loads(partial_summary_path.read_text(encoding="utf-8"))
    partial_stage = next(stage for stage in partial_summary["stages"] if stage["stage_name"] == "comparison")

    assert partial_checkpoint["status"] == "partial"
    assert partial_checkpoint["stage_states"]["comparison"] == "partial"
    assert partial_checkpoint["partial_stage_count"] == 1
    assert partial_checkpoint["stage_states"]["candidate_selection"] == "pending"
    assert partial_stage["details"]["failure"]["kind"] == "interrupted"
    assert partial_stage["details"]["failure"]["message"] == "manual stop"
    assert partial_summary["final_outcomes"]["partial_stage_names"] == ["comparison"]
    assert partial_summary["final_outcomes"]["resumable_stage_names"] == [
        "comparison",
        "candidate_selection",
        "portfolio",
        "candidate_review",
        "review",
    ]
    assert partial_stage["execution_metadata"]["resume"]["resumable"] is True
    assert partial_stage["execution_metadata"]["failure"]["kind"] == "interrupted"
    assert partial_stage["execution_metadata"]["retry"]["attempted"] is False

    resumed_result = run_research_campaign(config)
    print_summary(resumed_result)
    stdout = capsys.readouterr().out
    resumed_summary = json.loads(resumed_result.campaign_summary_path.read_text(encoding="utf-8"))
    resumed_stage = next(stage for stage in resumed_summary["stages"] if stage["stage_name"] == "comparison")

    assert resumed_stage["state"] == "completed"
    assert resumed_stage["details"]["retry"]["attempted"] is True
    assert resumed_stage["details"]["retry"]["previous_state"] == "partial"
    assert resumed_summary["final_outcomes"]["retry_stage_names"] == ["comparison"]
    assert resumed_stage["execution_metadata"]["retry"]["attempted"] is True
    assert resumed_stage["execution_metadata"]["retry"]["previous_state"] == "partial"
    assert "Stage States: completed=" in stdout
    assert "reused=" in stdout
    assert "Stage Details: preflight=reused | research=reused | comparison=completed" in stdout


def test_run_research_campaign_writes_stable_stitched_artifacts_across_identical_reruns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    alpha_catalog = tmp_path / "alphas.yml"
    strategy_config = tmp_path / "strategies.yml"
    portfolio_config = tmp_path / "portfolios.yml"
    features_root = tmp_path / "features_root"
    _write_feature_fixture(features_root)
    monkeypatch.setenv("FEATURES_ROOT", features_root.as_posix())

    alpha_catalog.write_text(
        """
alpha_a:
  alpha_name: alpha_a
  dataset: features_daily
  target_column: target_ret_1d
  feature_columns: [feature_ret_1d]
  model_type: cross_sectional_linear
  model_params: {}
  alpha_horizon: 5
alpha_b:
  alpha_name: alpha_b
  dataset: features_daily
  target_column: target_ret_1d
  feature_columns: [feature_ret_1d]
  model_type: cross_sectional_linear
  model_params: {}
  alpha_horizon: 5
""".strip(),
        encoding="utf-8",
    )
    strategy_config.write_text(
        """
strategy_a:
  dataset: features_daily
  parameters: {}
strategy_b:
  dataset: features_daily
  parameters: {}
""".strip(),
        encoding="utf-8",
    )
    portfolio_config.write_text(
        """
portfolios:
  candidate_portfolio:
    allocator: equal_weight
    components:
      - strategy_name: strategy_a
      - strategy_name: strategy_b
""".strip(),
        encoding="utf-8",
    )

    config = resolve_research_campaign_config(
        {
            "targets": {
                "alpha_names": ["alpha_b", "alpha_a"],
                "strategy_names": ["strategy_b", "strategy_a"],
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
            },
            "comparison": {
                "enabled": True,
                "top_k": 5,
            },
            "candidate_selection": {
                "enabled": True,
                "alpha_name": "alpha_a",
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
                "campaign_artifacts_root": (tmp_path / "campaign_artifacts").as_posix(),
                "comparison_output_path": (tmp_path / "comparisons").as_posix(),
                "portfolio_artifacts_root": (tmp_path / "portfolio_artifacts").as_posix(),
            },
        }
    )

    alpha_run_ids = {"alpha_a": "alpha_run_a", "alpha_b": "alpha_run_b"}
    strategy_run_ids = {"strategy_a": "strategy_run_a", "strategy_b": "strategy_run_b"}
    alpha_metrics = {
        "alpha_a": {"mean_ic": 0.11, "ic_ir": 1.25, "n_periods": 24, "sharpe_ratio": 0.8, "total_return": 0.14},
        "alpha_b": {"mean_ic": 0.09, "ic_ir": 1.05, "n_periods": 24, "sharpe_ratio": 0.7, "total_return": 0.12},
    }
    strategy_metrics = {
        "strategy_a": {"cumulative_return": 0.21, "sharpe_ratio": 1.4, "max_drawdown": -0.08},
        "strategy_b": {"cumulative_return": 0.18, "sharpe_ratio": 1.2, "max_drawdown": -0.09},
    }

    def write_text(path: Path, contents: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
        return path

    def fake_alpha_cli(argv: list[str]) -> SimpleNamespace:
        alpha_name = argv[argv.index("--alpha-name") + 1]
        artifact_dir = tmp_path / "alpha_artifacts" / alpha_run_ids[alpha_name]
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            alpha_name=alpha_name,
            run_id=alpha_run_ids[alpha_name],
            artifact_dir=artifact_dir,
            evaluation=SimpleNamespace(
                evaluation_result=SimpleNamespace(summary=alpha_metrics[alpha_name]),
                manifest={
                    "sleeve": {
                        "metric_summary": {
                            "sharpe_ratio": alpha_metrics[alpha_name]["sharpe_ratio"],
                            "total_return": alpha_metrics[alpha_name]["total_return"],
                        }
                    }
                },
            ),
        )

    def fake_strategy_cli(argv: list[str]) -> SimpleNamespace:
        strategy_name = argv[argv.index("--strategy") + 1]
        experiment_dir = tmp_path / "strategy_artifacts" / strategy_run_ids[strategy_name]
        experiment_dir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            strategy_name=strategy_name,
            run_id=strategy_run_ids[strategy_name],
            experiment_dir=experiment_dir,
            metrics=strategy_metrics[strategy_name],
        )

    def fake_alpha_compare(argv: list[str]) -> SimpleNamespace:
        output_root = Path(argv[argv.index("--output-path") + 1])
        return SimpleNamespace(
            comparison_id="alpha_cmp",
            csv_path=write_text(output_root / "leaderboard.csv", "alpha_name,score\nalpha_a,1.0\n"),
            json_path=write_text(output_root / "summary.json", '{"comparison":"alpha"}\n'),
        )

    def fake_strategy_compare(argv: list[str]) -> SimpleNamespace:
        output_root = Path(argv[argv.index("--output-path") + 1])
        return SimpleNamespace(
            comparison_id="strategy_cmp",
            csv_path=write_text(output_root / "leaderboard.csv", "strategy_name,score\nstrategy_a,1.0\n"),
            json_path=write_text(output_root / "summary.json", '{"comparison":"strategy"}\n'),
        )

    def fake_candidate_selection(argv: list[str]) -> SimpleNamespace:
        artifact_dir = tmp_path / "candidate_selection" / "candidate_run"
        summary_json = write_text(artifact_dir / "selection_summary.json", '{"selected_candidates":2}\n')
        manifest_json = write_text(artifact_dir / "manifest.json", '{"run_id":"candidate_run"}\n')
        return SimpleNamespace(
            run_id="candidate_run",
            artifact_dir=artifact_dir,
            summary_json=summary_json,
            manifest_json=manifest_json,
            primary_metric="ic_ir",
            universe_count=8,
            eligible_count=4,
            selected_count=2,
            rejected_count=2,
            pruned_by_redundancy=1,
        )

    def fake_portfolio(argv: list[str]) -> SimpleNamespace:
        experiment_dir = tmp_path / "portfolio_artifacts" / "portfolio_run"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            run_id="portfolio_run",
            experiment_dir=experiment_dir,
            portfolio_name="candidate_portfolio",
            component_count=2,
            metrics={"total_return": 0.16, "sharpe_ratio": 1.3, "max_drawdown": -0.07},
        )

    def fake_candidate_review(argv: list[str]) -> SimpleNamespace:
        review_dir = tmp_path / "candidate_review" / "candidate_run"
        review_dir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            review_dir=review_dir,
            manifest_json=write_text(review_dir / "manifest.json", '{"run_id":"candidate_review"}\n'),
            candidate_review_summary_json=write_text(
                review_dir / "candidate_review_summary.json",
                '{"selected_candidates":2,"rejected_candidates":2}\n',
            ),
            candidate_selection_run_id="candidate_run",
            portfolio_run_id="portfolio_run",
            total_candidates=4,
            selected_candidates=2,
            rejected_candidates=2,
        )

    def fake_review(argv: list[str]) -> SimpleNamespace:
        review_dir = tmp_path / "reviews" / "campaign_review"
        review_dir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            review_id="review_run",
            csv_path=write_text(review_dir / "leaderboard.csv", "run_type,run_id\nalpha_evaluation,alpha_run_a\n"),
            json_path=write_text(review_dir / "summary.json", '{"entry_count":5}\n'),
            manifest_path=write_text(review_dir / "manifest.json", '{"review_id":"review_run"}\n'),
            promotion_gate_path=write_text(
                review_dir / "promotion_gates.json",
                json.dumps(
                    {
                        "evaluation_status": "passed",
                        "promotion_status": "approved",
                        "gate_count": 3,
                        "passed_gate_count": 3,
                        "failed_gate_count": 0,
                        "missing_gate_count": 0,
                    },
                    sort_keys=True,
                )
                + "\n",
            ),
            entries=[
                SimpleNamespace(run_type="alpha_evaluation"),
                SimpleNamespace(run_type="alpha_evaluation"),
                SimpleNamespace(run_type="strategy"),
                SimpleNamespace(run_type="portfolio"),
                SimpleNamespace(run_type="portfolio"),
            ],
        )

    monkeypatch.setattr("src.cli.run_research_campaign.run_alpha_cli.run_cli", fake_alpha_cli)
    monkeypatch.setattr("src.cli.run_research_campaign.run_strategy_cli.run_cli", fake_strategy_cli)
    monkeypatch.setattr("src.cli.run_research_campaign.compare_alpha_cli.run_cli", fake_alpha_compare)
    monkeypatch.setattr("src.cli.run_research_campaign.compare_strategies_cli.run_cli", fake_strategy_compare)
    monkeypatch.setattr("src.cli.run_research_campaign.run_candidate_selection_cli.run_cli", fake_candidate_selection)
    monkeypatch.setattr("src.cli.run_research_campaign.run_portfolio_cli.run_cli", fake_portfolio)
    monkeypatch.setattr("src.cli.run_research_campaign.review_candidate_selection_cli.run_cli", fake_candidate_review)
    monkeypatch.setattr("src.cli.run_research_campaign.compare_research_cli.run_cli", fake_review)

    first_result = run_research_campaign(config)
    first_manifest_bytes = first_result.campaign_manifest_path.read_bytes()
    first_summary_bytes = first_result.campaign_summary_path.read_bytes()
    first_checkpoint = json.loads(first_result.campaign_checkpoint_path.read_text(encoding="utf-8"))
    first_preflight_bytes = first_result.preflight_summary_path.read_bytes()
    first_config_bytes = (first_result.campaign_artifact_dir / "campaign_config.json").read_bytes()

    second_result = run_research_campaign(config)

    assert second_result.preflight_summary_path.read_bytes() == first_preflight_bytes
    assert (second_result.campaign_artifact_dir / "campaign_config.json").read_bytes() == first_config_bytes

    checkpoint = json.loads(second_result.campaign_checkpoint_path.read_text(encoding="utf-8"))
    summary = json.loads(second_result.campaign_summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(second_result.campaign_manifest_path.read_text(encoding="utf-8"))

    assert second_result.campaign_manifest_path.read_bytes() != first_manifest_bytes
    assert second_result.campaign_summary_path.read_bytes() != first_summary_bytes
    assert checkpoint != first_checkpoint

    assert [record.stage_name for record in second_result.stage_records] == [
        "preflight",
        "research",
        "comparison",
        "candidate_selection",
        "portfolio",
        "candidate_review",
        "review",
    ]
    assert [record.status for record in second_result.stage_records] == [
        "reused",
        "reused",
        "reused",
        "reused",
        "reused",
        "reused",
        "reused",
    ]
    assert checkpoint["stage_states"] == {
        "candidate_review": "reused",
        "candidate_selection": "reused",
        "comparison": "reused",
        "portfolio": "reused",
        "preflight": "reused",
        "research": "reused",
        "review": "reused",
    }
    assert checkpoint["stage_input_fingerprints"] == first_checkpoint["stage_input_fingerprints"]
    assert summary["selected_run_ids"]["alpha_run_ids"] == ["alpha_run_a", "alpha_run_b"]
    assert summary["selected_run_ids"]["strategy_run_ids"] == ["strategy_run_a", "strategy_run_b"]
    assert [entry["run_id"] for entry in summary["key_metrics"]["alpha_runs"]] == ["alpha_run_a", "alpha_run_b"]
    assert [entry["run_id"] for entry in summary["key_metrics"]["strategy_runs"]] == ["strategy_run_a", "strategy_run_b"]
    assert summary["output_paths"]["alpha_comparison_csv"].endswith("comparisons/alpha/leaderboard.csv")
    assert summary["output_paths"]["strategy_comparison_summary"].endswith("comparisons/strategy/summary.json")
    assert summary["output_paths"]["candidate_selection_manifest"].endswith("candidate_selection/candidate_run/manifest.json")
    assert summary["output_paths"]["candidate_review_summary"].endswith(
        "candidate_review/candidate_run/candidate_review_summary.json"
    )
    assert summary["output_paths"]["review_manifest"].endswith("reviews/campaign_review/manifest.json")
    assert summary["output_paths"]["review_promotion_gates"].endswith("reviews/campaign_review/promotion_gates.json")
    assert summary["final_outcomes"]["review_promotion_gate_summary"] == {
        "evaluation_status": "passed",
        "failed_gate_count": 0,
        "gate_count": 3,
        "missing_gate_count": 0,
        "passed_gate_count": 3,
        "promotion_status": "approved",
    }
    assert summary["final_outcomes"]["review_counts_by_run_type"] == {
        "alpha_evaluation": 2,
        "portfolio": 2,
        "strategy": 1,
    }
    assert summary["final_outcomes"]["skipped_stage_names"] == []
    assert [stage["state"] for stage in summary["stages"]] == [
        "reused",
        "reused",
        "reused",
        "reused",
        "reused",
        "reused",
        "reused",
    ]
    assert summary["stage_execution"]["preflight"]["reuse"]["reused"] is True
    assert summary["stage_execution"]["preflight"]["reuse"]["decision"]["fingerprint_match"] is True
    assert summary["stage_execution"]["preflight"]["retry"]["attempted"] is False
    assert summary["stage_execution"]["preflight"]["skip"]["skipped"] is False
    assert manifest["stage_execution"]["preflight"] == summary["stage_execution"]["preflight"]
    assert manifest["artifact_files"] == [
        "campaign_config.json",
        "checkpoint.json",
        "manifest.json",
        "preflight_summary.json",
        "summary.json",
    ]


def test_run_research_campaign_changes_dependent_stage_fingerprints_when_inputs_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    alpha_catalog = tmp_path / "alphas.yml"
    strategy_config = tmp_path / "strategies.yml"
    portfolio_config = tmp_path / "portfolios.yml"
    features_root = tmp_path / "features_root"
    _write_feature_fixture(features_root)
    monkeypatch.setenv("FEATURES_ROOT", features_root.as_posix())

    alpha_catalog.write_text(
        """
alpha_one:
  alpha_name: alpha_one
  dataset: features_daily
  target_column: target_ret_1d
  feature_columns: [feature_ret_1d]
  model_type: cross_sectional_linear
  model_params: {}
  alpha_horizon: 5
""".strip(),
        encoding="utf-8",
    )
    strategy_config.write_text("strategy_one:\n  dataset: features_daily\n  parameters: {}\n", encoding="utf-8")
    portfolio_config.write_text(
        """
portfolios:
  candidate_portfolio:
    allocator: equal_weight
    components:
      - strategy_name: strategy_one
""".strip(),
        encoding="utf-8",
    )

    def build_config(*, max_candidates: int) -> object:
        return resolve_research_campaign_config(
            {
                "targets": {
                    "alpha_names": ["alpha_one"],
                    "strategy_names": ["strategy_one"],
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
                },
                "comparison": {
                    "enabled": True,
                    "top_k": 3,
                },
                "candidate_selection": {
                    "enabled": True,
                    "alpha_name": "alpha_one",
                    "max_candidates": max_candidates,
                    "output": {
                        "path": (tmp_path / f"candidate_selection_{max_candidates}").as_posix(),
                        "review_output_path": (tmp_path / f"candidate_review_{max_candidates}").as_posix(),
                    },
                },
                "portfolio": {
                    "enabled": True,
                    "from_candidate_selection": True,
                },
                "review": {
                    "output": {
                        "path": (tmp_path / f"reviews_{max_candidates}").as_posix(),
                        "emit_plots": False,
                    }
                },
                "outputs": {
                    "alpha_artifacts_root": (tmp_path / "alpha_artifacts").as_posix(),
                    "campaign_artifacts_root": (tmp_path / f"campaign_artifacts_{max_candidates}").as_posix(),
                    "comparison_output_path": (tmp_path / "comparisons").as_posix(),
                    "portfolio_artifacts_root": (tmp_path / "portfolio_artifacts").as_posix(),
                },
            }
        )

    monkeypatch.setattr(
        "src.cli.run_research_campaign.run_alpha_cli.run_cli",
        lambda argv: SimpleNamespace(alpha_name="alpha_one", run_id="alpha_run", artifact_dir=tmp_path / "alpha"),
    )
    monkeypatch.setattr(
        "src.cli.run_research_campaign.run_strategy_cli.run_cli",
        lambda argv: SimpleNamespace(
            strategy_name="strategy_one",
            run_id="strategy_run",
            experiment_dir=tmp_path / "strategy",
            metrics={"cumulative_return": 0.1, "sharpe_ratio": 1.0, "max_drawdown": -0.05},
        ),
    )
    monkeypatch.setattr(
        "src.cli.run_research_campaign.compare_alpha_cli.run_cli",
        lambda argv: SimpleNamespace(comparison_id="alpha_cmp"),
    )
    monkeypatch.setattr(
        "src.cli.run_research_campaign.compare_strategies_cli.run_cli",
        lambda argv: SimpleNamespace(comparison_id="strategy_cmp"),
    )
    monkeypatch.setattr(
        "src.cli.run_research_campaign.run_candidate_selection_cli.run_cli",
        lambda argv: SimpleNamespace(
            run_id="candidate_run",
            artifact_dir=tmp_path / "candidate_selection" / "candidate_run",
        ),
    )
    monkeypatch.setattr(
        "src.cli.run_research_campaign.run_portfolio_cli.run_cli",
        lambda argv: SimpleNamespace(
            run_id="portfolio_run",
            experiment_dir=tmp_path / "portfolio_artifacts" / "portfolio_run",
            portfolio_name="candidate_portfolio",
        ),
    )
    monkeypatch.setattr(
        "src.cli.run_research_campaign.review_candidate_selection_cli.run_cli",
        lambda argv: SimpleNamespace(
            review_dir=tmp_path / "candidate_review" / "candidate_run",
            candidate_selection_run_id="candidate_run",
            portfolio_run_id="portfolio_run",
        ),
    )
    monkeypatch.setattr(
        "src.cli.run_research_campaign.compare_research_cli.run_cli",
        lambda argv: SimpleNamespace(review_id="review_run"),
    )

    first_checkpoint = json.loads(
        run_research_campaign(build_config(max_candidates=2)).campaign_checkpoint_path.read_text(encoding="utf-8")
    )
    second_checkpoint = json.loads(
        run_research_campaign(build_config(max_candidates=3)).campaign_checkpoint_path.read_text(encoding="utf-8")
    )

    assert (
        first_checkpoint["stage_input_fingerprints"]["research"]
        == second_checkpoint["stage_input_fingerprints"]["research"]
    )
    assert (
        first_checkpoint["stage_input_fingerprints"]["comparison"]
        == second_checkpoint["stage_input_fingerprints"]["comparison"]
    )
    assert (
        first_checkpoint["stage_input_fingerprints"]["candidate_selection"]
        != second_checkpoint["stage_input_fingerprints"]["candidate_selection"]
    )
    assert (
        first_checkpoint["stage_input_fingerprints"]["portfolio"]
        != second_checkpoint["stage_input_fingerprints"]["portfolio"]
    )
    assert (
        first_checkpoint["stages"][3]["fingerprint_inputs"]["candidate_selection"]["max_candidates"]
        == 2
    )
    assert (
        second_checkpoint["stages"][3]["fingerprint_inputs"]["candidate_selection"]["max_candidates"]
        == 3
    )
