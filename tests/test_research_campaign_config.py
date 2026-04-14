from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.research_campaign import (
    ResearchCampaignConfigError,
    build_research_campaign_scenario_catalog,
    expand_research_campaign_scenarios,
    load_research_campaign_config,
    resolve_research_campaign_config,
)


def test_resolve_research_campaign_config_loads_defaults() -> None:
    config = resolve_research_campaign_config()

    assert config.dataset_selection.dataset is None
    assert config.comparison.alpha_view == "combined"
    assert config.candidate_selection.metric == "ic_ir"
    assert config.candidate_selection.artifacts_root == "artifacts/alpha"
    assert config.candidate_selection.output.path == "artifacts/candidate_selection"
    assert config.portfolio.enabled is False
    assert config.review.ranking.strategy_primary_metric == "sharpe_ratio"
    assert config.milestone_reporting.enabled is True
    assert config.milestone_reporting.output.include_markdown_report is True
    assert config.milestone_reporting.output.decision_log_render_formats == ("markdown", "text")
    assert config.outputs.alpha_artifacts_root == "artifacts/alpha"
    assert config.outputs.campaign_artifacts_root == "artifacts/research_campaigns"
    assert config.reuse_policy.enable_checkpoint_reuse is True
    assert config.scenarios.enabled is False
    assert config.scenarios.matrix == ()
    assert config.scenarios.include == ()
    assert config.reuse_policy.reuse_prior_stages == (
        "preflight",
        "research",
        "comparison",
        "candidate_selection",
        "portfolio",
        "candidate_review",
        "review",
    )


def test_load_research_campaign_config_supports_nested_payload_and_inheritance(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "research_campaign.yml"
    config_path.write_text(
        """
research_campaign:
  dataset_selection:
    dataset: features_daily
    timeframe: 1D
    evaluation_horizon: 5
    mapping_name: top_bottom_quantile_q20
  targets:
    alpha_names: [ml_alpha_q1]
    strategy_names: [momentum_v1]
    portfolio_names: [candidate_portfolio]
  outputs:
    alpha_artifacts_root: artifacts/alpha/q1
    review_output_path: artifacts/reviews/q1
  reuse_policy:
    force_rerun_stages: [research]
  candidate_selection:
    enabled: true
  portfolio:
    enabled: true
  review:
    filters:
      run_types: [alpha_evaluation, portfolio]
  milestone_reporting:
    decision_categories: [review_promotion]
    output:
      include_markdown_report: false
      decision_log_render_formats: [text]
    summary:
      include_stage_counts: false
""".strip(),
        encoding="utf-8",
    )

    config = load_research_campaign_config(config_path)

    assert config.candidate_selection.dataset == "features_daily"
    assert config.candidate_selection.timeframe == "1D"
    assert config.candidate_selection.evaluation_horizon == 5
    assert config.candidate_selection.mapping_name == "top_bottom_quantile_q20"
    assert config.candidate_selection.alpha_name == "ml_alpha_q1"
    assert config.candidate_selection.artifacts_root == "artifacts/alpha/q1"
    assert config.portfolio.portfolio_name == "candidate_portfolio"
    assert config.portfolio.timeframe == "1D"
    assert config.reuse_policy.force_rerun_stages == ("research",)
    assert config.review.filters.dataset == "features_daily"
    assert config.review.filters.portfolio_name == "candidate_portfolio"
    assert config.review.output.path == "artifacts/reviews/q1"
    assert config.review.filters.run_types == ["alpha_evaluation", "portfolio"]
    assert config.milestone_reporting.decision_categories == ("review_promotion",)
    assert config.milestone_reporting.output.include_markdown_report is False
    assert config.milestone_reporting.output.decision_log_render_formats == ("text",)
    assert config.milestone_reporting.summary.include_stage_counts is False
    assert config.scenarios.enabled is False


def test_resolve_research_campaign_config_applies_precedence_and_normalization(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "research_campaign.yml"
    config_path.write_text(
        """
research_campaign:
  targets:
    alpha_names: [" alpha_one ", "alpha_one", "alpha_two"]
    portfolio_names: [baseline_portfolio]
  comparison:
    enabled: true
    top_k: 10
    alpha_view: FORECAST
  candidate_selection:
    enabled: true
    allocation:
      method: risk_parity
      max_weight: 0.25
  outputs:
    candidate_selection_output_path: artifacts/candidate_selection/custom
    campaign_artifacts_root: artifacts/research_campaigns/custom
""".strip(),
        encoding="utf-8",
    )

    loaded = load_research_campaign_config(config_path).to_dict()
    config = resolve_research_campaign_config(
        loaded,
        cli_overrides={
            "reuse_policy": {
                "reuse_prior_stages": ["preflight", "research"],
                "invalidate_downstream_after_stages": ["research"],
            },
            "comparison": {"top_k": 3},
            "candidate_selection": {
                "allocation": {"rounding_decimals": 6},
                "output": {"registry_path": "artifacts/candidate_selection/custom/registry.jsonl"},
            },
            "portfolio": {"enabled": True},
            "milestone_reporting": {
                "decision_categories": ["campaign_execution"],
                "sections": {"open_questions": False},
            },
        },
    )

    assert config.targets.alpha_names == ("alpha_one", "alpha_two")
    assert config.comparison.enabled is True
    assert config.comparison.alpha_view == "forecast"
    assert config.comparison.top_k == 3
    assert config.candidate_selection.allocation.allocation_method == "risk_parity"
    assert config.candidate_selection.allocation.max_weight_per_candidate == pytest.approx(0.25)
    assert config.candidate_selection.allocation.allocation_rounding_decimals == 6
    assert config.candidate_selection.output.path == "artifacts/candidate_selection/custom"
    assert (
        config.candidate_selection.output.registry_path
        == "artifacts/candidate_selection/custom/registry.jsonl"
    )
    assert config.reuse_policy.reuse_prior_stages == ("preflight", "research")
    assert config.reuse_policy.invalidate_downstream_after_stages == ("research",)
    assert config.outputs.campaign_artifacts_root == "artifacts/research_campaigns/custom"
    assert config.portfolio.enabled is True
    assert config.milestone_reporting.decision_categories == ("campaign_execution",)
    assert config.milestone_reporting.sections.open_questions is False


def test_expand_research_campaign_scenarios_supports_matrix_and_explicit_includes() -> None:
    config = resolve_research_campaign_config(
        {
            "dataset_selection": {
                "dataset": "features_daily",
                "timeframe": "1D",
                "evaluation_horizon": 5,
            },
            "targets": {
                "alpha_names": ["ml_alpha_q1"],
                "portfolio_names": ["candidate_portfolio"],
            },
            "scenarios": {
                "enabled": True,
                "matrix": [
                    {
                        "name": "timeframe",
                        "path": "dataset_selection.timeframe",
                        "values": ["1D", "4H"],
                    },
                    {
                        "name": "top_k",
                        "path": "comparison.top_k",
                        "values": [5, 10],
                    },
                ],
                "include": [
                    {
                        "scenario_id": "sleeve_review",
                        "description": "Use sleeve ranking for the review-heavy pass.",
                        "overrides": {
                            "comparison": {"alpha_view": "sleeve"},
                            "review": {"filters": {"run_types": ["alpha_evaluation"]}},
                        },
                    }
                ],
            },
        }
    )

    scenarios = expand_research_campaign_scenarios(config)

    assert [scenario.scenario_id for scenario in scenarios] == [
        "scenario_0000_timeframe_1d__top_k_5",
        "scenario_0001_timeframe_1d__top_k_10",
        "scenario_0002_timeframe_4h__top_k_5",
        "scenario_0003_timeframe_4h__top_k_10",
        "sleeve_review",
    ]
    assert scenarios[0].source == "matrix"
    assert scenarios[0].sweep_values == {"timeframe": "1D", "top_k": 5}
    assert scenarios[0].config.dataset_selection.timeframe == "1D"
    assert scenarios[3].config.dataset_selection.timeframe == "4H"
    assert scenarios[3].config.comparison.top_k == 10
    assert scenarios[4].source == "include"
    assert scenarios[4].description == "Use sleeve ranking for the review-heavy pass."
    assert scenarios[4].config.comparison.alpha_view == "sleeve"
    assert scenarios[4].config.review.filters.run_types == ["alpha_evaluation"]
    assert all(len(scenario.fingerprint) == 16 for scenario in scenarios)
    assert scenarios[0].to_dict()["effective_config"]["dataset_selection"]["timeframe"] == "1D"


def test_build_research_campaign_scenario_catalog_emits_stable_effective_snapshots() -> None:
    config = resolve_research_campaign_config(
        {
            "dataset_selection": {
                "dataset": "features_daily",
                "timeframe": "1D",
                "evaluation_horizon": 5,
            },
            "targets": {
                "alpha_names": ["ml_alpha_q1"],
            },
            "scenarios": {
                "enabled": True,
                "matrix": [
                    {
                        "name": "timeframe",
                        "path": "dataset_selection.timeframe",
                        "values": ["1D", "4H"],
                    }
                ],
                "include": [
                    {
                        "scenario_id": "review_only",
                        "overrides": {
                            "comparison": {"alpha_view": "sleeve"},
                        },
                    }
                ],
            },
        }
    )

    catalog = build_research_campaign_scenario_catalog(config)

    assert catalog["scenario_count"] == 3
    assert catalog["scenarios_enabled"] is True
    assert len(catalog["base_fingerprint"]) == 16
    assert [scenario["scenario_id"] for scenario in catalog["scenarios"]] == [
        "scenario_0000_timeframe_1d",
        "scenario_0001_timeframe_4h",
        "review_only",
    ]
    assert catalog["scenarios"][0]["effective_config"]["dataset_selection"]["timeframe"] == "1D"
    assert catalog["scenarios"][1]["effective_config"]["dataset_selection"]["timeframe"] == "4H"
    assert catalog["scenarios"][2]["effective_config"]["comparison"]["alpha_view"] == "sleeve"


def test_expand_research_campaign_scenarios_returns_implicit_default_when_disabled() -> None:
    config = resolve_research_campaign_config(
        {
            "dataset_selection": {"dataset": "features_daily"},
            "comparison": {"enabled": True},
        }
    )

    scenarios = expand_research_campaign_scenarios(config)

    assert len(scenarios) == 1
    assert scenarios[0].scenario_id == "default"
    assert scenarios[0].source == "default"
    assert scenarios[0].config.scenarios.enabled is False
    assert scenarios[0].config.comparison.enabled is True


def test_resolve_research_campaign_config_validates_fields_explicitly() -> None:
    with pytest.raises(ResearchCampaignConfigError, match="unsupported keys"):
        resolve_research_campaign_config({"research_campaign": {"mystery": {}}})

    with pytest.raises(ResearchCampaignConfigError, match="comparison.alpha_view"):
        resolve_research_campaign_config({"comparison": {"alpha_view": "mystery"}})

    with pytest.raises(
        ResearchCampaignConfigError, match="candidate_selection.execution.strict_mode"
    ):
        resolve_research_campaign_config(
            {"candidate_selection": {"execution": {"strict_mode": "yes"}}}
        )

    with pytest.raises(
        ResearchCampaignConfigError, match="candidate_selection.redundancy.max_pairwise_correlation"
    ):
        resolve_research_campaign_config(
            {"candidate_selection": {"redundancy": {"max_correlation": 1.2}}}
        )

    with pytest.raises(ResearchCampaignConfigError, match="targets.alpha_names"):
        resolve_research_campaign_config({"targets": {"alpha_names": ["ok", ""]}})

    with pytest.raises(ResearchCampaignConfigError, match="reuse_policy.reuse_prior_stages"):
        resolve_research_campaign_config({"reuse_policy": {"reuse_prior_stages": ["review", "unknown"]}})

    with pytest.raises(
        ResearchCampaignConfigError, match="milestone_reporting.output.decision_log_render_formats"
    ):
        resolve_research_campaign_config(
            {"milestone_reporting": {"output": {"decision_log_render_formats": ["html"]}}}
        )

    with pytest.raises(ResearchCampaignConfigError, match="scenarios.matrix"):
        resolve_research_campaign_config(
            {
                "scenarios": {
                    "enabled": True,
                    "matrix": [
                        {"name": "timeframe", "path": "dataset_selection.timeframe", "values": ["1D"]},
                        {"name": "timeframe", "path": "comparison.top_k", "values": [5]},
                    ]
                }
            }
        )

    with pytest.raises(ResearchCampaignConfigError, match="duplicate scenario_id"):
        resolve_research_campaign_config(
            {
                "scenarios": {
                    "enabled": True,
                    "include": [
                        {"scenario_id": "review-pass", "overrides": {"comparison": {"enabled": True}}},
                        {"scenario_id": "review pass", "overrides": {"portfolio": {"enabled": True}}},
                    ],
                }
            }
        )

    with pytest.raises(ResearchCampaignConfigError, match="must define at least one matrix axis"):
        resolve_research_campaign_config({"scenarios": {"enabled": True}})

    with pytest.raises(ResearchCampaignConfigError, match="cannot target the 'scenarios' section"):
        resolve_research_campaign_config(
            {
                "scenarios": {
                    "enabled": True,
                    "matrix": [
                        {"name": "bad", "path": "scenarios.enabled", "values": [True]},
                    ],
                }
            }
        )

    with pytest.raises(ResearchCampaignConfigError, match="cannot override the 'scenarios' section"):
        resolve_research_campaign_config(
            {
                "scenarios": {
                    "enabled": True,
                    "include": [
                        {
                            "scenario_id": "bad-include",
                            "overrides": {"scenarios": {"enabled": False}},
                        }
                    ],
                }
            }
        )

    with pytest.raises(
        ResearchCampaignConfigError,
        match=r"scenarios\.include\[0\]' contains unsupported keys",
    ):
        resolve_research_campaign_config(
            {
                "scenarios": {
                    "enabled": True,
                    "matrix": [
                        {"name": "timeframe", "path": "dataset_selection.timeframe", "values": ["1D"]},
                    ],
                    "include": [
                        {
                            "scenario_id": "bad-include",
                            "override": {"comparison": {"enabled": True}},
                            "overrides": {"comparison": {"enabled": True}},
                        }
                    ],
                }
            }
        )

    with pytest.raises(
        ResearchCampaignConfigError,
        match=r"scenarios\.include\[0\]\.overrides' must be a mapping",
    ):
        resolve_research_campaign_config(
            {
                "scenarios": {
                    "enabled": True,
                    "matrix": [
                        {"name": "timeframe", "path": "dataset_selection.timeframe", "values": ["1D"]},
                    ],
                    "include": [
                        {
                            "scenario_id": "bad-overrides-type",
                            "overrides": ["comparison.enabled=true"],
                        }
                    ],
                }
            }
        )

    with pytest.raises(ResearchCampaignConfigError, match="duplicate scenario_id"):
        resolve_research_campaign_config(
            {
                "scenarios": {
                    "enabled": True,
                    "matrix": [{"name": "timeframe", "path": "dataset_selection.timeframe", "values": ["1D"]}],
                    "include": [
                        {
                            "scenario_id": "scenario_0000_timeframe_1d",
                            "overrides": {"comparison": {"enabled": True}},
                        }
                    ],
                }
            }
        )


def test_resolve_research_campaign_config_produces_stable_equivalent_payloads(
    tmp_path: Path,
) -> None:
    nested_config_path = tmp_path / "research_campaign.yml"
    nested_config_path.write_text(
        """
research_campaign:
  dataset_selection:
    dataset: features_daily
    timeframe: 1D
    evaluation_horizon: 5
    mapping_name: top_bottom_quantile_q20
  targets:
    alpha_names: [" alpha_one ", "alpha_one"]
    strategy_names: [" momentum_v1 "]
    portfolio_names: [candidate_portfolio]
  comparison:
    enabled: true
    alpha_view: COMBINED
  candidate_selection:
    enabled: true
  portfolio:
    enabled: true
  outputs:
    candidate_selection_output_path: artifacts/candidate_selection/custom
    campaign_artifacts_root: artifacts/research_campaigns/custom
    review_output_path: artifacts/reviews/custom
""".strip(),
        encoding="utf-8",
    )

    direct = resolve_research_campaign_config(
        {
            "dataset_selection": {
                "dataset": "features_daily",
                "timeframe": "1D",
                "evaluation_horizon": 5,
                "mapping_name": "top_bottom_quantile_q20",
            },
            "targets": {
                "alpha_names": ["alpha_one"],
                "strategy_names": ["momentum_v1"],
                "portfolio_names": ["candidate_portfolio"],
            },
            "comparison": {
                "enabled": True,
                "alpha_view": "combined",
            },
            "candidate_selection": {
                "enabled": True,
            },
            "portfolio": {
                "enabled": True,
            },
            "outputs": {
                "candidate_selection_output_path": "artifacts/candidate_selection/custom",
                "campaign_artifacts_root": "artifacts/research_campaigns/custom",
                "review_output_path": "artifacts/reviews/custom",
            },
        }
    )
    nested = resolve_research_campaign_config(load_research_campaign_config(nested_config_path).to_dict())

    assert direct.to_dict() == nested.to_dict()
    assert nested.targets.alpha_names == ("alpha_one",)
    assert nested.targets.strategy_names == ("momentum_v1",)
    assert nested.comparison.alpha_view == "combined"
    assert nested.candidate_selection.output.path == "artifacts/candidate_selection/custom"
    assert nested.review.output.path == "artifacts/reviews/custom"
