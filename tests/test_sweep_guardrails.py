"""Tests for sweep-campaign guardrails and scenario-expansion safety controls.

Covers:
  - max_scenarios config field enforcement
  - max_values_per_axis config field enforcement
  - Hard-cap (_SCENARIOS_HARD_MAX) enforcement via the config resolver
  - compute_scenario_expansion_size correctness
  - ScenarioExpansionPreflightResult written to disk
  - Friendly error messages
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config.research_campaign import (
    ResearchCampaignConfigError,
    build_research_campaign_scenario_catalog,
    compute_scenario_expansion_size,
    expand_research_campaign_scenarios,
    resolve_research_campaign_config,
)
from src.cli.run_research_campaign import (
    ORCHESTRATION_EXPANSION_PREFLIGHT_FILENAME,
    ScenarioExpansionPreflightResult,
    _run_orchestration_expansion_preflight,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _matrix_config(
    axes: list[dict],
    *,
    max_scenarios: int | None = None,
    max_values_per_axis: int | None = None,
    enabled: bool = True,
) -> dict:
    scenarios: dict = {"enabled": enabled, "matrix": axes}
    if max_scenarios is not None:
        scenarios["max_scenarios"] = max_scenarios
    if max_values_per_axis is not None:
        scenarios["max_values_per_axis"] = max_values_per_axis
    return {
        "targets": {"alpha_names": ["alpha_one"]},
        "scenarios": scenarios,
    }


# ---------------------------------------------------------------------------
# compute_scenario_expansion_size
# ---------------------------------------------------------------------------

class TestComputeScenarioExpansionSize:
    def test_disabled_scenarios_returns_zero_counts(self) -> None:
        config = resolve_research_campaign_config(
            {"targets": {"alpha_names": ["alpha_one"]}}
        )
        size = compute_scenario_expansion_size(config)

        assert size["matrix_axis_count"] == 0
        assert size["matrix_combination_count"] == 0
        assert size["include_count"] == 0
        assert size["total_scenario_count"] == 0
        assert size["exceeds_limit"] is False

    def test_single_axis_counts(self) -> None:
        config = resolve_research_campaign_config(
            _matrix_config(
                [{"name": "timeframe", "path": "dataset_selection.timeframe", "values": ["1D", "4H", "1H"]}]
            )
        )
        size = compute_scenario_expansion_size(config)

        assert size["matrix_axis_count"] == 1
        assert size["per_axis_value_counts"] == [3]
        assert size["matrix_combination_count"] == 3
        assert size["include_count"] == 0
        assert size["total_scenario_count"] == 3
        assert size["exceeds_limit"] is False

    def test_two_axis_cartesian_product(self) -> None:
        config = resolve_research_campaign_config(
            _matrix_config(
                [
                    {"name": "timeframe", "path": "dataset_selection.timeframe", "values": ["1D", "4H"]},
                    {"name": "top_k", "path": "comparison.top_k", "values": [5, 10, 20]},
                ]
            )
        )
        size = compute_scenario_expansion_size(config)

        assert size["matrix_axis_count"] == 2
        assert size["per_axis_value_counts"] == [2, 3]
        assert size["matrix_combination_count"] == 6
        assert size["total_scenario_count"] == 6

    def test_matrix_plus_includes(self) -> None:
        config = resolve_research_campaign_config(
            {
                "targets": {"alpha_names": ["alpha_one"]},
                "scenarios": {
                    "enabled": True,
                    "matrix": [
                        {"name": "timeframe", "path": "dataset_selection.timeframe", "values": ["1D", "4H"]},
                    ],
                    "include": [
                        {"scenario_id": "extra_pass", "overrides": {"comparison": {"enabled": True}}},
                        {"scenario_id": "sleeve_pass", "overrides": {"comparison": {"alpha_view": "sleeve"}}},
                    ],
                },
            }
        )
        size = compute_scenario_expansion_size(config)

        assert size["matrix_combination_count"] == 2
        assert size["include_count"] == 2
        assert size["total_scenario_count"] == 4

    def test_hard_max_and_configured_max_reported(self) -> None:
        config = resolve_research_campaign_config(
            _matrix_config(
                [{"name": "timeframe", "path": "dataset_selection.timeframe", "values": ["1D"]}],
                max_scenarios=50,
            )
        )
        size = compute_scenario_expansion_size(config)

        assert size["configured_max_scenarios"] == 50
        assert size["effective_max_scenarios"] == 50
        assert size["hard_max_scenarios"] == 1000

    def test_effective_max_uses_minimum_of_hard_and_configured(self) -> None:
        # configured_max above hard cap → effective_max = hard_max
        config = resolve_research_campaign_config(
            _matrix_config(
                [{"name": "timeframe", "path": "dataset_selection.timeframe", "values": ["1D"]}],
                max_scenarios=5000,
            )
        )
        size = compute_scenario_expansion_size(config)

        assert size["configured_max_scenarios"] == 5000
        assert size["effective_max_scenarios"] == 1000  # hard max wins

    def test_max_values_per_axis_reported(self) -> None:
        config = resolve_research_campaign_config(
            _matrix_config(
                [{"name": "timeframe", "path": "dataset_selection.timeframe", "values": ["1D"]}],
                max_values_per_axis=10,
            )
        )
        size = compute_scenario_expansion_size(config)

        assert size["max_values_per_axis"] == 10


# ---------------------------------------------------------------------------
# max_scenarios enforcement
# ---------------------------------------------------------------------------

class TestMaxScenariosEnforcement:
    def test_within_limit_passes(self) -> None:
        config = resolve_research_campaign_config(
            _matrix_config(
                [{"name": "timeframe", "path": "dataset_selection.timeframe", "values": ["1D", "4H"]}],
                max_scenarios=5,
            )
        )
        scenarios = expand_research_campaign_scenarios(config)
        assert len(scenarios) == 2

    def test_at_exact_limit_passes(self) -> None:
        config = resolve_research_campaign_config(
            _matrix_config(
                [{"name": "timeframe", "path": "dataset_selection.timeframe", "values": ["1D", "4H"]}],
                max_scenarios=2,
            )
        )
        scenarios = expand_research_campaign_scenarios(config)
        assert len(scenarios) == 2

    def test_exceeds_configured_max_raises(self) -> None:
        with pytest.raises(ResearchCampaignConfigError, match="exceeds the effective limit of 2"):
            resolve_research_campaign_config(
                _matrix_config(
                    [
                        {"name": "timeframe", "path": "dataset_selection.timeframe", "values": ["1D", "4H"]},
                        {"name": "top_k", "path": "comparison.top_k", "values": [5, 10]},
                    ],
                    max_scenarios=2,
                )
            )

    def test_exceeds_hard_max_raises(self) -> None:
        # 1001 individual include entries would exceed the hard cap.
        # Use a multi-axis matrix to approach the limit.
        # 32 values × 32 values = 1024 > 1000 (hard cap).
        values_32 = list(range(32))
        with pytest.raises(ResearchCampaignConfigError, match="exceeds the effective limit of 1000"):
            resolve_research_campaign_config(
                {
                    "targets": {"alpha_names": ["alpha_one"]},
                    "scenarios": {
                        "enabled": True,
                        "matrix": [
                            {"name": "a", "path": "comparison.top_k", "values": values_32},
                            {"name": "b", "path": "dataset_selection.evaluation_horizon", "values": values_32},
                        ],
                    },
                }
            )

    def test_error_message_includes_total_and_limit(self) -> None:
        with pytest.raises(ResearchCampaignConfigError) as exc_info:
            resolve_research_campaign_config(
                _matrix_config(
                    [
                        {"name": "timeframe", "path": "dataset_selection.timeframe", "values": ["1D", "4H", "1H"]},
                        {"name": "top_k", "path": "comparison.top_k", "values": [5, 10]},
                    ],
                    max_scenarios=4,
                )
            )
        message = str(exc_info.value)
        assert "6" in message      # total_scenario_count
        assert "4" in message      # configured limit

    def test_safety_net_in_expand_raises_for_programmatic_config(self) -> None:
        """expand_research_campaign_scenarios raises even without going through resolver."""
        from src.config.research_campaign import (
            CampaignScenariosConfig,
            CampaignScenarioMatrixAxisConfig,
        )
        # Build a config that bypasses the resolver's validation
        base = resolve_research_campaign_config(
            {"targets": {"alpha_names": ["alpha_one"]}}
        )
        # Patch scenarios directly to bypass resolver
        oversized_matrix = tuple(
            CampaignScenarioMatrixAxisConfig(
                name="top_k",
                path="comparison.top_k",
                values=tuple(range(1100)),
            )
            for _ in range(1)
        )
        from dataclasses import replace
        patched = replace(
            base,
            scenarios=CampaignScenariosConfig(
                enabled=True,
                matrix=oversized_matrix,
                include=(),
                max_scenarios=None,
                max_values_per_axis=None,
            ),
        )
        with pytest.raises(ResearchCampaignConfigError, match="exceeds the effective limit"):
            expand_research_campaign_scenarios(patched)


# ---------------------------------------------------------------------------
# max_values_per_axis enforcement
# ---------------------------------------------------------------------------

class TestMaxValuesPerAxisEnforcement:
    def test_within_per_axis_limit_passes(self) -> None:
        config = resolve_research_campaign_config(
            _matrix_config(
                [{"name": "timeframe", "path": "dataset_selection.timeframe", "values": ["1D", "4H"]}],
                max_values_per_axis=5,
            )
        )
        assert config.scenarios.max_values_per_axis == 5

    def test_exceeds_per_axis_limit_raises(self) -> None:
        with pytest.raises(
            ResearchCampaignConfigError,
            match="max_values_per_axis",
        ):
            resolve_research_campaign_config(
                _matrix_config(
                    [
                        {"name": "timeframe", "path": "dataset_selection.timeframe", "values": ["1D", "4H", "1H"]}
                    ],
                    max_values_per_axis=2,
                )
            )

    def test_error_message_names_the_axis(self) -> None:
        with pytest.raises(ResearchCampaignConfigError) as exc_info:
            resolve_research_campaign_config(
                _matrix_config(
                    [
                        {"name": "my_timeframe_axis", "path": "dataset_selection.timeframe", "values": ["1D", "4H"]}
                    ],
                    max_values_per_axis=1,
                )
            )
        assert "my_timeframe_axis" in str(exc_info.value)
        assert "2" in str(exc_info.value)   # axis has 2 values
        assert "1" in str(exc_info.value)   # limit is 1

    def test_per_axis_limit_not_enforced_on_include(self) -> None:
        # max_values_per_axis applies to matrix axes only, not explicit includes
        config = resolve_research_campaign_config(
            {
                "targets": {"alpha_names": ["alpha_one"]},
                "scenarios": {
                    "enabled": True,
                    "matrix": [
                        {"name": "flag", "path": "comparison.enabled", "values": [True]}
                    ],
                    "include": [
                        {"scenario_id": f"explicit_{i}", "overrides": {"comparison": {"enabled": True}}}
                        for i in range(10)
                    ],
                    "max_values_per_axis": 3,
                },
            }
        )
        # 10 include entries are fine — limit only applies per axis
        assert len(config.scenarios.include) == 10


# ---------------------------------------------------------------------------
# CampaignScenariosConfig serialization
# ---------------------------------------------------------------------------

class TestScenariosConfigSerialization:
    def test_max_fields_round_trip_through_to_dict(self) -> None:
        config = resolve_research_campaign_config(
            _matrix_config(
                [{"name": "timeframe", "path": "dataset_selection.timeframe", "values": ["1D"]}],
                max_scenarios=50,
                max_values_per_axis=5,
            )
        )
        payload = config.scenarios.to_dict()
        assert payload["max_scenarios"] == 50
        assert payload["max_values_per_axis"] == 5

    def test_none_limits_serialized_as_none(self) -> None:
        config = resolve_research_campaign_config(
            {"targets": {"alpha_names": ["alpha_one"]}}
        )
        payload = config.scenarios.to_dict()
        assert payload["max_scenarios"] is None
        assert payload["max_values_per_axis"] is None

    def test_scenario_catalog_includes_expansion_section(self) -> None:
        config = resolve_research_campaign_config(
            _matrix_config(
                [{"name": "timeframe", "path": "dataset_selection.timeframe", "values": ["1D", "4H"]}],
                max_scenarios=10,
            )
        )
        catalog = build_research_campaign_scenario_catalog(config)
        assert "expansion" in catalog
        expansion = catalog["expansion"]
        assert expansion["total_scenario_count"] == 2
        assert expansion["matrix_combination_count"] == 2
        assert expansion["effective_max_scenarios"] == 10

    def test_scenario_catalog_backward_compatible_keys_present(self) -> None:
        config = resolve_research_campaign_config(
            _matrix_config(
                [{"name": "timeframe", "path": "dataset_selection.timeframe", "values": ["1D"]}]
            )
        )
        catalog = build_research_campaign_scenario_catalog(config)
        assert "scenario_count" in catalog
        assert "scenarios_enabled" in catalog
        assert "base_fingerprint" in catalog
        assert "scenarios" in catalog


# ---------------------------------------------------------------------------
# ScenarioExpansionPreflightResult and helper
# ---------------------------------------------------------------------------

class TestOrchestrationExpansionPreflight:
    def test_preflight_writes_json_file(self, tmp_path: Path) -> None:
        config = resolve_research_campaign_config(
            _matrix_config(
                [{"name": "timeframe", "path": "dataset_selection.timeframe", "values": ["1D", "4H"]}]
            )
        )
        result = _run_orchestration_expansion_preflight(config, orchestration_artifact_dir=tmp_path)

        preflight_path = tmp_path / ORCHESTRATION_EXPANSION_PREFLIGHT_FILENAME
        assert preflight_path.exists()
        payload = json.loads(preflight_path.read_text(encoding="utf-8"))
        assert payload["status"] == "passed"
        assert payload["expansion"]["total_scenario_count"] == 2

    def test_preflight_returns_dataclass(self, tmp_path: Path) -> None:
        config = resolve_research_campaign_config(
            _matrix_config(
                [{"name": "timeframe", "path": "dataset_selection.timeframe", "values": ["1D"]}]
            )
        )
        result = _run_orchestration_expansion_preflight(config, orchestration_artifact_dir=tmp_path)

        assert isinstance(result, ScenarioExpansionPreflightResult)
        assert result.status == "passed"
        assert result.summary_path == tmp_path / ORCHESTRATION_EXPANSION_PREFLIGHT_FILENAME

    def test_preflight_summary_includes_limits(self, tmp_path: Path) -> None:
        config = resolve_research_campaign_config(
            _matrix_config(
                [{"name": "timeframe", "path": "dataset_selection.timeframe", "values": ["1D"]}],
                max_scenarios=25,
                max_values_per_axis=5,
            )
        )
        result = _run_orchestration_expansion_preflight(config, orchestration_artifact_dir=tmp_path)

        limits = result.summary["limits"]
        assert limits["configured_max_scenarios"] == 25
        assert limits["effective_max_scenarios"] == 25
        assert limits["hard_max_scenarios"] == 1000
        assert limits["max_values_per_axis"] == 5
        assert limits["exceeds_limit"] is False

    def test_preflight_summary_includes_per_axis_details(self, tmp_path: Path) -> None:
        config = resolve_research_campaign_config(
            _matrix_config(
                [
                    {"name": "timeframe", "path": "dataset_selection.timeframe", "values": ["1D", "4H"]},
                    {"name": "top_k", "path": "comparison.top_k", "values": [5, 10, 20]},
                ]
            )
        )
        result = _run_orchestration_expansion_preflight(config, orchestration_artifact_dir=tmp_path)

        per_axis = result.summary["expansion"]["per_axis_details"]
        assert len(per_axis) == 2
        assert per_axis[0]["name"] == "timeframe"
        assert per_axis[0]["value_count"] == 2
        assert per_axis[1]["name"] == "top_k"
        assert per_axis[1]["value_count"] == 3

    def test_preflight_summary_is_stable_json(self, tmp_path: Path) -> None:
        config = resolve_research_campaign_config(
            _matrix_config(
                [{"name": "timeframe", "path": "dataset_selection.timeframe", "values": ["1D"]}]
            )
        )
        _run_orchestration_expansion_preflight(config, orchestration_artifact_dir=tmp_path)

        preflight_path = tmp_path / ORCHESTRATION_EXPANSION_PREFLIGHT_FILENAME
        raw = preflight_path.read_text(encoding="utf-8")
        # Must parse without error
        payload = json.loads(raw)
        # Write again and compare (idempotent)
        preflight_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        assert preflight_path.read_text(encoding="utf-8") == json.dumps(payload, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Config field validation
# ---------------------------------------------------------------------------

class TestScenariosConfigFieldValidation:
    def test_max_scenarios_must_be_positive_int(self) -> None:
        with pytest.raises(ResearchCampaignConfigError, match="scenarios.max_scenarios"):
            resolve_research_campaign_config(
                _matrix_config(
                    [{"name": "timeframe", "path": "dataset_selection.timeframe", "values": ["1D"]}],
                    max_scenarios=0,  # 0 is not a positive int
                )
            )

    def test_max_values_per_axis_must_be_positive_int(self) -> None:
        with pytest.raises(ResearchCampaignConfigError, match="scenarios.max_values_per_axis"):
            resolve_research_campaign_config(
                _matrix_config(
                    [{"name": "timeframe", "path": "dataset_selection.timeframe", "values": ["1D"]}],
                    max_values_per_axis=0,
                )
            )

    def test_unknown_scenarios_keys_rejected(self) -> None:
        with pytest.raises(ResearchCampaignConfigError, match="unsupported keys"):
            resolve_research_campaign_config(
                {
                    "targets": {"alpha_names": ["alpha_one"]},
                    "scenarios": {
                        "enabled": True,
                        "matrix": [
                            {"name": "timeframe", "path": "dataset_selection.timeframe", "values": ["1D"]}
                        ],
                        "max_explosions": 42,  # unknown key
                    },
                }
            )

    def test_none_limits_are_accepted_as_unconfigured(self) -> None:
        config = resolve_research_campaign_config(
            _matrix_config(
                [{"name": "timeframe", "path": "dataset_selection.timeframe", "values": ["1D"]}]
            )
        )
        assert config.scenarios.max_scenarios is None
        assert config.scenarios.max_values_per_axis is None
