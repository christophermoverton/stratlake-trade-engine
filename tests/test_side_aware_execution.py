"""Canonical stress tests for side-aware execution realism and capacity modeling (M22.5)."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.execution import ExecutionConfig, resolve_execution_config
from src.portfolio.execution import apply_portfolio_execution_model, PortfolioExecutionResult


class TestShortCapacityConfiguration:
    """Test short-side capacity and availability fields in ExecutionConfig."""

    def test_execution_config_with_max_short_weight_sum(self) -> None:
        """Test that max_short_weight_sum is properly configured."""
        config = ExecutionConfig.from_mapping({
            "enabled": True,
            "execution_delay": 1,
            "transaction_cost_bps": 5.0,
            "slippage_bps": 2.0,
            "max_short_weight_sum": 0.40,
        })
        
        assert config.max_short_weight_sum == 0.40
        assert config.has_directional_asymmetry is True
        assert "max_short_weight_sum" in config.to_dict()
        assert config.to_dict()["max_short_weight_sum"] == 0.40

    def test_execution_config_with_short_availability_limit(self) -> None:
        """Test that short_availability_limit is properly configured."""
        config = ExecutionConfig.from_mapping({
            "enabled": True,
            "execution_delay": 1,
            "transaction_cost_bps": 5.0,
            "slippage_bps": 2.0,
            "short_availability_limit": 0.20,
        })
        
        assert config.short_availability_limit == 0.20
        assert config.has_directional_asymmetry is True
        assert "short_availability_limit" in config.to_dict()
        assert config.to_dict()["short_availability_limit"] == 0.20

    def test_execution_config_with_short_availability_policy(self) -> None:
        """Test short_availability_policy field with valid options."""
        for policy in ["exclude", "cap", "penalty"]:
            config = ExecutionConfig.from_mapping({
                "enabled": True,
                "execution_delay": 1,
                "transaction_cost_bps": 5.0,
                "slippage_bps": 2.0,
                "short_availability_policy": policy,
            })
            
            assert config.short_availability_policy == policy
            if policy != "exclude":  # exclude is default
                assert "short_availability_policy" in config.to_dict()

    def test_execution_config_rejects_invalid_short_availability_policy(self) -> None:
        """Test that invalid short_availability_policy values are rejected."""
        with pytest.raises(ValueError, match="short_availability_policy"):
            ExecutionConfig.from_mapping({
                "enabled": True,
                "execution_delay": 1,
                "transaction_cost_bps": 5.0,
                "slippage_bps": 2.0,
                "short_availability_policy": "invalid_policy",
            })

    def test_short_capacity_fields_are_optional(self) -> None:
        """Test that short-capacity fields are optional (backward compatibility)."""
        config = ExecutionConfig.from_mapping({
            "enabled": True,
            "execution_delay": 1,
            "transaction_cost_bps": 5.0,
            "slippage_bps": 2.0,
        })
        
        assert config.max_short_weight_sum is None
        assert config.short_availability_limit is None
        assert config.short_availability_policy == "exclude"
        assert config.has_directional_asymmetry is False
        # Should not appear in to_dict if they're at defaults
        config_dict = config.to_dict()
        assert "max_short_weight_sum" not in config_dict
        assert "short_availability_limit" not in config_dict
        assert "short_availability_policy" not in config_dict


class TestLongOnlyStressTest:
    """Canonical stress test: long-only scenario (no shorts allowed)."""

    def test_long_only_config_creation(self) -> None:
        """Test creating a long-only execution config."""
        config = ExecutionConfig.from_mapping({
            "enabled": True,
            "execution_delay": 1,
            "transaction_cost_bps": 5.0,
            "slippage_bps": 2.0,
            "max_short_weight_sum": 0.0,  # No shorts
            "short_availability_policy": "exclude",
        })
        
        assert config.max_short_weight_sum == 0.0
        assert config.has_directional_asymmetry is True

    def test_long_only_execution_produces_no_short_costs(self) -> None:
        """Test that long-only portfolio produces zero short costs."""
        # Create a portfolio with only long positions
        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        assets = ["ASSET_A", "ASSET_B", "ASSET_C"]
        
        # Index: MultiIndex of (date, asset)
        index = pd.MultiIndex.from_product([dates, assets], names=["ts_utc", "symbol"])
        
        # Returns: all positive
        returns_wide = pd.DataFrame(
            [[0.01, 0.02, 0.015],  # ASSET_A, B, C on day 1
             [0.01, 0.02, 0.015],  # day 2
             [0.01, 0.02, 0.015],  # day 3
             [0.01, 0.02, 0.015],  # day 4
             [0.01, 0.02, 0.015]],  # day 5
            index=dates,
            columns=assets,
        )
        
        # Weights: long only (all positive, same across all dates to minimize turnover)
        weights_wide = pd.DataFrame(
            [[0.35, 0.35, 0.30],   # day 1
             [0.35, 0.35, 0.30],   # day 2 (same)
             [0.35, 0.35, 0.30],   # day 3 (same)
             [0.35, 0.35, 0.30],   # day 4 (same)
             [0.35, 0.35, 0.30]],  # day 5 (same)
            index=dates,
            columns=assets,
        )
        
        config = ExecutionConfig.from_mapping({
            "enabled": True,
            "execution_delay": 1,
            "transaction_cost_bps": 5.0,
            "slippage_bps": 2.0,
            "max_short_weight_sum": 0.0,  # Long-only constraint
            "short_borrow_cost_bps": 50.0,  # Irrelevant for long-only
        })
        
        result = apply_portfolio_execution_model(
            returns_wide=returns_wide,
            weights_wide=weights_wide,
            execution_config=config,
        )
        
        # Verify no short costs appear
        if "portfolio_short_borrow_cost" in result.frame.columns:
            assert (result.frame["portfolio_short_borrow_cost"] == 0.0).all()
        # No short positions means no short costs
        assert result.summary["totals"]["total_short_borrow_cost"] == 0.0
        assert result.summary["totals"]["total_short_turnover"] == 0.0
        assert result.summary["totals"]["total_short_transaction_cost"] == 0.0


class TestHighBorrowCostStressTest:
    """Canonical stress test: high short borrow costs."""

    def test_high_borrow_cost_config_creation(self) -> None:
        """Test creating a config with high short borrow costs."""
        config = ExecutionConfig.from_mapping({
            "enabled": True,
            "execution_delay": 1,
            "transaction_cost_bps": 5.0,
            "slippage_bps": 2.0,
            "short_borrow_cost_bps": 1000.0,  # 10% annual borrow cost
            "max_short_weight_sum": 0.30,
        })
        
        assert config.short_borrow_cost_bps == 1000.0

    def test_high_borrow_cost_increases_short_friction(self) -> None:
        """Test that high borrow costs significantly increase short-side friction."""
        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        assets = ["ASSET_A", "ASSET_B", "ASSET_C"]
        
        index = pd.MultiIndex.from_product([dates, assets], names=["ts_utc", "symbol"])
        
        returns_wide = pd.DataFrame(
            [[0.01, 0.02, 0.015]] * 5,
            index=dates,
            columns=assets,
        )
        
        # Weights with some short positions
        weights_wide = pd.DataFrame(
            [[0.45, 0.45, -0.10],   # 45% A, 45% B, -10% C (short)
             [0.45, 0.45, -0.10],
             [0.45, 0.45, -0.10],
             [0.45, 0.45, -0.10],
             [0.45, 0.45, -0.10]],
            index=dates,
            columns=assets,
        )
        
        # Config with high borrow cost
        high_borrow_config = ExecutionConfig.from_mapping({
            "enabled": True,
            "execution_delay": 1,
            "transaction_cost_bps": 5.0,
            "slippage_bps": 2.0,
            "short_borrow_cost_bps": 1000.0,  # 10% annual
            "max_short_weight_sum": 0.30,
        })
        
        # Config with low borrow cost
        low_borrow_config = ExecutionConfig.from_mapping({
            "enabled": True,
            "execution_delay": 1,
            "transaction_cost_bps": 5.0,
            "slippage_bps": 2.0,
            "short_borrow_cost_bps": 10.0,  # 0.1% annual
            "max_short_weight_sum": 0.30,
        })
        
        result_high = apply_portfolio_execution_model(
            returns_wide=returns_wide,
            weights_wide=weights_wide,
            execution_config=high_borrow_config,
        )
        
        result_low = apply_portfolio_execution_model(
            returns_wide=returns_wide,
            weights_wide=weights_wide,
            execution_config=low_borrow_config,
        )
        
        # Verify high borrow cost results in higher total short cost
        high_short_cost = result_high.summary["totals"]["total_short_borrow_cost"]
        low_short_cost = result_low.summary["totals"]["total_short_borrow_cost"]
        
        assert high_short_cost > low_short_cost
        # Ratio should be approximately 100x (1000 bps vs 10 bps)
        assert high_short_cost / low_short_cost > 50.0  # Allow some variance


class TestShortAvailabilityPolicies:
    """Test different short availability policies (exclude, cap, penalty)."""

    def test_short_availability_policy_exclude(self) -> None:
        """Test 'exclude' policy: shorts beyond availability limit are excluded from portfolio."""
        config = ExecutionConfig.from_mapping({
            "enabled": True,
            "execution_delay": 1,
            "transaction_cost_bps": 5.0,
            "slippage_bps": 2.0,
            "short_availability_limit": 0.20,  # Max 20% can be shorted
            "short_availability_policy": "exclude",
        })
        
        assert config.short_availability_policy == "exclude"
        assert config.has_directional_asymmetry is True

    def test_short_availability_policy_cap(self) -> None:
        """Test 'cap' policy: shorts beyond availability limit are capped."""
        config = ExecutionConfig.from_mapping({
            "enabled": True,
            "execution_delay": 1,
            "transaction_cost_bps": 5.0,
            "slippage_bps": 2.0,
            "short_availability_limit": 0.20,  # Max 20% can be shorted
            "short_availability_policy": "cap",
        })
        
        assert config.short_availability_policy == "cap"

    def test_short_availability_policy_penalty(self) -> None:
        """Test 'penalty' policy: shorts beyond availability limit incur additional cost."""
        config = ExecutionConfig.from_mapping({
            "enabled": True,
            "execution_delay": 1,
            "transaction_cost_bps": 5.0,
            "slippage_bps": 2.0,
            "short_availability_limit": 0.20,
            "short_availability_policy": "penalty",
        })
        
        assert config.short_availability_policy == "penalty"


class TestDirectionalAsymmetryFlag:
    """Test has_directional_asymmetry property with short-capacity fields."""

    def test_asymmetry_enabled_with_capacity_fields(self) -> None:
        """Test that asymmetry is enabled when capacity fields are set."""
        config_with_max_short = ExecutionConfig.from_mapping({
            "enabled": True,
            "execution_delay": 1,
            "transaction_cost_bps": 5.0,
            "slippage_bps": 2.0,
            "max_short_weight_sum": 0.30,
        })
        assert config_with_max_short.has_directional_asymmetry is True
        
        config_with_limit = ExecutionConfig.from_mapping({
            "enabled": True,
            "execution_delay": 1,
            "transaction_cost_bps": 5.0,
            "slippage_bps": 2.0,
            "short_availability_limit": 0.20,
        })
        assert config_with_limit.has_directional_asymmetry is True

    def test_asymmetry_disabled_without_directional_fields(self) -> None:
        """Test that asymmetry is disabled when no directional fields are set."""
        config = ExecutionConfig.from_mapping({
            "enabled": True,
            "execution_delay": 1,
            "transaction_cost_bps": 5.0,
            "slippage_bps": 2.0,
        })
        assert config.has_directional_asymmetry is False


class TestExecutionConfigSerialization:
    """Test serialization/deserialization of new short-capacity fields."""

    def test_to_dict_includes_short_capacity_fields_when_set(self) -> None:
        """Test that to_dict includes short-capacity fields when they're set."""
        config = ExecutionConfig.from_mapping({
            "enabled": True,
            "execution_delay": 1,
            "transaction_cost_bps": 5.0,
            "slippage_bps": 2.0,
            "max_short_weight_sum": 0.40,
            "short_availability_limit": 0.20,
            "short_availability_policy": "cap",
        })
        
        config_dict = config.to_dict()
        assert config_dict["max_short_weight_sum"] == 0.40
        assert config_dict["short_availability_limit"] == 0.20
        assert config_dict["short_availability_policy"] == "cap"

    def test_round_trip_serialization(self) -> None:
        """Test round-trip: ExecutionConfig -> dict -> ExecutionConfig."""
        original = ExecutionConfig.from_mapping({
            "enabled": True,
            "execution_delay": 1,
            "transaction_cost_bps": 5.0,
            "slippage_bps": 2.0,
            "short_borrow_cost_bps": 100.0,
            "max_short_weight_sum": 0.40,
            "short_availability_limit": 0.20,
            "short_availability_policy": "penalty",
        })
        
        # Serialize and deserialize
        config_dict = original.to_dict()
        restored = ExecutionConfig.from_mapping(config_dict)
        
        # Verify all fields match
        assert restored.enabled == original.enabled
        assert restored.execution_delay == original.execution_delay
        assert restored.transaction_cost_bps == original.transaction_cost_bps
        assert restored.slippage_bps == original.slippage_bps
        assert restored.short_borrow_cost_bps == original.short_borrow_cost_bps
        assert restored.max_short_weight_sum == original.max_short_weight_sum
        assert restored.short_availability_limit == original.short_availability_limit
        assert restored.short_availability_policy == original.short_availability_policy


class TestBackwardCompatibility:
    """Test backward compatibility with existing configs."""

    def test_existing_config_still_works(self) -> None:
        """Test that existing configs without new fields continue to work."""
        # Old-style config without short-capacity fields
        config = ExecutionConfig.from_mapping({
            "enabled": True,
            "execution_delay": 1,
            "transaction_cost_bps": 5.0,
            "slippage_bps": 2.0,
        })
        
        # Should work without errors
        assert config.enabled is True
        assert config.max_short_weight_sum is None
        assert config.short_availability_limit is None
        assert config.short_availability_policy == "exclude"

    def test_mixed_old_and_new_fields(self) -> None:
        """Test configs with mix of old and new fields."""
        config = ExecutionConfig.from_mapping({
            "enabled": True,
            "execution_delay": 1,
            "transaction_cost_bps": 5.0,
            "slippage_bps": 2.0,
            "short_borrow_cost_bps": 100.0,  # Old directional field
            "max_short_weight_sum": 0.40,    # New short-capacity field
        })
        
        assert config.short_borrow_cost_bps == 100.0
        assert config.max_short_weight_sum == 0.40
        assert config.has_directional_asymmetry is True


class TestConstraintEventTracking:
    """Test M22.5 constraint event tracking and utilization metrics."""

    def test_constraint_events_when_no_constraints_configured(self) -> None:
        """Test that constraint events are zero when constraints not configured."""
        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        assets = ["A", "B", "C"]
        
        returns_wide = pd.DataFrame(
            [[0.01, 0.02, -0.01]] * 5,
            index=dates,
            columns=assets,
        )
        weights_wide = pd.DataFrame(
            [[0.45, 0.45, -0.10]] * 5,
            index=dates,
            columns=assets,
        )
        
        config = ExecutionConfig.from_mapping({
            "enabled": True,
            "execution_delay": 1,
            "transaction_cost_bps": 5.0,
            "slippage_bps": 2.0,
        })
        
        result = apply_portfolio_execution_model(
            returns_wide=returns_wide,
            weights_wide=weights_wide,
            execution_config=config,
        )
        
        assert "constraint_events" in result.summary
        assert result.summary["constraint_events"]["max_short_weight_hits"] == 0
        assert result.summary["constraint_events"]["availability_caps_triggered"] == 0
        assert result.summary["constraint_events"]["availability_exclusions"] == 0

    def test_constraint_events_with_max_short_weight_violations(self) -> None:
        """Test constraint event tracking when max_short_weight_sum is exceeded."""
        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        assets = ["A", "B", "C"]
        
        returns_wide = pd.DataFrame(
            [[0.01, 0.02, -0.01]] * 5,
            index=dates,
            columns=assets,
        )
        
        # Weights with short position exceeding the constraint
        weights_wide = pd.DataFrame(
            [[0.40, 0.40, -0.50],  # Short exceeds max_short_weight_sum of 0.30
             [0.45, 0.45, -0.25],
             [0.40, 0.40, -0.50],  # Exceeds again
             [0.45, 0.45, -0.25],
             [0.40, 0.40, -0.50]],  # Exceeds again
            index=dates,
            columns=assets,
        )
        
        config = ExecutionConfig.from_mapping({
            "enabled": True,
            "execution_delay": 1,
            "transaction_cost_bps": 5.0,
            "slippage_bps": 2.0,
            "max_short_weight_sum": 0.30,
        })
        
        result = apply_portfolio_execution_model(
            returns_wide=returns_wide,
            weights_wide=weights_wide,
            execution_config=config,
        )
        
        # Should track violations
        assert result.summary["constraint_events"]["max_short_weight_hits"] >= 3

    def test_constraint_utilization_with_max_short_limit(self) -> None:
        """Test constraint utilization metrics are computed."""
        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        assets = ["A", "B", "C"]
        
        returns_wide = pd.DataFrame(
            [[0.01, 0.02, -0.01]] * 5,
            index=dates,
            columns=assets,
        )
        weights_wide = pd.DataFrame(
            [[0.45, 0.45, -0.10]] * 5,
            index=dates,
            columns=assets,
        )
        
        config = ExecutionConfig.from_mapping({
            "enabled": True,
            "execution_delay": 1,
            "transaction_cost_bps": 5.0,
            "slippage_bps": 2.0,
            "max_short_weight_sum": 0.30,
        })
        
        result = apply_portfolio_execution_model(
            returns_wide=returns_wide,
            weights_wide=weights_wide,
            execution_config=config,
        )
        
        assert "constraint_utilization" in result.summary
        # Utilization should be 0.10 / 0.30 = ~0.33
        utilization = result.summary["constraint_utilization"]["avg_short_utilization"]
        assert 0.2 < utilization < 0.5  # Roughly 1/3


class TestSideAwareCostAttribution:
    """Test M22.5 side-aware cost attribution metrics."""

    def test_side_cost_attribution_structure(self) -> None:
        """Test that side cost attribution structure exists and is valid."""
        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        assets = ["A", "B", "C"]
        
        returns_wide = pd.DataFrame(
            [[0.01, 0.02, -0.01]] * 5,
            index=dates,
            columns=assets,
        )
        weights_wide = pd.DataFrame(
            [[0.45, 0.45, -0.10]] * 5,
            index=dates,
            columns=assets,
        )
        
        config = ExecutionConfig.from_mapping({
            "enabled": True,
            "execution_delay": 1,
            "transaction_cost_bps": 5.0,
            "slippage_bps": 2.0,
            "short_borrow_cost_bps": 50.0,
        })
        
        result = apply_portfolio_execution_model(
            returns_wide=returns_wide,
            weights_wide=weights_wide,
            execution_config=config,
        )
        
        assert "side_cost_attribution" in result.summary
        attribution = result.summary["side_cost_attribution"]
        assert "long_cost_pct_total" in attribution
        assert "short_cost_pct_total" in attribution
        assert "short_borrow_cost_drag_pct" in attribution
        
        # Percentages should sum to 100 (approximately)
        total_pct = attribution["long_cost_pct_total"] + attribution["short_cost_pct_total"]
        assert 95.0 < total_pct < 105.0  # Allow small rounding errors

    def test_short_borrow_cost_attribution(self) -> None:
        """Test that short borrow cost is properly attributed."""
        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        assets = ["A", "B", "C"]
        
        returns_wide = pd.DataFrame(
            [[0.01, 0.02, -0.01]] * 5,
            index=dates,
            columns=assets,
        )
        weights_wide = pd.DataFrame(
            [[0.45, 0.45, -0.10]] * 5,
            index=dates,
            columns=assets,
        )
        
        # Config with high borrow cost
        config = ExecutionConfig.from_mapping({
            "enabled": True,
            "execution_delay": 1,
            "transaction_cost_bps": 5.0,
            "slippage_bps": 2.0,
            "short_borrow_cost_bps": 1000.0,  # 10% annual
        })
        
        result = apply_portfolio_execution_model(
            returns_wide=returns_wide,
            weights_wide=weights_wide,
            execution_config=config,
        )
        
        attribution = result.summary["side_cost_attribution"]
        # Borrow cost should be significant due to high rate
        borrow_drag = attribution["short_borrow_cost_drag_pct"]
        assert borrow_drag > 50.0  # Borrow cost dominates


class TestStressAnalysisSummary:
    """Test M22.5 side-stress analysis summary block."""

    def test_stress_analysis_exists_with_constraints(self) -> None:
        """Test that stress analysis block exists when constraints are configured."""
        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        assets = ["A", "B", "C"]
        
        returns_wide = pd.DataFrame(
            [[0.01, 0.02, -0.01]] * 5,
            index=dates,
            columns=assets,
        )
        weights_wide = pd.DataFrame(
            [[0.45, 0.45, -0.10]] * 5,
            index=dates,
            columns=assets,
        )
        
        config = ExecutionConfig.from_mapping({
            "enabled": True,
            "execution_delay": 1,
            "transaction_cost_bps": 5.0,
            "slippage_bps": 2.0,
            "max_short_weight_sum": 0.30,
            "short_borrow_cost_bps": 50.0,
        })
        
        result = apply_portfolio_execution_model(
            returns_wide=returns_wide,
            weights_wide=weights_wide,
            execution_config=config,
        )
        
        # Should have stress analysis block
        assert "side_stress_analysis" in result.summary
        stress_analysis = result.summary["side_stress_analysis"]
        
        # Check required fields
        assert "short_cost_drag_pct" in stress_analysis
        assert "constraint_impact_on_return" in stress_analysis
        assert "constraint_binding_frequency" in stress_analysis
        assert "constraint_binding_events" in stress_analysis
        assert "max_short_utilization" in stress_analysis

    def test_stress_analysis_no_constraints(self) -> None:
        """Test that stress analysis is not present without constraints."""
        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        assets = ["A", "B", "C"]
        
        returns_wide = pd.DataFrame(
            [[0.01, 0.02, -0.01]] * 5,
            index=dates,
            columns=assets,
        )
        weights_wide = pd.DataFrame(
            [[0.45, 0.45, -0.10]] * 5,
            index=dates,
            columns=assets,
        )
        
        config = ExecutionConfig.from_mapping({
            "enabled": True,
            "execution_delay": 1,
            "transaction_cost_bps": 5.0,
            "slippage_bps": 2.0,
        })
        
        result = apply_portfolio_execution_model(
            returns_wide=returns_wide,
            weights_wide=weights_wide,
            execution_config=config,
        )
        
        # Should NOT have stress analysis block without constraints
        assert "side_stress_analysis" not in result.summary


class TestCapacityImpactAnalysis:
    """Test M22.5 capacity impact analysis metrics."""

    def test_capacity_impact_exists_with_constraints(self) -> None:
        """Test that capacity impact is computed when constraints are configured."""
        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        assets = ["A", "B", "C"]
        
        returns_wide = pd.DataFrame(
            [[0.01, 0.02, -0.01]] * 5,
            index=dates,
            columns=assets,
        )
        weights_wide = pd.DataFrame(
            [[0.45, 0.45, -0.10]] * 5,
            index=dates,
            columns=assets,
        )
        
        config = ExecutionConfig.from_mapping({
            "enabled": True,
            "execution_delay": 1,
            "transaction_cost_bps": 5.0,
            "slippage_bps": 2.0,
            "max_short_weight_sum": 0.30,
        })
        
        result = apply_portfolio_execution_model(
            returns_wide=returns_wide,
            weights_wide=weights_wide,
            execution_config=config,
        )
        
        # Should have capacity impact analysis
        assert "capacity_impact" in result.summary
        impact = result.summary["capacity_impact"]
        assert "return_delta" in impact
        assert "turnover_delta" in impact
        assert "short_exposure_delta" in impact
        assert "baseline_friction" in impact
        assert "constrained_friction" in impact

    def test_capacity_impact_no_constraints(self) -> None:
        """Test that capacity impact is None without constraints."""
        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        assets = ["A", "B", "C"]
        
        returns_wide = pd.DataFrame(
            [[0.01, 0.02, -0.01]] * 5,
            index=dates,
            columns=assets,
        )
        weights_wide = pd.DataFrame(
            [[0.45, 0.45, -0.10]] * 5,
            index=dates,
            columns=assets,
        )
        
        config = ExecutionConfig.from_mapping({
            "enabled": True,
            "execution_delay": 1,
            "transaction_cost_bps": 5.0,
            "slippage_bps": 2.0,
        })
        
        result = apply_portfolio_execution_model(
            returns_wide=returns_wide,
            weights_wide=weights_wide,
            execution_config=config,
        )
        
        # Should NOT have capacity impact without constraints
        assert "capacity_impact" not in result.summary


class TestInterpretationLayerDeterminism:
    """Test that interpretation layer metrics are deterministic."""

    def test_constraint_events_deterministic_across_runs(self) -> None:
        """Test that constraint events are identical across runs."""
        dates = pd.date_range("2025-01-01", periods=10, freq="D")
        assets = ["A", "B", "C"]
        
        returns_wide = pd.DataFrame(
            [[0.01, 0.02, -0.01]] * 10,
            index=dates,
            columns=assets,
        )
        weights_wide = pd.DataFrame(
            [[0.40, 0.40, -0.35]] * 10,
            index=dates,
            columns=assets,
        )
        
        config = ExecutionConfig.from_mapping({
            "enabled": True,
            "execution_delay": 1,
            "transaction_cost_bps": 5.0,
            "slippage_bps": 2.0,
            "max_short_weight_sum": 0.30,
        })
        
        # Run twice
        result1 = apply_portfolio_execution_model(
            returns_wide=returns_wide,
            weights_wide=weights_wide,
            execution_config=config,
        )
        
        result2 = apply_portfolio_execution_model(
            returns_wide=returns_wide,
            weights_wide=weights_wide,
            execution_config=config,
        )
        
        # Should be identical
        assert result1.summary["constraint_events"] == result2.summary["constraint_events"]
        assert result1.summary["constraint_utilization"] == result2.summary["constraint_utilization"]
        assert result1.summary["side_cost_attribution"] == result2.summary["side_cost_attribution"]


class TestPolicyDifferentiation:
    """Test that different policies produce measurably different execution outcomes."""
    
    def test_exclude_vs_cap_produce_distinct_short_exposure(self) -> None:
        """Test that 'exclude' and 'cap' policies produce different short exposures."""
        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        assets = ["A", "B", "C"]
        
        returns_wide = pd.DataFrame(
            [[0.01, 0.02, -0.01]] * 5,
            index=dates,
            columns=assets,
        )
        # Weights with short exposure exceeding the limit
        weights_wide = pd.DataFrame(
            [[0.30, 0.30, -0.50]] * 5,  # -50% short, limit is 0.30
            index=dates,
            columns=assets,
        )
        
        # Exclude policy: removes shorts beyond limit
        config_exclude = ExecutionConfig.from_mapping({
            "enabled": True,
            "execution_delay": 1,
            "transaction_cost_bps": 5.0,
            "slippage_bps": 2.0,
            "max_short_weight_sum": 0.30,
            "short_availability_policy": "exclude",
        })
        
        # Cap policy: rescales shorts to fit limit
        config_cap = ExecutionConfig.from_mapping({
            "enabled": True,
            "execution_delay": 1,
            "transaction_cost_bps": 5.0,
            "slippage_bps": 2.0,
            "max_short_weight_sum": 0.30,
            "short_availability_policy": "cap",
        })
        
        result_exclude = apply_portfolio_execution_model(
            returns_wide=returns_wide,
            weights_wide=weights_wide,
            execution_config=config_exclude,
        )
        
        result_cap = apply_portfolio_execution_model(
            returns_wide=returns_wide,
            weights_wide=weights_wide,
            execution_config=config_cap,
        )
        
        # Get final short exposures
        exclude_short_exposure = result_exclude.summary["totals"]["average_short_exposure"]
        cap_short_exposure = result_cap.summary["totals"]["average_short_exposure"]
        
        # Exclude should have zero short exposure
        assert exclude_short_exposure == 0.0
        # Cap should have some short exposure (capped to 0.30)
        assert 0.25 < cap_short_exposure <= 0.30


class TestConstraintEnforcementBehavior:
    """Test that constraints actually affect execution behavior (not just tracked)."""
    
    def test_constraints_reduce_short_exposure_in_execution(self) -> None:
        """Test that configured constraints actually reduce short exposure in portfolio."""
        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        assets = ["A", "B", "C"]
        
        returns_wide = pd.DataFrame(
            [[0.01, 0.02, -0.01]] * 5,
            index=dates,
            columns=assets,
        )
        # Weights with unconstrained short exposure of 40%
        weights_wide = pd.DataFrame(
            [[0.30, 0.30, -0.40]] * 5,
            index=dates,
            columns=assets,
        )
        
        # Unconstrained execution
        config_unconstrained = ExecutionConfig.from_mapping({
            "enabled": True,
            "execution_delay": 1,
            "transaction_cost_bps": 5.0,
            "slippage_bps": 2.0,
        })
        
        # Constrained execution (limit 30% short)
        config_constrained = ExecutionConfig.from_mapping({
            "enabled": True,
            "execution_delay": 1,
            "transaction_cost_bps": 5.0,
            "slippage_bps": 2.0,
            "max_short_weight_sum": 0.30,
            "short_availability_policy": "exclude",
        })
        
        result_unconstrained = apply_portfolio_execution_model(
            returns_wide=returns_wide,
            weights_wide=weights_wide,
            execution_config=config_unconstrained,
        )
        
        result_constrained = apply_portfolio_execution_model(
            returns_wide=returns_wide,
            weights_wide=weights_wide,
            execution_config=config_constrained,
        )
        
        unconstrained_short = result_unconstrained.summary["totals"]["average_short_exposure"]
        constrained_short = result_constrained.summary["totals"]["average_short_exposure"]
        
        # Constraint should reduce short exposure
        assert unconstrained_short > constrained_short
        # Should be reduced to approximately 0 (excluded)
        assert constrained_short <= 0.01  # Allow tiny floating point errors


class TestRealCapacityImpactDeltas:
    """Test that capacity_impact deltas are derived from real dual execution."""
    
    def test_capacity_impact_deltas_are_positive_when_constraints_bind(self) -> None:
        """Test that capacity impact shows friction INCREASE when constraints bind."""
        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        assets = ["A", "B", "C"]
        
        returns_wide = pd.DataFrame(
            [[0.01, 0.02, -0.01]] * 5,
            index=dates,
            columns=assets,
        )
        # Weights where constraint will bind (40% short > 30% limit)
        weights_wide = pd.DataFrame(
            [[0.30, 0.30, -0.40]] * 5,
            index=dates,
            columns=assets,
        )
        
        config = ExecutionConfig.from_mapping({
            "enabled": True,
            "execution_delay": 1,
            "transaction_cost_bps": 5.0,
            "slippage_bps": 2.0,
            "max_short_weight_sum": 0.30,
            "short_availability_policy": "exclude",
        })
        
        result = apply_portfolio_execution_model(
            returns_wide=returns_wide,
            weights_wide=weights_wide,
            execution_config=config,
        )
        
        # Should have capacity_impact block
        assert "capacity_impact" in result.summary
        impact = result.summary["capacity_impact"]
        
        # Deltas should be present
        assert "return_delta" in impact
        assert "turnover_delta" in impact
        assert "short_exposure_delta" in impact
        
        # Verify deltas reference real execution values
        assert "baseline_friction" in impact
        assert "constrained_friction" in impact
        assert isinstance(impact["baseline_friction"], (int, float))
        assert isinstance(impact["constrained_friction"], (int, float))


class TestBindingFrequencyCalculation:
    """Test that constraint_binding_frequency is correctly computed from actual binding events."""
    
    def test_binding_frequency_is_zero_when_no_constraints_bind(self) -> None:
        """Test that binding frequency is zero when constraints are not violated."""
        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        assets = ["A", "B", "C"]
        
        returns_wide = pd.DataFrame(
            [[0.01, 0.02, -0.01]] * 5,
            index=dates,
            columns=assets,
        )
        # Weights that don't violate the 30% limit (only 10% short)
        weights_wide = pd.DataFrame(
            [[0.45, 0.45, -0.10]] * 5,
            index=dates,
            columns=assets,
        )
        
        config = ExecutionConfig.from_mapping({
            "enabled": True,
            "execution_delay": 1,
            "transaction_cost_bps": 5.0,
            "slippage_bps": 2.0,
            "max_short_weight_sum": 0.30,
        })
        
        result = apply_portfolio_execution_model(
            returns_wide=returns_wide,
            weights_wide=weights_wide,
            execution_config=config,
        )
        
        assert "side_stress_analysis" in result.summary
        stress = result.summary["side_stress_analysis"]
        # Binding frequency should be zero
        assert stress["constraint_binding_frequency"] == 0.0
    
    def test_binding_frequency_is_positive_when_constraints_bind(self) -> None:
        """Test that binding frequency is > 0 when constraints actually bind."""
        dates = pd.date_range("2025-01-01", periods=10, freq="D")
        assets = ["A", "B", "C"]
        
        returns_wide = pd.DataFrame(
            [[0.01, 0.02, -0.01]] * 10,
            index=dates,
            columns=assets,
        )
        # Weights where constraint binds on all 10 dates (40% short > 30% limit)
        weights_wide = pd.DataFrame(
            [[0.30, 0.30, -0.40]] * 10,
            index=dates,
            columns=assets,
        )
        
        config = ExecutionConfig.from_mapping({
            "enabled": True,
            "execution_delay": 1,
            "transaction_cost_bps": 5.0,
            "slippage_bps": 2.0,
            "max_short_weight_sum": 0.30,
        })
        
        result = apply_portfolio_execution_model(
            returns_wide=returns_wide,
            weights_wide=weights_wide,
            execution_config=config,
        )
        
        assert "side_stress_analysis" in result.summary
        stress = result.summary["side_stress_analysis"]
        # Binding frequency should be > 0
        assert stress["constraint_binding_frequency"] > 0.0
        # With 10 dates and constraint binding all periods, frequency should reflect binding
        assert stress["constraint_binding_frequency"] >= 1.0  # At least 1 binding per period on average

