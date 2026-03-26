from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.runtime import resolve_runtime_config


def test_resolve_runtime_config_loads_repository_defaults() -> None:
    runtime = resolve_runtime_config()

    assert runtime.execution.to_dict() == {
        "enabled": False,
        "execution_delay": 1,
        "transaction_cost_bps": 0.0,
        "slippage_bps": 0.0,
    }
    assert runtime.sanity.strict_sanity_checks is False
    assert runtime.portfolio_validation.to_dict() == {
        "target_weight_sum": 1.0,
        "weight_sum_tolerance": 1e-8,
        "target_net_exposure": 1.0,
        "net_exposure_tolerance": 1e-8,
        "max_gross_exposure": 1.0,
        "max_leverage": 1.0,
        "max_single_sleeve_weight": None,
        "min_single_sleeve_weight": None,
        "max_abs_period_return": 1.0,
        "max_equity_multiple": 1_000_000.0,
        "strict_sanity_checks": False,
    }
    assert runtime.strict_mode.to_dict() == {"enabled": False, "source": "default"}


def test_resolve_runtime_config_applies_precedence_deterministically() -> None:
    runtime = resolve_runtime_config(
        {
            "execution": {"execution_delay": 2},
            "sanity": {"max_abs_period_return": 0.8},
            "validation": {"max_leverage": 0.9},
            "strict_mode": {"enabled": False},
        },
        cli_overrides={
            "execution": {"transaction_cost_bps": 12.0},
            "sanity": {"strict_sanity_checks": True},
            "portfolio_validation": {"max_leverage": 0.75},
        },
        cli_strict=True,
    )

    assert runtime.execution.to_dict() == {
        "enabled": True,
        "execution_delay": 2,
        "transaction_cost_bps": 12.0,
        "slippage_bps": 0.0,
    }
    assert runtime.sanity.max_abs_period_return == pytest.approx(0.8)
    assert runtime.sanity.strict_sanity_checks is True
    assert runtime.portfolio_validation.max_leverage == pytest.approx(0.75)
    assert runtime.portfolio_validation.strict_sanity_checks is True
    assert runtime.strict_mode.to_dict() == {"enabled": True, "source": "cli"}


def test_resolve_runtime_config_rejects_invalid_runtime_values() -> None:
    with pytest.raises(ValueError, match="transaction_cost_bps"):
        resolve_runtime_config({"execution": {"transaction_cost_bps": -1.0}})

    with pytest.raises(ValueError, match="max_leverage"):
        resolve_runtime_config({"validation": {"max_gross_exposure": 0.5, "max_leverage": 0.6}})

    with pytest.raises(ValueError, match="unsupported keys"):
        resolve_runtime_config({"runtime": {"execution": {"enabled": True}, "mystery": {}}})


def test_resolve_runtime_config_accepts_runtime_and_legacy_validation_aliases() -> None:
    runtime = resolve_runtime_config(
        {
            "runtime": {
                "execution": {"execution_delay": 3},
                "portfolio_validation": {"max_single_sleeve_weight": 0.4},
                "strict_mode": {"enabled": True, "source": "config"},
            }
        }
    )

    assert runtime.execution.execution_delay == 3
    assert runtime.portfolio_validation.max_single_sleeve_weight == pytest.approx(0.4)
    assert runtime.strict_mode.to_dict() == {"enabled": True, "source": "config"}
