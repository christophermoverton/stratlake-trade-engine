from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.execution import ExecutionConfig, resolve_execution_config


def test_execution_config_rejects_negative_bps_values() -> None:
    with pytest.raises(ValueError, match="transaction_cost_bps"):
        ExecutionConfig.from_mapping({"transaction_cost_bps": -1.0})

    with pytest.raises(ValueError, match="slippage_bps"):
        ExecutionConfig.from_mapping({"slippage_bps": -0.1})


def test_execution_config_rejects_zero_delay() -> None:
    with pytest.raises(ValueError, match="execution_delay"):
        ExecutionConfig.from_mapping({"execution_delay": 0})


def test_resolve_execution_config_merges_overrides_deterministically() -> None:
    config = resolve_execution_config(
        {"enabled": False, "execution_delay": 1},
        {"transaction_cost_bps": 8.0, "slippage_bps": 2.0},
    )

    assert config == ExecutionConfig(
        enabled=True,
        execution_delay=1,
        transaction_cost_bps=8.0,
        slippage_bps=2.0,
    )
