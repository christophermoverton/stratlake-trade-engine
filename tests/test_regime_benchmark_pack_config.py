from __future__ import annotations

from pathlib import Path

import pytest

from src.config.regime_benchmark_pack import (
    RegimeBenchmarkPackConfigError,
    load_regime_benchmark_pack_config,
)


def test_load_regime_benchmark_pack_config_resolves_policy_path(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.yml"
    policy_path.write_text(
        """
regime_policy:
  default:
    signal_scale: 1.0
    allocation_scale: 1.0
    alpha_weight_multiplier: 1.0
    volatility_target: 0.10
    gross_exposure_cap: 1.0
    max_component_weight: 1.0
    rebalance_enabled: true
    optimizer_override: null
    allocation_rule_override: null
    fallback_policy: baseline
  confidence:
    min_confidence: 0.60
    low_confidence_fallback: neutral
    ambiguous_fallback: reduce_exposure
    unsupported_fallback: baseline
  regimes: {}
""".strip(),
        encoding="utf-8",
        newline="\n",
    )
    config_path = tmp_path / "benchmark.yml"
    config_path.write_text(
        """
benchmark_name: test_regime_pack
dataset: features_daily
timeframe: 1D
start: "2026-01-01"
end: "2026-02-28"
variants:
  - name: static_baseline
    regime_source: static
  - name: policy_variant
    regime_source: policy
    policy_config: ./policy.yml
""".strip(),
        encoding="utf-8",
        newline="\n",
    )

    config = load_regime_benchmark_pack_config(config_path)

    assert config.benchmark_name == "test_regime_pack"
    assert config.features_root == "data"
    assert config.output_root == "artifacts/regime_benchmarks"
    assert config.variants[1].policy_config == policy_path.resolve().as_posix()


def test_policy_variant_requires_policy_config(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yml"
    config_path.write_text(
        """
benchmark_name: invalid_regime_pack
dataset: features_daily
timeframe: 1D
start: "2026-01-01"
end: "2026-02-28"
variants:
  - name: policy_variant
    regime_source: policy
""".strip(),
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(RegimeBenchmarkPackConfigError, match="requires 'policy_config'"):
        load_regime_benchmark_pack_config(config_path)


def test_policy_config_path_must_exist(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yml"
    config_path.write_text(
        """
benchmark_name: invalid_regime_pack
dataset: features_daily
timeframe: 1D
start: "2026-01-01"
end: "2026-02-28"
variants:
  - name: static_baseline
    regime_source: static
  - name: policy_variant
    regime_source: policy
    policy_config: ./missing.yml
""".strip(),
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(RegimeBenchmarkPackConfigError, match="missing path"):
        load_regime_benchmark_pack_config(config_path)
