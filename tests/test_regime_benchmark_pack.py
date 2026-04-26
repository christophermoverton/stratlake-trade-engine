from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli.run_regime_benchmark_pack import run_cli
from src.config.regime_benchmark_pack import load_regime_benchmark_pack_config
from src.research.regime_benchmark_pack import run_regime_benchmark_pack


def _write_feature_fixture(root: Path, *, periods: int = 60) -> Path:
    dataset_root = root / "curated" / "features_daily"
    timestamps = pd.date_range("2026-01-01", periods=periods, freq="D", tz="UTC")
    symbols = ("AAA", "BBB", "CCC")
    for symbol_index, symbol in enumerate(symbols):
        rows: list[dict[str, object]] = []
        for index, ts_utc in enumerate(timestamps):
            base = 100.0 + (symbol_index * 7.5)
            close = base + (index * (0.7 + (symbol_index * 0.11))) + ((index % 5) * 0.35)
            rows.append(
                {
                    "symbol": symbol,
                    "ts_utc": ts_utc,
                    "timeframe": "1D",
                    "date": ts_utc.strftime("%Y-%m-%d"),
                    "close": close,
                }
            )
        frame = pd.DataFrame(rows)
        output_dir = dataset_root / f"symbol={symbol}" / "year=2026"
        output_dir.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output_dir / "part-0.parquet", index=False)
    return root


def _write_policy_config(path: Path) -> Path:
    path.write_text(
        """
regime_aliases:
  risk_on:
    match:
      trend: uptrend
      stress: normal_stress
  risk_off:
    match:
      stress_state: correlation_stress
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
  regimes:
    risk_on:
      signal_scale: 1.10
      allocation_scale: 1.0
      alpha_weight_multiplier: 1.05
      volatility_target: 0.12
      gross_exposure_cap: 1.0
      max_component_weight: 1.0
      rebalance_enabled: true
      optimizer_override: equal_weight
      allocation_rule_override: pro_rata
      fallback_policy: baseline
    risk_off:
      signal_scale: 0.50
      allocation_scale: 0.60
      alpha_weight_multiplier: 0.75
      volatility_target: 0.06
      gross_exposure_cap: 0.60
      max_component_weight: 0.50
      rebalance_enabled: true
      optimizer_override: risk_parity
      allocation_rule_override: null
      fallback_policy: reduce_exposure
""".strip(),
        encoding="utf-8",
        newline="\n",
    )
    return path


def _write_config(
    tmp_path: Path,
    *,
    features_root: Path,
    policy_path: Path,
    variants: list[dict[str, object]] | None = None,
    periods: int = 60,
    relative_policy_path: bool = False,
) -> Path:
    policy_value = (
        Path(policy_path.name).as_posix()
        if relative_policy_path
        else policy_path.as_posix()
    )
    config = {
        "benchmark_name": "test_regime_policy_benchmark",
        "dataset": "features_daily",
        "timeframe": "1D",
        "start": "2026-01-01",
        "end": "2026-03-01" if periods >= 60 else "2026-01-20",
        "features_root": features_root.as_posix(),
        "output_root": (tmp_path / "artifacts").as_posix(),
        "variants": variants
        or [
            {"name": "static_baseline", "regime_source": "static"},
            {"name": "taxonomy_only", "regime_source": "taxonomy"},
            {"name": "gmm_classifier", "regime_source": "gmm"},
            {
                "name": "policy_optimized",
                "regime_source": "policy",
                "calibration_profile": "baseline",
                "policy_config": policy_value,
            },
        ],
        "gmm": {
            "n_components": 2,
            "random_state": 11,
            "min_observations": 30 if periods >= 60 else 30,
            "feature_columns": [
                "volatility_metric",
                "trend_metric",
                "drawdown_metric",
                "stress_correlation_metric",
                "stress_dispersion_metric",
            ],
        },
    }
    config_path = tmp_path / "benchmark.yml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8", newline="\n")
    return config_path


def test_regime_benchmark_pack_generates_multi_variant_matrix_and_manifest(tmp_path: Path) -> None:
    features_root = _write_feature_fixture(tmp_path / "features")
    policy_path = _write_policy_config(tmp_path / "policy.yml")
    config_path = _write_config(
        tmp_path,
        features_root=features_root,
        policy_path=policy_path,
        relative_policy_path=True,
    )

    config = load_regime_benchmark_pack_config(config_path)
    result = run_regime_benchmark_pack(config)

    matrix = pd.read_csv(result.benchmark_matrix_csv_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    stability = json.loads(result.stability_summary_path.read_text(encoding="utf-8"))
    transition = json.loads(result.transition_summary_path.read_text(encoding="utf-8"))
    persisted_config = json.loads(result.config_path.read_text(encoding="utf-8"))
    conditional = json.loads(result.conditional_performance_summary_path.read_text(encoding="utf-8"))
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    provenance = json.loads(result.provenance_path.read_text(encoding="utf-8"))

    assert matrix["variant_name"].tolist() == [
        "static_baseline",
        "taxonomy_only",
        "gmm_classifier",
        "policy_optimized",
    ]
    assert (result.output_root / "benchmark_matrix.json").exists()
    assert (result.output_root / "model_comparison.csv").exists()
    assert (result.output_root / "policy_comparison.csv").exists()
    assert manifest["variant_count"] == 4
    assert "benchmark_matrix.csv" in manifest["generated_files"]
    assert len(stability["variants"]) == 3
    assert len(transition["variants"]) == 3
    assert matrix.loc[matrix["variant_name"].eq("static_baseline"), "regime_count"].isna().all()
    assert matrix.loc[matrix["variant_name"].eq("policy_optimized"), "policy_turnover"].notna().all()
    assert persisted_config["variants"][-1]["policy_config"] == "policy.yml"
    assert conditional["enabled"] is True
    assert conditional["emitted"] is False
    assert "follow-up issue" in conditional["reason"]
    assert summary["warnings"]
    assert provenance["conditional_performance"]["emitted"] is False


def test_regime_benchmark_pack_is_deterministic_for_same_config(tmp_path: Path) -> None:
    features_root = _write_feature_fixture(tmp_path / "features")
    policy_path = _write_policy_config(tmp_path / "policy.yml")
    config_path = _write_config(tmp_path, features_root=features_root, policy_path=policy_path)
    config = load_regime_benchmark_pack_config(config_path)

    first = run_regime_benchmark_pack(config)
    second = run_regime_benchmark_pack(config)

    assert first.benchmark_run_id == second.benchmark_run_id
    assert first.output_root == second.output_root
    assert first.benchmark_matrix_csv_path.read_text(encoding="utf-8") == second.benchmark_matrix_csv_path.read_text(
        encoding="utf-8"
    )


def test_regime_benchmark_pack_fails_fast_when_gmm_inputs_are_insufficient(tmp_path: Path) -> None:
    features_root = _write_feature_fixture(tmp_path / "features_small", periods=20)
    policy_path = _write_policy_config(tmp_path / "policy.yml")
    config_path = _write_config(
        tmp_path,
        features_root=features_root,
        policy_path=policy_path,
        periods=20,
        variants=[
            {"name": "static_baseline", "regime_source": "static"},
            {"name": "gmm_classifier", "regime_source": "gmm"},
        ],
    )
    config = load_regime_benchmark_pack_config(config_path)

    with pytest.raises(Exception, match="at least 30 observations"):
        run_regime_benchmark_pack(config)


def test_regime_benchmark_cli_smoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    features_root = _write_feature_fixture(tmp_path / "features")
    policy_path = _write_policy_config(tmp_path / "policy.yml")
    config_path = _write_config(
        tmp_path,
        features_root=features_root,
        policy_path=policy_path,
        relative_policy_path=True,
    )

    result = run_cli(["--config", config_path.as_posix()])
    captured = capsys.readouterr()

    assert "Regime Benchmark Pack Summary" in captured.out
    assert "benchmark_matrix.csv" in captured.out
    assert result.benchmark_matrix_csv_path.exists()
