from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.regimes import (
    ADAPTIVE_POLICY_MANIFEST_FILENAME,
    ADAPTIVE_VS_STATIC_COMPARISON_FILENAME,
    REGIME_POLICY_DECISIONS_FILENAME,
    REGIME_POLICY_SUMMARY_FILENAME,
    RegimePolicyError,
    apply_regime_policy,
    load_adaptive_policy_manifest,
    load_adaptive_vs_static_comparison,
    load_regime_policy_decisions,
    load_regime_policy_summary,
    resolve_regime_policy_config,
    write_regime_policy_artifacts,
)

_RISK_ON = "volatility=low_volatility|trend=uptrend|drawdown_recovery=near_peak|stress=normal_stress"
_RISK_OFF = "volatility=high_volatility|trend=downtrend|drawdown_recovery=drawdown|stress=correlation_stress"
_RECOVERY = "volatility=normal_volatility|trend=sideways|drawdown_recovery=recovery|stress=normal_stress"
_UNDEFINED = "volatility=undefined|trend=undefined|drawdown_recovery=undefined|stress=undefined"


def _labels(sequence: list[str], *, include_symbol: bool = False) -> pd.DataFrame:
    ts = pd.date_range("2026-02-01", periods=len(sequence), freq="D", tz="UTC")
    rows: list[dict[str, object]] = []
    for index, label in enumerate(sequence):
        mapping = dict(part.split("=", 1) for part in label.split("|"))
        row = {
            "ts_utc": ts[index],
            "volatility_state": mapping["volatility"],
            "trend_state": mapping["trend"],
            "drawdown_recovery_state": mapping["drawdown_recovery"],
            "stress_state": mapping["stress"],
            "regime_label": label,
            "is_defined": all(value != "undefined" for value in mapping.values()),
            "volatility_metric": 0.10 + (0.01 * index) if mapping["volatility"] != "undefined" else float("nan"),
            "trend_metric": 0.02 - (0.01 * (index % 3)) if mapping["trend"] != "undefined" else float("nan"),
            "drawdown_metric": 0.01 * (index % 5) if mapping["drawdown_recovery"] != "undefined" else float("nan"),
            "stress_correlation_metric": 0.20 + (0.02 * (index % 4)) if mapping["stress"] != "undefined" else float("nan"),
            "stress_dispersion_metric": 0.25 + (0.01 * (index % 4)) if mapping["stress"] != "undefined" else float("nan"),
        }
        if include_symbol:
            row["symbol"] = "SPY" if index % 2 == 0 else "QQQ"
        rows.append(row)
    return pd.DataFrame(rows)


def _confidence(labels: pd.DataFrame) -> pd.DataFrame:
    frame = labels.loc[:, [column for column in ("symbol", "ts_utc", "regime_label") if column in labels.columns]].copy()
    frame["confidence_score"] = [0.95, 0.30, 0.75, 0.80][: len(frame)]
    frame["confidence_bucket"] = ["high", "low", "medium", "high"][: len(frame)]
    frame["fallback_flag"] = [False, True, True, False][: len(frame)]
    frame["fallback_reason"] = [None, "low_confidence", "ambiguous", None][: len(frame)]
    frame["predicted_label"] = frame["regime_label"]
    return frame


def _baseline(labels: pd.DataFrame, *, column: str = "strategy_return") -> pd.DataFrame:
    frame = labels.loc[:, [column for column in ("symbol", "ts_utc") if column in labels.columns]].copy()
    frame[column] = [0.02, -0.01, 0.015, -0.005][: len(frame)]
    frame["weight"] = [0.50, 0.40, 0.60, 0.55][: len(frame)]
    return frame


def _policy_config() -> dict[str, object]:
    return {
        "regime_aliases": {
            "risk_on": {
                "match": {
                    "stress": "normal_stress",
                    "trend": "uptrend",
                }
            },
            "risk_off": {
                "match": {
                    "stress_state": "correlation_stress",
                }
            },
            "crisis": {
                "match": {
                    "volatility_state": "high_volatility",
                    "stress_state": "correlation_stress",
                }
            },
        },
        "regime_policy": {
            "default": {
                "signal_scale": 1.0,
                "allocation_scale": 1.0,
                "alpha_weight_multiplier": 1.0,
                "volatility_target": 0.10,
                "gross_exposure_cap": 1.0,
                "max_component_weight": 1.0,
                "rebalance_enabled": True,
                "optimizer_override": None,
                "allocation_rule_override": None,
                "fallback_policy": "baseline",
            },
            "confidence": {
                "min_confidence": 0.60,
                "low_confidence_fallback": "neutral",
                "ambiguous_fallback": "reduce_exposure",
                "unsupported_fallback": "baseline",
            },
            "regimes": {
                "risk_on": {
                    "signal_scale": 1.10,
                    "allocation_scale": 1.00,
                    "alpha_weight_multiplier": 1.05,
                    "volatility_target": 0.12,
                    "gross_exposure_cap": 1.0,
                    "max_component_weight": 1.0,
                    "rebalance_enabled": True,
                    "optimizer_override": "equal_weight",
                    "allocation_rule_override": "pro_rata",
                    "fallback_policy": "baseline",
                },
                "risk_off": {
                    "signal_scale": 0.50,
                    "allocation_scale": 0.60,
                    "alpha_weight_multiplier": 0.75,
                    "volatility_target": 0.06,
                    "gross_exposure_cap": 0.60,
                    "max_component_weight": 0.50,
                    "rebalance_enabled": True,
                    "optimizer_override": "risk_parity",
                    "allocation_rule_override": None,
                    "fallback_policy": "reduce_exposure",
                },
            },
        },
    }


def test_policy_selection_by_regime_and_default_fallback() -> None:
    labels = _labels([_RISK_ON, _RISK_OFF, _RECOVERY])

    result = apply_regime_policy(labels, config=_policy_config())

    assert result.decisions["matched_policy_key"].tolist() == ["risk_on", "risk_off", "default"]
    assert result.decisions["matched_alias"].tolist() == ["risk_on", "risk_off", None]
    assert result.decisions["signal_scale"].tolist() == [1.1, 0.5, 1.0]
    assert result.decisions["allocation_rule_override"].tolist() == ["pro_rata", None, None]


def test_alias_matching_is_deterministic_and_uses_config_order_for_ties() -> None:
    labels = _labels([_RISK_OFF])
    config = _policy_config()
    config["regime_aliases"]["stress_first"] = {
        "match": {
            "stress_state": "correlation_stress",
        }
    }
    config["regime_policy"]["regimes"]["stress_first"] = config["regime_policy"]["regimes"]["risk_off"]

    result = apply_regime_policy(labels, config=config)

    assert result.decisions.loc[0, "matched_alias"] == "risk_off"
    assert result.decisions.loc[0, "matched_policy_key"] == "risk_off"


def test_unknown_alias_and_bad_optimizer_fail_fast() -> None:
    config = _policy_config()
    config["regime_policy"]["regimes"]["not_defined"] = config["regime_policy"]["regimes"]["risk_on"]
    with pytest.raises(RegimePolicyError, match="configured aliases"):
        resolve_regime_policy_config(config)

    bad_config = _policy_config()
    bad_config["regime_policy"]["regimes"]["risk_on"]["optimizer_override"] = "unsupported"
    with pytest.raises(RegimePolicyError, match="must be one of"):
        resolve_regime_policy_config(bad_config)


def test_fallback_behavior_handles_undefined_unstable_and_skip_adaptation() -> None:
    labels = _labels([_UNDEFINED, _RISK_OFF, _RECOVERY])
    config = _policy_config()
    config["regime_policy"]["default"]["fallback_policy"] = "skip_adaptation"

    result = apply_regime_policy(
        labels,
        config=config,
        is_unstable_profile=True,
        eligible_for_downstream_decisioning=False,
    )

    assert result.decisions.loc[0, "fallback_policy_applied"] == "skip_adaptation"
    assert result.decisions.loc[0, "signal_scale"] == 1.0
    assert result.decisions.loc[1, "fallback_policy_applied"] == "reduce_exposure"
    assert result.decisions.loc[1, "gross_exposure_cap"] == 0.5
    assert result.decisions["policy_reason"].str.contains("fallback").all()


def test_confidence_gated_routing_works_with_and_without_confidence_frame() -> None:
    labels = _labels([_RISK_ON, _RISK_OFF, _RECOVERY], include_symbol=True)
    with_confidence = apply_regime_policy(labels, config=_policy_config(), confidence_frame=_confidence(labels))
    without_confidence = apply_regime_policy(labels, config=_policy_config())

    assert with_confidence.decisions["confidence_fallback_reason"].tolist() == [None, "low_confidence", "ambiguous"]
    assert with_confidence.decisions.loc[1, "fallback_policy_applied"] == "neutral"
    assert with_confidence.decisions.loc[1, "signal_scale"] == 0.0
    assert with_confidence.decisions.loc[2, "fallback_policy_applied"] == "reduce_exposure"
    assert without_confidence.decisions["confidence_score"].isna().all()
    assert without_confidence.decisions["matched_policy_key"].tolist() == ["risk_on", "risk_off", "default"]


def test_adaptive_comparison_is_deterministic_for_strategy_and_portfolio_surfaces() -> None:
    labels = _labels([_RISK_ON, _RISK_OFF, _RECOVERY], include_symbol=True)
    baseline = _baseline(labels)

    first = apply_regime_policy(
        labels,
        config=_policy_config(),
        baseline_frame=baseline,
        surface="strategy",
    )
    second = apply_regime_policy(
        labels,
        config=_policy_config(),
        baseline_frame=baseline,
        surface="strategy",
    )

    pd.testing.assert_frame_equal(first.decisions, second.decisions)
    pd.testing.assert_frame_equal(first.comparison, second.comparison)
    assert first.decisions.loc[0, "adaptive_return"] == pytest.approx(0.022)
    assert first.decisions.loc[1, "adaptive_return"] == pytest.approx(-0.005)
    assert first.decisions["adaptive_equity"].iloc[-1] == pytest.approx(second.decisions["adaptive_equity"].iloc[-1])

    portfolio_result = apply_regime_policy(
        labels,
        config=_policy_config(),
        baseline_frame=_baseline(labels, column="portfolio_return"),
        surface="portfolio",
    )
    assert portfolio_result.decisions.loc[1, "adaptive_return"] == pytest.approx(-0.006)


def test_policy_artifacts_are_written_and_loaded_with_relative_inventory(tmp_path: Path) -> None:
    labels = _labels([_RISK_ON, _RISK_OFF, _RECOVERY], include_symbol=True)
    result = apply_regime_policy(
        labels,
        config=_policy_config(),
        confidence_frame=_confidence(labels),
        baseline_frame=_baseline(labels),
        surface="strategy",
        calibration_profile="baseline",
    )

    manifest = write_regime_policy_artifacts(
        tmp_path,
        result,
        run_id="policy_case",
        source_regime_artifact_references={"manifest_path": "../regimes/manifest.json"},
        calibration_profile="baseline",
        sensitivity_profile="baseline",
        confidence_artifact_references={"manifest_path": "../regime_ml/regime_model_manifest.json"},
    )

    assert (tmp_path / REGIME_POLICY_DECISIONS_FILENAME).exists()
    assert (tmp_path / REGIME_POLICY_SUMMARY_FILENAME).exists()
    assert (tmp_path / ADAPTIVE_VS_STATIC_COMPARISON_FILENAME).exists()
    assert (tmp_path / ADAPTIVE_POLICY_MANIFEST_FILENAME).exists()

    loaded_decisions = load_regime_policy_decisions(tmp_path)
    loaded_summary = load_regime_policy_summary(tmp_path)
    loaded_comparison = load_adaptive_vs_static_comparison(tmp_path)
    loaded_manifest = load_adaptive_policy_manifest(tmp_path)

    assert len(loaded_decisions) == len(result.decisions)
    assert loaded_summary["taxonomy_version"] == result.taxonomy_version
    assert len(loaded_comparison) == 1
    assert loaded_manifest == manifest
    assert loaded_manifest["file_inventory"][REGIME_POLICY_DECISIONS_FILENAME]["path"] == REGIME_POLICY_DECISIONS_FILENAME
    assert "sha256" not in loaded_manifest["file_inventory"][ADAPTIVE_POLICY_MANIFEST_FILENAME]


def test_policy_inputs_are_not_mutated() -> None:
    labels = _labels([_RISK_ON, _RISK_OFF, _RECOVERY], include_symbol=True)
    confidence = _confidence(labels)
    baseline = _baseline(labels)
    original_labels = labels.copy(deep=True)
    original_confidence = confidence.copy(deep=True)
    original_baseline = baseline.copy(deep=True)
    config = _policy_config()
    original_config = pd.json_normalize(config, sep=".").to_dict(orient="records")[0]

    apply_regime_policy(
        labels,
        config=config,
        confidence_frame=confidence,
        baseline_frame=baseline,
        surface="strategy",
    )

    pd.testing.assert_frame_equal(labels, original_labels)
    pd.testing.assert_frame_equal(confidence, original_confidence)
    pd.testing.assert_frame_equal(baseline, original_baseline)
    assert pd.json_normalize(config, sep=".").to_dict(orient="records")[0] == original_config


def test_policy_preserves_taxonomy_and_rejects_invalid_contracts() -> None:
    labels = _labels([_RISK_ON, _RISK_OFF])
    result = apply_regime_policy(labels, config=_policy_config())

    assert result.decisions["regime_label"].tolist() == labels["regime_label"].tolist()

    bad_labels = labels.copy(deep=True)
    bad_labels.loc[0, "regime_label"] = "volatility=bad|trend=uptrend|drawdown_recovery=near_peak|stress=normal_stress"
    with pytest.raises(Exception):
        apply_regime_policy(bad_labels, config=_policy_config())
