from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config.regime_promotion_gates import (
    RegimePromotionGateConfig,
    RegimePromotionGateConfigError,
    load_regime_promotion_gate_config,
)


def _write_config(tmp_path: Path, overrides: dict | None = None) -> Path:
    payload = {
        "gate_name": "test_regime_gates",
        "decision_policy": "strict_with_warnings",
        "missing_metric_policy": {
            "required_metric_missing": "fail",
            "optional_metric_missing": "warn",
            "not_applicable_metric": "ignore",
        },
        "confidence": {
            "enabled": True,
            "required_for_variants": ["gmm_classifier", "policy_optimized"],
            "min_mean_confidence": 0.55,
        },
        "entropy": {"enabled": False, "max_mean_entropy": 1.1},
        "stability": {"enabled": True, "min_avg_regime_duration": 3},
    }
    if overrides:
        payload.update(overrides)
    path = tmp_path / "gates.yml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8", newline="\n")
    return path


def test_load_regime_promotion_gate_config_valid_config(tmp_path: Path) -> None:
    config = load_regime_promotion_gate_config(_write_config(tmp_path))

    assert isinstance(config, RegimePromotionGateConfig)
    assert config.gate_name == "test_regime_gates"
    assert config.category("confidence").thresholds["min_mean_confidence"] == 0.55
    assert config.category("confidence").required_for_variants == (
        "gmm_classifier",
        "policy_optimized",
    )
    assert config.category("entropy").enabled is False


def test_regime_promotion_gate_config_rejects_unknown_root_key(tmp_path: Path) -> None:
    path = _write_config(tmp_path, {"surprise": True})

    with pytest.raises(RegimePromotionGateConfigError, match="unsupported keys"):
        load_regime_promotion_gate_config(path)


def test_regime_promotion_gate_config_rejects_invalid_threshold_type(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        {"confidence": {"enabled": True, "min_mean_confidence": "high"}},
    )

    with pytest.raises(RegimePromotionGateConfigError, match="must be numeric"):
        load_regime_promotion_gate_config(path)


def test_regime_promotion_gate_config_rejects_invalid_missing_metric_policy(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        {"missing_metric_policy": {"required_metric_missing": "explode"}},
    )

    with pytest.raises(RegimePromotionGateConfigError, match="must be one of"):
        load_regime_promotion_gate_config(path)


def test_regime_promotion_gate_config_rejects_unknown_category_key(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        {"stability": {"enabled": True, "min_avg_regime_duration": 3, "bad": 1}},
    )

    with pytest.raises(RegimePromotionGateConfigError, match="stability"):
        load_regime_promotion_gate_config(path)


def test_regime_promotion_gate_config_rejects_unknown_required_variant(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        {
            "confidence": {
                "enabled": True,
                "required_for_variants": ["unknown_variant"],
                "min_mean_confidence": 0.55,
            }
        },
    )

    with pytest.raises(RegimePromotionGateConfigError, match="unsupported variants"):
        load_regime_promotion_gate_config(path)
