from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config.regime_review_pack import (
    RegimeReviewPackConfig,
    RegimeReviewPackConfigError,
    load_regime_review_pack_config,
)


def _write_config(tmp_path: Path, overrides: dict | None = None) -> Path:
    payload = {
        "review_name": "test_review",
        "benchmark_path": "artifacts/regime_benchmarks/test_run",
        "promotion_gates_path": "artifacts/regime_benchmarks/test_run/promotion_gates",
        "output_root": "artifacts/regime_reviews",
        "ranking": {
            "decision_order": ["accepted", "accepted_with_warnings", "needs_review", "rejected"],
            "metric_priority": ["adaptive_vs_static_sharpe_delta", "transition_rate"],
            "tie_breakers": ["variant_name"],
        },
    }
    if overrides:
        payload.update(overrides)
    path = tmp_path / "review.yml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8", newline="\n")
    return path


def test_load_regime_review_pack_config_valid_config(tmp_path: Path) -> None:
    config = load_regime_review_pack_config(_write_config(tmp_path))

    assert isinstance(config, RegimeReviewPackConfig)
    assert config.review_name == "test_review"
    assert config.ranking.decision_order == ("accepted", "accepted_with_warnings", "needs_review", "rejected")
    assert config.report.write_markdown is True


def test_regime_review_pack_config_rejects_unknown_root_key(tmp_path: Path) -> None:
    path = _write_config(tmp_path, {"unexpected": True})

    with pytest.raises(RegimeReviewPackConfigError, match="unsupported keys"):
        load_regime_review_pack_config(path)


def test_regime_review_pack_config_rejects_invalid_decision_order(tmp_path: Path) -> None:
    path = _write_config(tmp_path, {"ranking": {"decision_order": ["accepted", "rejected"]}})

    with pytest.raises(RegimeReviewPackConfigError, match="decision_order"):
        load_regime_review_pack_config(path)


def test_regime_review_pack_config_rejects_invalid_metric_priority(tmp_path: Path) -> None:
    path = _write_config(tmp_path, {"ranking": {"metric_priority": ["made_up_metric"]}})

    with pytest.raises(RegimeReviewPackConfigError, match="unsupported metrics"):
        load_regime_review_pack_config(path)


def test_regime_review_pack_config_rejects_invalid_report_option(tmp_path: Path) -> None:
    path = _write_config(tmp_path, {"report": {"write_markdown": "yes"}})

    with pytest.raises(RegimeReviewPackConfigError, match="must be boolean"):
        load_regime_review_pack_config(path)


def test_regime_review_pack_config_defaults_report_options(tmp_path: Path) -> None:
    config = load_regime_review_pack_config(_write_config(tmp_path, {"report": {}}))

    assert config.report.include_rejected_candidates is True
    assert config.report.include_warning_summary is True
    assert config.report.include_evidence_index is True

