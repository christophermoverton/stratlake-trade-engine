from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config.regime_aware_candidate_selection import (
    RegimeAwareCandidateSelectionConfig,
    RegimeAwareCandidateSelectionConfigError,
    apply_regime_aware_candidate_selection_overrides,
    load_regime_aware_candidate_selection_config,
)


def _write_config(tmp_path: Path, overrides: dict | None = None) -> Path:
    payload = {
        "selection_name": "test_regime_selection",
        "source_review_pack": "artifacts/regime_reviews/test_review",
        "source_candidate_universe": {"candidate_metrics_path": "fixtures/candidates.csv"},
        "regime_context": {
            "regime_source": "calibrated_taxonomy",
            "allow_gmm_confidence_overlay": True,
            "min_regime_confidence": 0.55,
            "transition_window_bars": 5,
        },
        "selection_categories": {
            "global_performer": {"enabled": True, "max_candidates": 2, "min_global_score": 0.60},
            "regime_specialist": {
                "enabled": True,
                "max_candidates_per_regime": 2,
                "min_regime_score": 0.65,
                "min_regime_observations": 20,
            },
            "transition_resilient": {"enabled": True, "max_candidates": 2, "min_transition_score": 0.55},
            "defensive_fallback": {"enabled": True, "max_candidates": 2, "max_correlation_to_selected": 0.70},
        },
        "redundancy": {
            "enabled": True,
            "max_pairwise_correlation": 0.85,
            "apply_within_category": True,
            "apply_across_categories": True,
        },
        "allocation_hints": {
            "write_category_weight_hints": True,
            "default_category_budget": {
                "global_performer": 0.45,
                "regime_specialist": 0.30,
                "transition_resilient": 0.15,
                "defensive_fallback": 0.10,
            },
        },
        "output_root": "artifacts/candidate_selection",
    }
    if overrides:
        payload.update(overrides)
    path = tmp_path / "regime_selection.yml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8", newline="\n")
    return path


def test_load_regime_aware_candidate_selection_config_valid_config(tmp_path: Path) -> None:
    config = load_regime_aware_candidate_selection_config(_write_config(tmp_path))

    assert isinstance(config, RegimeAwareCandidateSelectionConfig)
    assert config.selection_name == "test_regime_selection"
    assert config.selection_categories.global_performer.max_candidates == 2
    assert config.allocation_hints.default_category_budget["global_performer"] == 0.45


def test_regime_aware_candidate_selection_config_rejects_unknown_root_key(tmp_path: Path) -> None:
    path = _write_config(tmp_path, {"unexpected": True})

    with pytest.raises(RegimeAwareCandidateSelectionConfigError, match="unsupported keys"):
        load_regime_aware_candidate_selection_config(path)


def test_regime_aware_candidate_selection_config_rejects_invalid_category_threshold(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        {"selection_categories": {"global_performer": {"min_global_score": 1.2}}},
    )

    with pytest.raises(RegimeAwareCandidateSelectionConfigError, match="min_global_score"):
        load_regime_aware_candidate_selection_config(path)


def test_regime_aware_candidate_selection_config_accepts_disabled_category(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        {"selection_categories": {"defensive_fallback": {"enabled": False, "max_candidates": 0}}},
    )

    config = load_regime_aware_candidate_selection_config(path)

    assert config.selection_categories.defensive_fallback.enabled is False
    assert config.selection_categories.defensive_fallback.max_candidates == 0


def test_regime_aware_candidate_selection_config_rejects_invalid_redundancy(tmp_path: Path) -> None:
    path = _write_config(tmp_path, {"redundancy": {"max_pairwise_correlation": -0.1}})

    with pytest.raises(RegimeAwareCandidateSelectionConfigError, match="max_pairwise_correlation"):
        load_regime_aware_candidate_selection_config(path)


def test_regime_aware_candidate_selection_config_validates_allocation_budget(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        {
            "allocation_hints": {
                "default_category_budget": {
                    "global_performer": 0.50,
                    "regime_specialist": 0.30,
                    "transition_resilient": 0.15,
                    "defensive_fallback": 0.10,
                }
            }
        },
    )

    with pytest.raises(RegimeAwareCandidateSelectionConfigError, match="sum to 1.0"):
        load_regime_aware_candidate_selection_config(path)


def test_regime_aware_candidate_selection_override_handling(tmp_path: Path) -> None:
    config = load_regime_aware_candidate_selection_config(_write_config(tmp_path))

    updated = apply_regime_aware_candidate_selection_overrides(
        config,
        source_review_pack="new/review",
        candidate_metrics_path="new/candidates.csv",
        output_root="new/output",
    )

    assert updated.source_review_pack == "new/review"
    assert updated.source_candidate_universe.candidate_metrics_path == "new/candidates.csv"
    assert updated.output_root == "new/output"
