from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.review import ReviewConfigError, load_review_config, resolve_review_config


def test_resolve_review_config_loads_repository_defaults() -> None:
    config = resolve_review_config()

    assert config.to_dict() == {
        "filters": {
            "alpha_name": None,
            "dataset": None,
            "portfolio_name": None,
            "run_types": ["alpha_evaluation", "strategy", "portfolio"],
            "strategy_name": None,
            "timeframe": None,
            "top_k_per_type": None,
        },
        "output": {
            "emit_plots": True,
            "path": None,
        },
        "promotion_gates": None,
        "ranking": {
            "alpha_evaluation_primary_metric": "ic_ir",
            "alpha_evaluation_secondary_metric": "mean_ic",
            "portfolio_primary_metric": "sharpe_ratio",
            "portfolio_secondary_metric": "total_return",
            "strategy_primary_metric": "sharpe_ratio",
            "strategy_secondary_metric": "total_return",
        },
    }


def test_resolve_review_config_applies_precedence_deterministically(tmp_path: Path) -> None:
    config_path = tmp_path / "review.yml"
    config_path.write_text(
        """
review:
  filters:
    run_types: [strategy]
    timeframe: 1D
  ranking:
    strategy_primary_metric: total_return
  output:
    emit_plots: false
""".strip(),
        encoding="utf-8",
    )

    config = resolve_review_config(
        load_review_config(config_path).to_dict(),
        cli_overrides={
            "filters": {"top_k_per_type": 2},
            "ranking": {"strategy_primary_metric": "sharpe_ratio"},
            "output": {"path": "artifacts/custom"},
        },
    )

    assert config.filters.run_types == ["strategy"]
    assert config.filters.timeframe == "1D"
    assert config.filters.top_k_per_type == 2
    assert config.ranking.strategy_primary_metric == "sharpe_ratio"
    assert config.output.emit_plots is False
    assert config.output.path == "artifacts/custom"


def test_resolve_review_config_validates_fields_explicitly() -> None:
    with pytest.raises(ReviewConfigError, match="filters.run_types"):
        resolve_review_config({"filters": {"run_types": ["mystery"]}})

    with pytest.raises(ReviewConfigError, match="ranking.strategy_primary_metric"):
        resolve_review_config({"ranking": {"strategy_primary_metric": ""}})

    with pytest.raises(ReviewConfigError, match="output.emit_plots"):
        resolve_review_config({"output": {"emit_plots": "yes"}})

    with pytest.raises(ReviewConfigError, match="unsupported keys"):
        resolve_review_config({"review": {"filters": {}, "mystery": {}}})
