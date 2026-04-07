from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.cli.run_candidate_selection import parse_args, resolve_cli_config, run_cli
from src.research.candidate_review.review import CandidateReviewArtifacts


def _pipeline_result_payload(root: Path) -> dict[str, object]:
    run_dir = root / "candidate_selection_test"
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "candidate_universe.csv",
        "selected_candidates.csv",
        "rejected_candidates.csv",
        "eligibility_filter_results.csv",
        "correlation_matrix.csv",
        "allocation_weights.csv",
        "selection_summary.json",
        "manifest.json",
    ):
        target = run_dir / name
        if name.endswith(".json"):
            target.write_text("{}", encoding="utf-8")
        else:
            target.write_text("", encoding="utf-8")

    return {
        "run_id": run_dir.name,
        "universe_count": 3,
        "eligible_count": 2,
        "rejected_count": 1,
        "selected_count": 2,
        "primary_metric": "ic_ir",
        "filters": {"dataset": "features_daily"},
        "thresholds": {"min_ic_ir": 0.1},
        "redundancy_thresholds": {"max_pairwise_correlation": 0.8},
        "stage_execution": {
            "universe": True,
            "eligibility": False,
            "redundancy": False,
            "allocation": False,
        },
        "redundancy_summary": {"pruned_by_redundancy": 0},
        "universe_csv": str(run_dir / "candidate_universe.csv"),
        "selected_csv": str(run_dir / "selected_candidates.csv"),
        "rejected_csv": str(run_dir / "rejected_candidates.csv"),
        "eligibility_csv": str(run_dir / "eligibility_filter_results.csv"),
        "correlation_csv": str(run_dir / "correlation_matrix.csv"),
        "allocation_csv": str(run_dir / "allocation_weights.csv"),
        "summary_json": str(run_dir / "selection_summary.json"),
        "manifest_json": str(run_dir / "manifest.json"),
        "artifact_dir": str(run_dir),
        "allocation_constraints": {},
        "allocation_summary": {
            "allocation_enabled": False,
            "allocation_method": "equal_weight",
            "allocated_candidates": 0,
        },
        "registry_entry": None,
    }


def test_parse_args_accepts_candidate_selection_surface() -> None:
    args = parse_args(
        [
            "--config",
            "configs/candidate.yml",
            "--dataset",
            "features_daily",
            "--timeframe",
            "1D",
            "--evaluation-horizon",
            "5",
            "--mapping-name",
            "rank_long_short",
            "--min-ic",
            "0.01",
            "--min-ic-ir",
            "0.2",
            "--max-pairwise-correlation",
            "0.85",
            "--allocation-method",
            "equal_weight",
            "--max-weight-per-candidate",
            "0.2",
            "--skip-redundancy",
            "--skip-allocation",
            "--enable-review",
            "--portfolio-run-id",
            "portfolio_abc",
            "--strict",
        ]
    )

    assert args.config == "configs/candidate.yml"
    assert args.dataset == "features_daily"
    assert args.timeframe == "1D"
    assert args.evaluation_horizon == 5
    assert args.mapping_name == "rank_long_short"
    assert args.min_mean_ic == pytest.approx(0.01)
    assert args.min_ic_ir == pytest.approx(0.2)
    assert args.max_pairwise_correlation == pytest.approx(0.85)
    assert args.allocation_method == "equal_weight"
    assert args.max_weight_per_candidate == pytest.approx(0.2)
    assert args.skip_redundancy is True
    assert args.skip_allocation is True
    assert args.enable_review is True
    assert args.portfolio_run_id == "portfolio_abc"
    assert args.strict is True


def test_resolve_cli_config_precedence_defaults_config_then_cli(tmp_path: Path) -> None:
    config_path = tmp_path / "candidate_config.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "candidate_selection": {
                    "dataset": "bars_daily",
                    "timeframe": "1D",
                    "evaluation_horizon": 5,
                    "eligibility": {
                        "min_ic": 0.01,
                        "min_ic_ir": 0.2,
                    },
                    "allocation": {
                        "method": "equal_weight",
                        "max_weight": 0.3,
                    },
                    "execution": {
                        "strict_mode": True,
                        "skip_redundancy": True,
                    },
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    args = parse_args(
        [
            "--config",
            str(config_path),
            "--dataset",
            "features_daily",
            "--max-weight-per-candidate",
            "0.2",
            "--skip-allocation",
        ]
    )
    resolved = resolve_cli_config(args)

    assert resolved["dataset"] == "features_daily"
    assert resolved["timeframe"] == "1D"
    assert resolved["evaluation_horizon"] == 5
    assert resolved["min_mean_ic"] == pytest.approx(0.01)
    assert resolved["min_ic_ir"] == pytest.approx(0.2)
    assert resolved["allocation_method"] == "equal_weight"
    assert resolved["max_weight_per_candidate"] == pytest.approx(0.2)
    assert resolved["strict_mode"] is True
    assert resolved["skip_redundancy"] is True
    assert resolved["skip_allocation"] is True


def test_run_cli_passes_stage_toggles_to_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_run_candidate_selection(**kwargs):
        captured.update(kwargs)
        return _pipeline_result_payload(tmp_path)

    monkeypatch.setattr("src.cli.run_candidate_selection.run_candidate_selection", _fake_run_candidate_selection)

    result = run_cli(["--skip-eligibility", "--skip-redundancy", "--skip-allocation"])

    assert captured["skip_eligibility"] is True
    assert captured["skip_redundancy"] is True
    assert captured["allocation_enabled"] is False
    assert result.stage_execution["eligibility"] is False
    assert result.stage_execution["redundancy"] is False
    assert result.stage_execution["allocation"] is False


def test_run_cli_candidate_selection_run_id_uses_existing_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "candidate_root"
    run_dir = root / "candidate_selection_existing"
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": "candidate_selection_existing",
        "allocation_applied": False,
        "config_snapshot": {"filters": {"dataset": "features_daily"}, "primary_metric": "ic_ir"},
        "stage_execution": {"universe": True, "eligibility": True, "redundancy": True, "allocation": False},
    }
    summary = {
        "run_id": "candidate_selection_existing",
        "total_candidates": 4,
        "eligible_candidates": 4,
        "rejected_candidates": 0,
        "selected_candidates": 2,
        "thresholds": {},
        "redundancy_thresholds": {},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "selection_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    for name in (
        "candidate_universe.csv",
        "selected_candidates.csv",
        "rejected_candidates.csv",
        "eligibility_filter_results.csv",
        "correlation_matrix.csv",
        "allocation_weights.csv",
    ):
        (run_dir / name).write_text("", encoding="utf-8")

    def _fail_if_called(**_: object):
        raise AssertionError("pipeline should not execute when using existing candidate-selection run")

    monkeypatch.setattr("src.cli.run_candidate_selection.run_candidate_selection", _fail_if_called)

    result = run_cli(["--output-path", str(root), "--candidate-selection-run-id", "candidate_selection_existing"])

    assert result.run_id == "candidate_selection_existing"
    assert result.universe_count == 4
    assert result.selected_count == 2


def test_run_cli_enable_review_calls_review_module(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _fake_run_candidate_selection(**kwargs):
        return _pipeline_result_payload(tmp_path)

    portfolio_dir = tmp_path / "portfolio_run"
    portfolio_dir.mkdir(parents=True, exist_ok=True)

    def _fake_review_candidate_selection(**kwargs):
        assert Path(kwargs["portfolio_artifact_dir"]) == portfolio_dir
        candidate_dir = Path(kwargs["candidate_selection_artifact_dir"])
        review_dir = candidate_dir / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        return CandidateReviewArtifacts(
            candidate_selection_run_id="candidate_selection_test",
            portfolio_run_id="portfolio_test",
            review_dir=review_dir,
            candidate_decisions_csv=review_dir / "candidate_decisions.csv",
            candidate_summary_csv=review_dir / "candidate_summary.csv",
            candidate_contributions_csv=review_dir / "candidate_contributions.csv",
            diversification_summary_json=review_dir / "diversification_summary.json",
            candidate_review_summary_json=review_dir / "candidate_review_summary.json",
            candidate_review_report_md=review_dir / "candidate_review_report.md",
            manifest_json=review_dir / "manifest.json",
            total_candidates=3,
            selected_candidates=2,
            rejected_candidates=1,
        )

    monkeypatch.setattr("src.cli.run_candidate_selection.run_candidate_selection", _fake_run_candidate_selection)
    monkeypatch.setattr("src.cli.run_candidate_selection.review_candidate_selection", _fake_review_candidate_selection)

    result = run_cli(["--enable-review", "--portfolio-path", str(portfolio_dir)])

    assert result.review_artifacts is not None
    assert result.review_artifacts.review_dir.name == "review"
