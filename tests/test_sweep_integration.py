from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.cli.run_research_campaign import run_research_campaign
from src.config.research_campaign import (
    compute_scenario_expansion_size,
    resolve_research_campaign_config,
)


def _write_feature_fixture(root: Path, dataset: str = "features_daily") -> None:
    dataset_root = root / "curated" / dataset / "symbol=AAA" / "year=2025"
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "part-0.parquet").write_text("fixture", encoding="utf-8")


def _build_sweep_config(
    *,
    tmp_path: Path,
    alpha_catalog: Path,
    strategy_config: Path,
) -> object:
    return resolve_research_campaign_config(
        {
            "targets": {
                "alpha_names": ["alpha_one"],
                "alpha_catalog_path": alpha_catalog.as_posix(),
                "strategy_config_path": strategy_config.as_posix(),
            },
            "dataset_selection": {
                "dataset": "features_daily",
                "timeframe": "1D",
                "evaluation_horizon": 5,
            },
            "comparison": {"enabled": False},
            "candidate_selection": {"enabled": False},
            "portfolio": {"enabled": False},
            "outputs": {
                "campaign_artifacts_root": (tmp_path / "campaign_artifacts").as_posix(),
            },
            "scenarios": {
                "enabled": True,
                "matrix": [
                    {
                        "name": "timeframe",
                        "path": "dataset_selection.timeframe",
                        "values": ["1D", "4H"],
                    }
                ],
                "include": [
                    {
                        "scenario_id": "include_review",
                        "overrides": {
                            "dataset_selection": {"timeframe": "30M"},
                        },
                    }
                ],
                "max_scenarios": 10,
                "max_values_per_axis": 5,
            },
        }
    )


def test_sweep_orchestration_manifest_artifacts_are_relative_and_consistent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    alpha_catalog = tmp_path / "alphas.yml"
    strategy_config = tmp_path / "strategies.yml"
    features_root = tmp_path / "features_root"
    alpha_catalog.write_text(
        """
alpha_one:
  alpha_name: alpha_one
  dataset: features_daily
  target_column: target_ret_1d
  feature_columns: [feature_ret_1d]
  model_type: cross_sectional_linear
  model_params: {}
  alpha_horizon: 1
""".strip(),
        encoding="utf-8",
    )
    strategy_config.write_text("{}", encoding="utf-8")
    _write_feature_fixture(features_root)
    monkeypatch.setenv("FEATURES_ROOT", features_root.as_posix())

    config = _build_sweep_config(
        tmp_path=tmp_path,
        alpha_catalog=alpha_catalog,
        strategy_config=strategy_config,
    )

    monkeypatch.setattr(
        "src.cli.run_research_campaign.run_alpha_cli.run_cli",
        lambda argv: SimpleNamespace(
            alpha_name="alpha_one",
            run_id=f"alpha_run_{argv[argv.index('--timeframe') + 1] if '--timeframe' in argv else 'include'}",
            artifact_dir=tmp_path / "alpha" / (argv[argv.index("--timeframe") + 1] if "--timeframe" in argv else "include"),
            evaluation=SimpleNamespace(
                evaluation_result=SimpleNamespace(summary={"mean_ic": 0.1, "ic_ir": 0.2}),
                manifest={},
            ),
        ),
    )
    monkeypatch.setattr(
        "src.cli.run_research_campaign.compare_research_cli.run_cli",
        lambda _argv: SimpleNamespace(review_id="review_run", entries=[]),
    )

    result = run_research_campaign(config)
    orchestration_manifest = json.loads(result.orchestration_manifest_path.read_text(encoding="utf-8"))

    artifact_files = list(orchestration_manifest["artifact_files"])
    assert artifact_files
    assert set(artifact_files) == set(orchestration_manifest["artifacts"])

    for artifact_path in artifact_files:
        assert "\\" not in artifact_path
        assert not Path(artifact_path).is_absolute()
        assert (result.orchestration_artifact_dir / artifact_path).exists()

    for artifact_key, metadata in orchestration_manifest["artifacts"].items():
        path_value = metadata.get("path")
        assert path_value == artifact_key
        assert isinstance(path_value, str)
        assert "\\" not in path_value
        assert not Path(path_value).is_absolute()


def test_sweep_orchestration_matrix_order_is_deterministic_when_metrics_tie(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    alpha_catalog = tmp_path / "alphas.yml"
    strategy_config = tmp_path / "strategies.yml"
    features_root = tmp_path / "features_root"
    alpha_catalog.write_text(
        """
alpha_one:
  alpha_name: alpha_one
  dataset: features_daily
  target_column: target_ret_1d
  feature_columns: [feature_ret_1d]
  model_type: cross_sectional_linear
  model_params: {}
  alpha_horizon: 1
""".strip(),
        encoding="utf-8",
    )
    strategy_config.write_text("{}", encoding="utf-8")
    _write_feature_fixture(features_root)
    monkeypatch.setenv("FEATURES_ROOT", features_root.as_posix())

    config = _build_sweep_config(
        tmp_path=tmp_path,
        alpha_catalog=alpha_catalog,
        strategy_config=strategy_config,
    )

    monkeypatch.setattr(
        "src.cli.run_research_campaign.run_alpha_cli.run_cli",
        lambda argv: SimpleNamespace(
            alpha_name="alpha_one",
            run_id=f"alpha_run_{argv[argv.index('--timeframe') + 1] if '--timeframe' in argv else 'include'}",
            artifact_dir=tmp_path / "alpha" / (argv[argv.index("--timeframe") + 1] if "--timeframe" in argv else "include"),
            evaluation=SimpleNamespace(
                evaluation_result=SimpleNamespace(summary={"mean_ic": 0.1, "ic_ir": 0.2}),
                manifest={},
            ),
        ),
    )
    monkeypatch.setattr(
        "src.cli.run_research_campaign.compare_research_cli.run_cli",
        lambda _argv: SimpleNamespace(review_id="review_run", entries=[]),
    )

    result = run_research_campaign(config)
    matrix_summary = json.loads(result.scenario_matrix_summary_path.read_text(encoding="utf-8"))

    actual_order = [row["scenario_id"] for row in matrix_summary["leaderboard"]]
    assert actual_order == sorted(actual_order)
    assert [row["rank"] for row in matrix_summary["leaderboard"]] == [1, 2, 3]
    assert all(row["ranking_metric"] == "alpha_ic_ir_max" for row in matrix_summary["leaderboard"])


def test_sweep_orchestration_reuses_all_then_reruns_only_tampered_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    alpha_catalog = tmp_path / "alphas.yml"
    strategy_config = tmp_path / "strategies.yml"
    features_root = tmp_path / "features_root"
    alpha_catalog.write_text(
        """
alpha_one:
  alpha_name: alpha_one
  dataset: features_daily
  target_column: target_ret_1d
  feature_columns: [feature_ret_1d]
  model_type: cross_sectional_linear
  model_params: {}
  alpha_horizon: 1
""".strip(),
        encoding="utf-8",
    )
    strategy_config.write_text("{}", encoding="utf-8")
    _write_feature_fixture(features_root)
    monkeypatch.setenv("FEATURES_ROOT", features_root.as_posix())

    config = _build_sweep_config(
        tmp_path=tmp_path,
        alpha_catalog=alpha_catalog,
        strategy_config=strategy_config,
    )

    first_alpha_calls: list[list[str]] = []
    second_alpha_calls: list[list[str]] = []

    monkeypatch.setattr(
        "src.cli.run_research_campaign.run_alpha_cli.run_cli",
        lambda argv: first_alpha_calls.append(list(argv))
        or SimpleNamespace(
            alpha_name="alpha_one",
            run_id=f"alpha_run_{len(first_alpha_calls):02d}",
            artifact_dir=tmp_path / "alpha" / f"{len(first_alpha_calls):02d}",
            evaluation=SimpleNamespace(
                evaluation_result=SimpleNamespace(summary={"mean_ic": 0.1, "ic_ir": 0.2}),
                manifest={},
            ),
        ),
    )
    monkeypatch.setattr(
        "src.cli.run_research_campaign.compare_research_cli.run_cli",
        lambda _argv: SimpleNamespace(review_id="review_run", entries=[]),
    )

    first_result = run_research_campaign(config)
    assert len(first_alpha_calls) == len(first_result.scenario_results)

    # Baseline rerun should fully reuse scenario checkpoints.
    monkeypatch.setattr(
        "src.cli.run_research_campaign.run_alpha_cli.run_cli",
        lambda argv: second_alpha_calls.append(list(argv))
        or SimpleNamespace(
            alpha_name="alpha_one",
            run_id=f"alpha_run_resume_{len(second_alpha_calls):02d}",
            artifact_dir=tmp_path / "alpha" / f"resume_{len(second_alpha_calls):02d}",
            evaluation=SimpleNamespace(
                evaluation_result=SimpleNamespace(summary={"mean_ic": 0.1, "ic_ir": 0.2}),
                manifest={},
            ),
        ),
    )
    second_result = run_research_campaign(config)
    assert second_alpha_calls == []

    # Tamper one scenario fingerprint provenance and confirm only that scenario reruns.
    tampered = first_result.scenario_results[0]
    checkpoint = json.loads(tampered.result.campaign_checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["scenario"]["fingerprint"] = "tampered_fingerprint"
    tampered.result.campaign_checkpoint_path.write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    third_alpha_calls: list[list[str]] = []
    monkeypatch.setattr(
        "src.cli.run_research_campaign.run_alpha_cli.run_cli",
        lambda argv: third_alpha_calls.append(list(argv))
        or SimpleNamespace(
            alpha_name="alpha_one",
            run_id=f"alpha_run_tampered_{len(third_alpha_calls):02d}",
            artifact_dir=tmp_path / "alpha" / f"tampered_{len(third_alpha_calls):02d}",
            evaluation=SimpleNamespace(
                evaluation_result=SimpleNamespace(summary={"mean_ic": 0.1, "ic_ir": 0.2}),
                manifest={},
            ),
        ),
    )

    third_result = run_research_campaign(config)
    assert len(third_alpha_calls) == 1

    first_run_ids = {
        scenario.scenario_id: scenario.result.campaign_run_id
        for scenario in first_result.scenario_results
    }
    third_run_ids = {
        scenario.scenario_id: scenario.result.campaign_run_id
        for scenario in third_result.scenario_results
    }
    assert third_run_ids == first_run_ids

    state_by_scenario = {
        scenario.scenario_id: scenario.result.campaign_checkpoint["stage_states"]
        for scenario in third_result.scenario_results
    }
    assert state_by_scenario[tampered.scenario_id]["research"] == "completed"
    assert state_by_scenario[tampered.scenario_id]["review"] == "completed"
    for scenario_id, stage_states in state_by_scenario.items():
        if scenario_id == tampered.scenario_id:
            continue
        assert stage_states["preflight"] == "reused"
        assert stage_states["research"] == "reused"
        assert stage_states["review"] == "reused"


def test_sweep_expansion_preflight_matches_size_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    alpha_catalog = tmp_path / "alphas.yml"
    strategy_config = tmp_path / "strategies.yml"
    features_root = tmp_path / "features_root"
    alpha_catalog.write_text(
        """
alpha_one:
  alpha_name: alpha_one
  dataset: features_daily
  target_column: target_ret_1d
  feature_columns: [feature_ret_1d]
  model_type: cross_sectional_linear
  model_params: {}
  alpha_horizon: 1
""".strip(),
        encoding="utf-8",
    )
    strategy_config.write_text("{}", encoding="utf-8")
    _write_feature_fixture(features_root)
    monkeypatch.setenv("FEATURES_ROOT", features_root.as_posix())

    config = _build_sweep_config(
        tmp_path=tmp_path,
        alpha_catalog=alpha_catalog,
        strategy_config=strategy_config,
    )
    expected_size = compute_scenario_expansion_size(config)

    monkeypatch.setattr(
        "src.cli.run_research_campaign.run_alpha_cli.run_cli",
        lambda _argv: SimpleNamespace(
            alpha_name="alpha_one",
            run_id="alpha_run",
            artifact_dir=tmp_path / "alpha",
            evaluation=SimpleNamespace(
                evaluation_result=SimpleNamespace(summary={"mean_ic": 0.1, "ic_ir": 0.2}),
                manifest={},
            ),
        ),
    )
    monkeypatch.setattr(
        "src.cli.run_research_campaign.compare_research_cli.run_cli",
        lambda _argv: SimpleNamespace(review_id="review_run", entries=[]),
    )

    result = run_research_campaign(config)

    assert result.expansion_preflight_path.exists()
    payload = json.loads(result.expansion_preflight_path.read_text(encoding="utf-8"))
    assert payload == result.expansion_preflight
    assert payload["expansion"]["matrix_axis_count"] == expected_size["matrix_axis_count"]
    assert payload["expansion"]["matrix_combination_count"] == expected_size["matrix_combination_count"]
    assert payload["expansion"]["include_count"] == expected_size["include_count"]
    assert payload["expansion"]["total_scenario_count"] == expected_size["total_scenario_count"]
    assert payload["limits"]["configured_max_scenarios"] == expected_size["configured_max_scenarios"]
    assert payload["limits"]["effective_max_scenarios"] == expected_size["effective_max_scenarios"]
    assert payload["limits"]["max_values_per_axis"] == expected_size["max_values_per_axis"]