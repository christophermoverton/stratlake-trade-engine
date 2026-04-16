from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.cli.run_pipeline import load_pipeline_spec, run_cli


REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_MATRIX_PIPELINE_PATH = REPO_ROOT / "configs" / "pipelines" / "scenario_matrix_pipeline.yml"


def _write_features_daily_dataset(root: Path) -> None:
    timestamps = pd.date_range("2025-01-01", periods=30, freq="D", tz="UTC")
    rows: list[dict[str, object]] = []
    symbols = ["AAA", "BBB", "CCC"]
    for symbol_index, symbol in enumerate(symbols):
        base = float(symbol_index + 1)
        for ts_index, ts_utc in enumerate(timestamps):
            close = 100.0 + base * 10.0 + ts_index * (base + 0.75)
            rows.append(
                {
                    "symbol": symbol,
                    "ts_utc": ts_utc,
                    "timeframe": "1D",
                    "date": ts_utc.strftime("%Y-%m-%d"),
                    "close": close,
                    "feature_ret_1d": base * 0.010 + ts_index * 0.0005,
                    "feature_ret_5d": base * 0.015 + ts_index * 0.0007,
                    "feature_ret_20d": base * 0.020 + ts_index * 0.0009,
                    "target_ret_1d": base * 0.008 + ts_index * 0.0004,
                    "target_ret_5d": base * 0.012 + ts_index * 0.0006,
                }
            )

    frame = pd.DataFrame(rows).sort_values(["symbol", "ts_utc"], kind="stable").reset_index(drop=True)
    for column in ("symbol", "timeframe", "date"):
        frame[column] = frame[column].astype("string")
    for symbol in symbols:
        symbol_frame = frame.loc[frame["symbol"].eq(symbol)].copy(deep=True)
        dataset_dir = root / "data" / "curated" / "features_daily" / f"symbol={symbol}" / "year=2025"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        symbol_frame.to_parquet(dataset_dir / "part-0.parquet", index=False)


def _artifact_bytes(root: Path, *, exclude_names: set[str] | None = None) -> dict[str, bytes]:
    excluded = set() if exclude_names is None else set(exclude_names)
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in excluded
    }


def test_scenario_matrix_pipeline_config_declares_expected_workflow() -> None:
    spec = load_pipeline_spec(SCENARIO_MATRIX_PIPELINE_PATH)

    assert spec.pipeline_id == "scenario_matrix_pipeline"
    assert [step.id for step in spec.steps] == [
        "alpha_rank_composite",
        "alpha_xgb",
        "alpha_lgbm",
        "strategy_momentum",
        "strategy_mean_reversion",
        "compare_alpha",
        "compare_strategies",
        "candidate_selection",
        "construct_portfolio",
    ]
    assert [step.module for step in spec.steps[:5]] == [
        "src.cli.run_alpha",
        "src.cli.run_alpha",
        "src.cli.run_alpha",
        "src.cli.run_strategy",
        "src.cli.run_strategy",
    ]
    assert spec.steps[5].module == "src.cli.compare_alpha"
    assert spec.steps[6].module == "src.cli.compare_strategies"
    assert spec.steps[7].module == "src.cli.run_candidate_selection"
    assert spec.steps[8].module == "src.cli.run_portfolio"
    assert spec.steps[5].depends_on == ("alpha_rank_composite", "alpha_xgb", "alpha_lgbm")
    assert spec.steps[6].depends_on == ("strategy_momentum", "strategy_mean_reversion")
    assert spec.steps[7].depends_on == ("compare_alpha",)
    assert spec.steps[8].depends_on == ("candidate_selection",)


def test_pipeline_runner_executes_scenario_matrix_flow_with_deterministic_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_features_daily_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)

    alpha_root = tmp_path / "outputs" / "alpha"
    comparison_root = tmp_path / "outputs" / "comparisons"
    candidate_root = tmp_path / "outputs" / "candidate_selection"
    portfolio_root = tmp_path / "outputs" / "portfolios"
    pipeline_config_path = tmp_path / "scenario_matrix_pipeline.yml"
    pipeline_config_path.write_text(
        f"""
id: scenario_matrix_pipeline_smoke
steps:
  - id: alpha_ret_1d
    adapter: python_module
    module: src.cli.run_alpha
    argv:
      - --config
      - {json.dumps((REPO_ROOT / "configs" / "alphas.yml").as_posix())}
      - --alpha-name
      - cs_linear_ret_1d
      - --mode
      - full
      - --artifacts-root
      - {json.dumps(alpha_root.as_posix())}
      - --start
      - "2025-01-20"
      - --end
      - "2025-01-31"
  - id: alpha_ret_1d_quantile
    adapter: python_module
    module: src.cli.run_alpha
    argv:
      - --config
      - {json.dumps((REPO_ROOT / "configs" / "alphas.yml").as_posix())}
      - --alpha-name
      - cs_linear_ret_1d
      - --mode
      - full
      - --artifacts-root
      - {json.dumps(alpha_root.as_posix())}
      - --signal-policy
      - top_bottom_quantile
      - --signal-quantile
      - "0.3"
      - --start
      - "2025-01-20"
      - --end
      - "2025-01-31"
  - id: strategy_momentum
    adapter: python_module
    module: src.cli.run_strategy
    argv:
      - --strategy
      - momentum_v1
      - --start
      - "2025-01-01"
      - --end
      - "2025-01-31"
  - id: compare_alpha
    depends_on:
      - alpha_ret_1d
      - alpha_ret_1d_quantile
    adapter: python_module
    module: src.cli.compare_alpha
    argv:
      - --from-registry
      - --artifacts-root
      - {json.dumps(alpha_root.as_posix())}
      - --view
      - combined
      - --metric
      - ic_ir
      - --output-path
      - {json.dumps((comparison_root / "alpha").as_posix())}
  - id: candidate_selection
    depends_on:
      - compare_alpha
    adapter: python_module
    module: src.cli.run_candidate_selection
    argv:
      - --artifacts-root
      - {json.dumps(alpha_root.as_posix())}
      - --metric
      - ic_ir
      - --max-candidates
      - "2"
      - --allocation-method
      - equal_weight
      - --output-path
      - {json.dumps(candidate_root.as_posix())}
  - id: construct_portfolio
    depends_on:
      - candidate_selection
    adapter: python_module
    module: src.cli.run_portfolio
    argv:
      - --portfolio-name
      - scenario_matrix_smoke_portfolio
      - --timeframe
      - 1D
      - --output-dir
      - {json.dumps(portfolio_root.as_posix())}
""".strip(),
        encoding="utf-8",
    )

    first = run_cli(["--config", str(pipeline_config_path)])
    pipeline_dir = tmp_path / "artifacts" / "pipelines" / first.pipeline_run_id

    manifest = json.loads((pipeline_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((pipeline_dir / "pipeline_metrics.json").read_text(encoding="utf-8"))
    lineage = json.loads((pipeline_dir / "lineage.json").read_text(encoding="utf-8"))
    state_payload = json.loads((pipeline_dir / "state.json").read_text(encoding="utf-8"))

    assert first.status == "completed"
    assert first.execution_order == (
        "alpha_ret_1d",
        "alpha_ret_1d_quantile",
        "strategy_momentum",
        "compare_alpha",
        "candidate_selection",
        "construct_portfolio",
    )
    assert pipeline_dir.exists()
    assert manifest["pipeline_run_id"] == first.pipeline_run_id
    assert manifest["status"] == "completed"
    assert [step["step_id"] for step in manifest["steps"]] == list(first.execution_order)
    assert metrics["pipeline_run_id"] == first.pipeline_run_id
    assert metrics["status"] == "completed"
    assert metrics["status_counts"] == {
        "completed": 6,
        "failed": 0,
        "reused": 0,
        "skipped": 0,
    }
    assert "nodes" in lineage and "edges" in lineage
    assert state_payload["state"]["candidate_selection_artifact_dir"].startswith(candidate_root.as_posix())
    assert state_payload["state"]["candidate_selection_run_id"]

    candidate_dir = Path(state_payload["state"]["candidate_selection_artifact_dir"])
    portfolio_step = next(step for step in manifest["steps"] if step["step_id"] == "construct_portfolio")
    assert portfolio_step["step_artifact_dir"].startswith(portfolio_root.as_posix())
    assert candidate_dir.exists()
    assert (candidate_dir / "selected_candidates.csv").exists()
    assert (candidate_dir / "manifest.json").exists()
    assert (Path(portfolio_step["step_artifact_dir"]) / "manifest.json").exists()

    first_snapshots = {
        "pipeline": _artifact_bytes(pipeline_dir),
        "candidate_selection": _artifact_bytes(candidate_root),
        "portfolio": _artifact_bytes(portfolio_root),
    }

    second = run_cli(["--config", str(pipeline_config_path)])
    second_pipeline_dir = tmp_path / "artifacts" / "pipelines" / second.pipeline_run_id
    second_snapshots = {
        "pipeline": _artifact_bytes(second_pipeline_dir),
        "candidate_selection": _artifact_bytes(candidate_root),
        "portfolio": _artifact_bytes(portfolio_root),
    }

    assert first.pipeline_run_id == second.pipeline_run_id
    assert first_snapshots == second_snapshots
