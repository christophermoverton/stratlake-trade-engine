from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterator

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cli.compare_alpha import run_cli as compare_alpha_cli  # noqa: E402
from src.cli.run_alpha import run_cli as run_alpha_cli  # noqa: E402
from src.cli.run_candidate_selection import run_cli as run_candidate_selection_cli  # noqa: E402
from src.cli.run_portfolio import run_cli as run_portfolio_cli  # noqa: E402
from src.cli.run_strategy import run_cli as run_strategy_cli  # noqa: E402
from src.data.load_features import FeaturePaths, load_features  # noqa: E402
from src.research.metrics import aggregate_strategy_returns  # noqa: E402
from src.research.regimes import (  # noqa: E402
    REGIME_DIMENSIONS,
    RegimeClassificationConfig,
    RegimeConditionalConfig,
    RegimeTransitionConfig,
    align_regimes_to_alpha_windows,
    align_regimes_to_portfolio_windows,
    align_regimes_to_strategy_timeseries,
    analyze_alpha_regime_transitions,
    analyze_portfolio_regime_transitions,
    analyze_strategy_regime_transitions,
    classify_market_regimes,
    compare_regime_results,
    evaluate_all_dimensions,
    load_regime_review_bundle,
    render_artifact_inventory_markdown,
    render_attribution_summary_markdown,
    render_comparison_summary_markdown,
    render_transition_highlights_markdown,
    summarize_regime_attribution,
    summarize_transition_attribution,
    write_regime_artifacts,
    write_regime_attribution_artifacts,
    write_regime_conditional_artifacts_multi_dimension,
    write_regime_transition_artifacts,
)


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "real_q1_2026_regime_aware_case_study"
SUMMARY_FILENAME = "summary.json"
INTERPRETATION_FILENAME = "final_interpretation.md"
ALPHA_CONFIG_PATH = REPO_ROOT / "configs" / "alphas_2026_q1.yml"
STRATEGY_NAME = "mean_reversion_v1_safe_2026_q1"
ALPHA_NAMES = (
    "ml_cross_sectional_xgb_2026_q1",
    "ml_cross_sectional_lgbm_2026_q1",
    "rank_composite_momentum_2026_q1",
)
PORTFOLIO_NAME = "real_world_m15_candidate_driven"
DATASET_NAME = "features_daily"
TIMEFRAME = "1D"
START_DATE = "2026-01-01"
END_DATE = "2026-04-03"
EVALUATION_HORIZON = 5
MAPPING_NAME = "top_bottom_quantile_q20"
MARKET_BASKET_SIZE = 12


@dataclass(frozen=True)
class RealCaseStudyArtifacts:
    summary: dict[str, Any]
    summary_path: Path
    output_root: Path
    regime_dir: Path
    strategy_dir: Path
    portfolio_dir: Path
    alpha_bundle_dirs: dict[str, Path]


@dataclass(frozen=True)
class AlphaSurfaceArtifacts:
    name: str
    result: Any
    ic_timeseries: pd.DataFrame
    sleeve_returns: pd.DataFrame
    artifact_dir: Path


def _normalize_json_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | str | int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, pd.Timestamp):
        timestamp = value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
        return timestamp.isoformat().replace("+00:00", "Z")
    if isinstance(value, dict):
        return {key: _normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    if pd.isna(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_normalize_json_value(payload), indent=2, sort_keys=True), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def _relative_to_output(path: Path, output_root: Path) -> str:
    try:
        relative = path.relative_to(output_root)
    except ValueError:
        return path.as_posix()
    return "." if str(relative) == "." else relative.as_posix()


def _preview(frame: pd.DataFrame, *, columns: list[str], limit: int = 5) -> list[dict[str, Any]]:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return []
    preview = frame.loc[:, available].head(limit).copy(deep=True)
    for column in preview.columns:
        if column == "ts_utc" or column.endswith("ts_utc"):
            preview[column] = pd.to_datetime(preview[column], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return _normalize_json_value(preview.to_dict(orient="records"))


@contextmanager
def example_environment() -> Iterator[None]:
    previous_features_root = os.environ.get("FEATURES_ROOT")
    previous_cwd = Path.cwd()
    os.environ["FEATURES_ROOT"] = str(REPO_ROOT / "data")
    os.chdir(REPO_ROOT)
    try:
        yield
    finally:
        os.chdir(previous_cwd)
        if previous_features_root is None:
            os.environ.pop("FEATURES_ROOT", None)
        else:
            os.environ["FEATURES_ROOT"] = previous_features_root


def _available_market_symbols(*, limit: int = MARKET_BASKET_SIZE) -> list[str]:
    dataset_root = FeaturePaths(root=REPO_ROOT / "data").dataset_root(DATASET_NAME)
    symbols = sorted(
        path.name.split("symbol=", 1)[1]
        for path in dataset_root.glob("symbol=*")
        if path.is_dir() and (path / "year=2026" / "part-0.parquet").exists()
    )
    if not symbols:
        raise FileNotFoundError(f"No 2026 feature partitions found under {dataset_root}")
    return symbols[:limit]


def load_market_data() -> tuple[pd.DataFrame, list[str]]:
    symbols = _available_market_symbols()
    frame = load_features(
        DATASET_NAME,
        start=START_DATE,
        end=END_DATE,
        symbols=symbols,
        paths=FeaturePaths(root=REPO_ROOT / "data"),
    )
    market = frame.loc[:, ["ts_utc", "symbol", "close"]].copy(deep=True)
    market["close"] = pd.to_numeric(market["close"], errors="coerce").astype("float64")
    market = market.dropna(subset=["ts_utc", "symbol", "close"]).reset_index(drop=True)
    return market, symbols


def _load_alpha_surface(result: Any) -> AlphaSurfaceArtifacts:
    artifact_dir = Path(result.artifact_dir)
    if not artifact_dir.is_absolute():
        artifact_dir = REPO_ROOT / artifact_dir
    ic_timeseries = pd.read_csv(artifact_dir / "ic_timeseries.csv")
    sleeve_returns = pd.read_csv(artifact_dir / "sleeve_returns.csv")
    return AlphaSurfaceArtifacts(
        name=result.alpha_name,
        result=result,
        ic_timeseries=ic_timeseries,
        sleeve_returns=sleeve_returns,
        artifact_dir=artifact_dir,
    )


def _run_alpha_suite(alpha_root: Path) -> dict[str, AlphaSurfaceArtifacts]:
    runs: dict[str, AlphaSurfaceArtifacts] = {}
    for alpha_name in ALPHA_NAMES:
        result = run_alpha_cli(
            [
                "--config",
                ALPHA_CONFIG_PATH.as_posix(),
                "--alpha-name",
                alpha_name,
                "--artifacts-root",
                alpha_root.as_posix(),
                "--signal-policy",
                "top_bottom_quantile",
                "--signal-quantile",
                "0.2",
            ]
        )
        runs[alpha_name] = _load_alpha_surface(result)
    return runs


def _rename_timestamp_column(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    normalized = frame.copy(deep=True)
    if "ts_utc" not in normalized.columns and source in normalized.columns:
        normalized = normalized.rename(columns={source: "ts_utc"})
    normalized["ts_utc"] = pd.to_datetime(normalized["ts_utc"], utc=True, errors="coerce")
    normalized = normalized.dropna(subset=["ts_utc"]).reset_index(drop=True)
    return normalized


def _write_bundle(
    bundle_dir: Path,
    *,
    labels: pd.DataFrame,
    regime_metadata: dict[str, Any],
    conditional_results: dict[str, Any],
    transition_result: Any,
    attribution: Any,
    transition_attribution: Any,
    comparison: Any | None,
    run_id: str,
    metadata: dict[str, Any],
) -> None:
    write_regime_artifacts(bundle_dir, labels, metadata=regime_metadata)
    write_regime_conditional_artifacts_multi_dimension(
        bundle_dir,
        conditional_results,
        run_id=run_id,
        extra_metadata=metadata,
    )
    write_regime_transition_artifacts(
        bundle_dir,
        transition_result,
        run_id=run_id,
        extra_metadata=metadata,
    )
    write_regime_attribution_artifacts(
        bundle_dir,
        attribution,
        transition=transition_attribution,
        comparison=comparison,
        run_id=run_id,
        extra_metadata=metadata,
    )


def _write_review_markdown(
    output_root: Path,
    *,
    strategy_bundle_dir: Path,
    portfolio_bundle_dir: Path,
    alpha_bundle_dirs: dict[str, Path],
    comparison_host_alpha: str,
) -> dict[str, str]:
    review_dir = output_root / "notebook_review"
    review_dir.mkdir(parents=True, exist_ok=True)

    strategy_bundle = load_regime_review_bundle(strategy_bundle_dir)
    portfolio_bundle = load_regime_review_bundle(portfolio_bundle_dir)
    host_bundle = load_regime_review_bundle(alpha_bundle_dirs[comparison_host_alpha])

    files: dict[str, Path] = {
        "strategy_inventory": review_dir / "strategy_inventory.md",
        "strategy_attribution_summary": review_dir / "strategy_attribution_summary.md",
        "strategy_transition_highlights": review_dir / "strategy_transition_highlights.md",
        "portfolio_inventory": review_dir / "portfolio_inventory.md",
        "portfolio_attribution_summary": review_dir / "portfolio_attribution_summary.md",
        "portfolio_transition_highlights": review_dir / "portfolio_transition_highlights.md",
        "alpha_comparison_summary": review_dir / "alpha_comparison_summary.md",
    }
    _write_text(files["strategy_inventory"], render_artifact_inventory_markdown(strategy_bundle))
    _write_text(files["strategy_attribution_summary"], render_attribution_summary_markdown(strategy_bundle))
    _write_text(files["strategy_transition_highlights"], render_transition_highlights_markdown(strategy_bundle))
    _write_text(files["portfolio_inventory"], render_artifact_inventory_markdown(portfolio_bundle))
    _write_text(files["portfolio_attribution_summary"], render_attribution_summary_markdown(portfolio_bundle))
    _write_text(files["portfolio_transition_highlights"], render_transition_highlights_markdown(portfolio_bundle))
    _write_text(files["alpha_comparison_summary"], render_comparison_summary_markdown(host_bundle))

    for alpha_name, bundle_dir in sorted(alpha_bundle_dirs.items()):
        bundle = load_regime_review_bundle(bundle_dir)
        safe_name = alpha_name.replace("-", "_")
        inventory = review_dir / f"{safe_name}_inventory.md"
        attribution = review_dir / f"{safe_name}_attribution_summary.md"
        transition = review_dir / f"{safe_name}_transition_highlights.md"
        _write_text(inventory, render_artifact_inventory_markdown(bundle))
        _write_text(attribution, render_attribution_summary_markdown(bundle))
        _write_text(transition, render_transition_highlights_markdown(bundle))
        files[f"{safe_name}_inventory"] = inventory
        files[f"{safe_name}_attribution_summary"] = attribution
        files[f"{safe_name}_transition_highlights"] = transition

    return {key: _relative_to_output(path, output_root) for key, path in sorted(files.items())}


def _best_run_summary(comparison: Any) -> dict[str, Any]:
    table = comparison.comparison_table.copy(deep=True)
    if table.empty:
        return {"run_id": None, "winner_count": 0}
    winners = table.loc[table["rank_within_regime"] == 1, ["run_id"]].copy()
    if winners.empty:
        return {"run_id": None, "winner_count": 0}
    counts = winners["run_id"].value_counts().sort_index()
    best_run_id = counts.sort_values(ascending=False, kind="stable").index[0]
    return {"run_id": str(best_run_id), "winner_count": int(counts.loc[best_run_id])}


def _artifact_tree(output_root: Path, *, include_native_artifacts: bool = True) -> list[str]:
    return sorted(
        str(path.relative_to(output_root)).replace("\\", "/")
        for path in output_root.rglob("*")
        if path.is_file() and (include_native_artifacts or "native_artifacts" not in path.parts)
    )


def _build_interpretation_markdown(summary: dict[str, Any]) -> str:
    strategy_best = summary["strategy"]["attribution"]["best_regime"]
    strategy_worst = summary["strategy"]["attribution"]["worst_regime"]
    portfolio_best = summary["portfolio"]["attribution"]["best_regime"]
    portfolio_worst = summary["portfolio"]["attribution"]["worst_regime"]
    alpha_best_run = summary["alpha_comparison"]["best_run"]
    worst_transition = summary["portfolio"]["transition"]["most_adverse_category"]
    lines = [
        "# Real Q1 2026 Regime-Aware Interpretation",
        "",
        "## Strategy Surface",
        (
            f"- The strongest strategy regime was `{strategy_best['regime_label']}` with "
            f"`{summary['strategy']['attribution']['primary_metric']}` of `{strategy_best['metric_value']}`."
        ),
        (
            f"- The weakest strategy regime was `{strategy_worst['regime_label']}` with "
            f"`{summary['strategy']['attribution']['primary_metric']}` of `{strategy_worst['metric_value']}`."
        ),
        (
            f"- Strategy fragility flag: `{str(summary['strategy']['attribution']['fragility_flag']).lower()}`. "
            f"Reason: {summary['strategy']['attribution']['fragility_reason']}."
        ),
        "",
        "## Alpha Surface",
        (
            f"- Across the compared Q1 2026 alpha runs, `{alpha_best_run['run_id']}` led the largest number of "
            f"regime slices (`winner_count={alpha_best_run['winner_count']}`)."
        ),
        (
            f"- Robust regime leadership was observed for: "
            f"{', '.join(summary['alpha_comparison']['robust_run_ids']) if summary['alpha_comparison']['robust_run_ids'] else 'none'}."
        ),
        (
            f"- Candidate selection carried the following alpha runs into the portfolio path: "
            f"{', '.join(summary['candidate_selection']['selected_alpha_names'])}."
        ),
        "",
        "## Portfolio Surface",
        (
            f"- The strongest portfolio regime was `{portfolio_best['regime_label']}` with "
            f"`{summary['portfolio']['attribution']['primary_metric']}` of `{portfolio_best['metric_value']}`."
        ),
        (
            f"- The weakest portfolio regime was `{portfolio_worst['regime_label']}` with "
            f"`{summary['portfolio']['attribution']['primary_metric']}` of `{portfolio_worst['metric_value']}`."
        ),
        (
            f"- The most adverse portfolio transition category was `{worst_transition['transition_category']}` "
            f"with metric value `{worst_transition['metric_value']}`."
        ),
        "",
        "## Caution Boundaries",
        "- These are descriptive regime slices over repository-available Q1 2026 research outputs, not causal claims about why performance changed.",
        "- The classifier uses a fixed deterministic market basket from `features_daily`, so the regime labels should be read as a project-level context surface rather than a claim about the whole market.",
        "- Sparse and empty regime slices remain visible and should limit confidence in any regime-specific interpretation.",
    ]
    return "\n".join(lines)


def run_case_study(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    reset_output: bool = True,
    verbose: bool = True,
) -> RealCaseStudyArtifacts:
    output_root = Path(output_root)
    if reset_output and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    regime_dir = output_root / "regime_bundle"
    strategy_dir = output_root / "strategy_bundle"
    portfolio_dir = output_root / "portfolio_bundle"
    alpha_bundle_dirs = {alpha_name: output_root / f"alpha_{alpha_name}_bundle" for alpha_name in ALPHA_NAMES}
    for directory in [regime_dir, strategy_dir, portfolio_dir, *alpha_bundle_dirs.values()]:
        directory.mkdir(parents=True, exist_ok=True)

    alpha_root = output_root / "native_artifacts" / "alpha"
    comparison_output = output_root / "native_artifacts" / "alpha_comparisons"
    candidate_output = output_root / "native_artifacts" / "candidate_selection"
    portfolio_output = output_root / "native_artifacts" / "portfolios"
    for directory in (alpha_root, comparison_output, candidate_output, portfolio_output):
        directory.mkdir(parents=True, exist_ok=True)

    with example_environment():
        market_data, market_symbols = load_market_data()
        strategy_run = run_strategy_cli(
            [
                "--strategy",
                STRATEGY_NAME,
                "--start",
                START_DATE,
                "--end",
                END_DATE,
            ]
        )
        strategy_frame = aggregate_strategy_returns(strategy_run.results_df)
        strategy_frame["ts_utc"] = pd.to_datetime(strategy_frame["ts_utc"], utc=True, errors="coerce")

        alpha_runs = _run_alpha_suite(alpha_root)
        compare_alpha_cli(
            [
                "--from-registry",
                "--artifacts-root",
                alpha_root.as_posix(),
                "--dataset",
                DATASET_NAME,
                "--timeframe",
                TIMEFRAME,
                "--evaluation-horizon",
                str(EVALUATION_HORIZON),
                "--mapping-name",
                MAPPING_NAME,
                "--view",
                "combined",
                "--metric",
                "ic_ir",
                "--sleeve-metric",
                "sharpe_ratio",
                "--output-path",
                comparison_output.as_posix(),
            ]
        )
        candidate_result = run_candidate_selection_cli(
            [
                "--artifacts-root",
                alpha_root.as_posix(),
                "--output-path",
                candidate_output.as_posix(),
                "--dataset",
                DATASET_NAME,
                "--timeframe",
                TIMEFRAME,
                "--evaluation-horizon",
                str(EVALUATION_HORIZON),
                "--mapping-name",
                MAPPING_NAME,
                "--metric",
                "ic_ir",
                "--min-ic",
                "-1.0",
                "--min-rank-ic",
                "-1.0",
                "--min-ic-ir",
                "-1.0",
                "--min-rank-ic-ir",
                "-1.0",
                "--min-history-length",
                "10",
                "--max-pairwise-correlation",
                "0.75",
                "--min-overlap-observations",
                "10",
                "--allocation-method",
                "equal_weight",
                "--max-weight-per-candidate",
                "0.50",
                "--min-allocation-candidate-count",
                "2",
            ]
        )
        portfolio_result = run_portfolio_cli(
            [
                "--from-candidate-selection",
                candidate_result.artifact_dir.as_posix(),
                "--portfolio-name",
                PORTFOLIO_NAME,
                "--output-dir",
                portfolio_output.as_posix(),
                "--timeframe",
                TIMEFRAME,
            ]
        )

    portfolio_frame = pd.read_csv(portfolio_result.experiment_dir / "portfolio_returns.csv")
    portfolio_frame = _rename_timestamp_column(portfolio_frame, source="ts")

    classification = classify_market_regimes(
        market_data,
        config=RegimeClassificationConfig(
            price_column="close",
            metadata={
                "case_study": "m24_8_real_q1_2026_regime_aware_case_study",
                "market_symbols": market_symbols,
                "source_dataset": DATASET_NAME,
            },
        ),
    )
    labels = classification.labels
    regime_metadata = {
        **classification.metadata,
        "case_study": "m24_8_real_q1_2026_regime_aware_case_study",
        "source_dataset": DATASET_NAME,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "market_symbols": market_symbols,
    }
    write_regime_artifacts(regime_dir, labels, metadata=regime_metadata)

    conditional_config = RegimeConditionalConfig(
        min_observations=3,
        metadata={"case_study": "m24_8_real_q1_2026_regime_aware_case_study"},
    )
    transition_config = RegimeTransitionConfig(
        pre_event_rows=2,
        post_event_rows=2,
        min_observations=3,
        metadata={"case_study": "m24_8_real_q1_2026_regime_aware_case_study"},
    )

    strategy_aligned = align_regimes_to_strategy_timeseries(strategy_frame, labels)
    strategy_results = evaluate_all_dimensions(
        strategy_aligned,
        surface="strategy",
        return_column="strategy_return",
        config=conditional_config,
    )
    strategy_transition = analyze_strategy_regime_transitions(
        strategy_aligned,
        labels,
        dimensions=list(REGIME_DIMENSIONS),
        config=transition_config,
    )
    strategy_attribution = summarize_regime_attribution(strategy_results["composite"], run_id="strategy")
    strategy_transition_attribution = summarize_transition_attribution(strategy_transition, run_id="strategy")
    _write_bundle(
        strategy_dir,
        labels=labels,
        regime_metadata={**regime_metadata, "bundle": "strategy"},
        conditional_results=strategy_results,
        transition_result=strategy_transition,
        attribution=strategy_attribution,
        transition_attribution=strategy_transition_attribution,
        comparison=None,
        run_id="strategy_bundle",
        metadata={"surface": "strategy", "strategy_name": STRATEGY_NAME},
    )

    portfolio_aligned = align_regimes_to_portfolio_windows(portfolio_frame, labels)
    portfolio_results = evaluate_all_dimensions(
        portfolio_aligned,
        surface="portfolio",
        return_column="portfolio_return",
        config=conditional_config,
    )
    portfolio_transition = analyze_portfolio_regime_transitions(
        portfolio_aligned,
        labels,
        dimensions=list(REGIME_DIMENSIONS),
        config=transition_config,
    )
    portfolio_attribution = summarize_regime_attribution(portfolio_results["composite"], run_id="portfolio")
    portfolio_transition_attribution = summarize_transition_attribution(portfolio_transition, run_id="portfolio")
    _write_bundle(
        portfolio_dir,
        labels=labels,
        regime_metadata={**regime_metadata, "bundle": "portfolio"},
        conditional_results=portfolio_results,
        transition_result=portfolio_transition,
        attribution=portfolio_attribution,
        transition_attribution=portfolio_transition_attribution,
        comparison=None,
        run_id="portfolio_bundle",
        metadata={"surface": "portfolio", "portfolio_name": PORTFOLIO_NAME},
    )

    selected_candidates = pd.read_csv(candidate_result.selected_csv)
    rejected_candidates = pd.read_csv(candidate_result.rejected_csv)
    allocation_weights = pd.read_csv(candidate_result.allocation_csv)
    selected_alpha_names = (
        selected_candidates["alpha_name"].astype("string").tolist()
        if "alpha_name" in selected_candidates.columns
        else []
    )

    alpha_conditional_results: dict[str, Any] = {}
    alpha_summary_rows: list[dict[str, Any]] = []
    comparison_host_alpha = ALPHA_NAMES[0]
    for alpha_name, alpha_artifacts in alpha_runs.items():
        alpha_frame = _rename_timestamp_column(alpha_artifacts.ic_timeseries, source="ts")
        alpha_aligned = align_regimes_to_alpha_windows(alpha_frame, labels)
        conditional_results = evaluate_all_dimensions(
            alpha_aligned,
            surface="alpha",
            config=conditional_config,
        )
        alpha_conditional_results[alpha_name] = conditional_results["composite"]
        alpha_transition = analyze_alpha_regime_transitions(
            alpha_aligned,
            labels,
            dimensions=list(REGIME_DIMENSIONS),
            config=transition_config,
        )
        alpha_attribution = summarize_regime_attribution(conditional_results["composite"], run_id=alpha_name)
        alpha_transition_attribution = summarize_transition_attribution(alpha_transition, run_id=alpha_name)
        alpha_summary_rows.append(
            {
                "alpha_name": alpha_name,
                "mean_ic": alpha_artifacts.result.evaluation.evaluation_result.summary["mean_ic"],
                "ic_ir": alpha_artifacts.result.evaluation.evaluation_result.summary["ic_ir"],
                "selected_in_portfolio": alpha_name in selected_alpha_names,
            }
        )
        _write_bundle(
            alpha_bundle_dirs[alpha_name],
            labels=labels,
            regime_metadata={**regime_metadata, "bundle": alpha_name},
            conditional_results=conditional_results,
            transition_result=alpha_transition,
            attribution=alpha_attribution,
            transition_attribution=alpha_transition_attribution,
            comparison=None,
            run_id=f"{alpha_name}_bundle",
            metadata={"surface": "alpha", "alpha_name": alpha_name},
        )

    alpha_comparison = compare_regime_results(
        alpha_conditional_results,
        surface="alpha",
        dimension="composite",
        metadata={"case_study": "m24_8_real_q1_2026_regime_aware_case_study"},
    )
    host_alpha_attribution = summarize_regime_attribution(alpha_conditional_results[comparison_host_alpha], run_id=comparison_host_alpha)
    host_alpha_transition = summarize_transition_attribution(
        analyze_alpha_regime_transitions(
            align_regimes_to_alpha_windows(_rename_timestamp_column(alpha_runs[comparison_host_alpha].ic_timeseries, source="ts"), labels),
            labels,
            dimensions=list(REGIME_DIMENSIONS),
            config=transition_config,
        ),
        run_id=comparison_host_alpha,
    )
    write_regime_attribution_artifacts(
        alpha_bundle_dirs[comparison_host_alpha],
        host_alpha_attribution,
        transition=host_alpha_transition,
        comparison=alpha_comparison,
        run_id=f"{comparison_host_alpha}_bundle",
        extra_metadata={"surface": "alpha", "alpha_name": comparison_host_alpha, "comparison_host": True},
    )

    notebook_review_files = _write_review_markdown(
        output_root,
        strategy_bundle_dir=strategy_dir,
        portfolio_bundle_dir=portfolio_dir,
        alpha_bundle_dirs=alpha_bundle_dirs,
        comparison_host_alpha=comparison_host_alpha,
    )

    alpha_best_run = _best_run_summary(alpha_comparison)
    alpha_warning_row_count = int(alpha_comparison.comparison_table["warning_flag"].astype("bool").sum())
    summary: dict[str, Any] = {
        "case_study": {
            "milestone": "M24.8",
            "name": "real_q1_2026_regime_aware_case_study",
            "extends": [
                "docs/examples/real_world_candidate_selection_portfolio_case_study.py",
                "docs/examples/real_world_campaign_case_study.py",
                "docs/examples/regime_aware_case_study.py",
            ],
        },
        "commands": [
            "python docs/examples/real_q1_2026_regime_aware_case_study.py",
            "python docs/examples/real_q1_2026_regime_aware_case_study.py --output-root docs/examples/output/real_q1_2026_regime_aware_case_study",
        ],
        "input_path": {
            "dataset": DATASET_NAME,
            "features_root": "data/curated/features_daily",
            "date_window": {"start": START_DATE, "end_exclusive": END_DATE},
            "market_symbols": market_symbols,
            "strategy_name": STRATEGY_NAME,
            "alpha_names": list(ALPHA_NAMES),
            "portfolio_name": PORTFOLIO_NAME,
        },
        "paths": {
            "output_root": ".",
            "regime_bundle": _relative_to_output(regime_dir, output_root),
            "strategy_bundle": _relative_to_output(strategy_dir, output_root),
            "portfolio_bundle": _relative_to_output(portfolio_dir, output_root),
            "alpha_bundles": {name: _relative_to_output(path, output_root) for name, path in sorted(alpha_bundle_dirs.items())},
            "notebook_review_dir": "notebook_review",
            "final_interpretation": INTERPRETATION_FILENAME,
        },
        "classification": {
            "defined_row_count": int(labels["is_defined"].sum()),
            "undefined_row_count": int((~labels["is_defined"]).sum()),
            "label_preview": _preview(
                labels,
                columns=["ts_utc", "volatility_state", "trend_state", "drawdown_recovery_state", "stress_state", "regime_label"],
            ),
            "state_distribution": {
                "volatility": labels["volatility_state"].astype("string").value_counts(dropna=False).sort_index().to_dict(),
                "trend": labels["trend_state"].astype("string").value_counts(dropna=False).sort_index().to_dict(),
                "drawdown_recovery": labels["drawdown_recovery_state"].astype("string").value_counts(dropna=False).sort_index().to_dict(),
                "stress": labels["stress_state"].astype("string").value_counts(dropna=False).sort_index().to_dict(),
            },
        },
        "strategy": {
            "surface": "included",
            "row_count": int(len(strategy_frame)),
            "alignment_preview": _preview(strategy_aligned, columns=["ts_utc", "strategy_return", "regime_label", "regime_alignment_status"]),
            "conditional_dimensions": sorted(strategy_results),
            "attribution": {
                "primary_metric": strategy_attribution.summary["primary_metric"],
                "best_regime": strategy_attribution.summary["best_regime"],
                "worst_regime": strategy_attribution.summary["worst_regime"],
                "fragility_flag": strategy_attribution.summary["fragility_flag"],
                "fragility_reason": strategy_attribution.summary["fragility_reason"],
            },
            "transition": {
                "event_count": int(len(strategy_transition.events)),
                "primary_metric": strategy_transition_attribution.summary["primary_metric"],
                "most_adverse_category": strategy_transition_attribution.summary["worst_transition_category"],
            },
        },
        "alpha": {
            "surface": "included",
            "runs": _normalize_json_value(alpha_summary_rows),
            "comparison_host_alpha": comparison_host_alpha,
        },
        "alpha_comparison": {
            "best_run": alpha_best_run,
            "robust_run_ids": alpha_comparison.summary["robust_run_ids"],
            "regime_winners": alpha_comparison.summary["regime_winners"],
            "warning_row_count": alpha_warning_row_count,
        },
        "portfolio": {
            "surface": "included",
            "component_count": int(getattr(portfolio_result, "component_count", 0)),
            "alignment_preview": _preview(portfolio_aligned, columns=["ts_utc", "portfolio_return", "regime_label", "regime_alignment_status"]),
            "attribution": {
                "primary_metric": portfolio_attribution.summary["primary_metric"],
                "best_regime": portfolio_attribution.summary["best_regime"],
                "worst_regime": portfolio_attribution.summary["worst_regime"],
                "fragility_flag": portfolio_attribution.summary["fragility_flag"],
                "fragility_reason": portfolio_attribution.summary["fragility_reason"],
            },
            "transition": {
                "event_count": int(len(portfolio_transition.events)),
                "primary_metric": portfolio_transition_attribution.summary["primary_metric"],
                "most_adverse_category": portfolio_transition_attribution.summary["worst_transition_category"],
            },
        },
        "candidate_selection": {
            "selected_alpha_names": selected_alpha_names,
            "selected_count": int(len(selected_candidates)),
            "rejected_count": int(len(rejected_candidates)),
            "allocation_weight_sum": float(pd.to_numeric(allocation_weights["allocation_weight"], errors="coerce").sum()),
        },
        "notebook_review": notebook_review_files,
        "artifact_tree": _artifact_tree(output_root, include_native_artifacts=False),
        "limitations": [
            "The market regime surface is built from a deterministic fixed basket of repository-available Q1 2026 `features_daily` symbols rather than a bespoke benchmark index series.",
            "The strategy surface is a direct real-data strategy run, while the alpha and portfolio surfaces extend the prior real-world candidate-selection workflow.",
            "Interpretation is descriptive and evidence-based; sparse and empty slices should limit confidence.",
        ],
    }

    interpretation_markdown = _build_interpretation_markdown(summary)
    _write_text(output_root / INTERPRETATION_FILENAME, interpretation_markdown)
    summary["artifact_tree"] = _artifact_tree(output_root, include_native_artifacts=False)

    summary_path = output_root / SUMMARY_FILENAME
    _write_json(summary_path, summary)

    if verbose:
        print(json.dumps(summary["paths"], indent=2, sort_keys=True))
        print(f"summary_path={summary_path.as_posix()}")

    return RealCaseStudyArtifacts(
        summary=summary,
        summary_path=summary_path,
        output_root=output_root,
        regime_dir=regime_dir,
        strategy_dir=strategy_dir,
        portfolio_dir=portfolio_dir,
        alpha_bundle_dirs=alpha_bundle_dirs,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the real Q1 2026 regime-aware case study.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where the deterministic example artifacts should be written.",
    )
    parser.add_argument(
        "--no-reset-output",
        action="store_true",
        help="Keep any existing output root instead of clearing it before writing artifacts.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_case_study(output_root=args.output_root, reset_output=not args.no_reset_output, verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
