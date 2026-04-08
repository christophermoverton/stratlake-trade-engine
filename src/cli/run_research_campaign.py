from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Iterator, Sequence

from src.cli import compare_alpha as compare_alpha_cli
from src.cli import compare_research as compare_research_cli
from src.cli import compare_strategies as compare_strategies_cli
from src.cli import review_candidate_selection as review_candidate_selection_cli
from src.cli import run_alpha as run_alpha_cli
from src.cli import run_candidate_selection as run_candidate_selection_cli
from src.cli import run_portfolio as run_portfolio_cli
from src.cli import run_strategy as run_strategy_cli
from src.config.research_campaign import (
    ResearchCampaignConfig,
    ResearchCampaignConfigError,
    load_research_campaign_config,
    resolve_research_campaign_config,
)


@dataclass(frozen=True)
class CampaignStageRecord:
    stage_name: str
    status: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResearchCampaignRunResult:
    config: ResearchCampaignConfig
    stage_records: tuple[CampaignStageRecord, ...]
    alpha_results: tuple[Any, ...]
    strategy_results: tuple[Any, ...]
    alpha_comparison_result: Any | None
    strategy_comparison_result: Any | None
    candidate_selection_result: Any | None
    portfolio_result: Any | None
    candidate_review_result: Any | None
    review_result: Any | None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one deterministic research campaign from a unified YAML/JSON config: "
            "preflight, research, comparison, candidate selection, portfolio, and review."
        )
    )
    parser.add_argument(
        "--config",
        help=(
            "Optional research campaign config path. "
            "Defaults to configs/research_campaign.yml."
        ),
    )
    return parser.parse_args(argv)


def resolve_cli_config(config_path: str | Path | None = None) -> ResearchCampaignConfig:
    if config_path is None:
        return resolve_research_campaign_config()
    loaded = load_research_campaign_config(Path(config_path))
    return resolve_research_campaign_config(loaded.to_dict())


def run_cli(argv: Sequence[str] | None = None) -> ResearchCampaignRunResult:
    args = parse_args(argv)
    config = resolve_cli_config(args.config)
    result = run_research_campaign(config)
    print_summary(result)
    return result


def run_research_campaign(config: ResearchCampaignConfig) -> ResearchCampaignRunResult:
    records: list[CampaignStageRecord] = []

    preflight_details = _run_preflight(config)
    records.append(
        CampaignStageRecord(
            stage_name="preflight",
            status="completed",
            details=preflight_details,
        )
    )

    alpha_results = _run_alpha_research(config)
    strategy_results = _run_strategy_research(config)
    records.append(
        CampaignStageRecord(
            stage_name="research",
            status="completed",
            details={
                "alpha_runs": len(alpha_results),
                "strategy_runs": len(strategy_results),
            },
        )
    )

    alpha_comparison_result = None
    strategy_comparison_result = None
    if config.comparison.enabled:
        alpha_comparison_result = _run_alpha_comparison(config)
        strategy_comparison_result = _run_strategy_comparison(config)
    records.append(
        CampaignStageRecord(
            stage_name="comparison",
            status="completed" if config.comparison.enabled else "skipped",
            details={
                "alpha_comparison_id": None
                if alpha_comparison_result is None
                else str(alpha_comparison_result.comparison_id),
                "strategy_comparison_id": None
                if strategy_comparison_result is None
                else str(strategy_comparison_result.comparison_id),
            },
        )
    )

    candidate_selection_result = None
    if config.candidate_selection.enabled:
        candidate_selection_result = _run_candidate_selection(config)
    records.append(
        CampaignStageRecord(
            stage_name="candidate_selection",
            status="completed" if candidate_selection_result is not None else "skipped",
            details={
                "run_id": None
                if candidate_selection_result is None
                else str(candidate_selection_result.run_id)
            },
        )
    )

    portfolio_result = None
    if config.portfolio.enabled:
        portfolio_result = _run_portfolio(config, candidate_selection_result=candidate_selection_result)
    records.append(
        CampaignStageRecord(
            stage_name="portfolio",
            status="completed" if portfolio_result is not None else "skipped",
            details={"run_id": None if portfolio_result is None else str(portfolio_result.run_id)},
        )
    )

    candidate_review_result = None
    if config.candidate_selection.enabled and config.candidate_selection.execution.enable_review:
        candidate_review_result = _run_candidate_review(
            config,
            candidate_selection_result=candidate_selection_result,
            portfolio_result=portfolio_result,
        )

    review_result = _run_research_review(config)
    records.append(
        CampaignStageRecord(
            stage_name="review",
            status="completed",
            details={
                "candidate_review_dir": None
                if candidate_review_result is None
                else str(candidate_review_result.review_dir),
                "review_id": str(review_result.review_id),
            },
        )
    )

    return ResearchCampaignRunResult(
        config=config,
        stage_records=tuple(records),
        alpha_results=tuple(alpha_results),
        strategy_results=tuple(strategy_results),
        alpha_comparison_result=alpha_comparison_result,
        strategy_comparison_result=strategy_comparison_result,
        candidate_selection_result=candidate_selection_result,
        portfolio_result=portfolio_result,
        candidate_review_result=candidate_review_result,
        review_result=review_result,
    )


def print_summary(result: ResearchCampaignRunResult) -> None:
    print("Research Campaign Summary")
    print("-------------------------")
    print(f"Preflight: ok ({len(result.stage_records)} stages tracked)")
    print(
        f"Research: alpha_runs={len(result.alpha_results)} | "
        f"strategy_runs={len(result.strategy_results)}"
    )

    alpha_comparison_id = (
        None
        if result.alpha_comparison_result is None
        else str(result.alpha_comparison_result.comparison_id)
    )
    strategy_comparison_id = (
        None
        if result.strategy_comparison_result is None
        else str(result.strategy_comparison_result.comparison_id)
    )
    print(
        "Comparison: "
        f"alpha={alpha_comparison_id or 'skipped'} | "
        f"strategy={strategy_comparison_id or 'skipped'}"
    )

    candidate_run_id = (
        None
        if result.candidate_selection_result is None
        else str(result.candidate_selection_result.run_id)
    )
    portfolio_run_id = (
        None if result.portfolio_result is None else str(result.portfolio_result.run_id)
    )
    print(
        "Selection/Portfolio: "
        f"candidate={candidate_run_id or 'skipped'} | "
        f"portfolio={portfolio_run_id or 'skipped'}"
    )

    candidate_review_dir = (
        None
        if result.candidate_review_result is None
        else str(result.candidate_review_result.review_dir)
    )
    print(
        "Review: "
        f"candidate_review={candidate_review_dir or 'skipped'} | "
        f"review_id={result.review_result.review_id}"
    )


def main() -> None:
    try:
        run_cli()
    except (ResearchCampaignConfigError, ValueError) as exc:
        print(_format_run_failure(exc), file=sys.stderr)
        raise SystemExit(1) from exc


def _format_run_failure(exc: Exception) -> str:
    message = str(exc).strip()
    if message.startswith("Run failed:"):
        return message
    return f"Run failed: {message}"


def _run_preflight(config: ResearchCampaignConfig) -> dict[str, Any]:
    alpha_names = config.targets.alpha_names
    strategy_names = config.targets.strategy_names

    if not alpha_names and not strategy_names:
        raise ValueError(
            "Campaign requires at least one alpha or strategy target in targets.alpha_names or targets.strategy_names."
        )

    if config.candidate_selection.enabled and not config.candidate_selection.alpha_name:
        raise ValueError(
            "Candidate selection requires one resolved alpha_name via candidate_selection.alpha_name or a single targets.alpha_names entry."
        )

    if config.portfolio.enabled and not config.portfolio.portfolio_name:
        raise ValueError(
            "Portfolio stage requires one resolved portfolio_name via portfolio.portfolio_name or a single targets.portfolio_names entry."
        )

    if config.portfolio.enabled and config.portfolio.from_candidate_selection and not config.candidate_selection.enabled:
        raise ValueError(
            "portfolio.from_candidate_selection requires candidate_selection.enabled to be true in the same campaign."
        )

    if (
        config.candidate_selection.enabled
        and config.candidate_selection.execution.enable_review
        and not config.portfolio.enabled
    ):
        raise ValueError(
            "candidate_selection.execution.enable_review requires portfolio.enabled so the review stage has portfolio artifacts."
        )

    _require_existing_path(config.targets.alpha_catalog_path, field_name="targets.alpha_catalog_path")
    _require_existing_path(
        config.targets.strategy_config_path,
        field_name="targets.strategy_config_path",
    )
    if config.portfolio.enabled and not config.portfolio.from_candidate_selection:
        _require_existing_path(
            config.targets.portfolio_config_path,
            field_name="targets.portfolio_config_path",
        )

    return {
        "alpha_target_count": len(alpha_names),
        "strategy_target_count": len(strategy_names),
        "candidate_selection_enabled": config.candidate_selection.enabled,
        "portfolio_enabled": config.portfolio.enabled,
        "review_enabled": True,
    }


def _run_alpha_research(config: ResearchCampaignConfig) -> list[Any]:
    results: list[Any] = []
    for alpha_name in config.targets.alpha_names:
        argv = ["--alpha-name", alpha_name, "--config", config.targets.alpha_catalog_path]
        if config.dataset_selection.dataset is not None:
            argv.extend(["--dataset", config.dataset_selection.dataset])
        if config.time_windows.start is not None:
            argv.extend(["--start", config.time_windows.start])
        if config.time_windows.end is not None:
            argv.extend(["--end", config.time_windows.end])
        if config.time_windows.train_start is not None:
            argv.extend(["--train-start", config.time_windows.train_start])
        if config.time_windows.train_end is not None:
            argv.extend(["--train-end", config.time_windows.train_end])
        if config.time_windows.predict_start is not None:
            argv.extend(["--predict-start", config.time_windows.predict_start])
        if config.time_windows.predict_end is not None:
            argv.extend(["--predict-end", config.time_windows.predict_end])
        if config.dataset_selection.tickers_path is not None:
            argv.extend(["--tickers", config.dataset_selection.tickers_path])
        results.append(run_alpha_cli.run_cli(argv))
    return results


def _run_strategy_research(config: ResearchCampaignConfig) -> list[Any]:
    results: list[Any] = []
    with _strategy_config_override(Path(config.targets.strategy_config_path)):
        for strategy_name in config.targets.strategy_names:
            argv = ["--strategy", strategy_name]
            if config.time_windows.start is not None:
                argv.extend(["--start", config.time_windows.start])
            if config.time_windows.end is not None:
                argv.extend(["--end", config.time_windows.end])
            results.append(run_strategy_cli.run_cli(argv))
    return results


def _run_alpha_comparison(config: ResearchCampaignConfig) -> Any | None:
    if not config.targets.alpha_names:
        return None

    argv = ["--from-registry", "--view", config.comparison.alpha_view, "--metric", config.comparison.alpha_metric]
    argv.extend(["--sleeve-metric", config.comparison.alpha_sleeve_metric])
    if len(config.targets.alpha_names) == 1:
        argv.extend(["--alpha-name", config.targets.alpha_names[0]])
    if config.dataset_selection.dataset is not None:
        argv.extend(["--dataset", config.dataset_selection.dataset])
    if config.dataset_selection.timeframe is not None:
        argv.extend(["--timeframe", config.dataset_selection.timeframe])
    if config.dataset_selection.evaluation_horizon is not None:
        argv.extend(
            [
                "--evaluation-horizon",
                str(config.dataset_selection.evaluation_horizon),
            ]
        )
    if config.dataset_selection.mapping_name is not None:
        argv.extend(["--mapping-name", config.dataset_selection.mapping_name])
    if config.outputs.alpha_artifacts_root:
        argv.extend(["--artifacts-root", config.outputs.alpha_artifacts_root])
    output_path = _comparison_output_path(config, stage_name="alpha")
    if output_path is not None:
        argv.extend(["--output-path", output_path.as_posix()])
    return compare_alpha_cli.run_cli(argv)


def _run_strategy_comparison(config: ResearchCampaignConfig) -> Any | None:
    if not config.targets.strategy_names:
        return None

    argv = [
        "--strategies",
        *config.targets.strategy_names,
        "--metric",
        config.comparison.strategy_metric,
    ]
    if config.comparison.top_k is not None:
        argv.extend(["--top-k", str(config.comparison.top_k)])
    if config.comparison.from_registry:
        argv.append("--from-registry")
    else:
        if config.time_windows.start is not None:
            argv.extend(["--start", config.time_windows.start])
        if config.time_windows.end is not None:
            argv.extend(["--end", config.time_windows.end])
    output_path = _comparison_output_path(config, stage_name="strategy")
    if output_path is not None:
        argv.extend(["--output-path", output_path.as_posix()])
    with _strategy_config_override(Path(config.targets.strategy_config_path)):
        return compare_strategies_cli.run_cli(argv)


def _run_candidate_selection(config: ResearchCampaignConfig) -> Any:
    candidate = config.candidate_selection
    argv = [
        "--artifacts-root",
        candidate.artifacts_root,
        "--alpha-name",
        str(candidate.alpha_name),
        "--metric",
        candidate.metric,
        "--output-path",
        candidate.output.path,
    ]
    if candidate.dataset is not None:
        argv.extend(["--dataset", candidate.dataset])
    if candidate.timeframe is not None:
        argv.extend(["--timeframe", candidate.timeframe])
    if candidate.evaluation_horizon is not None:
        argv.extend(["--evaluation-horizon", str(candidate.evaluation_horizon)])
    if candidate.mapping_name is not None:
        argv.extend(["--mapping-name", candidate.mapping_name])
    if candidate.max_candidates is not None:
        argv.extend(["--max-candidates", str(candidate.max_candidates)])
    if candidate.eligibility.min_mean_ic is not None:
        argv.extend(["--min-ic", str(candidate.eligibility.min_mean_ic)])
    if candidate.eligibility.min_mean_rank_ic is not None:
        argv.extend(["--min-rank-ic", str(candidate.eligibility.min_mean_rank_ic)])
    if candidate.eligibility.min_ic_ir is not None:
        argv.extend(["--min-ic-ir", str(candidate.eligibility.min_ic_ir)])
    if candidate.eligibility.min_rank_ic_ir is not None:
        argv.extend(["--min-rank-ic-ir", str(candidate.eligibility.min_rank_ic_ir)])
    if candidate.eligibility.min_history_length is not None:
        argv.extend(
            [
                "--min-history-length",
                str(candidate.eligibility.min_history_length),
            ]
        )
    if candidate.eligibility.min_coverage is not None:
        argv.extend(["--min-coverage", str(candidate.eligibility.min_coverage)])
    if candidate.redundancy.max_pairwise_correlation is not None:
        argv.extend(
            [
                "--max-pairwise-correlation",
                str(candidate.redundancy.max_pairwise_correlation),
            ]
        )
    if candidate.redundancy.min_overlap_observations is not None:
        argv.extend(
            [
                "--min-overlap-observations",
                str(candidate.redundancy.min_overlap_observations),
            ]
        )
    argv.extend(["--allocation-method", candidate.allocation.allocation_method])
    if candidate.allocation.max_weight_per_candidate is not None:
        argv.extend(
            [
                "--max-weight-per-candidate",
                str(candidate.allocation.max_weight_per_candidate),
            ]
        )
    if candidate.allocation.min_allocation_candidate_count is not None:
        argv.extend(
            [
                "--min-allocation-candidate-count",
                str(candidate.allocation.min_allocation_candidate_count),
            ]
        )
    if candidate.allocation.min_allocation_weight is not None:
        argv.extend(
            [
                "--min-allocation-weight",
                str(candidate.allocation.min_allocation_weight),
            ]
        )
    argv.extend(
        [
            "--allocation-weight-sum-tolerance",
            str(candidate.allocation.allocation_weight_sum_tolerance),
            "--allocation-rounding-decimals",
            str(candidate.allocation.allocation_rounding_decimals),
        ]
    )
    if candidate.execution.strict_mode:
        argv.append("--strict")
    if candidate.execution.skip_eligibility:
        argv.append("--skip-eligibility")
    if candidate.execution.skip_redundancy:
        argv.append("--skip-redundancy")
    if candidate.execution.skip_allocation:
        argv.append("--skip-allocation")
    if candidate.execution.from_registry:
        argv.append("--from-registry")
    if candidate.execution.register_run:
        argv.append("--register-run")
    if candidate.execution.no_markdown_review:
        argv.append("--no-markdown-review")
    if candidate.output.registry_path is not None:
        argv.extend(["--registry-path", candidate.output.registry_path])
    return run_candidate_selection_cli.run_cli(argv)


def _run_portfolio(config: ResearchCampaignConfig, *, candidate_selection_result: Any | None) -> Any:
    portfolio = config.portfolio
    argv = [
        "--portfolio-name",
        str(portfolio.portfolio_name),
        "--timeframe",
        str(portfolio.timeframe),
    ]
    if portfolio.from_candidate_selection:
        if candidate_selection_result is None:
            raise ValueError("Portfolio stage requires a completed candidate selection result.")
        argv.extend(
            [
                "--from-candidate-selection",
                Path(candidate_selection_result.artifact_dir).as_posix(),
            ]
        )
    else:
        argv.extend(["--portfolio-config", config.targets.portfolio_config_path])
        if portfolio.from_registry:
            argv.append("--from-registry")
    if portfolio.evaluation_path is not None:
        argv.extend(["--evaluation", portfolio.evaluation_path])
    if portfolio.optimizer_method is not None:
        argv.extend(["--optimizer-method", portfolio.optimizer_method])
    if config.outputs.portfolio_artifacts_root:
        argv.extend(["--output-dir", config.outputs.portfolio_artifacts_root])
    return run_portfolio_cli.run_cli(argv)


def _run_candidate_review(
    config: ResearchCampaignConfig,
    *,
    candidate_selection_result: Any | None,
    portfolio_result: Any | None,
) -> Any:
    if candidate_selection_result is None or portfolio_result is None:
        raise ValueError("Candidate review requires completed candidate selection and portfolio stages.")

    argv = [
        "--candidate-selection-path",
        Path(candidate_selection_result.artifact_dir).as_posix(),
        "--portfolio-path",
        Path(portfolio_result.experiment_dir).as_posix(),
    ]
    review_output_path = config.candidate_selection.output.review_output_path
    if review_output_path is not None:
        argv.extend(["--output-path", review_output_path])
    if config.candidate_selection.execution.no_markdown_review:
        argv.append("--no-markdown-report")
    return review_candidate_selection_cli.run_cli(argv)


def _run_research_review(config: ResearchCampaignConfig) -> Any:
    review = config.review
    argv = ["--from-registry"]
    if review.filters.run_types:
        argv.extend(["--run-types", *review.filters.run_types])
    if review.filters.timeframe is not None:
        argv.extend(["--timeframe", review.filters.timeframe])
    if review.filters.dataset is not None:
        argv.extend(["--dataset", review.filters.dataset])
    if review.filters.alpha_name is not None:
        argv.extend(["--alpha-name", review.filters.alpha_name])
    if review.filters.strategy_name is not None:
        argv.extend(["--strategy-name", review.filters.strategy_name])
    if review.filters.portfolio_name is not None:
        argv.extend(["--portfolio-name", review.filters.portfolio_name])
    if review.filters.top_k_per_type is not None:
        argv.extend(["--top-k", str(review.filters.top_k_per_type)])
    argv.extend(
        [
            "--alpha-metric",
            review.ranking.alpha_evaluation_primary_metric,
            "--alpha-secondary-metric",
            review.ranking.alpha_evaluation_secondary_metric,
            "--strategy-metric",
            review.ranking.strategy_primary_metric,
            "--strategy-secondary-metric",
            review.ranking.strategy_secondary_metric,
            "--portfolio-metric",
            review.ranking.portfolio_primary_metric,
            "--portfolio-secondary-metric",
            review.ranking.portfolio_secondary_metric,
        ]
    )
    if review.output.path is not None:
        argv.extend(["--output-path", review.output.path])
    if not review.output.emit_plots:
        argv.append("--disable-plots")
    return compare_research_cli.run_cli(argv)


def _require_existing_path(path_text: str, *, field_name: str) -> None:
    path = Path(path_text)
    if not path.exists():
        raise ValueError(f"{field_name} does not exist: {path.as_posix()}")


def _comparison_output_path(
    config: ResearchCampaignConfig,
    *,
    stage_name: str,
) -> Path | None:
    if config.outputs.comparison_output_path is None:
        return None
    return Path(config.outputs.comparison_output_path) / stage_name


@contextmanager
def _strategy_config_override(path: Path) -> Iterator[None]:
    default_path = Path(run_strategy_cli.STRATEGIES_CONFIG)
    if path == default_path:
        yield
        return

    original_const = run_strategy_cli.STRATEGIES_CONFIG
    original_loader = run_strategy_cli.load_strategies_config

    def _load_override(_: Path = path) -> dict[str, dict[str, Any]]:
        return original_loader(path)

    run_strategy_cli.STRATEGIES_CONFIG = path
    run_strategy_cli.load_strategies_config = _load_override
    try:
        yield
    finally:
        run_strategy_cli.STRATEGIES_CONFIG = original_const
        run_strategy_cli.load_strategies_config = original_loader


if __name__ == "__main__":
    main()
