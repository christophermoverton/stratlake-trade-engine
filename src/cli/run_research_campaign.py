from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
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
from src.data.load_features import FeaturePaths, SUPPORTED_FEATURE_DATASETS
from src.research.alpha.catalog import load_alphas_config
from src.research.alpha_eval.registry import alpha_evaluation_registry_path
from src.research.candidate_selection.registry import candidate_selection_registry_path
from src.research.experiment_tracker import ARTIFACTS_ROOT as STRATEGY_ARTIFACTS_ROOT
from src.research.registry import default_registry_path, load_registry

PREFLIGHT_SUMMARY_FILENAME = "preflight_summary.json"
CAMPAIGN_CONFIG_FILENAME = "campaign_config.json"


@dataclass(frozen=True)
class CampaignPreflightCheck:
    check_id: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CampaignPreflightResult:
    status: str
    summary_path: Path
    summary: dict[str, Any]
    checks: tuple[CampaignPreflightCheck, ...]


@dataclass(frozen=True)
class CampaignStageRecord:
    stage_name: str
    status: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResearchCampaignRunResult:
    config: ResearchCampaignConfig
    campaign_run_id: str
    campaign_artifact_dir: Path
    preflight_summary_path: Path
    preflight_summary: dict[str, Any]
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
    campaign_run_id = _build_campaign_run_id(config)
    campaign_artifact_dir = _campaign_artifact_dir(config, campaign_run_id=campaign_run_id)
    _persist_campaign_config(campaign_artifact_dir, config)

    preflight_result = _run_preflight(config, campaign_run_id=campaign_run_id, campaign_artifact_dir=campaign_artifact_dir)
    records.append(
        CampaignStageRecord(
            stage_name="preflight",
            status="completed" if preflight_result.status == "passed" else "failed",
            details=preflight_result.summary,
        )
    )
    if preflight_result.status != "passed":
        failures = [
            check.message
            for check in preflight_result.checks
            if check.status == "failed"
        ]
        raise ValueError(
            "Campaign preflight failed. "
            f"See {preflight_result.summary_path.as_posix()}. "
            + (" | ".join(failures) if failures else "One or more checks failed.")
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
        campaign_run_id=campaign_run_id,
        campaign_artifact_dir=campaign_artifact_dir,
        preflight_summary_path=preflight_result.summary_path,
        preflight_summary=preflight_result.summary,
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
    print(
        "Preflight: "
        f"{result.preflight_summary.get('status', 'unknown')} "
        f"({len(result.stage_records)} stages tracked) | "
        f"summary={result.preflight_summary_path.as_posix()}"
    )
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


def _run_preflight(
    config: ResearchCampaignConfig,
    *,
    campaign_run_id: str,
    campaign_artifact_dir: Path,
) -> CampaignPreflightResult:
    checks: list[CampaignPreflightCheck] = []
    alpha_catalog = {}
    strategy_catalog = {}
    portfolio_config_payload: dict[str, Any] = {}

    def add_check(
        check_id: str,
        ok: bool,
        message: str,
        **details: Any,
    ) -> None:
        checks.append(
            CampaignPreflightCheck(
                check_id=check_id,
                status="passed" if ok else "failed",
                message=message,
                details=_normalize_jsonable(details),
            )
        )

    alpha_names = config.targets.alpha_names
    strategy_names = config.targets.strategy_names

    add_check(
        "targets.present",
        bool(alpha_names or strategy_names),
        (
            "Campaign has at least one alpha or strategy target."
            if alpha_names or strategy_names
            else "Campaign requires at least one alpha or strategy target in targets.alpha_names or targets.strategy_names."
        ),
        alpha_target_count=len(alpha_names),
        strategy_target_count=len(strategy_names),
    )
    add_check(
        "candidate_selection.alpha_name",
        (not config.candidate_selection.enabled) or bool(config.candidate_selection.alpha_name),
        (
            "Candidate selection alpha_name resolved."
            if (not config.candidate_selection.enabled) or bool(config.candidate_selection.alpha_name)
            else "Candidate selection requires one resolved alpha_name via candidate_selection.alpha_name or a single targets.alpha_names entry."
        ),
        enabled=config.candidate_selection.enabled,
        alpha_name=config.candidate_selection.alpha_name,
    )
    add_check(
        "portfolio.portfolio_name",
        (not config.portfolio.enabled) or bool(config.portfolio.portfolio_name),
        (
            "Portfolio name resolved."
            if (not config.portfolio.enabled) or bool(config.portfolio.portfolio_name)
            else "Portfolio stage requires one resolved portfolio_name via portfolio.portfolio_name or a single targets.portfolio_names entry."
        ),
        enabled=config.portfolio.enabled,
        portfolio_name=config.portfolio.portfolio_name,
    )
    add_check(
        "portfolio.from_candidate_selection",
        (not config.portfolio.enabled)
        or (not config.portfolio.from_candidate_selection)
        or config.candidate_selection.enabled,
        (
            "Portfolio candidate-selection dependency is satisfied."
            if (not config.portfolio.enabled)
            or (not config.portfolio.from_candidate_selection)
            or config.candidate_selection.enabled
            else "portfolio.from_candidate_selection requires candidate_selection.enabled to be true in the same campaign."
        ),
        portfolio_enabled=config.portfolio.enabled,
        from_candidate_selection=config.portfolio.from_candidate_selection,
        candidate_selection_enabled=config.candidate_selection.enabled,
    )
    add_check(
        "candidate_review.requires_portfolio",
        (not config.candidate_selection.enabled)
        or (not config.candidate_selection.execution.enable_review)
        or config.portfolio.enabled,
        (
            "Candidate review dependency is satisfied."
            if (not config.candidate_selection.enabled)
            or (not config.candidate_selection.execution.enable_review)
            or config.portfolio.enabled
            else "candidate_selection.execution.enable_review requires portfolio.enabled so the review stage has portfolio artifacts."
        ),
        candidate_selection_enabled=config.candidate_selection.enabled,
        enable_review=config.candidate_selection.execution.enable_review,
        portfolio_enabled=config.portfolio.enabled,
    )

    alpha_catalog_path = Path(config.targets.alpha_catalog_path)
    strategy_config_path = Path(config.targets.strategy_config_path)
    portfolio_config_path = Path(config.targets.portfolio_config_path)

    add_check(
        "paths.alpha_catalog",
        alpha_catalog_path.exists(),
        (
            f"targets.alpha_catalog_path exists: {alpha_catalog_path.as_posix()}"
            if alpha_catalog_path.exists()
            else f"targets.alpha_catalog_path does not exist: {alpha_catalog_path.as_posix()}"
        ),
        path=alpha_catalog_path,
    )
    if alpha_catalog_path.exists():
        try:
            alpha_catalog = load_alphas_config(alpha_catalog_path)
            add_check(
                "catalog.alpha.load",
                True,
                f"Loaded alpha catalog from {alpha_catalog_path.as_posix()}.",
                alpha_count=len(alpha_catalog),
            )
        except Exception as exc:
            add_check(
                "catalog.alpha.load",
                False,
                f"Failed to load alpha catalog: {exc}",
                path=alpha_catalog_path,
            )
    for alpha_name in alpha_names:
        add_check(
            f"catalog.alpha.target.{alpha_name}",
            alpha_name in alpha_catalog,
            (
                f"Resolved alpha target '{alpha_name}'."
                if alpha_name in alpha_catalog
                else f"Unknown alpha '{alpha_name}' in targets.alpha_names."
            ),
            alpha_name=alpha_name,
        )

    add_check(
        "paths.strategy_config",
        strategy_config_path.exists(),
        (
            f"targets.strategy_config_path exists: {strategy_config_path.as_posix()}"
            if strategy_config_path.exists()
            else f"targets.strategy_config_path does not exist: {strategy_config_path.as_posix()}"
        ),
        path=strategy_config_path,
    )
    if strategy_config_path.exists():
        try:
            strategy_catalog = run_strategy_cli.load_strategies_config(strategy_config_path)
            add_check(
                "catalog.strategy.load",
                True,
                f"Loaded strategy config from {strategy_config_path.as_posix()}.",
                strategy_count=len(strategy_catalog),
            )
        except Exception as exc:
            add_check(
                "catalog.strategy.load",
                False,
                f"Failed to load strategy config: {exc}",
                path=strategy_config_path,
            )
    for strategy_name in strategy_names:
        add_check(
            f"catalog.strategy.target.{strategy_name}",
            strategy_name in strategy_catalog,
            (
                f"Resolved strategy target '{strategy_name}'."
                if strategy_name in strategy_catalog
                else f"Unknown strategy '{strategy_name}' in targets.strategy_names."
            ),
            strategy_name=strategy_name,
        )

    require_portfolio_config = config.portfolio.enabled and not config.portfolio.from_candidate_selection
    add_check(
        "paths.portfolio_config",
        (not require_portfolio_config) or portfolio_config_path.exists(),
        (
            f"targets.portfolio_config_path exists: {portfolio_config_path.as_posix()}"
            if (not require_portfolio_config) or portfolio_config_path.exists()
            else f"targets.portfolio_config_path does not exist: {portfolio_config_path.as_posix()}"
        ),
        required=require_portfolio_config,
        path=portfolio_config_path,
    )
    if require_portfolio_config and portfolio_config_path.exists():
        try:
            portfolio_config_payload = run_portfolio_cli.load_portfolio_config(portfolio_config_path)
            add_check(
                "catalog.portfolio.load",
                True,
                f"Loaded portfolio config from {portfolio_config_path.as_posix()}.",
                path=portfolio_config_path,
            )
        except Exception as exc:
            add_check(
                "catalog.portfolio.load",
                False,
                f"Failed to load portfolio config: {exc}",
                path=portfolio_config_path,
            )
        if config.portfolio.portfolio_name is not None and portfolio_config_payload:
            try:
                run_portfolio_cli.resolve_portfolio_definition(
                    portfolio_config_payload,
                    portfolio_name=config.portfolio.portfolio_name,
                )
                add_check(
                    "catalog.portfolio.target",
                    True,
                    f"Resolved portfolio target '{config.portfolio.portfolio_name}'.",
                    portfolio_name=config.portfolio.portfolio_name,
                )
            except Exception as exc:
                add_check(
                    "catalog.portfolio.target",
                    False,
                    f"Failed to resolve portfolio target '{config.portfolio.portfolio_name}': {exc}",
                    portfolio_name=config.portfolio.portfolio_name,
                )

    feature_paths = FeaturePaths()
    dataset_consumers = _collect_required_datasets(
        config,
        alpha_catalog=alpha_catalog,
        strategy_catalog=strategy_catalog,
    )
    dataset_records: list[dict[str, Any]] = []
    for dataset_name, consumers in sorted(dataset_consumers.items()):
        parquet_count = 0
        dataset_root: Path | None = None
        dataset_ok = False
        message = ""
        try:
            dataset_root = feature_paths.dataset_root(dataset_name)
            dataset_ok = dataset_root.exists()
            if dataset_ok:
                parquet_count = sum(1 for _ in dataset_root.glob("**/*.parquet"))
                dataset_ok = parquet_count > 0
            message = (
                f"Dataset '{dataset_name}' is available."
                if dataset_ok
                else f"Dataset '{dataset_name}' has no parquet files under {dataset_root.as_posix()}."
            )
        except Exception as exc:
            message = f"Dataset '{dataset_name}' failed validation: {exc}"
        add_check(
            f"dataset.{dataset_name}",
            dataset_ok,
            message,
            dataset=dataset_name,
            consumers=consumers,
            dataset_root=None if dataset_root is None else dataset_root.as_posix(),
            parquet_file_count=parquet_count,
            features_root=feature_paths.root.as_posix(),
        )
        dataset_records.append(
            {
                "dataset": dataset_name,
                "consumers": list(consumers),
                "dataset_root": None if dataset_root is None else dataset_root.as_posix(),
                "parquet_file_count": parquet_count,
            }
        )

    artifact_roots = _campaign_artifact_roots(config)
    for label, root in artifact_roots.items():
        ok, message = _ensure_directory(root)
        add_check(
            f"artifacts.{label}",
            ok,
            message,
            path=root.as_posix(),
        )

    if config.candidate_selection.enabled and config.candidate_selection.execution.from_registry:
        registry_path = (
            Path(config.candidate_selection.output.registry_path)
            if config.candidate_selection.output.registry_path is not None
            else candidate_selection_registry_path(config.candidate_selection.output.path)
        )
        ok, message, entry_count = _validate_registry_path(registry_path)
        add_check(
            "registry.candidate_selection",
            ok,
            message,
            path=registry_path.as_posix(),
            entry_count=entry_count,
        )

    if config.portfolio.enabled and config.portfolio.from_registry and not strategy_names:
        strategy_registry_path = default_registry_path(STRATEGY_ARTIFACTS_ROOT)
        ok, message, entry_count = _validate_registry_path(strategy_registry_path)
        add_check(
            "registry.strategy",
            ok,
            message,
            path=strategy_registry_path.as_posix(),
            entry_count=entry_count,
        )

    failed_checks = [check for check in checks if check.status == "failed"]
    summary = {
        "campaign_run_id": campaign_run_id,
        "status": "failed" if failed_checks else "passed",
        "campaign_artifact_dir": campaign_artifact_dir.as_posix(),
        "config_path": (campaign_artifact_dir / CAMPAIGN_CONFIG_FILENAME).as_posix(),
        "environment": {
            "cwd": Path.cwd().as_posix(),
            "features_root": feature_paths.root.as_posix(),
            "strategy_artifacts_root": STRATEGY_ARTIFACTS_ROOT.as_posix(),
            "alpha_registry_path": alpha_evaluation_registry_path(config.outputs.alpha_artifacts_root).as_posix(),
        },
        "targets": {
            "alpha_names": list(alpha_names),
            "strategy_names": list(strategy_names),
            "portfolio_names": list(config.targets.portfolio_names),
            "resolved_candidate_selection_alpha_name": config.candidate_selection.alpha_name,
            "resolved_portfolio_name": config.portfolio.portfolio_name,
        },
        "datasets": dataset_records,
        "artifact_roots": {
            key: value.as_posix()
            for key, value in artifact_roots.items()
        },
        "check_counts": {
            "total": len(checks),
            "passed": sum(1 for check in checks if check.status == "passed"),
            "failed": len(failed_checks),
        },
        "failed_checks": [check.check_id for check in failed_checks],
        "checks": [
            {
                "check_id": check.check_id,
                "status": check.status,
                "message": check.message,
                "details": check.details,
            }
            for check in checks
        ],
    }
    summary_path = campaign_artifact_dir / PREFLIGHT_SUMMARY_FILENAME
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return CampaignPreflightResult(
        status=str(summary["status"]),
        summary_path=summary_path,
        summary=summary,
        checks=tuple(checks),
    )


def _run_alpha_research(config: ResearchCampaignConfig) -> list[Any]:
    results: list[Any] = []
    for alpha_name in config.targets.alpha_names:
        argv = ["--alpha-name", alpha_name, "--config", config.targets.alpha_catalog_path]
        if config.outputs.alpha_artifacts_root:
            argv.extend(["--artifacts-root", config.outputs.alpha_artifacts_root])
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


def _build_campaign_run_id(config: ResearchCampaignConfig) -> str:
    payload = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"research_campaign_{digest}"


def _campaign_artifact_dir(config: ResearchCampaignConfig, *, campaign_run_id: str) -> Path:
    return Path(config.outputs.campaign_artifacts_root) / campaign_run_id


def _persist_campaign_config(campaign_artifact_dir: Path, config: ResearchCampaignConfig) -> Path:
    campaign_artifact_dir.mkdir(parents=True, exist_ok=True)
    path = campaign_artifact_dir / CAMPAIGN_CONFIG_FILENAME
    path.write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _collect_required_datasets(
    config: ResearchCampaignConfig,
    *,
    alpha_catalog: dict[str, dict[str, Any]],
    strategy_catalog: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    dataset_consumers: dict[str, list[str]] = {}

    def register(dataset_name: str | None, consumer: str) -> None:
        if dataset_name is None:
            return
        normalized = str(dataset_name).strip()
        if not normalized:
            return
        dataset_consumers.setdefault(normalized, [])
        if consumer not in dataset_consumers[normalized]:
            dataset_consumers[normalized].append(consumer)

    for alpha_name in config.targets.alpha_names:
        alpha_config = alpha_catalog.get(alpha_name, {})
        register(
            config.dataset_selection.dataset or alpha_config.get("dataset"),
            f"alpha:{alpha_name}",
        )

    for strategy_name in config.targets.strategy_names:
        strategy_config = strategy_catalog.get(strategy_name, {})
        register(
            strategy_config.get("dataset"),
            f"strategy:{strategy_name}",
        )

    if config.dataset_selection.dataset and not config.targets.alpha_names:
        register(config.dataset_selection.dataset, "campaign.dataset_selection")

    return {
        dataset: consumers
        for dataset, consumers in dataset_consumers.items()
        if dataset in SUPPORTED_FEATURE_DATASETS or consumers
    }


def _campaign_artifact_roots(config: ResearchCampaignConfig) -> dict[str, Path]:
    roots: dict[str, Path] = {
        "campaign_artifacts_root": Path(config.outputs.campaign_artifacts_root),
        "alpha_artifacts_root": Path(config.outputs.alpha_artifacts_root),
        "candidate_selection_artifacts_root": Path(config.candidate_selection.artifacts_root),
        "candidate_selection_output_path": Path(config.candidate_selection.output.path),
        "portfolio_artifacts_root": Path(config.outputs.portfolio_artifacts_root),
    }
    if config.outputs.comparison_output_path is not None:
        roots["comparison_output_path"] = Path(config.outputs.comparison_output_path)
    if config.review.output.path is not None:
        roots["review_output_path"] = Path(config.review.output.path)
    if config.candidate_selection.output.review_output_path is not None:
        roots["candidate_review_output_path"] = Path(config.candidate_selection.output.review_output_path)
    return roots


def _ensure_directory(path: Path) -> tuple[bool, str]:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return False, f"Failed to create or access directory {path.as_posix()}: {exc}"
    if not path.exists() or not path.is_dir():
        return False, f"Path is not a directory: {path.as_posix()}"
    return True, f"Directory is available: {path.as_posix()}"


def _validate_registry_path(path: Path) -> tuple[bool, str, int]:
    try:
        entries = load_registry(path)
    except Exception as exc:
        return False, f"Failed to load registry {path.as_posix()}: {exc}", 0
    if not path.exists():
        return False, f"Registry path does not exist: {path.as_posix()}", 0
    if not entries:
        return False, f"Registry path has no entries: {path.as_posix()}", 0
    return True, f"Registry is available: {path.as_posix()}", len(entries)


def _normalize_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _normalize_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_normalize_jsonable(item) for item in value]
    return value


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
