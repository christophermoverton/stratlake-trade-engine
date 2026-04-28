from __future__ import annotations

from pathlib import Path
from typing import Sequence

from src.execution.result import ExecutionResult, summarize_execution_result


def generate_regime_review_pack(
    *,
    config_path: str | Path | None = None,
    benchmark_path: str | Path | None = None,
    promotion_gates_path: str | Path | None = None,
    output_root: str | Path | None = None,
    review_name: str | None = None,
) -> ExecutionResult:
    from src.config.regime_review_pack import (
        RegimeReviewPackConfig,
        load_regime_review_pack_config,
    )
    from src.research.regime_review_pack import run_regime_review_pack

    if config_path is not None:
        config = load_regime_review_pack_config(Path(config_path))
        if benchmark_path is not None or promotion_gates_path is not None or output_root is not None:
            config = RegimeReviewPackConfig(
                review_name=review_name or config.review_name,
                benchmark_path=Path(benchmark_path or config.benchmark_path).as_posix(),
                promotion_gates_path=Path(promotion_gates_path or config.promotion_gates_path).as_posix(),
                output_root=Path(output_root or config.output_root).as_posix(),
                ranking=config.ranking,
                report=config.report,
            )
    else:
        if benchmark_path is None or promotion_gates_path is None:
            raise ValueError("Either --config or both --benchmark-path and --promotion-gates-path are required.")
        config = RegimeReviewPackConfig(
            review_name=review_name or "regime_policy_review",
            benchmark_path=Path(benchmark_path).as_posix(),
            promotion_gates_path=Path(promotion_gates_path).as_posix(),
            output_root=Path(output_root or "artifacts/regime_reviews").as_posix(),
        )

    raw_result = run_regime_review_pack(config)
    return summarize_execution_result(
        workflow="regime_review_pack",
        raw_result=raw_result,
        run_id=raw_result.review_run_id,
        name=raw_result.review_name,
        artifact_dir=raw_result.output_dir,
        manifest_path=raw_result.manifest_path,
        metrics=dict(raw_result.review_summary.get("decision_counts", {})),
        output_paths={
            "leaderboard_csv": raw_result.leaderboard_csv_path,
            "leaderboard_json": raw_result.leaderboard_json_path,
            "decision_log_json": raw_result.decision_log_path,
            "review_summary_json": raw_result.review_summary_path,
            "accepted_variants_csv": raw_result.accepted_variants_path,
            "warning_variants_csv": raw_result.warning_variants_path,
            "needs_review_variants_csv": raw_result.needs_review_variants_path,
            "rejected_candidates_csv": raw_result.rejected_candidates_path,
            "evidence_index_json": raw_result.evidence_index_path,
            "config_json": raw_result.config_path,
            "manifest_json": raw_result.manifest_path,
            "report_md": raw_result.report_path,
            "review_inputs_json": raw_result.review_inputs_path,
        },
        extra={
            "source_benchmark_run_id": raw_result.source_benchmark_run_id,
            "source_gate_config_name": raw_result.source_gate_config_name,
        },
    )


def generate_regime_review_pack_from_cli_args(args) -> ExecutionResult:
    return generate_regime_review_pack(
        config_path=args.config,
        benchmark_path=args.benchmark_path,
        promotion_gates_path=args.promotion_gates_path,
        output_root=args.output_root,
        review_name=args.review_name,
    )


def generate_regime_review_pack_from_argv(argv: Sequence[str] | None = None) -> ExecutionResult:
    from src.cli.generate_regime_review_pack import parse_args

    return generate_regime_review_pack_from_cli_args(parse_args(argv))

