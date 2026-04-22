from __future__ import annotations

from pathlib import Path
from typing import Sequence

from src.execution.result import ExecutionResult


def run_deterministic_rerun_validation(
    *,
    repo_root: str | Path = ".",
    workdir: str | Path = "artifacts/qa/rerun_workdir",
    output: str | Path = "artifacts/qa/deterministic_rerun.json",
) -> ExecutionResult:
    """Run deterministic rerun validation and write the standard JSON report."""

    from src.validation.deterministic_rerun import (
        run_deterministic_rerun_validation as run_validation,
        write_deterministic_rerun_report,
    )

    report = run_validation(repo_root=Path(repo_root), output_root=Path(workdir))
    output_path = write_deterministic_rerun_report(report, Path(output))
    return ExecutionResult(
        workflow="deterministic_rerun_validation",
        run_id="deterministic_rerun",
        name="deterministic_rerun_validation",
        artifact_dir=output_path.parent,
        metrics=dict(report),
        output_paths={"report_json": output_path},
        raw_result=report,
    )


def run_milestone_validation(
    *,
    repo_root: str | Path = ".",
    bundle_dir: str | Path = "artifacts/qa/milestone_validation_bundle",
    include_full_pytest: bool = False,
) -> ExecutionResult:
    """Run milestone validation bundle generation through the shared API."""

    from src.validation.milestone_bundle import build_milestone_validation_bundle

    summary = build_milestone_validation_bundle(
        repo_root=Path(repo_root),
        bundle_dir=Path(bundle_dir),
        include_full_pytest=include_full_pytest,
    )
    return ExecutionResult(
        workflow="milestone_validation",
        run_id="milestone_validation",
        name="milestone_validation",
        artifact_dir=Path(str(summary["bundle_dir"])),
        metrics=dict(summary),
        raw_result=summary,
    )


def run_deterministic_rerun_validation_from_cli_args(args) -> ExecutionResult:
    return run_deterministic_rerun_validation(
        repo_root=args.repo_root,
        workdir=args.workdir,
        output=args.output,
    )


def run_milestone_validation_from_cli_args(args) -> ExecutionResult:
    return run_milestone_validation(
        repo_root=args.repo_root,
        bundle_dir=args.bundle_dir,
        include_full_pytest=bool(args.include_full_pytest),
    )


def run_deterministic_rerun_validation_from_argv(argv: Sequence[str] | None = None) -> ExecutionResult:
    from src.cli.run_deterministic_rerun_validation import parse_args

    return run_deterministic_rerun_validation_from_cli_args(parse_args(argv))


def run_milestone_validation_from_argv(argv: Sequence[str] | None = None) -> ExecutionResult:
    from src.cli.run_milestone_validation import parse_args

    return run_milestone_validation_from_cli_args(parse_args(argv))
