from __future__ import annotations

from pathlib import Path
from typing import Sequence

from src.execution.result import ExecutionResult


def run_docs_path_lint(
    *,
    repo_root: str | Path = ".",
    output: str | Path = "artifacts/qa/docs_path_lint.json",
) -> ExecutionResult:
    """Run guarded docs/path lint validation and write the standard JSON report."""

    from src.validation.docs_path_lint import lint_guarded_surfaces, write_docs_path_lint_report

    report = lint_guarded_surfaces(repo_root=Path(repo_root))
    output_path = write_docs_path_lint_report(report, Path(output))
    return ExecutionResult(
        workflow="docs_path_lint",
        run_id="docs_path_lint",
        name="docs_path_lint",
        artifact_dir=output_path.parent,
        metrics=dict(report),
        output_paths={"report_json": output_path},
        extra={
            "guarded_surfaces": list(report.get("guarded_surfaces", [])),
            "guarded_file_count": report.get("guarded_file_count"),
            "finding_count": report.get("finding_count"),
        },
        raw_result=report,
    )


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
        extra={
            "target_count": report.get("target_count"),
            "pass_count": report.get("pass_count"),
        },
        raw_result=report,
    )


def run_cross_layer_validation(
    *,
    repo_root: str | Path = ".",
    workdir: str | Path = "artifacts/qa/m28_cross_layer_validation",
    output: str | Path = "artifacts/qa/cross_layer_validation_report.json",
    scenarios: Sequence[str] | None = None,
) -> ExecutionResult:
    """Run representative M28 cross-layer validation and write the JSON report."""

    from src.validation.cross_layer import (
        run_cross_layer_validation as run_validation,
        write_cross_layer_validation_report,
    )

    report = run_validation(
        repo_root=Path(repo_root),
        output_root=Path(workdir),
        scenarios=scenarios,
    )
    output_path = write_cross_layer_validation_report(report, Path(output))
    return ExecutionResult(
        workflow="cross_layer_validation",
        run_id="cross_layer_validation",
        name="cross_layer_validation",
        artifact_dir=output_path.parent,
        metrics=dict(report),
        output_paths={"report_json": output_path},
        extra={
            "scenario_count": report.get("scenario_count"),
            "pass_count": report.get("pass_count"),
        },
        raw_result=report,
    )


def run_milestone_validation(
    *,
    repo_root: str | Path = ".",
    bundle_dir: str | Path = "artifacts/qa/milestone_validation_bundle",
    include_full_pytest: bool = False,
    include_cross_layer_validation: bool = False,
) -> ExecutionResult:
    """Run milestone validation bundle generation through the shared API."""

    from src.validation.milestone_bundle import build_milestone_validation_bundle

    summary = build_milestone_validation_bundle(
        repo_root=Path(repo_root),
        bundle_dir=Path(bundle_dir),
        include_full_pytest=include_full_pytest,
        include_cross_layer_validation=include_cross_layer_validation,
    )
    resolved_bundle_dir = Path(str(summary["bundle_dir"]))
    output_paths = {
        key: value
        for key, value in {
            "summary_json": resolved_bundle_dir / "summary.json",
            "docs_path_lint_json": _bundle_path(summary, "docs_path_lint"),
            "deterministic_rerun_json": _bundle_path(summary, "deterministic_rerun"),
            "cross_layer_validation_json": _bundle_path(summary, "cross_layer_validation"),
        }.items()
        if value is not None
    }
    return ExecutionResult(
        workflow="milestone_validation",
        run_id="milestone_validation",
        name="milestone_validation",
        artifact_dir=resolved_bundle_dir,
        metrics=dict(summary),
        output_paths=output_paths,
        extra={
            "checks": dict(summary.get("checks", {})),
            "include_full_pytest": summary.get("include_full_pytest"),
            "include_cross_layer_validation": summary.get("include_cross_layer_validation"),
            "pytest_targets": list(summary.get("pytest_targets", [])),
        },
        raw_result=summary,
    )


def run_docs_path_lint_from_cli_args(args) -> ExecutionResult:
    return run_docs_path_lint(
        repo_root=args.repo_root,
        output=args.output,
    )


def run_deterministic_rerun_validation_from_cli_args(args) -> ExecutionResult:
    return run_deterministic_rerun_validation(
        repo_root=args.repo_root,
        workdir=args.workdir,
        output=args.output,
    )


def run_cross_layer_validation_from_cli_args(args) -> ExecutionResult:
    return run_cross_layer_validation(
        repo_root=args.repo_root,
        workdir=args.workdir,
        output=args.output,
        scenarios=tuple(args.scenario) if args.scenario else None,
    )


def run_milestone_validation_from_cli_args(args) -> ExecutionResult:
    return run_milestone_validation(
        repo_root=args.repo_root,
        bundle_dir=args.bundle_dir,
        include_full_pytest=bool(args.include_full_pytest),
        include_cross_layer_validation=bool(getattr(args, "include_cross_layer_validation", False)),
    )


def run_docs_path_lint_from_argv(argv: Sequence[str] | None = None) -> ExecutionResult:
    from src.cli.run_docs_path_lint import parse_args

    return run_docs_path_lint_from_cli_args(parse_args(argv))


def run_deterministic_rerun_validation_from_argv(argv: Sequence[str] | None = None) -> ExecutionResult:
    from src.cli.run_deterministic_rerun_validation import parse_args

    return run_deterministic_rerun_validation_from_cli_args(parse_args(argv))


def run_cross_layer_validation_from_argv(argv: Sequence[str] | None = None) -> ExecutionResult:
    from src.cli.run_cross_layer_validation import parse_args

    return run_cross_layer_validation_from_cli_args(parse_args(argv))


def run_milestone_validation_from_argv(argv: Sequence[str] | None = None) -> ExecutionResult:
    from src.cli.run_milestone_validation import parse_args

    return run_milestone_validation_from_cli_args(parse_args(argv))


def _bundle_path(summary: dict[str, object], check_name: str) -> Path | None:
    bundle_dir = Path(str(summary["bundle_dir"]))
    checks = summary.get("checks", {})
    if not isinstance(checks, dict):
        return None
    check = checks.get(check_name)
    if not isinstance(check, dict):
        return None
    report_path = check.get("report_path")
    if not isinstance(report_path, str) or not report_path:
        return None
    return bundle_dir / report_path
