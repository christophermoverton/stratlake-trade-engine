from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence

from src.validation.deterministic_rerun import (
    run_deterministic_rerun_validation,
    write_deterministic_rerun_report,
)
from src.validation.docs_path_lint import lint_guarded_surfaces, write_docs_path_lint_report


@dataclass(frozen=True)
class CommandResult:
    name: str
    command: list[str]
    status: str
    exit_code: int
    duration_seconds: float
    log_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_milestone_validation_bundle(
    *,
    repo_root: str | Path,
    bundle_dir: str | Path,
    include_full_pytest: bool = False,
    pytest_targets: Sequence[str] | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    bundle = Path(bundle_dir).resolve()
    checks_dir = bundle / "checks"
    checks_dir.mkdir(parents=True, exist_ok=True)

    started_at = _now_utc()

    docs_lint_report = lint_guarded_surfaces(root)
    docs_lint_path = write_docs_path_lint_report(docs_lint_report, checks_dir / "docs_path_lint.json")

    rerun_report = run_deterministic_rerun_validation(
        repo_root=root,
        output_root=checks_dir / "rerun_workdir",
    )
    rerun_path = write_deterministic_rerun_report(rerun_report, checks_dir / "deterministic_rerun.json")

    command_results: list[CommandResult] = []
    command_results.append(
        _run_command(
            name="ruff",
            command=[
                sys.executable,
                "-m",
                "ruff",
                "check",
                "src/validation/",
                "src/cli/run_docs_path_lint.py",
                "src/cli/run_deterministic_rerun_validation.py",
                "src/cli/run_milestone_validation.py",
                "tests/test_docs_path_portability.py",
                "tests/test_m22_deterministic_rerun_validation.py",
                "tests/test_canonical_pipeline_examples.py",
            ],
            cwd=root,
            log_path=checks_dir / "ruff.log",
        )
    )

    selected_targets = tuple(
        pytest_targets
        or (
            "tests/test_docs_path_portability.py",
            "tests/test_m22_deterministic_rerun_validation.py",
            "tests/test_canonical_pipeline_examples.py",
        )
    )
    command_results.append(
        _run_command(
            name="pytest_milestone_slice",
            command=[
                sys.executable,
                "-m",
                "pytest",
                "--tb=short",
                "-q",
                *selected_targets,
                "--junitxml",
                str(checks_dir / "pytest_milestone_slice_junit.xml"),
            ],
            cwd=root,
            log_path=checks_dir / "pytest_milestone_slice.log",
        )
    )

    if include_full_pytest:
        command_results.append(
            _run_command(
                name="pytest_full",
                command=[
                    sys.executable,
                    "-m",
                    "pytest",
                    "--tb=short",
                    "-q",
                    "--junitxml",
                    str(checks_dir / "pytest_full_junit.xml"),
                ],
                cwd=root,
                log_path=checks_dir / "pytest_full.log",
            )
        )

    command_status = all(result.status == "passed" for result in command_results)
    overall_passed = (
        docs_lint_report["status"] == "passed"
        and rerun_report["status"] == "passed"
        and command_status
    )

    finished_at = _now_utc()
    summary = {
        "run_type": "milestone_validation_bundle",
        "schema_version": 1,
        "status": "passed" if overall_passed else "failed",
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "bundle_dir": bundle.as_posix(),
        "checks": {
            "docs_path_lint": {
                "status": docs_lint_report["status"],
                "report_path": docs_lint_path.relative_to(bundle).as_posix(),
                "finding_count": docs_lint_report["finding_count"],
                "guarded_file_count": docs_lint_report["guarded_file_count"],
            },
            "deterministic_rerun": {
                "status": rerun_report["status"],
                "report_path": rerun_path.relative_to(bundle).as_posix(),
                "target_count": rerun_report["target_count"],
                "pass_count": rerun_report["pass_count"],
            },
            "commands": [result.to_dict() for result in command_results],
        },
        "pytest_targets": list(selected_targets),
        "include_full_pytest": include_full_pytest,
    }

    summary_path = bundle / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _run_command(*, name: str, command: list[str], cwd: Path, log_path: Path) -> CommandResult:
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    duration = time.perf_counter() - started

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(
            [
                f"$ {' '.join(command)}",
                "",
                "[stdout]",
                completed.stdout.rstrip(),
                "",
                "[stderr]",
                completed.stderr.rstrip(),
                "",
                f"exit_code={completed.returncode}",
            ]
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    return CommandResult(
        name=name,
        command=command,
        status="passed" if completed.returncode == 0 else "failed",
        exit_code=completed.returncode,
        duration_seconds=round(duration, 3),
        log_path=log_path.as_posix(),
    )


def _now_utc() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
