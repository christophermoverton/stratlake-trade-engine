"""CI-safe M39 first-run configuration profile example.

This example packages the M39 profile, validation, doctor, and explain helpers
into one clean first-run path. It does not require live market data,
credentials, network access, or external services, and it does not execute
workflow engines.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cli.validate_config import parse_args as parse_validate_args
from src.cli.validate_config import validate_config_from_args
from src.config.doctor import run_environment_doctor, write_environment_doctor_report
from src.config.explain import (
    SUPPORTED_EXPLAIN_WORKFLOWS,
    build_runtime_explain_report,
    write_runtime_explain_report,
)
from src.config.profiles import SUPPORTED_RUNTIME_PROFILES


EXAMPLE_NAME = "m39_first_run_configuration_profile_example"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "docs" / "examples" / "output" / EXAMPLE_NAME
REPORT_FILENAMES = {
    "validation": "config_validation.json",
    "doctor": "environment_doctor.json",
    "explain": "strategy_explain.json",
    "synthetic_probe": "synthetic_first_run_probe.json",
    "summary": "summary.json",
}


def run_m39_first_run_configuration_profile_example(
    *,
    profile: str = "ci",
    workflow: str = "strategy",
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    reset_output: bool = True,
) -> dict[str, Any]:
    """Run a deterministic first-run readiness example and return its summary."""

    if profile not in SUPPORTED_RUNTIME_PROFILES:
        supported = ", ".join(sorted(SUPPORTED_RUNTIME_PROFILES))
        raise ValueError(f"Unsupported profile {profile!r}. Supported profiles: {supported}.")
    if workflow not in SUPPORTED_EXPLAIN_WORKFLOWS:
        supported = ", ".join(sorted(SUPPORTED_EXPLAIN_WORKFLOWS))
        raise ValueError(f"Unsupported workflow {workflow!r}. Supported workflows: {supported}.")

    root = Path(output_root)
    if reset_output:
        _reset_output_root(root)
    root.mkdir(parents=True, exist_ok=True)

    validation_report = validate_config_from_args(parse_validate_args(["--profile", profile]))
    validation_path = root / REPORT_FILENAMES["validation"]
    _write_json(validation_path, validation_report)

    doctor_path = root / REPORT_FILENAMES["doctor"]
    doctor_report = run_environment_doctor(profile, output_path=doctor_path)
    write_environment_doctor_report(doctor_report, doctor_path)
    doctor_payload = doctor_report.to_json_dict()

    explain_path = root / REPORT_FILENAMES["explain"]
    explain_report = build_runtime_explain_report(profile, workflow=workflow, output_path=explain_path)
    write_runtime_explain_report(explain_report, explain_path)
    explain_payload = explain_report.to_json_dict()

    synthetic_probe = _build_synthetic_probe(
        profile=profile,
        workflow=workflow,
        validation_report=validation_report,
        doctor_report=doctor_payload,
        explain_report=explain_payload,
    )
    synthetic_probe_path = root / REPORT_FILENAMES["synthetic_probe"]
    _write_json(synthetic_probe_path, synthetic_probe)

    summary = _build_summary(
        profile=profile,
        workflow=workflow,
        output_root=root,
        validation_report=validation_report,
        doctor_report=doctor_payload,
        explain_report=explain_payload,
        synthetic_probe=synthetic_probe,
    )
    _write_json(root / REPORT_FILENAMES["summary"], summary)
    _assert_portable_payload(summary, output_root=root)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the CI-safe M39 first-run configuration profile example."
    )
    parser.add_argument(
        "--profile",
        choices=tuple(sorted(SUPPORTED_RUNTIME_PROFILES)),
        default="ci",
        help="Runtime profile to validate, doctor, and explain.",
    )
    parser.add_argument(
        "--workflow",
        choices=tuple(sorted(SUPPORTED_EXPLAIN_WORKFLOWS)),
        default="strategy",
        help="Workflow subject for the explain report.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Generated example output directory.",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Keep existing files under the output root.",
    )
    args = parser.parse_args(argv)

    summary = run_m39_first_run_configuration_profile_example(
        profile=args.profile,
        workflow=args.workflow,
        output_root=args.output_root,
        reset_output=not args.no_reset,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


def _build_synthetic_probe(
    *,
    profile: str,
    workflow: str,
    validation_report: dict[str, Any],
    doctor_report: dict[str, Any],
    explain_report: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "profile_validation": validation_report["status"],
        "environment_doctor": doctor_report["status"],
        "runtime_explain": explain_report["status"],
    }
    passed = all(status == "passed" for status in checks.values())
    return {
        "status": "passed" if passed else "failed",
        "schema_version": 1,
        "run_type": "m39_first_run_synthetic_probe",
        "authoritative": False,
        "profile": profile,
        "workflow": workflow,
        "checks": checks,
        "synthetic_inputs": [
            {
                "name": "starter_profile",
                "value": profile,
                "purpose": "Validate the selected non-secret profile contract.",
            },
            {
                "name": "workflow_subject",
                "value": workflow,
                "purpose": "Explain assumptions for a representative workflow subject.",
            },
        ],
        "safety": {
            "workflows_executed": False,
            "canonical_artifacts_mutated": False,
            "requires_network": False,
            "requires_credentials": False,
            "requires_live_market_data": False,
        },
        "note": "This synthetic first-run probe verifies configuration readiness only.",
    }


def _build_summary(
    *,
    profile: str,
    workflow: str,
    output_root: Path,
    validation_report: dict[str, Any],
    doctor_report: dict[str, Any],
    explain_report: dict[str, Any],
    synthetic_probe: dict[str, Any],
) -> dict[str, Any]:
    statuses = {
        "profile_validation": validation_report["status"],
        "environment_doctor": doctor_report["status"],
        "runtime_explain": explain_report["status"],
        "synthetic_probe": synthetic_probe["status"],
    }
    passed = all(status == "passed" for status in statuses.values())
    report_paths = {
        name: _display_output_path(output_root / filename, output_root=output_root)
        for name, filename in sorted(REPORT_FILENAMES.items())
    }
    return {
        "status": "passed" if passed else "failed",
        "schema_version": 1,
        "run_type": "m39_first_run_configuration_profile_example",
        "authoritative": False,
        "profile": profile,
        "workflow": workflow,
        "output_root": _display_output_path(output_root, output_root=output_root),
        "reports": report_paths,
        "statuses": statuses,
        "starter_templates": {
            "profiles": sorted(SUPPORTED_RUNTIME_PROFILES),
            "profile_directory": "configs/profiles",
        },
        "artifact_boundaries": {
            "direct_scan": explain_report["artifact_boundaries"]["direct_scan"],
            "derived_outputs_authoritative": False,
            "mutates_canonical_artifacts": False,
        },
        "safety": {
            "workflows_executed": False,
            "canonical_artifacts_mutated": False,
            "requires_network": False,
            "requires_credentials": False,
            "requires_live_market_data": False,
        },
        "advisory": True,
        "next_steps": [
            "Inspect config_validation.json for resolved profile values.",
            "Inspect environment_doctor.json for readiness warnings or skips.",
            "Inspect strategy_explain.json before running a real workflow.",
        ],
    }


def _reset_output_root(output_root: Path) -> None:
    if not output_root.exists():
        return
    resolved = output_root.resolve()
    default_parent = (REPO_ROOT / "docs" / "examples" / "output").resolve()
    try:
        resolved.relative_to(default_parent)
    except ValueError:
        raise ValueError(
            "Refusing to reset output outside docs/examples/output. "
            "Pass a fresh output directory or use --no-reset."
        ) from None
    shutil.rmtree(resolved)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _assert_portable_payload(payload, output_root=path.parent)


def _display_output_path(path: Path, *, output_root: Path) -> str:
    candidate = path
    if not candidate.is_absolute():
        return candidate.as_posix()
    try:
        return candidate.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        pass
    try:
        relative = candidate.resolve().relative_to(output_root.resolve()).as_posix()
    except ValueError:
        return f"<external>/{candidate.name}"
    if relative == ".":
        return f"<external>/{output_root.name}"
    return f"<output>/{relative}"


def _assert_portable_payload(payload: dict[str, Any], *, output_root: Path) -> None:
    serialized = json.dumps(payload, sort_keys=True)
    invalid_fragments = (
        str(output_root),
        "file://",
        "\\",
        "C:/",
        "../",
        '": "/',
    )
    for fragment in invalid_fragments:
        if fragment and fragment in serialized:
            raise AssertionError(f"Generated first-run output contains non-portable path: {fragment}")


if __name__ == "__main__":
    raise SystemExit(main())
