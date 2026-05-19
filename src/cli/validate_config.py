from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from src.config.profiles import RuntimeProfileError, load_runtime_profile
from src.config.resolution import ConfigResolutionError, resolve_runtime_profile_config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate StratLake runtime profile and resolved configuration contracts."
    )
    profile_group = parser.add_mutually_exclusive_group()
    profile_group.add_argument(
        "--profile",
        choices=("local", "ci", "notebook", "pipeline"),
        help="Supported runtime profile name to validate.",
    )
    profile_group.add_argument(
        "--profile-path",
        help="Explicit runtime profile YAML path to validate.",
    )
    parser.add_argument(
        "--output",
        help="Optional deterministic JSON report path. Prefer artifacts/_derived/config_validation/.",
    )
    return parser.parse_args(argv)


def run_cli(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    report = validate_config_from_args(args)
    _emit_report(report, args.output)
    _print_summary(report)
    if report["status"] != "passed":
        raise SystemExit(1)
    return report


def validate_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    profile_name = args.profile
    profile_path = args.profile_path
    try:
        if profile_path is not None:
            loaded_profile = load_runtime_profile(profile_path)
            result = resolve_runtime_profile_config(profile_path=profile_path)
            profile_payload = {
                "name": loaded_profile.profile,
                "path": _display_path(profile_path),
            }
        else:
            result = resolve_runtime_profile_config(profile_name)
            profile_payload = {
                "name": result.profile_name,
                "path": result.profile_path,
            }
    except (RuntimeProfileError, ConfigResolutionError, FileNotFoundError, OSError, ValueError) as exc:
        report = _failure_report(
            message=_safe_message(str(exc), profile_path),
            profile_name=profile_name,
            profile_path=profile_path,
        )
        return report

    resolved = result.to_json_dict()
    return {
        "status": "passed",
        "schema_version": 1,
        "validated": True,
        "authoritative": False,
        "run_type": "config_validation",
        "profile": profile_payload,
        "findings": [],
        "resolved_config": resolved["config"],
        "provenance": resolved["provenance"],
        "precedence": resolved["precedence"],
        "artifact_boundaries": resolved["artifact_boundaries"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run_cli(argv)
    except SystemExit as exc:
        code = exc.code
        return int(code) if isinstance(code, int) else 1
    return 0


def _failure_report(
    *,
    message: str,
    profile_name: str | None,
    profile_path: str | None,
) -> dict[str, Any]:
    return {
        "status": "failed",
        "schema_version": 1,
        "validated": False,
        "authoritative": False,
        "run_type": "config_validation",
        "profile": {
            "name": profile_name,
            "path": None if profile_path is None else _display_path(profile_path),
        },
        "findings": [
            {
                "severity": "error",
                "message": message,
            }
        ],
    }


def _emit_report(report: dict[str, Any], output: str | None) -> None:
    if output is None:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _print_summary(report: dict[str, Any]) -> None:
    print(f"config_validation_status: {report['status']}", file=sys.stderr)
    print(f"finding_count: {len(report.get('findings', []))}", file=sys.stderr)


def _display_path(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        return candidate.as_posix()
    return f"<external>/{candidate.name}"


def _safe_message(message: str, profile_path: str | None) -> str:
    if profile_path is None:
        return message
    candidate = Path(profile_path)
    if not candidate.is_absolute():
        return message
    return message.replace(str(candidate), _display_path(candidate))


if __name__ == "__main__":
    raise SystemExit(main())
