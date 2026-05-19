from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from src.config.profiles import RuntimeProfileError, load_runtime_profile
from src.config.resolution import (
    ConfigResolutionError,
    ConfigResolutionResult,
    resolve_runtime_profile_config,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CHECK_STATUSES = ("pass", "warning", "fail", "skipped")


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    status: str
    message: str
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "status": self.status,
            "message": self.message,
        }
        if self.details:
            payload["details"] = _canonicalize(self.details)
        return payload


@dataclass(frozen=True)
class EnvironmentDoctorReport:
    status: str
    profile: dict[str, Any]
    checks: tuple[DoctorCheck, ...]
    resolved_config: dict[str, Any] | None
    artifact_boundaries: dict[str, Any] | None
    output_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        checks = [check.to_dict() for check in self.checks]
        return _canonicalize(
            {
                "status": self.status,
                "schema_version": 1,
                "run_type": "environment_doctor",
                "authoritative": False,
                "profile": self.profile,
                "checks": checks,
                "finding_counts": _finding_counts(checks),
                "artifact_boundaries": self.artifact_boundaries,
                "resolved_config": self.resolved_config,
                "workflows_executed": False,
                "output_path": self.output_path,
            }
        )

    def to_json_dict(self) -> dict[str, Any]:
        return self.to_dict()

    def to_json(self) -> str:
        return json.dumps(self.to_json_dict(), indent=2, sort_keys=True)


def run_environment_doctor(
    profile: str | None = None,
    *,
    profile_path: str | Path | None = None,
    output_path: str | Path | None = None,
    repo_root: str | Path = REPO_ROOT,
) -> EnvironmentDoctorReport:
    """
    Build an advisory environment-readiness report without executing workflows.

    The doctor does not load .env, call external services, require credentials,
    require live market data, mutate canonical artifacts, or create output
    directories. CLI code owns optional report writing.
    """

    root = Path(repo_root)
    checks: list[DoctorCheck] = []
    profile_payload = _initial_profile_payload(profile, profile_path)
    resolved_result: ConfigResolutionResult | None = None

    checks.append(_python_runtime_check())
    checks.extend(_importability_checks())

    try:
        if profile_path is not None:
            loaded_profile = load_runtime_profile(str(profile_path))
            resolved_result = resolve_runtime_profile_config(profile_path=profile_path)
            profile_payload = {
                "name": loaded_profile.profile,
                "path": _display_path(profile_path),
            }
        else:
            resolved_result = resolve_runtime_profile_config(profile)
            profile_payload = {
                "name": resolved_result.profile_name,
                "path": resolved_result.profile_path,
            }
        checks.append(
            DoctorCheck(
                name="profile_resolves",
                status="pass",
                message="Runtime profile and resolved configuration loaded successfully.",
            )
        )
    except (RuntimeProfileError, ConfigResolutionError, FileNotFoundError, OSError, ValueError) as exc:
        checks.append(
            DoctorCheck(
                name="profile_resolves",
                status="fail",
                message=_safe_message(str(exc), profile_path),
            )
        )
        checks.append(
            DoctorCheck(
                name="readiness_checks",
                status="skipped",
                message="Configuration-dependent readiness checks were skipped because profile resolution failed.",
            )
        )
        return _build_report(
            checks=checks,
            profile=profile_payload,
            resolved_result=None,
            output_path=output_path,
        )

    checks.extend(_boundary_checks(resolved_result))
    checks.extend(_path_portability_checks(resolved_result))
    checks.extend(_workflow_config_checks(resolved_result, root))
    checks.extend(_optional_root_checks(resolved_result, root))
    checks.append(_output_path_check(output_path))

    return _build_report(
        checks=checks,
        profile=profile_payload,
        resolved_result=resolved_result,
        output_path=output_path,
    )


def write_environment_doctor_report(
    report: EnvironmentDoctorReport,
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.to_json() + "\n", encoding="utf-8")
    return output


def _build_report(
    *,
    checks: list[DoctorCheck],
    profile: dict[str, Any],
    resolved_result: ConfigResolutionResult | None,
    output_path: str | Path | None,
) -> EnvironmentDoctorReport:
    status = "failed" if any(check.status == "fail" for check in checks) else "passed"
    resolved_payload = None if resolved_result is None else resolved_result.to_json_dict()
    return EnvironmentDoctorReport(
        status=status,
        profile=profile,
        checks=tuple(sorted(checks, key=lambda check: check.name)),
        resolved_config=None if resolved_payload is None else resolved_payload["config"],
        artifact_boundaries=None if resolved_payload is None else resolved_payload["artifact_boundaries"],
        output_path=None if output_path is None else _display_path(output_path),
    )


def _python_runtime_check() -> DoctorCheck:
    major, minor, micro = sys.version_info[:3]
    status = "pass" if (major, minor) >= (3, 10) else "fail"
    return DoctorCheck(
        name="python_runtime",
        status=status,
        message=f"Python runtime is {major}.{minor}.{micro}.",
        details={
            "python_version": f"{major}.{minor}.{micro}",
            "minimum_supported": "3.10",
        },
    )


def _importability_checks() -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    for module_name in (
        "src.config.profiles",
        "src.config.resolution",
        "src.config.runtime",
        "src.cli.validate_config",
    ):
        spec = importlib.util.find_spec(module_name)
        checks.append(
            DoctorCheck(
                name=f"module_importable:{module_name}",
                status="pass" if spec is not None else "fail",
                message=(
                    f"Module {module_name} is importable."
                    if spec is not None
                    else f"Module {module_name} is not importable."
                ),
            )
        )
    return checks


def _boundary_checks(result: ConfigResolutionResult) -> list[DoctorCheck]:
    boundaries = result.config.boundaries
    expectations = {
        "direct_scan": True,
        "derived_outputs_authoritative": False,
        "mutates_canonical_artifacts": False,
        "requires_network": False,
        "requires_credentials": False,
        "requires_live_market_data": False,
    }
    checks: list[DoctorCheck] = []
    for key, expected in sorted(expectations.items()):
        actual = boundaries.get(key)
        checks.append(
            DoctorCheck(
                name=f"boundary:{key}",
                status="pass" if actual is expected else "fail",
                message=(
                    f"Boundary {key} is {actual}."
                    if actual is expected
                    else f"Boundary {key} is {actual}; expected {expected}."
                ),
                details={"expected": expected, "actual": actual},
            )
        )
    return checks


def _path_portability_checks(result: ConfigResolutionResult) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    config = result.config.to_dict()
    path_fields: dict[str, Any] = {}
    for key, value in config["settings"].items():
        if key.endswith("_root") or key == "duckdb_path":
            path_fields[f"settings.{key}"] = value
    for key, value in config["workflow_configs"].items():
        path_fields[f"workflow_configs.{key}"] = value
    review_path = config.get("review", {}).get("output", {}).get("path")
    if review_path is not None:
        path_fields["review.output.path"] = review_path

    for field, value in sorted(path_fields.items()):
        if value is None or value == ":memory:":
            checks.append(
                DoctorCheck(
                    name=f"path_portable:{field}",
                    status="skipped",
                    message=f"Path field {field} is not filesystem-backed in this profile.",
                )
            )
            continue
        try:
            _validate_portable_path(str(value))
        except ValueError as exc:
            checks.append(
                DoctorCheck(
                    name=f"path_portable:{field}",
                    status="fail",
                    message=str(exc),
                )
            )
        else:
            checks.append(
                DoctorCheck(
                    name=f"path_portable:{field}",
                    status="pass",
                    message=f"Path field {field} is portable and repository-relative.",
                )
            )
    return checks


def _workflow_config_checks(result: ConfigResolutionResult, repo_root: Path) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    for key, value in sorted(result.config.workflow_configs.items()):
        path = repo_root / value
        if path.exists() and path.is_file():
            status = "pass"
            message = f"Workflow config {key} exists."
        elif key == "config_dir" and path.exists() and path.is_dir():
            status = "pass"
            message = "Workflow config directory exists."
        else:
            status = "warning"
            message = f"Workflow config reference {key} was not found; no workflow was executed."
        checks.append(
            DoctorCheck(
                name=f"workflow_config_exists:{key}",
                status=status,
                message=message,
                details={"path": str(value)},
            )
        )
    return checks


def _optional_root_checks(result: ConfigResolutionResult, repo_root: Path) -> list[DoctorCheck]:
    settings = result.config.settings
    checks = [
        _optional_read_root_check(
            "features_root",
            settings.get("features_root"),
            repo_root,
            missing_status="skipped",
            missing_message="Features root does not exist; this is acceptable before feature generation.",
        ),
        _optional_read_root_check(
            "marketlake_root",
            settings.get("marketlake_root"),
            repo_root,
            missing_status="skipped",
            missing_message="Marketlake root does not exist; clean CI profiles do not require live data.",
        ),
        _writable_root_check(settings.get("artifacts_root"), repo_root),
    ]
    return checks


def _optional_read_root_check(
    name: str,
    value: Any,
    repo_root: Path,
    *,
    missing_status: str,
    missing_message: str,
) -> DoctorCheck:
    if value is None:
        return DoctorCheck(
            name=f"optional_root:{name}",
            status="skipped",
            message=f"{name} is not configured.",
        )
    path = repo_root / str(value)
    if not path.exists():
        return DoctorCheck(
            name=f"optional_root:{name}",
            status=missing_status,
            message=missing_message,
            details={"path": str(value)},
        )
    if path.is_dir():
        return DoctorCheck(
            name=f"optional_root:{name}",
            status="pass",
            message=f"{name} exists and is a directory.",
            details={"path": str(value)},
        )
    return DoctorCheck(
        name=f"optional_root:{name}",
        status="warning",
        message=f"{name} exists but is not a directory.",
        details={"path": str(value)},
    )


def _writable_root_check(value: Any, repo_root: Path) -> DoctorCheck:
    if value is None:
        return DoctorCheck(
            name="artifacts_root_writable",
            status="skipped",
            message="artifacts_root is not configured.",
        )
    relative = str(value)
    path = repo_root / relative
    target = path if path.exists() else _nearest_existing_parent(path, repo_root)
    if target is None:
        return DoctorCheck(
            name="artifacts_root_writable",
            status="warning",
            message="No existing parent was found for artifacts_root; no directories were created.",
            details={"path": relative},
        )
    writable = target.is_dir() and _is_writable_directory(target)
    return DoctorCheck(
        name="artifacts_root_writable",
        status="pass" if writable else "warning",
        message=(
            "artifacts_root target or nearest existing parent appears writable."
            if writable
            else "artifacts_root target or nearest existing parent is not writable."
        ),
        details={
            "path": relative,
            "checked_path": _display_path(target),
        },
    )


def _output_path_check(output_path: str | Path | None) -> DoctorCheck:
    if output_path is None:
        return DoctorCheck(
            name="output_path_recommendation",
            status="skipped",
            message="No output path was requested; the doctor will not write a report by default.",
        )
    display = _display_path(output_path)
    try:
        normalized = _validate_portable_path(Path(output_path).as_posix())
    except ValueError:
        return DoctorCheck(
            name="output_path_recommendation",
            status="warning",
            message="Output path is not repository-relative; report path is sanitized.",
            details={"path": display},
        )
    status = "pass" if "/_derived/" in f"/{normalized}" else "warning"
    return DoctorCheck(
        name="output_path_recommendation",
        status=status,
        message=(
            "Output path is under a derived-output location."
            if status == "pass"
            else "Prefer writing doctor reports under artifacts/_derived/environment_readiness/."
        ),
        details={"path": normalized},
    )


def _initial_profile_payload(profile: str | None, profile_path: str | Path | None) -> dict[str, Any]:
    return {
        "name": profile,
        "path": None if profile_path is None else _display_path(profile_path),
    }


def _display_path(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        return candidate.as_posix()
    try:
        return candidate.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return f"<external>/{candidate.name}"


def _safe_message(message: str, profile_path: str | Path | None) -> str:
    if profile_path is None:
        return message
    candidate = Path(profile_path)
    if not candidate.is_absolute():
        return message
    return message.replace(str(candidate), _display_path(candidate))


def _validate_portable_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    first = path.parts[0] if path.parts else ""
    invalid = (
        not normalized
        or normalized != value
        or normalized.startswith("/")
        or "://" in normalized
        or normalized.startswith("~")
        or any(part in {"", ".", ".."} for part in path.parts)
        or (len(first) == 2 and first[1] == ":")
    )
    if invalid:
        raise ValueError(f"Path value must be portable and repository-relative: {value}")
    return path.as_posix()


def _nearest_existing_parent(path: Path, stop_at: Path) -> Path | None:
    current = path
    stop = stop_at.resolve()
    while not current.exists():
        if current == current.parent:
            return None
        try:
            current.resolve().relative_to(stop)
        except ValueError:
            return None
        current = current.parent
    return current


def _is_writable_directory(path: Path) -> bool:
    try:
        return path.is_dir() and os_access_write(path)
    except OSError:
        return False


def os_access_write(path: Path) -> bool:
    # Wrapped for tests without importing workflow modules or touching files.
    import os

    return os.access(path, os.W_OK)


def _finding_counts(checks: list[dict[str, Any]]) -> dict[str, int]:
    return {
        status: sum(1 for check in checks if check.get("status") == status)
        for status in CHECK_STATUSES
    }


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    if isinstance(value, tuple):
        return [_canonicalize(item) for item in value]
    return value
