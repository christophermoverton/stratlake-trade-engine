from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

ARCHIVE_MARKERS = (
    "manifest.json",
    "archive_index.json",
    "checksums.json",
    "restore_plan.json",
)
DEFAULT_SECRET_NAMES = (
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
)


@dataclass(frozen=True)
class NotebookDoctorCheck:
    name: str
    status: str
    severity: str
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "status": self.status,
            "severity": self.severity,
            "message": self.message,
        }
        if self.details:
            payload["details"] = _stable_jsonable(self.details)
        return payload


@dataclass(frozen=True)
class NotebookDoctorResult:
    schema_version: int
    status: str
    read_only: bool
    root: str
    marketlake_root: str | None
    drive_root: str | None
    archive_root: str | None
    archive_destination_root: str | None
    checks: tuple[NotebookDoctorCheck, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "read_only": self.read_only,
            "root": self.root,
            "marketlake_root": self.marketlake_root,
            "drive_root": self.drive_root,
            "archive_root": self.archive_root,
            "archive_destination_root": self.archive_destination_root,
            "checks": [check.to_dict() for check in self.checks],
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "summary": _stable_jsonable(self.summary),
        }


def run_notebook_doctor(
    *,
    root: str | Path,
    marketlake_root: str | Path | None = None,
    drive_root: str | Path | None = None,
    archive_root: str | Path | None = None,
    archive_destination_root: str | Path | None = None,
    universe: str | Path | None = None,
    check_configs: bool = False,
    check_universe: bool = False,
    check_drive: bool = False,
    check_archives: bool = False,
    check_marketlake: bool = True,
    check_secrets: bool = False,
    secret_names: Sequence[str] = (),
) -> NotebookDoctorResult:
    root_path = Path(root).expanduser().resolve()
    marketlake_path = (
        None if marketlake_root is None else Path(marketlake_root).expanduser().resolve()
    )
    drive_path = None if drive_root is None else Path(drive_root).expanduser().resolve()
    archive_path = None if archive_root is None else Path(archive_root).expanduser().resolve()
    archive_dest_path = (
        None
        if archive_destination_root is None
        else Path(archive_destination_root).expanduser().resolve()
    )

    checks: list[NotebookDoctorCheck] = []

    checks.append(
        NotebookDoctorCheck(
            name="read_only_boundaries",
            status="pass",
            severity="info",
            message=(
                "Notebook doctor is read-only: no .env or os.environ mutation, no Drive mounting, "
                "no Google API calls, no hidden sync, and no data or artifact writes."
            ),
            details={"read_only": True},
        )
    )

    root_ok = _check_root(root_path, checks)
    _check_session_metadata(root_path, checks)

    universe_info: dict[str, Any] = {}
    if check_configs:
        universe_info = _check_configs(root_path, checks)

    if check_universe:
        _check_universe(
            root=root_path,
            explicit_universe=universe,
            checks=checks,
            preloaded_universe=universe_info,
        )

    if check_marketlake or marketlake_path is not None:
        _check_marketlake_root(marketlake_path, checks)

    if check_drive or drive_path is not None:
        _check_drive_root(drive_path, checks, explicit_required=check_drive)

    if check_archives:
        _check_archives(
            root=root_path,
            archive_root_path=archive_path,
            archive_destination_root=archive_dest_path,
            checks=checks,
        )

    if check_secrets:
        _check_secrets(secret_names, checks)

    status = _overall_status(checks)
    errors = tuple(dict.fromkeys(check.message for check in checks if check.status == "fail"))
    warnings = tuple(dict.fromkeys(check.message for check in checks if check.status == "warn"))

    check_counts = {
        "pass": sum(1 for check in checks if check.status == "pass"),
        "warn": sum(1 for check in checks if check.status == "warn"),
        "fail": sum(1 for check in checks if check.status == "fail"),
    }
    summary = {
        "check_count": len(checks),
        "check_counts": check_counts,
        "root_ready": root_ok,
        "next_steps": {
            "marketlake_handoff_validator": "stratlake-validate-marketlake-handoff",
            "archive_restore_planning": "stratlake-session-archive-restore-bootstrap --dry-run --json",
        },
    }

    return NotebookDoctorResult(
        schema_version=1,
        status=status,
        read_only=True,
        root=root_path.as_posix(),
        marketlake_root=None if marketlake_path is None else marketlake_path.as_posix(),
        drive_root=None if drive_path is None else drive_path.as_posix(),
        archive_root=None if archive_path is None else archive_path.as_posix(),
        archive_destination_root=None
        if archive_dest_path is None
        else archive_dest_path.as_posix(),
        checks=tuple(checks),
        errors=errors,
        warnings=warnings,
        summary=summary,
    )


def _check_root(root: Path, checks: list[NotebookDoctorCheck]) -> bool:
    if not root.exists():
        checks.append(
            NotebookDoctorCheck(
                name="root_exists",
                status="fail",
                severity="error",
                message=f"StratLake root is missing: {root.as_posix()}.",
                details={"root": root.as_posix()},
            )
        )
        return False
    if not root.is_dir():
        checks.append(
            NotebookDoctorCheck(
                name="root_directory",
                status="fail",
                severity="error",
                message=f"StratLake root is not a directory: {root.as_posix()}.",
                details={"root": root.as_posix()},
            )
        )
        return False

    try:
        next(root.iterdir())
        readable = True
    except StopIteration:
        readable = True
    except OSError as exc:
        checks.append(
            NotebookDoctorCheck(
                name="root_readable",
                status="fail",
                severity="error",
                message=f"StratLake root is not readable: {exc}",
                details={"root": root.as_posix()},
            )
        )
        return False

    if readable:
        checks.append(
            NotebookDoctorCheck(
                name="root_readable",
                status="pass",
                severity="info",
                message=f"StratLake root is readable: {root.as_posix()}.",
                details={"root": root.as_posix()},
            )
        )

    for relative in ("configs", ".stratlake", "artifacts", "data"):
        path = root / relative
        if path.is_dir():
            checks.append(
                NotebookDoctorCheck(
                    name=f"root_subdir:{relative}",
                    status="pass",
                    severity="info",
                    message=f"Expected root subdirectory exists: {relative}.",
                    details={"path": path.as_posix()},
                )
            )
        else:
            checks.append(
                NotebookDoctorCheck(
                    name=f"root_subdir:{relative}",
                    status="warn",
                    severity="warning",
                    message=(
                        f"Expected root subdirectory is missing: {relative}. "
                        "Use session/bootstrap commands if this is a fresh workspace."
                    ),
                    details={"path": path.as_posix()},
                )
            )
    return True


def _check_session_metadata(root: Path, checks: list[NotebookDoctorCheck]) -> None:
    session_path = root / ".stratlake" / "session.json"
    path_resolution_path = root / ".stratlake" / "path_resolution.json"
    _check_json_file(
        name="session_metadata",
        path=session_path,
        missing_status="warn",
        checks=checks,
    )
    _check_json_file(
        name="path_resolution_metadata",
        path=path_resolution_path,
        missing_status="warn",
        checks=checks,
    )


def _check_configs(root: Path, checks: list[NotebookDoctorCheck]) -> dict[str, Any]:
    configs_root = root / "configs"
    required = ("paths.yml", "universe.yml")
    optional = ("strategies.yml", "evaluation.yml", "session.yml")
    loaded: dict[str, Any] = {}

    for file_name in required + optional:
        path = configs_root / file_name
        missing_status = "fail" if file_name in required else "warn"
        if not path.exists():
            checks.append(
                NotebookDoctorCheck(
                    name=f"config:{file_name}",
                    status=missing_status,
                    severity="error" if missing_status == "fail" else "warning",
                    message=f"Config file is missing: {path.as_posix()}.",
                    details={"path": path.as_posix(), "required": file_name in required},
                )
            )
            continue
        if not path.is_file():
            checks.append(
                NotebookDoctorCheck(
                    name=f"config:{file_name}",
                    status="fail",
                    severity="error",
                    message=f"Config path is not a file: {path.as_posix()}.",
                    details={"path": path.as_posix()},
                )
            )
            continue
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError, ValueError) as exc:
            checks.append(
                NotebookDoctorCheck(
                    name=f"config:{file_name}",
                    status="fail",
                    severity="error",
                    message=f"Config file is not valid YAML: {path.as_posix()} ({exc}).",
                    details={"path": path.as_posix()},
                )
            )
            continue

        checks.append(
            NotebookDoctorCheck(
                name=f"config:{file_name}",
                status="pass",
                severity="info",
                message=f"Config file is readable YAML: {path.as_posix()}.",
                details={
                    "path": path.as_posix(),
                    "mapping": isinstance(payload, dict),
                },
            )
        )
        loaded[file_name] = payload
    return loaded


def _check_universe(
    *,
    root: Path,
    explicit_universe: str | Path | None,
    checks: list[NotebookDoctorCheck],
    preloaded_universe: Mapping[str, Any],
) -> None:
    if explicit_universe is None:
        universe_path = root / "configs" / "universe.yml"
    else:
        candidate = Path(explicit_universe).expanduser()
        universe_path = candidate if candidate.is_absolute() else (root / candidate)
        universe_path = universe_path.resolve()

    if not universe_path.exists() or not universe_path.is_file():
        checks.append(
            NotebookDoctorCheck(
                name="universe_file",
                status="fail",
                severity="error",
                message=f"Universe config file is missing: {universe_path.as_posix()}.",
                details={"path": universe_path.as_posix()},
            )
        )
        return

    payload = preloaded_universe.get("universe.yml")
    if payload is None:
        try:
            payload = yaml.safe_load(universe_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError, ValueError) as exc:
            checks.append(
                NotebookDoctorCheck(
                    name="universe_file",
                    status="fail",
                    severity="error",
                    message=f"Universe config is not valid YAML: {exc}",
                    details={"path": universe_path.as_posix()},
                )
            )
            return

    if not isinstance(payload, dict):
        checks.append(
            NotebookDoctorCheck(
                name="universe_file",
                status="fail",
                severity="error",
                message="Universe config must be a mapping object.",
                details={"path": universe_path.as_posix()},
            )
        )
        return

    symbols, source, source_path = _resolve_universe_symbols(
        payload=payload,
        root=root,
        universe_path=universe_path,
    )
    if not symbols:
        checks.append(
            NotebookDoctorCheck(
                name="universe_symbols",
                status="fail",
                severity="error",
                message="Universe resolved zero symbols.",
                details={
                    "source": source,
                    "source_path": source_path,
                    "symbol_count": 0,
                },
            )
        )
        return

    checks.append(
        NotebookDoctorCheck(
            name="universe_symbols",
            status="pass",
            severity="info",
            message=f"Universe resolved {len(symbols)} symbol(s) from {source}.",
            details={
                "source": source,
                "source_path": source_path,
                "symbol_count": len(symbols),
                "sample_symbols": symbols[:10],
            },
        )
    )


def _resolve_universe_symbols(
    *,
    payload: Mapping[str, Any],
    root: Path,
    universe_path: Path,
) -> tuple[list[str], str, str | None]:
    symbols: set[str] = set()
    for key in ("symbols", "universe_symbols"):
        value = payload.get(key)
        if isinstance(value, list):
            symbols.update(_normalize_symbols(value))

    tickers_value = payload.get("tickers")
    if isinstance(tickers_value, list):
        symbols.update(_normalize_symbols(tickers_value))

    nested = payload.get("universe")
    if isinstance(nested, Mapping):
        nested_symbols = nested.get("symbols")
        if isinstance(nested_symbols, list):
            symbols.update(_normalize_symbols(nested_symbols))

    if symbols:
        return sorted(symbols), "symbols", universe_path.as_posix()

    ticker_file = _resolve_ticker_file(payload, root)
    if ticker_file is None:
        return [], "none", None

    if not ticker_file.exists() or not ticker_file.is_file():
        return [], "ticker_file_missing", ticker_file.as_posix()

    lines = ticker_file.read_text(encoding="utf-8").splitlines()
    resolved = sorted({line.strip().upper() for line in lines if line.strip()})
    return resolved, "ticker_file", ticker_file.as_posix()


def _resolve_ticker_file(payload: Mapping[str, Any], root: Path) -> Path | None:
    for key in ("tickers_file", "tickers_path", "tickers"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            candidate = Path(value).expanduser()
            return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    return None


def _normalize_symbols(values: Sequence[Any]) -> set[str]:
    return {str(value).strip().upper() for value in values if str(value).strip()}


def _check_marketlake_root(marketlake_root: Path | None, checks: list[NotebookDoctorCheck]) -> None:
    if marketlake_root is None:
        checks.append(
            NotebookDoctorCheck(
                name="marketlake_root",
                status="warn",
                severity="warning",
                message=(
                    "MarketLake root is not provided. Provide --marketlake-root for data readiness and "
                    "use stratlake-validate-marketlake-handoff for detailed symbol/date coverage."
                ),
            )
        )
        return

    if not marketlake_root.exists():
        checks.append(
            NotebookDoctorCheck(
                name="marketlake_root",
                status="fail",
                severity="error",
                message=f"MarketLake root is missing: {marketlake_root.as_posix()}.",
                details={"marketlake_root": marketlake_root.as_posix()},
            )
        )
        return
    if not marketlake_root.is_dir():
        checks.append(
            NotebookDoctorCheck(
                name="marketlake_root",
                status="fail",
                severity="error",
                message=f"MarketLake root is not a directory: {marketlake_root.as_posix()}.",
                details={"marketlake_root": marketlake_root.as_posix()},
            )
        )
        return

    try:
        next(marketlake_root.iterdir())
    except StopIteration:
        pass
    except OSError as exc:
        checks.append(
            NotebookDoctorCheck(
                name="marketlake_root",
                status="fail",
                severity="error",
                message=f"MarketLake root is not readable: {exc}",
                details={"marketlake_root": marketlake_root.as_posix()},
            )
        )
        return

    if any((marketlake_root / marker).exists() for marker in ARCHIVE_MARKERS):
        checks.append(
            NotebookDoctorCheck(
                name="marketlake_root_not_archive_pack",
                status="fail",
                severity="error",
                message=(
                    "MarketLake root appears to be an archive pack directory; use a curated data root "
                    "instead."
                ),
                details={"marketlake_root": marketlake_root.as_posix()},
            )
        )
    else:
        checks.append(
            NotebookDoctorCheck(
                name="marketlake_root_ready",
                status="pass",
                severity="info",
                message=f"MarketLake root is readable: {marketlake_root.as_posix()}.",
                details={"marketlake_root": marketlake_root.as_posix()},
            )
        )

    datasets = {
        "bars_daily": (marketlake_root / "bars_daily").is_dir(),
        "bars_1m": (marketlake_root / "bars_1m").is_dir(),
    }
    if any(datasets.values()):
        checks.append(
            NotebookDoctorCheck(
                name="marketlake_dataset_dirs",
                status="pass",
                severity="info",
                message="At least one expected curated dataset directory is present.",
                details=datasets,
            )
        )
    else:
        checks.append(
            NotebookDoctorCheck(
                name="marketlake_dataset_dirs",
                status="warn",
                severity="warning",
                message=(
                    "Expected dataset directories bars_daily/bars_1m were not found. "
                    "Run stratlake-validate-marketlake-handoff for detailed data coverage diagnostics."
                ),
                details=datasets,
            )
        )


def _check_drive_root(
    drive_root: Path | None,
    checks: list[NotebookDoctorCheck],
    *,
    explicit_required: bool,
) -> None:
    if drive_root is None:
        if explicit_required:
            checks.append(
                NotebookDoctorCheck(
                    name="drive_root",
                    status="fail",
                    severity="error",
                    message="Drive root was requested with --check-drive but no --drive-root was provided.",
                )
            )
        else:
            checks.append(
                NotebookDoctorCheck(
                    name="drive_root",
                    status="warn",
                    severity="warning",
                    message="Drive root is not provided.",
                )
            )
        return

    if not drive_root.exists():
        checks.append(
            NotebookDoctorCheck(
                name="drive_root",
                status="fail" if explicit_required else "warn",
                severity="error" if explicit_required else "warning",
                message=f"Drive root is missing: {drive_root.as_posix()}.",
                details={"drive_root": drive_root.as_posix()},
            )
        )
        return
    if not drive_root.is_dir():
        checks.append(
            NotebookDoctorCheck(
                name="drive_root",
                status="fail",
                severity="error",
                message=f"Drive root is not a directory: {drive_root.as_posix()}.",
                details={"drive_root": drive_root.as_posix()},
            )
        )
        return

    try:
        next(drive_root.iterdir())
    except StopIteration:
        pass
    except OSError as exc:
        checks.append(
            NotebookDoctorCheck(
                name="drive_root",
                status="fail",
                severity="error",
                message=f"Drive root is not readable: {exc}",
                details={"drive_root": drive_root.as_posix()},
            )
        )
        return

    looks_like_colab = drive_root.as_posix().startswith("/content/drive/MyDrive/")
    checks.append(
        NotebookDoctorCheck(
            name="drive_root_readable",
            status="pass",
            severity="info",
            message=f"Drive root is readable: {drive_root.as_posix()}.",
            details={"looks_like_colab_mount": looks_like_colab},
        )
    )
    if not looks_like_colab:
        checks.append(
            NotebookDoctorCheck(
                name="drive_root_colab_shape",
                status="warn",
                severity="warning",
                message=(
                    "Drive root does not match /content/drive/MyDrive/... Colab shape. "
                    "This is acceptable for non-Colab environments."
                ),
                details={"drive_root": drive_root.as_posix()},
            )
        )


def _check_archives(
    *,
    root: Path,
    archive_root_path: Path | None,
    archive_destination_root: Path | None,
    checks: list[NotebookDoctorCheck],
) -> None:
    if archive_root_path is None and archive_destination_root is None:
        checks.append(
            NotebookDoctorCheck(
                name="archives_inputs",
                status="fail",
                severity="error",
                message=(
                    "Archive checks requested but neither --archive-root nor "
                    "--archive-destination-root was provided."
                ),
            )
        )
        return

    if archive_root_path is not None:
        if not archive_root_path.exists() or not archive_root_path.is_dir():
            checks.append(
                NotebookDoctorCheck(
                    name="archive_root",
                    status="fail",
                    severity="error",
                    message=f"Archive root is missing or not a directory: {archive_root_path.as_posix()}.",
                    details={"archive_root": archive_root_path.as_posix()},
                )
            )
        else:
            markers = {marker: (archive_root_path / marker).is_file() for marker in ARCHIVE_MARKERS}
            if all(markers.values()):
                checks.append(
                    NotebookDoctorCheck(
                        name="archive_root_markers",
                        status="pass",
                        severity="info",
                        message="Archive root contains expected M43 marker files.",
                        details={"archive_root": archive_root_path.as_posix(), **markers},
                    )
                )
            else:
                checks.append(
                    NotebookDoctorCheck(
                        name="archive_root_markers",
                        status="fail",
                        severity="error",
                        message="Archive root is missing one or more expected M43 marker files.",
                        details={"archive_root": archive_root_path.as_posix(), **markers},
                    )
                )

            if archive_root_path == root:
                checks.append(
                    NotebookDoctorCheck(
                        name="archive_root_not_target_root",
                        status="fail",
                        severity="error",
                        message="Archive root must not be the same as the target workspace root.",
                        details={
                            "archive_root": archive_root_path.as_posix(),
                            "target_root": root.as_posix(),
                        },
                    )
                )
            elif _is_relative_to(archive_root_path, root):
                checks.append(
                    NotebookDoctorCheck(
                        name="archive_root_not_under_target_root",
                        status="fail",
                        severity="error",
                        message="Archive root must not be nested under the target workspace root.",
                        details={
                            "archive_root": archive_root_path.as_posix(),
                            "target_root": root.as_posix(),
                        },
                    )
                )

    if archive_destination_root is not None:
        if not archive_destination_root.exists() or not archive_destination_root.is_dir():
            checks.append(
                NotebookDoctorCheck(
                    name="archive_destination_root",
                    status="fail",
                    severity="error",
                    message=(
                        "Archive destination root is missing or not a directory: "
                        f"{archive_destination_root.as_posix()}."
                    ),
                    details={"archive_destination_root": archive_destination_root.as_posix()},
                )
            )
        else:
            checks.append(
                NotebookDoctorCheck(
                    name="archive_destination_root",
                    status="pass",
                    severity="info",
                    message=f"Archive destination root is readable: {archive_destination_root.as_posix()}.",
                    details={"archive_destination_root": archive_destination_root.as_posix()},
                )
            )
            if archive_destination_root == root or _is_relative_to(archive_destination_root, root):
                checks.append(
                    NotebookDoctorCheck(
                        name="archive_destination_overlap",
                        status="warn",
                        severity="warning",
                        message=(
                            "Archive destination root overlaps the target workspace root. "
                            "Prefer a separate mounted archive path."
                        ),
                        details={
                            "archive_destination_root": archive_destination_root.as_posix(),
                            "target_root": root.as_posix(),
                        },
                    )
                )


def _check_secrets(secret_names: Sequence[str], checks: list[NotebookDoctorCheck]) -> None:
    requested = sorted({name.strip() for name in secret_names if name.strip()})
    names = requested or list(DEFAULT_SECRET_NAMES)
    for secret_name in names:
        value = os.environ.get(secret_name)
        state = "SET" if bool(value) else "NOT_SET"
        checks.append(
            NotebookDoctorCheck(
                name=f"secret:{secret_name}",
                status="pass" if state == "SET" else "warn",
                severity="info" if state == "SET" else "warning",
                message=f"Secret {secret_name} is {state}.",
                details={"secret_name": secret_name, "state": state},
            )
        )


def _check_json_file(
    *,
    name: str,
    path: Path,
    missing_status: str,
    checks: list[NotebookDoctorCheck],
) -> None:
    if not path.exists():
        checks.append(
            NotebookDoctorCheck(
                name=name,
                status=missing_status,
                severity="warning" if missing_status == "warn" else "error",
                message=f"Metadata file is missing: {path.as_posix()}.",
                details={"path": path.as_posix()},
            )
        )
        return
    if not path.is_file():
        checks.append(
            NotebookDoctorCheck(
                name=name,
                status="fail",
                severity="error",
                message=f"Metadata path is not a file: {path.as_posix()}.",
                details={"path": path.as_posix()},
            )
        )
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        checks.append(
            NotebookDoctorCheck(
                name=name,
                status="fail",
                severity="error",
                message=f"Metadata file is not valid JSON: {path.as_posix()} ({exc}).",
                details={"path": path.as_posix()},
            )
        )
        return
    checks.append(
        NotebookDoctorCheck(
            name=name,
            status="pass",
            severity="info",
            message=f"Metadata file is readable JSON: {path.as_posix()}.",
            details={"path": path.as_posix(), "mapping": isinstance(payload, dict)},
        )
    )


def _overall_status(checks: Sequence[NotebookDoctorCheck]) -> str:
    if any(check.status == "fail" for check in checks):
        return "fail"
    if any(check.status == "warn" for check in checks):
        return "warn"
    return "pass"


def _is_relative_to(path: Path, potential_parent: Path) -> bool:
    try:
        path.relative_to(potential_parent)
        return True
    except ValueError:
        return False


def _stable_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _stable_jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_stable_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_stable_jsonable(item) for item in value]
    if isinstance(value, set):
        return [_stable_jsonable(item) for item in sorted(value)]
    if isinstance(value, Path):
        return value.as_posix()
    return value
