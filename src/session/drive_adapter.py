from __future__ import annotations

from dataclasses import dataclass
import fnmatch
import hashlib
import json
from pathlib import Path
import shutil
from typing import Iterable, Mapping, Sequence

from src.session.contracts import NotebookProjectSession
from src.session.paths import load_session, resolve_session_paths

DRIVE_SYNC_MANIFEST_SCHEMA_VERSION = 1
DEFAULT_OPERATION_ID = "latest"
DEFAULT_INCLUDE_CATEGORIES = ("session_metadata",)
COPY_CATEGORIES = (
    "configs",
    "contracts",
    "docs",
    "artifacts",
    "derived_artifacts",
    "features",
    "market_data",
    "session_metadata",
)
SENSITIVE_NAME_FRAGMENTS = (
    "credential",
    "credentials",
    "secret",
    "secrets",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "private_key",
)
EXCLUDED_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".ipynb_checkpoints",
}
EXCLUDED_FILE_NAMES = {".env"}
EXCLUDED_PATTERNS = (
    "*.pyc",
    "*.pyo",
    "*.tmp",
    "*.temp",
    "*.swp",
    "*~",
)


@dataclass(frozen=True)
class SessionCopyItem:
    operation: str
    category: str
    relative_path: str
    source_path: Path
    destination_path: Path
    size_bytes: int
    sha256: str
    status: str
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "category": self.category,
            "relative_path": self.relative_path,
            "source_path": self.source_path.as_posix(),
            "destination_path": self.destination_path.as_posix(),
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "status": self.status,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SessionCopyPlan:
    operation: str
    dry_run: bool
    local_root: Path
    drive_root: Path
    include_categories: tuple[str, ...]
    exclude_rules: tuple[str, ...]
    items: tuple[SessionCopyItem, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "dry_run": self.dry_run,
            "local_root": self.local_root.as_posix(),
            "drive_root": self.drive_root.as_posix(),
            "include_categories": list(self.include_categories),
            "exclude_rules": list(self.exclude_rules),
            "items": [item.to_dict() for item in self.items],
        }


@dataclass(frozen=True)
class SessionCopyResult:
    plan: SessionCopyPlan
    copied_count: int
    skipped_count: int
    overwritten_count: int
    manifest_path: Path | None

    def to_dict(self) -> dict[str, object]:
        return {
            **self.plan.to_dict(),
            "copied_count": self.copied_count,
            "skipped_count": self.skipped_count,
            "overwritten_count": self.overwritten_count,
            "manifest_path": None if self.manifest_path is None else self.manifest_path.as_posix(),
        }


def plan_session_copy(
    *,
    operation: str,
    root: Path | str | NotebookProjectSession,
    drive_root: Path | str | None = None,
    include_categories: Iterable[str] | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> SessionCopyPlan:
    operation = _validate_operation(operation)
    session = root if isinstance(root, NotebookProjectSession) else load_session(root)
    paths = resolve_session_paths(session)
    local_root = Path(paths["project_root"].resolved_path).resolve()
    effective_drive_root = _resolve_drive_root(drive_root, paths)
    categories = _normalize_categories(include_categories)
    items = _build_plan_items(
        operation=operation,
        local_root=local_root,
        drive_root=effective_drive_root,
        paths=paths,
        include_categories=categories,
        force=force,
        dry_run=dry_run,
    )
    return SessionCopyPlan(
        operation=operation,
        dry_run=dry_run,
        local_root=local_root,
        drive_root=effective_drive_root,
        include_categories=categories,
        exclude_rules=_exclude_rules(),
        items=tuple(items),
    )


def export_session_to_drive(
    *,
    root: Path | str | NotebookProjectSession,
    drive_root: Path | str | None = None,
    include_categories: Iterable[str] | None = None,
    force: bool = False,
    dry_run: bool = False,
    operation_id: str = DEFAULT_OPERATION_ID,
    write_manifest: bool | None = None,
) -> SessionCopyResult:
    return _execute_session_copy(
        operation="export",
        root=root,
        drive_root=drive_root,
        include_categories=include_categories,
        force=force,
        dry_run=dry_run,
        operation_id=operation_id,
        write_manifest=write_manifest,
    )


def import_session_from_drive(
    *,
    root: Path | str | NotebookProjectSession,
    drive_root: Path | str | None = None,
    include_categories: Iterable[str] | None = None,
    force: bool = False,
    dry_run: bool = False,
    operation_id: str = DEFAULT_OPERATION_ID,
    write_manifest: bool | None = None,
) -> SessionCopyResult:
    return _execute_session_copy(
        operation="import",
        root=root,
        drive_root=drive_root,
        include_categories=include_categories,
        force=force,
        dry_run=dry_run,
        operation_id=operation_id,
        write_manifest=write_manifest,
    )


def write_drive_sync_manifest(
    plan_or_result: SessionCopyPlan | SessionCopyResult,
    *,
    operation_id: str = DEFAULT_OPERATION_ID,
) -> Path:
    plan = plan_or_result.plan if isinstance(plan_or_result, SessionCopyResult) else plan_or_result
    manifest_path = _manifest_path(plan.local_root, plan.operation, operation_id)
    _ensure_under_root(manifest_path, plan.local_root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(_manifest_data(plan), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _execute_session_copy(
    *,
    operation: str,
    root: Path | str | NotebookProjectSession,
    drive_root: Path | str | None,
    include_categories: Iterable[str] | None,
    force: bool,
    dry_run: bool,
    operation_id: str,
    write_manifest: bool | None,
) -> SessionCopyResult:
    plan = plan_session_copy(
        operation=operation,
        root=root,
        drive_root=drive_root,
        include_categories=include_categories,
        force=force,
        dry_run=dry_run,
    )
    copied = 0
    skipped = 0
    overwritten = 0
    if dry_run:
        for item in plan.items:
            if item.status.startswith("would_"):
                continue
            skipped += 1
    else:
        for item in plan.items:
            if item.status == "skipped":
                skipped += 1
                continue
            item.destination_path.parent.mkdir(parents=True, exist_ok=True)
            destination_existed = item.destination_path.exists()
            shutil.copy2(item.source_path, item.destination_path)
            if destination_existed:
                overwritten += 1
            else:
                copied += 1

    should_write_manifest = (not dry_run) if write_manifest is None else write_manifest
    manifest_path = write_drive_sync_manifest(plan, operation_id=operation_id) if should_write_manifest else None
    if dry_run:
        copied = 0
        overwritten = 0
        skipped = len([item for item in plan.items if item.status == "skipped"])
    return SessionCopyResult(
        plan=plan,
        copied_count=copied,
        skipped_count=skipped,
        overwritten_count=overwritten,
        manifest_path=manifest_path,
    )


def _build_plan_items(
    *,
    operation: str,
    local_root: Path,
    drive_root: Path,
    paths: Mapping[str, object],
    include_categories: Sequence[str],
    force: bool,
    dry_run: bool,
) -> list[SessionCopyItem]:
    items: list[SessionCopyItem] = []
    for category in include_categories:
        source_root, destination_root = _category_roots(
            operation=operation,
            category=category,
            local_root=local_root,
            drive_root=drive_root,
            paths=paths,
        )
        if not source_root.exists():
            continue
        for source in _iter_category_files(source_root, category):
            relative = source.relative_to(source_root).as_posix()
            destination = (destination_root / relative).resolve()
            _ensure_under_root(destination, destination_root.resolve())
            status, reason = _planned_status(destination, force=force, dry_run=dry_run)
            items.append(
                SessionCopyItem(
                    operation=operation,
                    category=category,
                    relative_path=relative,
                    source_path=source.resolve(),
                    destination_path=destination,
                    size_bytes=source.stat().st_size,
                    sha256=_sha256(source),
                    status=status,
                    reason=reason,
                )
            )
    return sorted(items, key=lambda item: (item.category, item.relative_path))


def _category_roots(
    *,
    operation: str,
    category: str,
    local_root: Path,
    drive_root: Path,
    paths: Mapping[str, object],
) -> tuple[Path, Path]:
    local_category_root = _local_category_root(category, local_root, paths)
    drive_category_root = _drive_category_root(
        category=category,
        local_category_root=local_category_root,
        local_root=local_root,
        drive_root=drive_root,
    )
    if operation == "export":
        return local_category_root, drive_category_root
    return drive_category_root, local_category_root


def _local_category_root(
    category: str,
    local_root: Path,
    paths: Mapping[str, object],
) -> Path:
    if category == "configs":
        return Path(paths["configs_root"].resolved_path).resolve()  # type: ignore[attr-defined]
    if category == "contracts":
        return (local_root / "contracts").resolve()
    if category == "docs":
        return (local_root / "docs").resolve()
    if category == "artifacts":
        return Path(paths["artifacts_root"].resolved_path).resolve()  # type: ignore[attr-defined]
    if category == "derived_artifacts":
        return (Path(paths["artifacts_root"].resolved_path) / "_derived").resolve()  # type: ignore[attr-defined]
    if category == "features":
        return Path(paths["features_root"].resolved_path).resolve()  # type: ignore[attr-defined]
    if category == "market_data":
        return Path(paths["marketlake_root"].resolved_path).resolve()  # type: ignore[attr-defined]
    if category == "session_metadata":
        return (local_root / ".stratlake").resolve()
    raise ValueError(f"Unsupported copy category: {category}")


def _drive_category_root(
    *,
    category: str,
    local_category_root: Path,
    local_root: Path,
    drive_root: Path,
) -> Path:
    try:
        relative = local_category_root.relative_to(local_root)
    except ValueError:
        return (drive_root / category).resolve()
    return (drive_root / relative).resolve()


def _iter_category_files(root: Path, category: str) -> Iterable[Path]:
    for path in sorted(root.rglob("*"), key=lambda candidate: candidate.as_posix()):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(root).parts
        if _is_excluded(path.name, relative_parts):
            continue
        if category == "artifacts" and relative_parts and relative_parts[0] == "_derived":
            continue
        yield path


def _is_excluded(name: str, relative_parts: Sequence[str]) -> bool:
    lowered_parts = [part.lower() for part in relative_parts]
    lowered_name = name.lower()
    if any(part in EXCLUDED_DIR_NAMES for part in lowered_parts):
        return True
    if lowered_name in EXCLUDED_FILE_NAMES:
        return True
    if any(fragment in lowered_name for fragment in SENSITIVE_NAME_FRAGMENTS):
        return True
    return any(fnmatch.fnmatch(lowered_name, pattern) for pattern in EXCLUDED_PATTERNS)


def _planned_status(destination: Path, *, force: bool, dry_run: bool) -> tuple[str, str | None]:
    if destination.exists() and not force:
        return "skipped", "destination_exists"
    if dry_run:
        return ("would_overwrite", None) if destination.exists() else ("would_copy", None)
    return ("overwritten", None) if destination.exists() else ("copied", None)


def _normalize_categories(include_categories: Iterable[str] | None) -> tuple[str, ...]:
    requested = set(DEFAULT_INCLUDE_CATEGORIES)
    if include_categories is not None:
        requested.update(include_categories)
    unknown = sorted(requested - set(COPY_CATEGORIES))
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"Unknown session copy categories: {joined}")
    return tuple(category for category in COPY_CATEGORIES if category in requested)


def _resolve_drive_root(
    drive_root: Path | str | None,
    paths: Mapping[str, object],
) -> Path:
    if drive_root is not None:
        return Path(drive_root).expanduser().resolve()
    drive = paths.get("drive_root")
    if drive is None:
        raise ValueError("A drive root is required. Pass --drive-root or configure drive_root in the session.")
    return Path(drive.resolved_path).resolve()  # type: ignore[attr-defined]


def _manifest_data(plan: SessionCopyPlan) -> dict[str, object]:
    return {
        "schema_version": DRIVE_SYNC_MANIFEST_SCHEMA_VERSION,
        "authoritative": False,
        **plan.to_dict(),
    }


def _manifest_path(local_root: Path, operation: str, operation_id: str) -> Path:
    safe_operation_id = _safe_operation_id(operation_id)
    return (
        local_root
        / "artifacts"
        / "_derived"
        / "notebook_sessions"
        / f"{operation}_{safe_operation_id}"
        / "drive_sync_manifest.json"
    ).resolve()


def _safe_operation_id(operation_id: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in operation_id)
    return cleaned or DEFAULT_OPERATION_ID


def _validate_operation(operation: str) -> str:
    if operation not in {"export", "import"}:
        raise ValueError(f"Unsupported session copy operation: {operation}")
    return operation


def _exclude_rules() -> tuple[str, ...]:
    return (
        ".env",
        "*credential*",
        "*secret*",
        "*api_key*",
        "*apikey*",
        "*access_token*",
        "*refresh_token*",
        "*private_key*",
        "__pycache__/",
        ".pytest_cache/",
        ".ruff_cache/",
        ".mypy_cache/",
        ".ipynb_checkpoints/",
        "*.pyc",
        "*.pyo",
        "*.tmp",
        "*.temp",
        "*.swp",
        "*~",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_under_root(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Refusing to copy outside root {root.as_posix()}: {path.as_posix()}") from exc
