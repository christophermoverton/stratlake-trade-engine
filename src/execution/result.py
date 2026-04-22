from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExecutionResult:
    """Notebook-friendly summary for one deterministic workflow execution."""

    workflow: str
    run_id: str
    name: str
    artifact_dir: Path | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    manifest_path: Path | None = None
    registry_path: Path | None = None
    output_paths: dict[str, Path] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    raw_result: Any = None

    def to_dict(self, *, include_raw_result: bool = False) -> dict[str, Any]:
        """Return a JSON-safe summary payload."""

        payload: dict[str, Any] = {
            "workflow": self.workflow,
            "run_id": self.run_id,
            "name": self.name,
            "artifact_dir": _path_or_none(self.artifact_dir),
            "metrics": _json_safe(self.metrics),
            "manifest_path": _path_or_none(self.manifest_path),
            "registry_path": _path_or_none(self.registry_path),
            "output_paths": {
                key: value.as_posix()
                for key, value in sorted(self.output_paths.items())
                if value is not None
            },
            "extra": _json_safe(self.extra),
        }
        if include_raw_result:
            payload["raw_result"] = self.raw_result
        return payload


def summarize_execution_result(
    *,
    workflow: str,
    raw_result: Any,
    name: str | None = None,
    run_id: str | None = None,
    artifact_dir: Path | None = None,
    metrics: dict[str, Any] | None = None,
    manifest_path: Path | None = None,
    registry_path: Path | None = None,
    output_paths: dict[str, Path | None] | None = None,
    extra: dict[str, Any] | None = None,
) -> ExecutionResult:
    """Build a standard execution summary from an existing workflow result object."""

    resolved_artifact_dir = artifact_dir or _optional_path(getattr(raw_result, "artifact_dir", None))
    if resolved_artifact_dir is None:
        resolved_artifact_dir = _optional_path(getattr(raw_result, "experiment_dir", None))
    resolved_manifest_path = manifest_path
    if resolved_manifest_path is None and resolved_artifact_dir is not None:
        candidate = resolved_artifact_dir / "manifest.json"
        resolved_manifest_path = candidate if candidate.exists() else candidate

    return ExecutionResult(
        workflow=workflow,
        run_id=str(run_id or getattr(raw_result, "run_id", "")),
        name=str(name or _result_name(raw_result)),
        artifact_dir=resolved_artifact_dir,
        metrics=dict(metrics if metrics is not None else _result_metrics(raw_result)),
        manifest_path=resolved_manifest_path,
        registry_path=registry_path,
        output_paths={
            key: value
            for key, value in (output_paths or {}).items()
            if value is not None
        },
        extra=dict(extra or {}),
        raw_result=raw_result,
    )


def _result_name(raw_result: Any) -> str:
    for attribute in ("strategy_name", "alpha_name", "portfolio_name", "pipeline_id", "pack_run_id"):
        value = getattr(raw_result, attribute, None)
        if value is not None:
            return str(value)
    return raw_result.__class__.__name__


def _result_metrics(raw_result: Any) -> dict[str, Any]:
    metrics = getattr(raw_result, "metrics", None)
    if isinstance(metrics, dict):
        return metrics
    evaluation_result = getattr(raw_result, "evaluation_result", None)
    summary = getattr(evaluation_result, "summary", None)
    if isinstance(summary, dict):
        return summary
    summary = getattr(raw_result, "summary", None)
    if isinstance(summary, dict):
        return summary
    return {}


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    return value if isinstance(value, Path) else Path(str(value))


def _path_or_none(value: Path | None) -> str | None:
    return None if value is None else value.as_posix()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return str(value)
    return value
