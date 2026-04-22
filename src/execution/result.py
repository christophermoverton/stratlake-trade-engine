from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable


_METRICS_OUTPUT_KEYS = (
    "metrics_json",
    "alpha_metrics_json",
    "aggregate_metrics_json",
    "pipeline_metrics_json",
    "report_json",
)
_SUMMARY_OUTPUT_KEYS = (
    "summary_json",
    "benchmark_matrix_summary",
    "qa_summary_json",
    "training_summary_json",
    "report_json",
)


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

    def output_keys(self) -> tuple[str, ...]:
        """Return available named output keys in deterministic order."""

        return tuple(sorted(self.output_paths))

    def has_output(self, key: str) -> bool:
        """Return whether a named output path is present on this result."""

        return key in self.output_paths

    def output_path(self, key: str, *, must_exist: bool = False) -> Path:
        """Return one named output path, raising clearly when it is unavailable."""

        if key not in self.output_paths:
            available = ", ".join(self.output_keys()) or "<none>"
            raise KeyError(f"ExecutionResult has no output path named {key!r}; available outputs: {available}")
        path = self.output_paths[key]
        if must_exist and not path.exists():
            raise FileNotFoundError(f"Output path {key!r} does not exist: {path}")
        return path

    def artifact_path(self, *parts: str | Path, must_exist: bool = False) -> Path:
        """Resolve a path under artifact_dir without creating or mutating anything."""

        if self.artifact_dir is None:
            raise ValueError("ExecutionResult has no artifact_dir.")
        path = self.artifact_dir.joinpath(*parts)
        _ensure_under_root(path, self.artifact_dir)
        if must_exist and not path.exists():
            raise FileNotFoundError(f"Artifact path does not exist: {path}")
        return path

    def load_manifest(self) -> dict[str, Any]:
        """Load the result manifest JSON from manifest_path."""

        if self.manifest_path is None:
            raise ValueError("ExecutionResult has no manifest_path.")
        payload = load_json_artifact(self.manifest_path)
        if not isinstance(payload, dict):
            raise ValueError(f"Manifest JSON must contain an object: {self.manifest_path}")
        return payload

    def load_output_json(self, key: str) -> Any:
        """Load JSON from a named output path."""

        return load_json_artifact(self.output_path(key, must_exist=True))

    def load_metrics_json(self, key: str | None = None) -> dict[str, Any]:
        """Load a metrics/report JSON payload from a named output path."""

        selected_key = key or self._first_available_output(_METRICS_OUTPUT_KEYS, label="metrics JSON")
        payload = self.load_output_json(selected_key)
        if not isinstance(payload, dict):
            raise ValueError(f"Metrics JSON output {selected_key!r} must contain an object.")
        return payload

    def load_summary_json(self, key: str | None = None) -> dict[str, Any]:
        """Load a summary-style JSON payload from a named output path."""

        selected_key = key or self._first_available_output(_SUMMARY_OUTPUT_KEYS, label="summary JSON")
        payload = self.load_output_json(selected_key)
        if not isinstance(payload, dict):
            raise ValueError(f"Summary JSON output {selected_key!r} must contain an object.")
        return payload

    def load_comparison_json(self, key: str = "comparison_json") -> dict[str, Any]:
        """Load a comparison JSON payload when the workflow exposed one."""

        payload = self.load_output_json(key)
        if not isinstance(payload, dict):
            raise ValueError(f"Comparison JSON output {key!r} must contain an object.")
        return payload

    def notebook_summary(self) -> dict[str, Any]:
        """Return a compact JSON-safe payload for notebook display."""

        return {
            "workflow": self.workflow,
            "run_id": self.run_id,
            "name": self.name,
            "artifact_dir": _path_or_none(self.artifact_dir),
            "manifest_path": _path_or_none(self.manifest_path),
            "metrics": _json_safe(self.metrics),
            "output_keys": list(self.output_keys()),
            "extra": _json_safe(self.extra),
        }

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

    def _first_available_output(self, candidates: Iterable[str], *, label: str) -> str:
        for key in candidates:
            if key in self.output_paths:
                return key
        available = ", ".join(self.output_keys()) or "<none>"
        expected = ", ".join(candidates)
        raise KeyError(
            f"ExecutionResult has no {label} output. Expected one of: {expected}; "
            f"available outputs: {available}"
        )


def load_json_artifact(path: str | Path) -> Any:
    """Load a local JSON artifact without side effects."""

    resolved_path = Path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"JSON artifact does not exist: {resolved_path}")
    return json.loads(resolved_path.read_text(encoding="utf-8"))


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


def _ensure_under_root(path: Path, root: Path) -> None:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
        raise ValueError(f"Artifact path escapes artifact_dir: {path}")
