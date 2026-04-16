from __future__ import annotations

from collections import deque
import csv
from dataclasses import dataclass
import json
import hashlib
import importlib
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import yaml

from src.contracts.validate import validate_json
from src.research.registry import canonicalize_value, serialize_canonical_json


@dataclass(frozen=True)
class PipelineStepSpec:
    """One pipeline step resolved from YAML."""

    id: str
    adapter: str
    module: str
    argv: tuple[str, ...]
    depends_on: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        return canonicalize_value(
            {
                "id": self.id,
                "depends_on": list(self.depends_on),
                "adapter": self.adapter,
                "module": self.module,
                "argv": list(self.argv),
            }
        )


@dataclass(frozen=True)
class PipelineSpec:
    """Deterministic pipeline specification resolved from YAML."""

    pipeline_id: str
    steps: tuple[PipelineStepSpec, ...]
    parameters: dict[str, Any] | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineSpec":
        resolved_path = Path(path)
        with resolved_path.open("r", encoding="utf-8") as file_obj:
            payload = yaml.safe_load(file_obj) or {}

        if not isinstance(payload, dict):
            raise ValueError("Pipeline spec must deserialize to a mapping.")

        nested = payload.get("pipeline")
        if nested is not None:
            if not isinstance(nested, dict):
                raise ValueError("pipeline section must be a mapping when provided.")
            payload = nested

        return cls.from_mapping(payload, default_pipeline_id=resolved_path.stem)

    @classmethod
    def from_mapping(
        cls,
        payload: dict[str, Any],
        *,
        default_pipeline_id: str = "pipeline",
    ) -> "PipelineSpec":
        if not isinstance(payload, dict):
            raise ValueError("Pipeline spec payload must be a mapping.")

        pipeline_id = str(payload.get("id") or default_pipeline_id).strip()
        if not pipeline_id:
            raise ValueError("Pipeline spec id must be non-empty.")

        raw_steps = payload.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ValueError("Pipeline spec must include a non-empty steps list.")

        steps = tuple(_parse_step(raw_step) for raw_step in raw_steps)
        _validate_steps(steps)
        raw_parameters = payload.get("parameters")
        parameters: dict[str, Any] | None = None
        if raw_parameters is not None:
            if not isinstance(raw_parameters, dict):
                raise ValueError("Pipeline spec parameters must be a mapping when provided.")
            parameters = canonicalize_value(dict(raw_parameters))
        return cls(pipeline_id=pipeline_id, steps=steps, parameters=parameters)

    def to_payload(self) -> dict[str, Any]:
        return canonicalize_value(
            {
                "id": self.pipeline_id,
                "steps": [step.to_payload() for step in self.steps],
                "parameters": self.parameters,
            }
        )


@dataclass(frozen=True)
class PipelineStepResult:
    """One executed pipeline step."""

    step_id: str
    module: str
    argv: tuple[str, ...]
    depends_on: tuple[str, ...]
    result: Any


@dataclass(frozen=True)
class PipelineStepMetrics:
    """Deterministic timing and status metrics for one pipeline step."""

    step_id: str
    started_at_unix: float | None
    ended_at_unix: float | None
    duration_seconds: float
    status: str

    def to_payload(self) -> dict[str, Any]:
        return canonicalize_value(
            {
                "step_id": self.step_id,
                "started_at_unix": self.started_at_unix,
                "ended_at_unix": self.ended_at_unix,
                "duration_seconds": self.duration_seconds,
                "status": self.status,
            }
        )


@dataclass(frozen=True)
class PipelineRunResult:
    """Structured result returned from one pipeline CLI run."""

    pipeline_id: str
    pipeline_run_id: str
    status: str
    execution_order: tuple[str, ...]
    spec: PipelineSpec
    step_results: tuple[PipelineStepResult, ...]
    artifact_dir: Path
    manifest_path: Path
    pipeline_metrics_path: Path
    lineage_path: Path


class PipelineRunner:
    """Run one validated PipelineSpec deterministically."""

    def __init__(self, spec: PipelineSpec) -> None:
        self._spec = spec

    @property
    def spec(self) -> PipelineSpec:
        return self._spec

    def pipeline_run_id(self) -> str:
        payload = {
            "pipeline_id": self._spec.pipeline_id,
            "spec": self._spec.to_payload(),
        }
        digest = hashlib.sha256(serialize_canonical_json(payload).encode("utf-8")).hexdigest()[:12]
        return f"{_sanitize_name_component(self._spec.pipeline_id)}_pipeline_{digest}"

    def artifact_dir(self) -> Path:
        return (Path.cwd() / "artifacts" / "pipelines" / self.pipeline_run_id()).resolve()

    def execution_order(self) -> tuple[str, ...]:
        declaration_index = {step.id: index for index, step in enumerate(self._spec.steps)}
        remaining_dependencies = {
            step.id: len(step.depends_on)
            for step in self._spec.steps
        }
        dependents: dict[str, list[str]] = {step.id: [] for step in self._spec.steps}
        for step in self._spec.steps:
            for dependency in step.depends_on:
                dependents[dependency].append(step.id)

        ready = deque(
            step.id
            for step in self._spec.steps
            if remaining_dependencies[step.id] == 0
        )
        ordered: list[str] = []
        while ready:
            step_id = ready.popleft()
            ordered.append(step_id)
            newly_ready: list[str] = []
            for dependent_id in dependents[step_id]:
                remaining_dependencies[dependent_id] -= 1
                if remaining_dependencies[dependent_id] == 0:
                    newly_ready.append(dependent_id)
            for dependent_id in sorted(newly_ready, key=lambda item: declaration_index[item]):
                ready.append(dependent_id)

        if len(ordered) != len(self._spec.steps):
            raise ValueError("Pipeline spec contains at least one dependency cycle.")
        return tuple(ordered)

    def run(self) -> PipelineRunResult:
        pipeline_run_id = self.pipeline_run_id()
        artifact_dir = self.artifact_dir()
        ordered_ids = self.execution_order()
        step_by_id = {step.id: step for step in self._spec.steps}
        step_results: list[PipelineStepResult] = []
        step_metrics_by_id: dict[str, PipelineStepMetrics] = {
            step_id: PipelineStepMetrics(
                step_id=step_id,
                started_at_unix=None,
                ended_at_unix=None,
                duration_seconds=0.0,
                status="skipped",
            )
            for step_id in ordered_ids
        }
        failure: BaseException | None = None
        pipeline_started_at = _timestamp_now()

        for step_id in ordered_ids:
            step = step_by_id[step_id]
            step_started_at = _timestamp_now()
            try:
                module = importlib.import_module(step.module)
                run_cli = getattr(module, "run_cli", None)
                if not callable(run_cli):
                    raise ValueError(f"Pipeline step module '{step.module}' must expose callable run_cli(argv).")
                result = run_cli(list(step.argv))
                step_ended_at = _timestamp_now()
                step_status = _status_from_step_result(result)
                step_results.append(
                    PipelineStepResult(
                        step_id=step.id,
                        module=step.module,
                        argv=step.argv,
                        depends_on=step.depends_on,
                        result=result,
                    )
                )
                step_metrics_by_id[step.id] = PipelineStepMetrics(
                    step_id=step.id,
                    started_at_unix=step_started_at,
                    ended_at_unix=step_ended_at,
                    duration_seconds=_duration_seconds(step_started_at, step_ended_at),
                    status=step_status,
                )
            except BaseException as exc:
                failure = exc
                step_ended_at = _timestamp_now()
                step_metrics_by_id[step.id] = PipelineStepMetrics(
                    step_id=step.id,
                    started_at_unix=step_started_at,
                    ended_at_unix=step_ended_at,
                    duration_seconds=_duration_seconds(step_started_at, step_ended_at),
                    status="failed",
                )
                break

        status = "failed" if failure is not None else "completed"
        pipeline_ended_at = _timestamp_now()
        result = PipelineRunResult(
            pipeline_id=self._spec.pipeline_id,
            pipeline_run_id=pipeline_run_id,
            status=status,
            execution_order=ordered_ids,
            spec=self._spec,
            step_results=tuple(step_results),
            artifact_dir=artifact_dir,
            manifest_path=artifact_dir / "manifest.json",
            pipeline_metrics_path=artifact_dir / "pipeline_metrics.json",
            lineage_path=artifact_dir / "lineage.json",
        )
        _write_pipeline_artifacts(
            result,
            pipeline_started_at=pipeline_started_at,
            pipeline_ended_at=pipeline_ended_at,
            step_metrics_by_id=step_metrics_by_id,
        )
        if failure is not None:
            raise failure
        return result


def _parse_step(payload: Any) -> PipelineStepSpec:
    if not isinstance(payload, dict):
        raise ValueError("Each pipeline step must be a mapping.")

    step_id = str(payload.get("id") or "").strip()
    if not step_id:
        raise ValueError("Each pipeline step must define a non-empty id.")

    adapter = str(payload.get("adapter") or "").strip()
    if adapter != "python_module":
        raise ValueError(
            f"Pipeline step '{step_id}' uses unsupported adapter {adapter!r}; only 'python_module' is supported."
        )

    module = str(payload.get("module") or "").strip()
    if not module:
        raise ValueError(f"Pipeline step '{step_id}' must define a non-empty module.")

    raw_argv = payload.get("argv") or []
    if not isinstance(raw_argv, list):
        raise ValueError(f"Pipeline step '{step_id}' argv must be provided as a list.")

    raw_depends_on = payload.get("depends_on") or []
    if isinstance(raw_depends_on, str):
        depends_on = (raw_depends_on,)
    elif isinstance(raw_depends_on, list):
        depends_on = tuple(str(item) for item in raw_depends_on)
    else:
        raise ValueError(f"Pipeline step '{step_id}' depends_on must be a string or list.")

    return PipelineStepSpec(
        id=step_id,
        adapter=adapter,
        module=module,
        argv=tuple(str(item) for item in raw_argv),
        depends_on=depends_on,
    )


def _validate_steps(steps: Sequence[PipelineStepSpec]) -> None:
    step_ids = [step.id for step in steps]
    duplicate_ids = sorted({step_id for step_id in step_ids if step_ids.count(step_id) > 1})
    if duplicate_ids:
        formatted = ", ".join(duplicate_ids)
        raise ValueError(f"Pipeline spec contains duplicate step ids: {formatted}.")

    known_ids = set(step_ids)
    for step in steps:
        if len(set(step.depends_on)) != len(step.depends_on):
            raise ValueError(f"Pipeline step '{step.id}' contains duplicate depends_on entries.")
        unknown = sorted(dependency for dependency in step.depends_on if dependency not in known_ids)
        if unknown:
            formatted = ", ".join(unknown)
            raise ValueError(f"Pipeline step '{step.id}' depends on unknown step ids: {formatted}.")
        if step.id in step.depends_on:
            raise ValueError(f"Pipeline step '{step.id}' cannot depend on itself.")


def _sanitize_name_component(name: str) -> str:
    cleaned = "".join(char if char.isalnum() else "_" for char in name.strip().lower())
    normalized = "_".join(part for part in cleaned.split("_") if part)
    return normalized or "pipeline"


def _write_pipeline_artifacts(
    result: PipelineRunResult,
    *,
    pipeline_started_at: float,
    pipeline_ended_at: float,
    step_metrics_by_id: Mapping[str, PipelineStepMetrics],
) -> None:
    manifest_payload = _build_pipeline_manifest_payload(result, step_metrics_by_id=step_metrics_by_id)
    metrics_payload = _build_pipeline_metrics_payload(
        result,
        pipeline_started_at=pipeline_started_at,
        pipeline_ended_at=pipeline_ended_at,
        step_metrics_by_id=step_metrics_by_id,
    )
    lineage_payload = _build_pipeline_lineage_payload(result)
    _validate_pipeline_artifacts(
        manifest_payload=manifest_payload,
        metrics_payload=metrics_payload,
        lineage_payload=lineage_payload,
    )
    result.artifact_dir.mkdir(parents=True, exist_ok=True)
    _write_json(result.manifest_path, manifest_payload)
    _write_json(result.pipeline_metrics_path, metrics_payload)
    _write_json(result.lineage_path, lineage_payload)


def _build_pipeline_manifest_payload(
    result: PipelineRunResult,
    *,
    step_metrics_by_id: Mapping[str, PipelineStepMetrics],
) -> dict[str, Any]:
    step_results_by_id = {step_result.step_id: step_result for step_result in result.step_results}
    steps_payload: list[dict[str, Any]] = []
    for step_id in result.execution_order:
        step_spec = next(step for step in result.spec.steps if step.id == step_id)
        step_result = step_results_by_id.get(step_id)
        step_metrics = step_metrics_by_id[step_id]
        normalized_outputs = _normalize_step_outputs(None if step_result is None else step_result.result)
        references = _extract_step_artifact_references(None if step_result is None else step_result.result)
        steps_payload.append(
            canonicalize_value(
                {
                    "step_id": step_id,
                    "status": step_metrics.status,
                    "outputs": normalized_outputs,
                    "step_artifact_dir": references["step_artifact_dir"],
                    "step_manifest_path": references["step_manifest_path"],
                    "module": step_spec.module,
                    "depends_on": list(step_spec.depends_on),
                }
            )
        )

    return canonicalize_value(
        {
            "pipeline_run_id": result.pipeline_run_id,
            "pipeline_name": result.pipeline_id,
            "run_type": "pipeline_manifest",
            "schema_version": 1,
            "status": result.status,
            "steps": steps_payload,
        }
    )


def _build_pipeline_metrics_payload(
    result: PipelineRunResult,
    *,
    pipeline_started_at: float,
    pipeline_ended_at: float,
    step_metrics_by_id: Mapping[str, PipelineStepMetrics],
) -> dict[str, Any]:
    ordered_step_metrics = [step_metrics_by_id[step_id] for step_id in result.execution_order]
    row_counts = _build_row_counts(result)
    payload = {
        "run_type": "pipeline_metrics",
        "schema_version": 1,
        "pipeline_run_id": result.pipeline_run_id,
        "pipeline_name": result.pipeline_id,
        "status": result.status,
        "started_at_unix": pipeline_started_at,
        "ended_at_unix": pipeline_ended_at,
        "duration_seconds": _duration_seconds(pipeline_started_at, pipeline_ended_at),
        "steps": [step_metrics.to_payload() for step_metrics in ordered_step_metrics],
        "step_durations_seconds": {
            step_metrics.step_id: step_metrics.duration_seconds
            for step_metrics in ordered_step_metrics
        },
        "status_counts": _status_counts(ordered_step_metrics),
    }
    if row_counts:
        payload["row_counts"] = row_counts
    return canonicalize_value(payload)


def _build_pipeline_lineage_payload(result: PipelineRunResult) -> dict[str, Any]:
    step_results_by_id = {step_result.step_id: step_result for step_result in result.step_results}
    step_status_by_id = _build_lineage_step_statuses(result)

    nodes_by_id: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, str]] = []

    dataset_nodes, dataset_edges = _build_dataset_lineage_nodes(result.spec)
    for node in dataset_nodes:
        nodes_by_id[node["id"]] = node
    edges.extend(dataset_edges)

    for step in result.spec.steps:
        step_result = step_results_by_id.get(step.id)
        normalized_outputs = _normalize_step_outputs(None if step_result is None else step_result.result)
        step_node = {
            "id": f"step:{step.id}",
            "type": "step",
            "metadata": canonicalize_value(
                {
                    "module": step.module,
                    "status": step_status_by_id.get(step.id, "completed" if step_result is not None else "skipped"),
                }
            ),
        }
        if normalized_outputs != {}:
            step_node["outputs"] = normalized_outputs
        nodes_by_id[step_node["id"]] = canonicalize_value(step_node)

        for dependency_step_id in step.depends_on:
            edges.append(
                {
                    "from": f"step:{dependency_step_id}",
                    "to": f"step:{step.id}",
                    "type": "depends_on",
                }
            )

        references = _extract_step_artifact_references(None if step_result is None else step_result.result)
        artifact_dir = references["step_artifact_dir"]
        if artifact_dir is not None:
            artifact_node_id = f"artifact:{artifact_dir}"
            nodes_by_id[artifact_node_id] = canonicalize_value(
                {
                    "id": artifact_node_id,
                    "type": "artifact",
                    "metadata": {
                        "path": artifact_dir,
                        "manifest_path": references["step_manifest_path"],
                    },
                }
            )
            edges.append(
                {
                    "from": f"step:{step.id}",
                    "to": artifact_node_id,
                    "type": "produces",
                }
            )

    nodes = [nodes_by_id[node_id] for node_id in sorted(nodes_by_id)]
    ordered_edges = sorted(
        (canonicalize_value(edge) for edge in edges),
        key=lambda item: (str(item["from"]), str(item["to"]), str(item["type"])),
    )
    return canonicalize_value(
        {
            "run_type": "pipeline_lineage",
            "schema_version": 1,
            "pipeline_run_id": result.pipeline_run_id,
            "nodes": nodes,
            "edges": ordered_edges,
        }
    )


def _build_lineage_step_statuses(result: PipelineRunResult) -> dict[str, str]:
    statuses = {
        step_result.step_id: _status_from_step_result(step_result.result)
        for step_result in result.step_results
    }
    if result.status != "failed":
        return statuses

    completed_count = len(result.step_results)
    if completed_count < len(result.execution_order):
        failed_step_id = result.execution_order[completed_count]
        statuses[failed_step_id] = "failed"
        for step_id in result.execution_order[completed_count + 1:]:
            statuses[step_id] = "skipped"
    return statuses


def _build_dataset_lineage_nodes(spec: PipelineSpec) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    parameters = spec.parameters if isinstance(spec.parameters, Mapping) else {}
    datasets = parameters.get("datasets")
    if not isinstance(datasets, Mapping):
        return [], []

    step_ids = {step.id for step in spec.steps}
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    for dataset_name in sorted(str(name).strip() for name in datasets if str(name).strip()):
        dataset_node_id = f"dataset:{dataset_name}"
        nodes.append(
            canonicalize_value(
                {
                    "id": dataset_node_id,
                    "type": "dataset",
                    "metadata": {"name": dataset_name},
                }
            )
        )
        consumer_step_ids = _dataset_consumer_step_ids(datasets[dataset_name], step_ids=step_ids)
        for step_id in consumer_step_ids:
            edges.append(
                {
                    "from": dataset_node_id,
                    "to": f"step:{step_id}",
                    "type": "depends_on",
                }
            )
    return nodes, edges


def _dataset_consumer_step_ids(value: Any, *, step_ids: set[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        candidates = [str(item) for item in value]
    elif isinstance(value, Mapping):
        consumers = value.get("consumers", value.get("steps", value.get("step_ids")))
        if isinstance(consumers, str):
            candidates = [consumers]
        elif isinstance(consumers, Sequence) and not isinstance(consumers, str | bytes):
            candidates = [str(item) for item in consumers]
        else:
            candidates = []
    else:
        candidates = []

    normalized = sorted({candidate.strip() for candidate in candidates if candidate.strip() in step_ids})
    return tuple(normalized)


def _normalize_step_outputs(value: Any) -> Any:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return canonicalize_value(dict(value))
    if isinstance(value, tuple | list):
        return canonicalize_value(list(value))
    if isinstance(value, str | int | float | bool | Path):
        return canonicalize_value(value)
    if hasattr(value, "__dict__"):
        return canonicalize_value(
            {
                key: attribute
                for key, attribute in vars(value).items()
                if not key.startswith("_")
            }
        )
    return canonicalize_value({"repr": repr(value)})


def _status_from_step_result(value: Any) -> str:
    mapping = _result_mapping(value)
    status = mapping.get("status")
    if isinstance(status, str):
        normalized = status.strip().lower()
        if normalized in {"completed", "skipped", "reused"}:
            return normalized
    return "completed"


def _build_row_counts(result: PipelineRunResult) -> dict[str, int]:
    row_counts: dict[str, int] = {}
    for step_result in result.step_results:
        row_count = _infer_row_count(step_result.result)
        if row_count is not None:
            row_counts[step_result.step_id] = row_count
    return canonicalize_value(row_counts)


def _infer_row_count(value: Any) -> int | None:
    direct_row_count = _direct_row_count(value)
    if direct_row_count is not None:
        return direct_row_count

    if isinstance(value, Mapping):
        return _sum_row_counts(value.values())
    if isinstance(value, (list, tuple)):
        return _sum_row_counts(value)
    if hasattr(value, "__dict__"):
        return _sum_row_counts(
            attribute
            for key, attribute in vars(value).items()
            if not key.startswith("_")
        )
    return None


def _sum_row_counts(values: Sequence[Any] | Any) -> int | None:
    total = 0
    found = False
    for item in values:
        item_row_count = _infer_row_count(item)
        if item_row_count is not None:
            total += item_row_count
            found = True
    if found:
        return total
    return None


def _direct_row_count(value: Any) -> int | None:
    csv_row_count = _csv_row_count(value)
    if csv_row_count is not None:
        return csv_row_count

    shape = getattr(value, "shape", None)
    if isinstance(shape, tuple) and shape and isinstance(shape[0], int):
        return int(shape[0])
    return None


def _csv_row_count(value: Any) -> int | None:
    candidate: Path | None = None
    if isinstance(value, Path):
        candidate = value
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped:
            candidate = Path(stripped)
    if candidate is None or candidate.suffix.lower() != ".csv" or not candidate.exists():
        return None

    with candidate.open("r", encoding="utf-8", newline="") as file_obj:
        row_total = sum(1 for _ in csv.reader(file_obj))
    return max(row_total - 1, 0)


def _status_counts(step_metrics: Sequence[PipelineStepMetrics]) -> dict[str, int]:
    counts = {status: 0 for status in ("completed", "failed", "skipped", "reused")}
    for step_metric in step_metrics:
        counts[step_metric.status] += 1
    return counts


def _timestamp_now() -> float:
    return round(time.time(), 6)


def _duration_seconds(started_at_unix: float | None, ended_at_unix: float | None) -> float:
    if started_at_unix is None or ended_at_unix is None:
        return 0.0
    return round(max(0.0, ended_at_unix - started_at_unix), 6)


def _extract_step_artifact_references(value: Any) -> dict[str, str | None]:
    mapping = _result_mapping(value)
    artifact_dir = _extract_path_value(
        mapping,
        keys=("artifact_dir", "step_artifact_dir", "output_dir", "campaign_artifact_dir", "orchestration_artifact_dir"),
    )
    manifest_path = _extract_path_value(
        mapping,
        keys=("manifest_json", "manifest_path", "step_manifest_path", "campaign_manifest_path", "orchestration_manifest_path"),
    )

    output_paths = mapping.get("output_paths")
    if manifest_path is None and isinstance(output_paths, Mapping):
        manifest_path = _extract_path_value(output_paths, keys=("manifest_json", "manifest_path"))

    if artifact_dir is None and manifest_path is not None:
        artifact_dir = str(Path(manifest_path).parent.as_posix())

    return canonicalize_value(
        {
            "step_artifact_dir": artifact_dir,
            "step_manifest_path": manifest_path,
        }
    )


def _result_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return {
            key: attribute
            for key, attribute in vars(value).items()
            if not key.startswith("_")
        }
    return {}


def _extract_path_value(mapping: Mapping[str, Any], *, keys: Sequence[str]) -> str | None:
    for key in keys:
        candidate = mapping.get(key)
        if isinstance(candidate, Path):
            return candidate.as_posix()
        if isinstance(candidate, str):
            normalized = candidate.strip()
            if normalized:
                return Path(normalized).as_posix()
    return None


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(canonicalize_value(payload), indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )


def _validate_pipeline_artifacts(
    *,
    manifest_payload: dict[str, Any],
    metrics_payload: dict[str, Any],
    lineage_payload: dict[str, Any],
) -> None:
    contracts_dir = Path(__file__).resolve().parents[2] / "contracts"
    validate_json(manifest_payload, contracts_dir / "pipeline_manifest.schema.json")
    validate_json(metrics_payload, contracts_dir / "pipeline_metrics.schema.json")
    validate_json(lineage_payload, contracts_dir / "lineage.schema.json")
