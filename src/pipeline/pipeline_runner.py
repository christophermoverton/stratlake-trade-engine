from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import importlib
from pathlib import Path
from typing import Any, Sequence

import yaml

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
        return cls(pipeline_id=pipeline_id, steps=steps)

    def to_payload(self) -> dict[str, Any]:
        return canonicalize_value(
            {
                "id": self.pipeline_id,
                "steps": [step.to_payload() for step in self.steps],
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
class PipelineRunResult:
    """Structured result returned from one pipeline CLI run."""

    pipeline_id: str
    pipeline_run_id: str
    execution_order: tuple[str, ...]
    spec: PipelineSpec
    step_results: tuple[PipelineStepResult, ...]


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
        ordered_ids = self.execution_order()
        step_by_id = {step.id: step for step in self._spec.steps}
        step_results: list[PipelineStepResult] = []
        for step_id in ordered_ids:
            step = step_by_id[step_id]
            module = importlib.import_module(step.module)
            run_cli = getattr(module, "run_cli", None)
            if not callable(run_cli):
                raise ValueError(f"Pipeline step module '{step.module}' must expose callable run_cli(argv).")
            result = run_cli(list(step.argv))
            step_results.append(
                PipelineStepResult(
                    step_id=step.id,
                    module=step.module,
                    argv=step.argv,
                    depends_on=step.depends_on,
                    result=result,
                )
            )
        return PipelineRunResult(
            pipeline_id=self._spec.pipeline_id,
            pipeline_run_id=self.pipeline_run_id(),
            execution_order=ordered_ids,
            spec=self._spec,
            step_results=tuple(step_results),
        )


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
