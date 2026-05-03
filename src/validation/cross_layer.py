from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
import os
from pathlib import Path
import re
import runpy
from typing import Any, Mapping, Sequence

from src.artifacts.safety import atomic_write_json, ensure_output_root_available, mark_run_completed, mark_run_started
from src.execution.result import ExecutionResult, load_json_artifact


DEFAULT_BENCHMARK_CONFIG = "configs/benchmark_packs/m22_scale_repro.yml"
DEFAULT_SCENARIOS: tuple[str, ...] = (
    "benchmark_pack_cli_api",
    "notebook_benchmark_api",
    "prefect_wrapper_api",
)

_JSON_OUTPUT_KEYS = (
    "config_json",
    "dataset_summary_json",
    "summary_json",
    "manifest_json",
    "checkpoint_json",
    "inventory_json",
    "batch_plan_json",
    "benchmark_matrix_summary",
)
_CSV_OUTPUT_KEYS = ("benchmark_matrix_csv",)
_UNSTABLE_KEYS = {
    "artifact_dir",
    "csv_path",
    "current_inventory_path",
    "duration_seconds",
    "finished_at_utc",
    "log_path",
    "marker_path",
    "recorded_at_utc",
    "reference_inventory_path",
    "started_at_utc",
    "stderr",
    "stdout",
    "summary_path",
}
_DIGEST_KEYS = {"aggregate_digest", "sha256", "size_bytes"}
_GENERATED_ID_KEYS = {
    "benchmark_matrix_id",
    "campaign_run_id",
    "comparison_id",
    "fingerprint",
    "input_fingerprint",
    "orchestration_run_id",
    "pack_run_id",
    "review_id",
    "run_id",
}
_MARKER_FILENAMES = {"_RUNNING.json", "_SUCCESS.json", "_FAILED.json"}


@dataclass(frozen=True)
class CrossLayerScenarioResult:
    name: str
    left_layer: str
    right_layer: str
    status: str
    differences: list[str]
    left_digest: str
    right_digest: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_artifact_payload(payload: Any, *, root: str | Path | None = None) -> Any:
    """Normalize a persisted artifact payload to its stable comparison contract."""

    roots = (Path(root).resolve(),) if root is not None else ()
    return _normalize_value(payload, roots=roots, key=None)


def normalize_execution_result(result: ExecutionResult, *, root: str | Path | None = None) -> dict[str, Any]:
    """Return stable, machine-readable fields from an ExecutionResult."""

    root_path = Path(root).resolve() if root is not None else result.artifact_dir
    roots = (root_path.resolve(),) if root_path is not None else ()
    return {
        "workflow": result.workflow,
        "name": _normalize_value(result.name, roots=roots, key="name"),
        "metrics": _normalize_value(result.metrics, roots=roots, key="metrics"),
        "output_keys": list(result.output_keys()),
        "extra": _normalize_value(result.extra, roots=roots, key="extra"),
    }


def compare_normalized_payloads(left: Any, right: Any) -> list[str]:
    """Return stable path-addressed differences between two normalized payloads."""

    differences: list[str] = []
    _collect_differences(left, right, path="$", differences=differences)
    return differences


def run_cross_layer_validation(
    *,
    repo_root: str | Path = ".",
    output_root: str | Path = "artifacts/qa/m28_cross_layer_validation",
    scenarios: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run representative cross-layer parity checks and return a JSON-safe report."""

    root = Path(repo_root).resolve()
    run_root = Path(output_root).resolve()
    ensure_output_root_available(run_root, collision_policy="reuse")
    mark_run_started(run_root, {"run_type": "cross_layer_validation"})
    work_root = _next_attempt_root(run_root / "attempts", prefix=f"pid_{os.getpid()}")

    selected = tuple(scenarios or DEFAULT_SCENARIOS)
    unknown = sorted(set(selected).difference(DEFAULT_SCENARIOS))
    if unknown:
        raise ValueError(f"Unknown cross-layer validation scenario(s): {', '.join(unknown)}")

    scenario_results: list[CrossLayerScenarioResult] = []
    references: dict[str, tuple[ExecutionResult, dict[str, Any]]] = {}

    def reference(stop_after_batches: int | None) -> tuple[ExecutionResult, dict[str, Any]]:
        key = "full" if stop_after_batches is None else f"stop_after_{stop_after_batches}"
        if key not in references:
            result = _run_api_benchmark(
                config_path=root / DEFAULT_BENCHMARK_CONFIG,
                output_root=work_root / "references" / key,
                stop_after_batches=stop_after_batches,
            )
            references[key] = (
                result,
                _benchmark_contract(result, root=result.artifact_dir),
            )
        return references[key]

    if "benchmark_pack_cli_api" in selected:
        _api_result, api_contract = reference(1)
        cli_result = _run_cli_benchmark(
            config_path=root / DEFAULT_BENCHMARK_CONFIG,
            output_root=work_root / "benchmark_pack_cli",
            stop_after_batches=1,
        )
        scenario_results.append(
            _compare_scenario(
                name="benchmark_pack_cli_api",
                left_layer="cli",
                right_layer="src.execution_api",
                left=_benchmark_contract(cli_result, root=cli_result.artifact_dir),
                right=api_contract,
            )
        )

    if "notebook_benchmark_api" in selected:
        _api_result, api_contract = reference(1)
        notebook_result = _run_notebook_benchmark(
            repo_root=root,
            output_root=work_root / "notebook_benchmark",
        )
        scenario_results.append(
            _compare_scenario(
                name="notebook_benchmark_api",
                left_layer="notebook_style_callable",
                right_layer="src.execution_api",
                left=_benchmark_contract(notebook_result, root=notebook_result.artifact_dir),
                right=api_contract,
            )
        )

    if "prefect_wrapper_api" in selected:
        _api_result, api_contract = reference(None)
        wrapper_result = _run_prefect_wrapper_benchmark(
            repo_root=root,
            output_root=work_root / "prefect_wrapper",
        )
        scenario_results.append(
            _compare_scenario(
                name="prefect_wrapper_api",
                left_layer="prefect_fallback_callable",
                right_layer="src.execution_api",
                left=_benchmark_contract(wrapper_result, root=wrapper_result.artifact_dir),
                right=api_contract,
            )
        )

    pass_count = sum(1 for result in scenario_results if result.status == "passed")
    report = {
        "run_type": "cross_layer_validation",
        "schema_version": 1,
        "status": "passed" if pass_count == len(scenario_results) else "failed",
        "scenario_count": len(scenario_results),
        "pass_count": pass_count,
        "scenarios": [result.to_dict() for result in scenario_results],
        "comparison_contract": {
            "stable_fields": [
                "workflow",
                "logical benchmark-pack config identity",
                "named output keys",
                "summary status and counts",
                "manifest schema and artifact groups",
                "batch plan fields",
                "benchmark matrix rows and columns",
                "inventory relative entry paths",
            ],
            "normalized_or_ignored_fields": [
                "absolute output roots",
                "artifact-root-specific path prefixes",
                "status marker timestamps",
                "transient stdout/stderr",
                "temporary files",
                "inventory file hashes and byte sizes",
            ],
        },
        "limitations": [
            "Representative benchmark-pack parity only; not exhaustive for every workflow config.",
            "Does not prove distributed locking or production scheduler deployment readiness.",
            "Does not replace full pytest, deterministic rerun validation, or milestone validation.",
        ],
    }
    mark_run_completed(run_root, {"run_type": "cross_layer_validation", "status": report["status"]})
    return report


def write_cross_layer_validation_report(report: Mapping[str, Any], output_path: str | Path) -> Path:
    return atomic_write_json(output_path, dict(report), sort_keys=True)


def concise_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_type": report.get("run_type"),
        "status": report.get("status"),
        "scenario_count": report.get("scenario_count"),
        "pass_count": report.get("pass_count"),
        "failed_scenarios": [
            scenario.get("name")
            for scenario in report.get("scenarios", [])
            if isinstance(scenario, Mapping) and scenario.get("status") != "passed"
        ],
    }


def _next_attempt_root(parent: Path, *, prefix: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    candidate = parent / prefix
    if not candidate.exists():
        return candidate
    index = 2
    while True:
        candidate = parent / f"{prefix}_{index:03d}"
        if not candidate.exists():
            return candidate
        index += 1


def _run_api_benchmark(
    *,
    config_path: Path,
    output_root: Path,
    stop_after_batches: int | None,
) -> ExecutionResult:
    from src.execution import run_benchmark_pack

    return run_benchmark_pack(
        config_path,
        output_root=output_root,
        stop_after_batches=stop_after_batches,
    )


def _run_cli_benchmark(
    *,
    config_path: Path,
    output_root: Path,
    stop_after_batches: int | None,
) -> ExecutionResult:
    from src.execution.benchmark import run_benchmark_pack_from_argv

    argv = [
        "--config",
        config_path.as_posix(),
        "--output-root",
        output_root.as_posix(),
    ]
    if stop_after_batches is not None:
        argv.extend(["--stop-after-batches", str(stop_after_batches)])
    return run_benchmark_pack_from_argv(argv)


def _run_notebook_benchmark(*, repo_root: Path, output_root: Path) -> ExecutionResult:
    namespace = runpy.run_path(repo_root / "docs" / "examples" / "notebooks" / "m28_benchmark_pack_execution_api.py")
    result, inspection = namespace["run_benchmark_pack_notebook_cell"](output_root=output_root)
    if "notebook_summary" not in inspection:
        raise ValueError("Notebook benchmark callable did not return the expected inspection payload.")
    return result


def _run_prefect_wrapper_benchmark(*, repo_root: Path, output_root: Path) -> ExecutionResult:
    namespace = runpy.run_path(repo_root / "docs" / "examples" / "pipelines" / "m28_prefect_regime_research_flow.py")
    callable_fn = namespace["run_m28_prefect_example"]
    callable_fn.__globals__["build_m28_prefect_output_root"] = lambda flow_run_id="manual", attempt=1: output_root
    return callable_fn("cross-layer-validation", 1)


def _benchmark_contract(result: ExecutionResult, *, root: str | Path | None) -> dict[str, Any]:
    artifact_root = Path(root).resolve() if root is not None else result.artifact_dir
    contract: dict[str, Any] = {
        "execution_result": normalize_execution_result(result, root=artifact_root),
        "artifacts": {},
    }
    for key in _JSON_OUTPUT_KEYS:
        if not result.has_output(key):
            continue
        path = result.output_path(key)
        if not path.exists():
            continue
        payload = load_json_artifact(path)
        if key == "inventory_json":
            payload = _inventory_contract(payload)
        contract["artifacts"][key] = normalize_artifact_payload(payload, root=artifact_root)
    for key in _CSV_OUTPUT_KEYS:
        if result.has_output(key) and result.output_path(key).exists():
            contract["artifacts"][key] = _csv_contract(result.output_path(key), root=artifact_root)
    return contract


def _inventory_contract(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {"entries": []}
    entries = [
        {"path": str(entry["path"])}
        for entry in payload.get("entries", [])
        if isinstance(entry, Mapping) and "path" in entry and Path(str(entry["path"])).name not in _MARKER_FILENAMES
    ]
    return {
        "file_count": len(entries),
        "entries": sorted(entries, key=lambda item: item["path"]),
    }


def _csv_contract(path: Path, *, root: Path | None) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [
            {
                key: _normalize_value(
                    value,
                    roots=(root.resolve(),) if root is not None else (),
                    key=key,
                )
                for key, value in sorted(row.items())
            }
            for row in reader
        ]
    return {
        "columns": list(reader.fieldnames or []),
        "row_count": len(rows),
        "rows": sorted(rows, key=lambda row: json.dumps(row, sort_keys=True, separators=(",", ":"))),
    }


def _compare_scenario(
    *,
    name: str,
    left_layer: str,
    right_layer: str,
    left: dict[str, Any],
    right: dict[str, Any],
) -> CrossLayerScenarioResult:
    differences = compare_normalized_payloads(left, right)
    return CrossLayerScenarioResult(
        name=name,
        left_layer=left_layer,
        right_layer=right_layer,
        status="passed" if not differences else "failed",
        differences=differences,
        left_digest=_stable_digest(left),
        right_digest=_stable_digest(right),
    )


def _normalize_value(value: Any, *, roots: Sequence[Path], key: str | None) -> Any:
    if key in _GENERATED_ID_KEYS or (key is not None and key.endswith("_fingerprint")):
        return "<GENERATED_ID>"
    if key in _UNSTABLE_KEYS or key in _DIGEST_KEYS:
        return "<IGNORED>"
    if isinstance(value, Mapping):
        return {
            str(item_key): _normalize_value(item_value, roots=roots, key=str(item_key))
            for item_key, item_value in sorted(value.items(), key=lambda item: str(item[0]))
            if Path(str(item_key)).name not in _MARKER_FILENAMES
        }
    if isinstance(value, list):
        return [_normalize_value(item, roots=roots, key=key) for item in value]
    if isinstance(value, tuple):
        return [_normalize_value(item, roots=roots, key=key) for item in value]
    if isinstance(value, Path):
        return _normalize_text(value.as_posix(), roots=roots)
    if isinstance(value, str):
        if Path(value).name in _MARKER_FILENAMES:
            return "<STATUS_MARKER>"
        return _normalize_text(value, roots=roots)
    return value


def _normalize_text(text: str, *, roots: Sequence[Path]) -> str:
    normalized = text.replace("\\", "/")
    for root in roots:
        root_text = root.as_posix()
        normalized = normalized.replace(root_text, "<OUTPUT_ROOT>")
    normalized = re.sub(r"<OUTPUT_ROOT>/+", "<OUTPUT_ROOT>/", normalized)
    normalized = re.sub(r"research_campaign_orchestration_[0-9a-f]{12}", "research_campaign_orchestration_<ID>", normalized)
    normalized = re.sub(r"research_campaign_[0-9a-f]{12}", "research_campaign_<ID>", normalized)
    normalized = re.sub(r"registry_single_[A-Za-z0-9_]+_[0-9a-f]{12}", "registry_single_<ID>", normalized)
    normalized = re.sub(r"registry_review_[0-9a-f]{12}", "registry_review_<ID>", normalized)
    return normalized


def _collect_differences(left: Any, right: Any, *, path: str, differences: list[str]) -> None:
    if type(left) is not type(right):
        differences.append(f"{path}: type {type(left).__name__} != {type(right).__name__}")
        return
    if isinstance(left, Mapping):
        left_keys = set(left)
        right_keys = set(right)
        for key in sorted(left_keys - right_keys):
            differences.append(f"{path}.{key}: missing from right")
        for key in sorted(right_keys - left_keys):
            differences.append(f"{path}.{key}: missing from left")
        for key in sorted(left_keys & right_keys):
            _collect_differences(left[key], right[key], path=f"{path}.{key}", differences=differences)
        return
    if isinstance(left, list):
        if len(left) != len(right):
            differences.append(f"{path}: length {len(left)} != {len(right)}")
            return
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _collect_differences(left_item, right_item, path=f"{path}[{index}]", differences=differences)
        return
    if left != right:
        differences.append(f"{path}: {left!r} != {right!r}")


def _stable_digest(payload: Any) -> str:
    import hashlib

    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


__all__ = [
    "DEFAULT_SCENARIOS",
    "CrossLayerScenarioResult",
    "compare_normalized_payloads",
    "concise_summary",
    "normalize_artifact_payload",
    "normalize_execution_result",
    "run_cross_layer_validation",
    "write_cross_layer_validation_report",
]
