from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import re
import sys
from typing import Any, Sequence


DEFAULT_CANONICAL_TARGETS: tuple[tuple[str, str], ...] = (
    ("docs/examples/pipelines/baseline_reference/pipeline.py", "runs"),
    ("docs/examples/pipelines/robustness_scenario_sweep/pipeline.py", "run"),
    ("docs/examples/pipelines/declarative_builder/pipeline.py", "equivalence"),
)


@dataclass(frozen=True)
class DeterministicRerunTargetResult:
    module_path: str
    expected_summary_key: str
    status: str
    reason: str | None
    first_summary_sha256: str | None
    second_summary_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_deterministic_rerun_validation(
    repo_root: str | Path,
    output_root: str | Path,
    targets: Sequence[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    run_root = Path(output_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    selected_targets = tuple(targets or DEFAULT_CANONICAL_TARGETS)
    target_results: list[DeterministicRerunTargetResult] = []
    for module_path, expected_summary_key in selected_targets:
        target_results.append(
            _run_single_target(
                repo_root=root,
                run_root=run_root,
                module_path=module_path,
                expected_summary_key=expected_summary_key,
            )
        )

    pass_count = sum(1 for result in target_results if result.status == "passed")
    payload = {
        "run_type": "deterministic_rerun_validation",
        "schema_version": 1,
        "status": "passed" if pass_count == len(target_results) else "failed",
        "target_count": len(target_results),
        "pass_count": pass_count,
        "targets": [result.to_dict() for result in target_results],
    }
    return payload


def write_deterministic_rerun_report(report: dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _run_single_target(
    *,
    repo_root: Path,
    run_root: Path,
    module_path: str,
    expected_summary_key: str,
) -> DeterministicRerunTargetResult:
    resolved_module_path = (repo_root / module_path).resolve()
    if not resolved_module_path.exists():
        return DeterministicRerunTargetResult(
            module_path=module_path,
            expected_summary_key=expected_summary_key,
            status="failed",
            reason="module path does not exist",
            first_summary_sha256=None,
            second_summary_sha256=None,
        )

    module = _load_example_module(resolved_module_path)
    target_name = resolved_module_path.parent.name
    first_output = run_root / target_name / "first"
    second_output = run_root / target_name / "second"

    first_artifacts = module.run_example(output_root=first_output, verbose=False, reset_output=True)
    second_artifacts = module.run_example(output_root=second_output, verbose=False, reset_output=True)
    first_summary = _normalize_summary_payload(first_artifacts.summary, roots=(first_output, second_output))
    second_summary = _normalize_summary_payload(second_artifacts.summary, roots=(first_output, second_output))

    if expected_summary_key not in first_summary:
        return DeterministicRerunTargetResult(
            module_path=module_path,
            expected_summary_key=expected_summary_key,
            status="failed",
            reason=f"missing expected summary key: {expected_summary_key}",
            first_summary_sha256=None,
            second_summary_sha256=None,
        )

    if expected_summary_key not in second_summary:
        return DeterministicRerunTargetResult(
            module_path=module_path,
            expected_summary_key=expected_summary_key,
            status="failed",
            reason=f"missing expected summary key: {expected_summary_key}",
            first_summary_sha256=None,
            second_summary_sha256=None,
        )

    first_digest = _stable_digest(first_summary)
    second_digest = _stable_digest(second_summary)
    if first_summary != second_summary:
        return DeterministicRerunTargetResult(
            module_path=module_path,
            expected_summary_key=expected_summary_key,
            status="failed",
            reason="normalized summaries differ across reruns",
            first_summary_sha256=first_digest,
            second_summary_sha256=second_digest,
        )

    return DeterministicRerunTargetResult(
        module_path=module_path,
        expected_summary_key=expected_summary_key,
        status="passed",
        reason=None,
        first_summary_sha256=first_digest,
        second_summary_sha256=second_digest,
    )


def _load_example_module(path: Path):
    module_name = f"m22_validation_{path.stem}_{abs(hash(path.as_posix()))}"
    spec = spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to load module from path: {path.as_posix()}")
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _stable_digest(payload: Any) -> str:
    import hashlib

    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _normalize_summary_payload(payload: Any, *, roots: Sequence[Path]) -> Any:
    if isinstance(payload, dict):
        return {key: _normalize_summary_payload(value, roots=roots) for key, value in sorted(payload.items())}
    if isinstance(payload, list):
        return [_normalize_summary_payload(value, roots=roots) for value in payload]
    if isinstance(payload, str):
        value = payload
        for root in roots:
            value = _replace_root(value, root)
        return value
    return payload


def _replace_root(text: str, root: Path) -> str:
    candidate_variants = {
        root.as_posix(),
        str(root),
        root.as_posix().rstrip("/"),
        str(root).rstrip("\\"),
    }
    normalized = text
    for variant in sorted(candidate_variants, key=len, reverse=True):
        if not variant:
            continue
        normalized = normalized.replace(variant, "<OUTPUT_ROOT>")
    # Also normalize escaped Windows separators that may appear in serialized JSON strings.
    normalized = re.sub(r"<OUTPUT_ROOT>[\\/]+", "<OUTPUT_ROOT>/", normalized)
    return normalized
