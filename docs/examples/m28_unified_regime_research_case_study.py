from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CASE_STUDY_NAME = "m28_unified_regime_research_case_study"
MILESTONE = "M28.6"
REGIME_BENCHMARK_CONFIG = "configs/regime_benchmark_packs/m26_regime_policy_benchmark.yml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "docs" / "examples" / "output" / CASE_STUDY_NAME

# Named artifacts written by the case study to the output root.
CASE_STUDY_OUTPUTS = {
    "manifest_json": "manifest.json",
    "summary_json": "summary.json",
    "validation_report_json": "validation_report.json",
    "cross_layer_comparison_json": "cross_layer_comparison.json",
    "artifact_index_json": "artifact_index.json",
}

# Workflow stages executed by this case study.
WORKFLOW_STAGES = [
    "regime_benchmark_pack",
    "cross_layer_validation",
    "case_study_artifact_assembly",
]

# Existing StratLake execution surfaces called by this case study.
EXECUTION_SURFACES_CALLED = [
    "src.execution.regime_benchmark.run_regime_benchmark_pack",
    "src.execution.run_cross_layer_validation",
]

CASE_STUDY_LIMITATIONS = [
    "This case study uses a fixture-backed regime benchmark config and does not require live market data.",
    "Cross-layer validation covers representative benchmark-pack parity (M28.5 scenarios) only.",
    "Regime benchmark variants are deterministic diagnostic examples, not trading recommendations.",
    "Stress tests, promotion gates, review packs, and market simulation are deferred workflow stages.",
    "This case study does not imply live trading or production deployment readiness.",
    "The case study is intended for research workflow validation and M28.6 capstone documentation.",
]


def run_m28_unified_regime_research_case_study(
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    reset_output: bool = True,
    include_cross_layer_validation: bool = True,
    stop_after_or_limit_options_if_needed: bool = True,
) -> dict[str, Any]:
    """Run the Milestone 28 unified regime research case study.

    This is the M28.6 capstone callable. It orchestrates existing StratLake
    execution surfaces to demonstrate that one execution system can be reached
    from script, notebook, and pipeline/orchestrator entry points.

    Parameters
    ----------
    output_root:
        Root directory for case study outputs. Defaults to
        ``docs/examples/output/m28_unified_regime_research_case_study/``.
    reset_output:
        When ``True`` (default), remove and recreate the output root before
        running. Set ``False`` to preserve existing artifacts.
    include_cross_layer_validation:
        When ``True`` (default), run the M28.5 cross-layer validation as a
        secondary stage and include its report as
        ``cross_layer_comparison.json`` in the output root.
    stop_after_or_limit_options_if_needed:
        Reserved for future use. When ``True`` (default), the case study may
        apply lightweight limits such as ``stop_after_batches`` to keep CI
        runtimes manageable.

    Returns
    -------
    dict
        JSON-safe dict with output paths, summary, and validation report.
        Notebook users can inspect returned paths and load artifact JSON
        through the returned path keys.
    """
    output_root = Path(output_root).resolve()
    if reset_output and output_root.exists():
        _reset_output_root(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Stage 1: Regime benchmark pack
    # ------------------------------------------------------------------
    from src.execution.regime_benchmark import run_regime_benchmark_pack  # noqa: PLC0415

    regime_output_root = output_root / "regime_benchmark"
    regime_result = run_regime_benchmark_pack(
        REGIME_BENCHMARK_CONFIG,
        output_root=regime_output_root,
    )
    regime_manifest = regime_result.load_manifest()
    regime_benchmark_summary = regime_result.load_output_json("benchmark_summary_json")

    # ------------------------------------------------------------------
    # Stage 2: Cross-layer validation (M28.5 foundation)
    # ------------------------------------------------------------------
    cross_layer_report: dict[str, Any] = {
        "skipped": True,
        "reason": "include_cross_layer_validation=False",
    }
    if include_cross_layer_validation:
        from src.execution import run_cross_layer_validation  # noqa: PLC0415

        cl_result = run_cross_layer_validation(
            workdir=str(output_root / "cross_layer_workdir"),
            output=str(output_root / CASE_STUDY_OUTPUTS["cross_layer_comparison_json"]),
        )
        cross_layer_report = (
            cl_result.load_output_json("report_json")
            if cl_result.has_output("report_json")
            and cl_result.output_path("report_json").exists()
            else cl_result.to_dict()
        )
        # Ensure the artifact is always written to the canonical output path.
        _write_json(
            output_root / CASE_STUDY_OUTPUTS["cross_layer_comparison_json"],
            cross_layer_report,
        )
    else:
        _write_json(
            output_root / CASE_STUDY_OUTPUTS["cross_layer_comparison_json"],
            cross_layer_report,
        )

    # ------------------------------------------------------------------
    # Stage 3: Case study artifact assembly
    # ------------------------------------------------------------------
    summary = _case_study_summary(
        regime_result=regime_result,
        regime_benchmark_summary=regime_benchmark_summary,
        cross_layer_report=cross_layer_report,
    )
    validation_report = _validation_report(
        regime_result=regime_result,
        cross_layer_report=cross_layer_report,
    )
    manifest = _case_study_manifest(
        output_root=output_root,
        regime_result=regime_result,
        regime_manifest=regime_manifest,
        summary=summary,
    )
    artifact_index = _artifact_index(
        output_root=output_root,
        regime_result=regime_result,
    )

    _write_json(output_root / CASE_STUDY_OUTPUTS["summary_json"], summary)
    _write_json(output_root / CASE_STUDY_OUTPUTS["validation_report_json"], validation_report)
    _write_json(output_root / CASE_STUDY_OUTPUTS["manifest_json"], manifest)
    _write_json(output_root / CASE_STUDY_OUTPUTS["artifact_index_json"], artifact_index)

    return {
        "case_study_name": CASE_STUDY_NAME,
        "milestone": MILESTONE,
        "output_root": _relative_path(output_root),
        "manifest_path": _relative_path(output_root / CASE_STUDY_OUTPUTS["manifest_json"]),
        "summary_path": _relative_path(output_root / CASE_STUDY_OUTPUTS["summary_json"]),
        "validation_report_path": _relative_path(
            output_root / CASE_STUDY_OUTPUTS["validation_report_json"]
        ),
        "cross_layer_comparison_path": _relative_path(
            output_root / CASE_STUDY_OUTPUTS["cross_layer_comparison_json"]
        ),
        "artifact_index_path": _relative_path(
            output_root / CASE_STUDY_OUTPUTS["artifact_index_json"]
        ),
        "regime_benchmark_result": regime_result.notebook_summary(),
        "summary": summary,
        "validation_report": validation_report,
    }


# ---------------------------------------------------------------------------
# Private helpers — artifact assembly only, no workflow logic
# ---------------------------------------------------------------------------


def _case_study_summary(
    *,
    regime_result: Any,
    regime_benchmark_summary: Mapping[str, Any],
    cross_layer_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the machine-readable case study summary."""
    cross_layer_status = cross_layer_report.get("status", "skipped")
    cross_layer_pass = cross_layer_report.get("pass_count", 0)
    cross_layer_total = cross_layer_report.get("scenario_count", 0)
    return _sort_keys(
        {
            "case_study_name": CASE_STUDY_NAME,
            "milestone": MILESTONE,
            "workflow_stages": WORKFLOW_STAGES,
            "execution_surfaces_called": EXECUTION_SURFACES_CALLED,
            "regime_benchmark": {
                "workflow": regime_result.workflow,
                "run_id": regime_result.run_id,
                "name": regime_result.name,
                "variant_count": regime_result.extra.get("variant_count"),
                "output_keys": list(regime_result.output_keys()),
                "benchmark_name": regime_benchmark_summary.get("benchmark_name"),
                "regime_sources": regime_benchmark_summary.get("regime_sources"),
                "policy_comparison_available": regime_result.has_output("policy_comparison_csv"),
                "calibration_comparison_available": regime_result.has_output(
                    "calibration_comparison_csv"
                ),
            },
            "cross_layer_validation": {
                "status": cross_layer_status,
                "scenario_count": cross_layer_total,
                "pass_count": cross_layer_pass,
                "skipped": bool(cross_layer_report.get("skipped")),
            },
            "limitations": CASE_STUDY_LIMITATIONS,
            "research_orientation_note": (
                "This case study is research and validation oriented. "
                "It does not imply live trading or production deployment readiness."
            ),
        }
    )


def _validation_report(
    *,
    regime_result: Any,
    cross_layer_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the machine-readable validation report for the case study."""
    regime_manifest_valid = regime_result.manifest_path is not None and regime_result.manifest_path.exists()
    regime_output_keys = list(regime_result.output_keys())
    required_regime_outputs = [
        "benchmark_matrix_csv",
        "benchmark_matrix_json",
        "benchmark_summary_json",
        "manifest_json",
    ]
    missing_regime_outputs = [k for k in required_regime_outputs if k not in regime_output_keys]

    cross_layer_passed = not cross_layer_report.get("skipped") and cross_layer_report.get(
        "status"
    ) == "passed"
    case_study_outputs_present = list(CASE_STUDY_OUTPUTS.keys())

    overall_status = (
        "passed"
        if regime_manifest_valid
        and not missing_regime_outputs
        and (cross_layer_report.get("skipped") or cross_layer_passed)
        else "warning"
    )
    return _sort_keys(
        {
            "run_type": "m28_unified_regime_research_case_study_validation",
            "schema_version": 1,
            "status": overall_status,
            "regime_benchmark_checks": {
                "manifest_present": regime_manifest_valid,
                "missing_required_outputs": missing_regime_outputs,
                "all_required_outputs_present": not missing_regime_outputs,
            },
            "cross_layer_validation_checks": {
                "skipped": bool(cross_layer_report.get("skipped")),
                "status": cross_layer_report.get("status", "skipped"),
                "pass_count": cross_layer_report.get("pass_count", 0),
                "scenario_count": cross_layer_report.get("scenario_count", 0),
                "passed": cross_layer_passed,
            },
            "case_study_artifact_schema": {
                "expected_outputs": case_study_outputs_present,
            },
            "limitations": CASE_STUDY_LIMITATIONS,
        }
    )


def _case_study_manifest(
    *,
    output_root: Path,
    regime_result: Any,
    regime_manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the machine-readable case study manifest."""
    output_artifact_paths = {
        key: _relative_path(output_root / filename)
        for key, filename in CASE_STUDY_OUTPUTS.items()
    }
    regime_artifact_paths: dict[str, str] = {}
    for key in regime_result.output_keys():
        path = regime_result.output_path(key)
        if path is not None:
            regime_artifact_paths[f"regime_benchmark_{key}"] = _relative_path(path)

    return _sort_keys(
        {
            "case_study_name": CASE_STUDY_NAME,
            "milestone": MILESTONE,
            "workflow_stages": WORKFLOW_STAGES,
            "execution_surfaces_called": EXECUTION_SURFACES_CALLED,
            "regime_benchmark_run_id": regime_result.run_id,
            "regime_benchmark_name": regime_result.name,
            "output_artifact_paths": output_artifact_paths,
            "regime_artifact_paths": regime_artifact_paths,
            "case_study_summary": {
                k: summary[k]
                for k in ("case_study_name", "milestone", "workflow_stages", "cross_layer_validation")
                if k in summary
            },
        }
    )


def _artifact_index(
    *,
    output_root: Path,
    regime_result: Any,
) -> dict[str, Any]:
    """Build an evidence index of all artifacts written by the case study."""
    entries: list[dict[str, str]] = []

    # Case study top-level outputs
    for key, filename in sorted(CASE_STUDY_OUTPUTS.items()):
        path = output_root / filename
        entries.append(
            {
                "key": key,
                "path": _relative_path(path),
                "group": "case_study",
                "present": str(path.exists()),
            }
        )

    # Regime benchmark outputs
    for key in sorted(regime_result.output_keys()):
        path = regime_result.output_path(key)
        entries.append(
            {
                "key": f"regime_benchmark_{key}",
                "path": _relative_path(path),
                "group": "regime_benchmark",
                "present": str(path.exists()),
            }
        )

    return _sort_keys(
        {
            "case_study_name": CASE_STUDY_NAME,
            "milestone": MILESTONE,
            "entry_count": len(entries),
            "entries": entries,
        }
    )


def _sort_keys(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a dict with top-level keys sorted for deterministic serialization."""
    return dict(sorted(payload.items()))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )


def _relative_path(path: str | Path) -> str:
    resolved = Path(path)
    if not resolved.is_absolute():
        return resolved.as_posix()
    try:
        return resolved.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.name


def _reset_output_root(path: Path) -> None:
    resolved = path.resolve()
    allowed_root = (REPO_ROOT / "docs" / "examples" / "output").resolve()
    if allowed_root not in [resolved, *resolved.parents]:
        raise ValueError(
            f"Refusing to reset output outside docs/examples/output: {_relative_path(resolved)}"
        )
    shutil.rmtree(resolved)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the M28.6 unified regime research case study."
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT.as_posix(),
        help="Root directory for case study outputs.",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Preserve the existing output directory instead of resetting it.",
    )
    parser.add_argument(
        "--skip-cross-layer-validation",
        action="store_true",
        help="Skip the M28.5 cross-layer validation stage.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate structure and imports without running the full regime benchmark workflow. "
            "Useful for lightweight CI checks."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    if args.dry_run:
        _dry_run_structural_check()
        return

    result = run_m28_unified_regime_research_case_study(
        output_root=Path(args.output_root),
        reset_output=not args.no_reset,
        include_cross_layer_validation=not args.skip_cross_layer_validation,
    )
    summary = result["summary"]
    cl = summary.get("cross_layer_validation", {})
    print(
        f"M28.6 unified regime research case study completed | "
        f"regime_benchmark_run_id={result['regime_benchmark_result']['run_id']} | "
        f"variant_count={summary['regime_benchmark'].get('variant_count')} | "
        f"cross_layer_status={cl.get('status', 'skipped')} | "
        f"output={result['output_root']}"
    )


def _dry_run_structural_check() -> None:
    """Validate structural invariants without executing any workflow."""
    config_path = REPO_ROOT / REGIME_BENCHMARK_CONFIG
    assert config_path.exists(), f"Regime benchmark config not found: {REGIME_BENCHMARK_CONFIG}"
    assert CASE_STUDY_OUTPUTS, "CASE_STUDY_OUTPUTS must be non-empty."
    assert all(WORKFLOW_STAGES), "WORKFLOW_STAGES must be non-empty strings."
    assert all(EXECUTION_SURFACES_CALLED), "EXECUTION_SURFACES_CALLED must be non-empty strings."

    # Validate that the execution surfaces are importable.
    from src.execution.regime_benchmark import run_regime_benchmark_pack  # noqa: F401, PLC0415
    from src.execution import run_cross_layer_validation  # noqa: F401, PLC0415

    print(
        f"M28.6 dry-run structural check passed | "
        f"case_study={CASE_STUDY_NAME} | "
        f"config={REGIME_BENCHMARK_CONFIG} | "
        f"workflow_stages={WORKFLOW_STAGES}"
    )


if __name__ == "__main__":
    main()
