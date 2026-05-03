from __future__ import annotations

import json
from pathlib import Path

from src.execution.result import ExecutionResult
from src.validation.cross_layer import (
    compare_normalized_payloads,
    normalize_artifact_payload,
    run_cross_layer_validation,
    write_cross_layer_validation_report,
)


def test_normalization_removes_root_specific_paths_and_marker_timestamps(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    payload = {
        "summary_path": first_root / "summary.json",
        "output_paths": {"summary": (first_root / "summary.json").as_posix()},
        "campaign_run_id": "research_campaign_aaaaaaaaaaaa",
        "fingerprint": "abc123",
        "relative_path": "batches/batch_0000/artifacts/research_campaigns/research_campaign_orchestration_aaaaaaaaaaaa/summary.json",
        "_SUCCESS.json": {"recorded_at_utc": "2026-05-03T01:02:03Z"},
        "inventory": {
            "entries": [
                {"path": "summary.json", "sha256": "abc", "size_bytes": 10},
            ],
            "aggregate_digest": "digest",
        },
    }

    left = normalize_artifact_payload(payload, root=first_root)
    right = normalize_artifact_payload(
        {
            **payload,
            "summary_path": second_root / "summary.json",
            "output_paths": {"summary": (second_root / "summary.json").as_posix()},
            "campaign_run_id": "research_campaign_bbbbbbbbbbbb",
            "fingerprint": "def456",
            "relative_path": "batches/batch_0000/artifacts/research_campaigns/research_campaign_orchestration_bbbbbbbbbbbb/summary.json",
            "_SUCCESS.json": {"recorded_at_utc": "2026-05-03T09:08:07Z"},
            "inventory": {
                "entries": [
                    {"path": "summary.json", "sha256": "def", "size_bytes": 99},
                ],
                "aggregate_digest": "other",
            },
        },
        root=second_root,
    )

    assert compare_normalized_payloads(left, right) == []
    assert "<OUTPUT_ROOT>/summary.json" in json.dumps(left, sort_keys=True)
    assert "2026-05-03T01:02:03Z" not in json.dumps(left, sort_keys=True)
    assert "abc" not in json.dumps(left, sort_keys=True)


def test_compare_normalized_payloads_reports_stable_differences() -> None:
    differences = compare_normalized_payloads(
        {"summary": {"status": "completed", "scenario_count": 3}},
        {"summary": {"status": "completed", "scenario_count": 4}},
    )

    assert differences == ["$.summary.scenario_count: 3 != 4"]


def test_cross_layer_validation_reports_representative_benchmark_pack_pass(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_result(output_root: Path, *, status: str = "partial") -> ExecutionResult:
        return _write_benchmark_contract(output_root, status=status)

    monkeypatch.setattr(
        "src.validation.cross_layer._run_api_benchmark",
        lambda config_path, output_root, stop_after_batches: fake_result(
            output_root,
            status="completed" if stop_after_batches is None else "partial",
        ),
    )
    monkeypatch.setattr(
        "src.validation.cross_layer._run_cli_benchmark",
        lambda config_path, output_root, stop_after_batches: fake_result(output_root, status="partial"),
    )
    monkeypatch.setattr(
        "src.validation.cross_layer._run_notebook_benchmark",
        lambda repo_root, output_root: fake_result(output_root, status="partial"),
    )
    monkeypatch.setattr(
        "src.validation.cross_layer._run_prefect_wrapper_benchmark",
        lambda repo_root, output_root: fake_result(output_root, status="completed"),
    )

    report = run_cross_layer_validation(repo_root=tmp_path, output_root=tmp_path / "workdir")

    assert report["status"] == "passed"
    assert report["scenario_count"] == 3
    assert report["pass_count"] == 3
    assert [scenario["name"] for scenario in report["scenarios"]] == [
        "benchmark_pack_cli_api",
        "notebook_benchmark_api",
        "prefect_wrapper_api",
    ]
    assert str(tmp_path) not in json.dumps(report, sort_keys=True)


def test_cross_layer_validation_fails_when_stable_artifact_diverges(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "src.validation.cross_layer._run_api_benchmark",
        lambda config_path, output_root, stop_after_batches: _write_benchmark_contract(
            output_root,
            status="partial",
            scenario_count=3,
        ),
    )
    monkeypatch.setattr(
        "src.validation.cross_layer._run_cli_benchmark",
        lambda config_path, output_root, stop_after_batches: _write_benchmark_contract(
            output_root,
            status="partial",
            scenario_count=4,
        ),
    )

    report = run_cross_layer_validation(
        repo_root=tmp_path,
        output_root=tmp_path / "workdir",
        scenarios=("benchmark_pack_cli_api",),
    )

    assert report["status"] == "failed"
    assert report["pass_count"] == 0
    assert any("scenario_count" in diff for diff in report["scenarios"][0]["differences"])


def test_cross_layer_validation_report_writes_deterministic_json(tmp_path: Path) -> None:
    report = {
        "run_type": "cross_layer_validation",
        "schema_version": 1,
        "status": "passed",
        "scenario_count": 1,
        "pass_count": 1,
        "scenarios": [],
    }
    output_path = write_cross_layer_validation_report(report, tmp_path / "report.json")

    assert output_path == tmp_path / "report.json"
    assert json.loads(output_path.read_text(encoding="utf-8")) == report
    assert output_path.read_text(encoding="utf-8").splitlines()[0] == "{"


def _write_benchmark_contract(
    output_root: Path,
    *,
    status: str,
    scenario_count: int = 3,
) -> ExecutionResult:
    output_root.mkdir(parents=True, exist_ok=True)
    pack_run_id = "m22_scale_repro_stable"
    summary = {
        "run_type": "benchmark_pack",
        "pack_id": "m22_scale_repro",
        "pack_run_id": pack_run_id,
        "status": status,
        "batch_count": 1,
        "scenario_count": scenario_count,
        "batch_status_counts": {status: 1},
        "dataset": {"features_root": (output_root / "data").as_posix()},
        "benchmark_matrix": {
            "benchmark_matrix_id": pack_run_id,
            "row_count": scenario_count,
            "csv_path": (output_root / "benchmark_matrix.csv").as_posix(),
            "summary_path": (output_root / "benchmark_matrix.json").as_posix(),
        },
        "output_paths": {
            "summary": (output_root / "summary.json").as_posix(),
            "manifest": (output_root / "manifest.json").as_posix(),
        },
        "inventory": {
            "file_count": 2,
            "aggregate_digest": "root-specific",
            "entries": [
                {"path": "summary.json", "sha256": "root-specific", "size_bytes": 10},
                {"path": "benchmark_matrix.csv", "sha256": "root-specific", "size_bytes": 10},
            ],
        },
    }
    manifest = {
        "run_type": "benchmark_pack",
        "pack_id": "m22_scale_repro",
        "pack_run_id": pack_run_id,
        "status": status,
        "artifact_files": ["summary.json", "manifest.json", "benchmark_matrix.csv"],
        "artifact_groups": {"benchmark_pack": ["summary.json", "manifest.json"]},
        "summary_path": "summary.json",
        "batch_status_counts": {status: 1},
        "batches": [
            {
                "batch_id": "batch_0000",
                "status": status,
                "orchestration_summary_path": (output_root / "batches" / "summary.json").as_posix(),
                "scenario_ids": ["baseline", "stress", "recovery"][:scenario_count],
            }
        ],
    }
    batch_plan = {
        "pack_id": "m22_scale_repro",
        "batch_size": 3,
        "batch_count": 1,
        "scenario_count": scenario_count,
        "batches": [{"batch_id": "batch_0000", "scenario_count": scenario_count}],
    }
    matrix_summary = {
        "benchmark_matrix_id": pack_run_id,
        "row_count": scenario_count,
        "csv_path": (output_root / "benchmark_matrix.csv").as_posix(),
        "summary_path": (output_root / "benchmark_matrix.json").as_posix(),
        "leaderboard": [{"rank": index, "scenario_id": f"scenario_{index}"} for index in range(1, scenario_count + 1)],
    }
    inventory = summary["inventory"]
    checkpoint = {
        "run_type": "benchmark_pack_checkpoint",
        "pack_run_id": pack_run_id,
        "status": status,
        "output_root": output_root.as_posix(),
        "batch_count": 1,
    }
    config = {"pack_id": "m22_scale_repro"}
    dataset_summary = {"features_root": (output_root / "data").as_posix(), "parquet_file_count": 1}

    paths = {
        "summary_json": output_root / "summary.json",
        "manifest_json": output_root / "manifest.json",
        "checkpoint_json": output_root / "checkpoint.json",
        "inventory_json": output_root / "inventory.json",
        "batch_plan_json": output_root / "batch_plan.json",
        "benchmark_matrix_summary": output_root / "benchmark_matrix.json",
        "config_json": output_root / "benchmark_pack_config.json",
        "dataset_summary_json": output_root / "dataset_summary.json",
        "benchmark_matrix_csv": output_root / "benchmark_matrix.csv",
    }
    for path, payload in (
        (paths["summary_json"], summary),
        (paths["manifest_json"], manifest),
        (paths["checkpoint_json"], checkpoint),
        (paths["inventory_json"], inventory),
        (paths["batch_plan_json"], batch_plan),
        (paths["benchmark_matrix_summary"], matrix_summary),
        (paths["config_json"], config),
        (paths["dataset_summary_json"], dataset_summary),
    ):
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    paths["benchmark_matrix_csv"].write_text(
        "rank,scenario_id\n"
        + "".join(f"{index},scenario_{index}\n" for index in range(1, scenario_count + 1)),
        encoding="utf-8",
    )

    return ExecutionResult(
        workflow="benchmark_pack",
        run_id=pack_run_id,
        name=pack_run_id,
        artifact_dir=output_root,
        metrics=summary,
        manifest_path=paths["manifest_json"],
        output_paths=paths,
        extra={"pack_id": "m22_scale_repro", "status": status, "comparison": None},
    )
