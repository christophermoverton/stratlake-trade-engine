from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli.generate_regime_review_pack import run_cli
from src.config.regime_review_pack import (
    RegimeReviewPackConfig,
    RegimeReviewReportConfig,
    load_regime_review_pack_config,
)
from src.research.regime_review_pack import run_regime_review_pack


def _base_row(name: str, variant_type: str, decision_metrics: dict[str, object] | None = None) -> dict[str, object]:
    row = {
        "benchmark_run_id": "test_run",
        "variant_name": name,
        "variant_type": variant_type,
        "regime_source": "policy" if variant_type == "policy_optimized" else "gmm",
        "calibration_profile": "baseline" if variant_type == "policy_optimized" else None,
        "classifier_model": "gaussian_mixture_model",
        "policy_name": "policy_default" if variant_type == "policy_optimized" else None,
        "mean_confidence": 0.70,
        "mean_entropy": 0.50,
        "transition_rate": 0.10,
        "avg_regime_duration": 8.0,
        "dominant_regime_share": 0.60,
        "adaptive_vs_static_return_delta": None,
        "adaptive_vs_static_sharpe_delta": None,
        "adaptive_vs_static_max_drawdown_delta": None,
        "policy_turnover": None,
    }
    row.update(decision_metrics or {})
    return row


def _write_source_artifacts(
    tmp_path: Path,
    *,
    include_optional: bool = True,
    matrix_format: str = "csv",
) -> tuple[Path, Path]:
    benchmark_path = tmp_path / "artifacts" / "regime_benchmarks" / "test_run"
    gates_path = benchmark_path / "promotion_gates"
    gates_path.mkdir(parents=True)
    matrix_rows = [
        _base_row("accepted_policy", "policy_optimized", {"adaptive_vs_static_sharpe_delta": 0.30, "adaptive_vs_static_return_delta": 0.05, "adaptive_vs_static_max_drawdown_delta": 0.02, "policy_turnover": 0.20}),
        _base_row("warning_gmm", "gmm_classifier", {"adaptive_vs_static_sharpe_delta": 0.10}),
        _base_row("review_gmm", "gmm_classifier", {"mean_confidence": 0.62}),
        _base_row("rejected_gmm", "gmm_classifier", {"mean_confidence": 0.40}),
    ]
    if matrix_format == "json":
        (benchmark_path / "benchmark_matrix.json").write_text(
            json.dumps(
                {
                    "benchmark_run_id": "test_run",
                    "columns": list(matrix_rows[0]),
                    "row_count": len(matrix_rows),
                    "rows": matrix_rows,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
            newline="\n",
        )
    else:
        pd.DataFrame(matrix_rows).to_csv(benchmark_path / "benchmark_matrix.csv", index=False)
    (benchmark_path / "benchmark_summary.json").write_text(
        json.dumps({"benchmark_run_id": "test_run", "benchmark_name": "test_benchmark"}, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    if include_optional:
        (benchmark_path / "artifact_provenance.json").write_text("{}", encoding="utf-8", newline="\n")
    gate_rows = []
    decisions = [
        ("accepted_policy", "accepted", "pass", "passed", "Metric mean_confidence passed."),
        ("warning_gmm", "accepted_with_warnings", "warn", "failed", "Metric mean_entropy failed."),
        ("review_gmm", "needs_review", "review", "needs_review", "Metric transition_rate failed."),
        ("rejected_gmm", "rejected", "fail", "failed", "Metric mean_confidence failed."),
    ]
    for name, _decision, impact, status, message in decisions:
        gate_rows.append(
            {
                "variant_name": name,
                "variant_type": "policy_optimized" if name == "accepted_policy" else "gmm_classifier",
                "regime_source": "policy" if name == "accepted_policy" else "gmm",
                "calibration_profile": "baseline" if name == "accepted_policy" else None,
                "classifier_model": "gaussian_mixture_model",
                "policy_name": "policy_default" if name == "accepted_policy" else None,
                "gate_name": "min_mean_confidence" if name == "rejected_gmm" else "max_mean_entropy",
                "gate_category": "confidence" if name == "rejected_gmm" else "entropy",
                "metric_name": "mean_confidence" if name == "rejected_gmm" else "mean_entropy",
                "metric_value": 0.40 if name == "rejected_gmm" else 0.50,
                "threshold_operator": ">=" if name == "rejected_gmm" else "<=",
                "threshold_value": 0.55 if name == "rejected_gmm" else 0.45,
                "severity": "failure" if impact == "fail" else "warning",
                "status": status,
                "decision_impact": impact,
                "message": message,
            }
        )
    gate_rows.append({**gate_rows[0], "gate_name": "max_policy_turnover", "gate_category": "policy_turnover", "metric_name": "policy_turnover", "status": "not_applicable", "decision_impact": "ignore", "message": "Metric policy_turnover missing for gmm_classifier and treated as not_applicable."})
    pd.DataFrame(gate_rows).to_csv(gates_path / "gate_results.csv", index=False)
    decision_summary = {
        "benchmark_run_id": "test_run",
        "gate_config_name": "test_gates",
        "decision_counts": {"accepted": 1, "accepted_with_warnings": 1, "needs_review": 1, "rejected": 1},
        "variant_decisions": [
            {"variant_name": name, "variant_type": "policy_optimized" if name == "accepted_policy" else "gmm_classifier", "decision": decision, "primary_reasons": [message]}
            for name, decision, _impact, _status, message in decisions
        ],
    }
    (gates_path / "decision_summary.json").write_text(json.dumps(decision_summary, indent=2), encoding="utf-8", newline="\n")
    if include_optional:
        (gates_path / "gate_results.json").write_text(json.dumps({"rows": gate_rows}, indent=2), encoding="utf-8", newline="\n")
        pd.DataFrame([row for row in gate_rows if row["decision_impact"] == "fail"]).to_csv(gates_path / "failed_gates.csv", index=False)
        pd.DataFrame([row for row in gate_rows if row["decision_impact"] == "warn"]).to_csv(gates_path / "warning_gates.csv", index=False)
        (gates_path / "manifest.json").write_text("{}", encoding="utf-8", newline="\n")
    return benchmark_path, gates_path


def _config(tmp_path: Path, benchmark_path: Path, gates_path: Path) -> RegimeReviewPackConfig:
    return RegimeReviewPackConfig(
        review_name="test_review",
        benchmark_path=benchmark_path.as_posix(),
        promotion_gates_path=gates_path.as_posix(),
        output_root=(tmp_path / "reviews").as_posix(),
    )


def test_regime_review_pack_generates_review_artifacts(tmp_path: Path) -> None:
    benchmark_path, gates_path = _write_source_artifacts(tmp_path)

    result = run_regime_review_pack(_config(tmp_path, benchmark_path, gates_path))

    leaderboard = pd.read_csv(result.leaderboard_csv_path)
    summary = json.loads(result.review_summary_path.read_text(encoding="utf-8"))
    decision_log = json.loads(result.decision_log_path.read_text(encoding="utf-8"))
    evidence = json.loads(result.evidence_index_path.read_text(encoding="utf-8"))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert leaderboard["variant_name"].tolist() == ["accepted_policy", "warning_gmm", "review_gmm", "rejected_gmm"]
    assert summary["decision_counts"]["rejected"] == 1
    assert decision_log["rows"][0]["passed_gates"]
    assert "benchmark_matrix" in evidence["source_artifacts"]
    assert manifest["artifact_type"] == "regime_review_pack"
    assert result.report_path is not None and result.report_path.exists()


def test_regime_review_pack_evidence_uses_json_matrix_when_csv_absent(tmp_path: Path) -> None:
    benchmark_path, gates_path = _write_source_artifacts(tmp_path, matrix_format="json")

    result = run_regime_review_pack(_config(tmp_path, benchmark_path, gates_path))
    evidence = json.loads(result.evidence_index_path.read_text(encoding="utf-8"))

    assert evidence["source_artifacts"]["benchmark_matrix"].endswith("benchmark_matrix.json")
    assert not evidence["source_artifacts"]["benchmark_matrix"].endswith("benchmark_matrix.csv")


def test_regime_review_pack_evidence_lists_all_generated_artifacts(tmp_path: Path) -> None:
    benchmark_path, gates_path = _write_source_artifacts(tmp_path)

    result = run_regime_review_pack(_config(tmp_path, benchmark_path, gates_path))
    evidence = json.loads(result.evidence_index_path.read_text(encoding="utf-8"))

    generated = evidence["generated_review_artifacts"]
    assert "review_inputs.json" in generated
    assert "report.md" in generated
    assert set(generated) == set(result.review_summary["generated_artifacts"])


def test_regime_review_pack_can_disable_markdown_report(tmp_path: Path) -> None:
    benchmark_path, gates_path = _write_source_artifacts(tmp_path)
    config = RegimeReviewPackConfig(
        review_name="test_review_no_report",
        benchmark_path=benchmark_path.as_posix(),
        promotion_gates_path=gates_path.as_posix(),
        output_root=(tmp_path / "reviews").as_posix(),
        report=RegimeReviewReportConfig(write_markdown=False),
    )

    result = run_regime_review_pack(config)
    evidence = json.loads(result.evidence_index_path.read_text(encoding="utf-8"))

    assert result.report_path is None
    assert not (result.output_dir / "report.md").exists()
    assert "report.md" not in evidence["generated_review_artifacts"]
    assert "review_inputs.json" in evidence["generated_review_artifacts"]


def test_regime_review_pack_report_warning_section_toggle(tmp_path: Path) -> None:
    benchmark_path, gates_path = _write_source_artifacts(tmp_path)
    default_result = run_regime_review_pack(_config(tmp_path, benchmark_path, gates_path))
    assert default_result.report_path is not None
    assert "Warning Variants" in default_result.report_path.read_text(encoding="utf-8")

    config = RegimeReviewPackConfig(
        review_name="test_review_without_warning_section",
        benchmark_path=benchmark_path.as_posix(),
        promotion_gates_path=gates_path.as_posix(),
        output_root=(tmp_path / "reviews").as_posix(),
        report=RegimeReviewReportConfig(include_warning_summary=False),
    )

    result = run_regime_review_pack(config)

    assert result.report_path is not None
    assert "Warning Variants" not in result.report_path.read_text(encoding="utf-8")


def test_regime_review_pack_filtered_variant_files(tmp_path: Path) -> None:
    benchmark_path, gates_path = _write_source_artifacts(tmp_path)

    result = run_regime_review_pack(_config(tmp_path, benchmark_path, gates_path))

    assert pd.read_csv(result.accepted_variants_path)["variant_name"].tolist() == ["accepted_policy"]
    assert pd.read_csv(result.warning_variants_path)["variant_name"].tolist() == ["warning_gmm"]
    assert pd.read_csv(result.needs_review_variants_path)["variant_name"].tolist() == ["review_gmm"]
    rejected = pd.read_csv(result.rejected_candidates_path)
    assert rejected["variant_name"].tolist() == ["rejected_gmm"]
    assert rejected["primary_failure_category"].tolist() == ["confidence"]


def test_regime_review_pack_is_deterministic(tmp_path: Path) -> None:
    benchmark_path, gates_path = _write_source_artifacts(tmp_path)
    config = _config(tmp_path, benchmark_path, gates_path)

    first = run_regime_review_pack(config)
    second = run_regime_review_pack(config)

    assert first.review_run_id == second.review_run_id
    assert first.leaderboard_csv_path.read_text(encoding="utf-8") == second.leaderboard_csv_path.read_text(encoding="utf-8")
    assert first.decision_log_path.read_text(encoding="utf-8") == second.decision_log_path.read_text(encoding="utf-8")


def test_regime_review_pack_missing_optional_artifacts_warn(tmp_path: Path) -> None:
    benchmark_path, gates_path = _write_source_artifacts(tmp_path, include_optional=False)

    result = run_regime_review_pack(_config(tmp_path, benchmark_path, gates_path))
    evidence = json.loads(result.evidence_index_path.read_text(encoding="utf-8"))

    assert evidence["missing_optional_artifacts"]
    assert "artifact_provenance" in evidence["missing_optional_artifacts"][0] or evidence["missing_optional_artifacts"]


def test_regime_review_pack_missing_required_artifact_fails(tmp_path: Path) -> None:
    benchmark_path, gates_path = _write_source_artifacts(tmp_path)
    (gates_path / "decision_summary.json").unlink()

    with pytest.raises(FileNotFoundError, match="decision_summary"):
        run_regime_review_pack(_config(tmp_path, benchmark_path, gates_path))


def test_regime_review_pack_cli_smoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    benchmark_path, gates_path = _write_source_artifacts(tmp_path)

    result = run_cli(
        [
            "--benchmark-path",
            benchmark_path.as_posix(),
            "--promotion-gates-path",
            gates_path.as_posix(),
            "--output-root",
            (tmp_path / "reviews").as_posix(),
        ]
    )
    captured = capsys.readouterr()

    assert "Regime Review Pack Summary" in captured.out
    assert result.leaderboard_csv_path.exists()


def test_regime_review_pack_cli_config_driven(tmp_path: Path) -> None:
    benchmark_path, gates_path = _write_source_artifacts(tmp_path)
    path = tmp_path / "review.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "review_name": "config_review",
                "benchmark_path": benchmark_path.as_posix(),
                "promotion_gates_path": gates_path.as_posix(),
                "output_root": (tmp_path / "reviews").as_posix(),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
        newline="\n",
    )

    result = run_cli(["--config", path.as_posix()])

    assert result.review_name == "config_review"
    assert result.review_summary_path.exists()


def test_regime_review_pack_config_loader_supports_explicit_paths(tmp_path: Path) -> None:
    benchmark_path, gates_path = _write_source_artifacts(tmp_path)
    path = tmp_path / "review.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "review_name": "config_review",
                "benchmark_path": benchmark_path.as_posix(),
                "promotion_gates_path": gates_path.as_posix(),
                "output_root": (tmp_path / "reviews").as_posix(),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
        newline="\n",
    )

    config = load_regime_review_pack_config(path)

    assert config.benchmark_path == benchmark_path.as_posix()
    assert config.promotion_gates_path == gates_path.as_posix()
