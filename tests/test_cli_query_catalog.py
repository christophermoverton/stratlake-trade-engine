from __future__ import annotations

import json
from pathlib import Path

import src.cli.query_catalog as query_catalog_cli
from src.catalog.models import CatalogRecord, CatalogValidationStatus


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, payloads: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(payload, sort_keys=True) for payload in payloads) + "\n", encoding="utf-8")


def _make_strategy(
    artifacts_root: Path,
    run_id: str,
    *,
    strategy_name: str = "momentum_v1",
    sharpe_ratio: float = 1.2,
    status_marker: str = "_SUCCESS.json",
) -> Path:
    run_root = artifacts_root / "strategies" / run_id
    _write_json(run_root / status_marker, {"run_id": run_id, "status": "completed"})
    _write_json(
        run_root / "manifest.json",
        {"run_id": run_id, "strategy_name": strategy_name, "artifacts": ["metrics.json", "summary.json"]},
    )
    _write_json(run_root / "metrics.json", {"sharpe_ratio": sharpe_ratio})
    _write_json(run_root / "summary.json", {"strategy_name": strategy_name, "start_ts": "2024-01-01"})
    return run_root


def _make_portfolio(
    artifacts_root: Path,
    run_id: str,
    *,
    component_run_ids: list[str] | None = None,
    portfolio_name: str = "risk_parity",
) -> Path:
    run_root = artifacts_root / "portfolios" / run_id
    _write_json(run_root / "_SUCCESS.json", {"run_id": run_id, "status": "completed"})
    _write_json(
        run_root / "manifest.json",
        {
            "run_id": run_id,
            "portfolio_name": portfolio_name,
            "component_run_ids": component_run_ids or [],
            "artifacts": ["summary.json"],
        },
    )
    _write_json(
        run_root / "summary.json",
        {"portfolio_name": portfolio_name, "component_run_ids": component_run_ids or []},
    )
    return run_root


def _make_template(artifacts_root: Path, run_id: str = "template_1") -> None:
    _write_jsonl(
        artifacts_root / "registry" / "portfolios.jsonl",
        [
            {
                "run_id": run_id,
                "run_type": "portfolio_template",
                "portfolio_name": "template_portfolio",
            }
        ],
    )


def test_cli_json_output_for_run_type_filter(tmp_path: Path, capsys) -> None:
    _make_strategy(tmp_path, "strategy_1")
    _make_portfolio(tmp_path, "portfolio_1")

    code = query_catalog_cli.main(
        ["--repo-root", str(tmp_path), "--artifacts-root", ".", "--run-type", "strategy", "--format", "json"]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert [record["run_id"] for record in payload] == ["strategy_1"]


def test_cli_table_output_smoke(tmp_path: Path, capsys) -> None:
    _make_strategy(tmp_path, "strategy_1")

    code = query_catalog_cli.main(["--repo-root", str(tmp_path), "--artifacts-root", ".", "--format", "table"])

    assert code == 0
    out = capsys.readouterr().out
    assert "catalog_id\trun_id\trun_type\tstatus" in out
    assert "strategy_1" in out


def test_cli_summary_output(tmp_path: Path, capsys) -> None:
    _make_strategy(tmp_path, "strategy_1")
    _make_portfolio(tmp_path, "portfolio_1")

    code = query_catalog_cli.main(
        ["--repo-root", str(tmp_path), "--artifacts-root", ".", "--summary", "--format", "json"]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["total_count"] == 2
    assert payload["by_run_type"] == {"portfolio": 1, "strategy": 1}


def test_cli_metric_filter(tmp_path: Path, capsys) -> None:
    _make_strategy(tmp_path, "strategy_high", sharpe_ratio=1.4)
    _make_strategy(tmp_path, "strategy_low", sharpe_ratio=0.6)

    code = query_catalog_cli.main(
        [
            "--repo-root",
            str(tmp_path),
            "--artifacts-root",
            ".",
            "--min-metric",
            "sharpe_ratio",
            "1.0",
            "--format",
            "json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert [record["run_id"] for record in payload] == ["strategy_high"]


def test_cli_include_templates(tmp_path: Path, capsys) -> None:
    _make_strategy(tmp_path, "strategy_1")
    _make_template(tmp_path)

    default_code = query_catalog_cli.main(["--repo-root", str(tmp_path), "--artifacts-root", ".", "--format", "json"])
    default_payload = json.loads(capsys.readouterr().out)
    include_code = query_catalog_cli.main(
        ["--repo-root", str(tmp_path), "--artifacts-root", ".", "--include-templates", "--format", "json"]
    )
    include_payload = json.loads(capsys.readouterr().out)

    assert default_code == 0
    assert include_code == 0
    assert [record["run_type"] for record in default_payload] == ["strategy"]
    assert sorted(record["run_type"] for record in include_payload) == ["portfolio_template", "strategy"]


def test_cli_related_upstream_downstream(tmp_path: Path, capsys) -> None:
    _make_strategy(tmp_path, "strategy_1")
    _make_portfolio(tmp_path, "portfolio_1", component_run_ids=["strategy_1"])

    upstream_code = query_catalog_cli.main(
        [
            "--repo-root",
            str(tmp_path),
            "--artifacts-root",
            ".",
            "--related",
            "portfolio_1",
            "--direction",
            "upstream",
            "--format",
            "json",
        ]
    )
    upstream_payload = json.loads(capsys.readouterr().out)
    downstream_code = query_catalog_cli.main(
        [
            "--repo-root",
            str(tmp_path),
            "--artifacts-root",
            ".",
            "--related",
            "strategy_1",
            "--direction",
            "downstream",
            "--edge-type",
            "portfolio_component",
            "--format",
            "json",
        ]
    )
    downstream_payload = json.loads(capsys.readouterr().out)

    assert upstream_code == 0
    assert downstream_code == 0
    assert [record["run_id"] for record in upstream_payload] == ["strategy_1"]
    assert [record["run_id"] for record in downstream_payload] == ["portfolio_1"]


def test_cli_invalid_related_target_returns_nonzero_cleanly(tmp_path: Path, capsys) -> None:
    _make_strategy(tmp_path, "strategy_1")

    code = query_catalog_cli.main(["--repo-root", str(tmp_path), "--artifacts-root", ".", "--related", "missing"])

    captured = capsys.readouterr()
    assert code == 2
    assert "Related target not found: missing" in captured.err
    assert "Traceback" not in captured.err


def test_cli_related_summary_summarizes_related_result_set(tmp_path: Path, capsys) -> None:
    _make_strategy(tmp_path, "strategy_a")
    _make_strategy(tmp_path, "strategy_b")
    _make_strategy(tmp_path, "unrelated_strategy")
    _make_portfolio(tmp_path, "portfolio_1", component_run_ids=["strategy_a", "strategy_b"])

    code = query_catalog_cli.main(
        [
            "--repo-root",
            str(tmp_path),
            "--artifacts-root",
            ".",
            "--related",
            "portfolio_1",
            "--direction",
            "upstream",
            "--summary",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["total_count"] == 2
    assert payload["by_run_type"] == {"strategy": 2}
    assert "portfolio" not in payload["by_run_type"]


def test_cli_builds_catalog_once_for_related(monkeypatch, capsys) -> None:
    records = [
        _record("strategy_a", "strategy"),
        _record("portfolio_1", "portfolio", metadata={"component_run_ids": ["strategy_a"]}),
    ]
    calls = 0

    def fake_build_catalog(*args, **kwargs):
        nonlocal calls
        calls += 1
        return records

    monkeypatch.setattr(query_catalog_cli, "build_catalog", fake_build_catalog)

    code = query_catalog_cli.main(["--related", "portfolio_1", "--direction", "upstream", "--summary"])

    assert code == 0
    assert calls == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["total_count"] == 1
    assert payload["by_run_type"] == {"strategy": 1}


def _record(
    run_id: str,
    run_type: str,
    *,
    metadata: dict | None = None,
) -> CatalogRecord:
    return CatalogRecord(
        catalog_id=f"catalog_{run_id}",
        run_id=run_id,
        run_type=run_type,
        status="completed",
        artifact_root=f"artifacts/{run_type}/{run_id}",
        source_registry_path=None,
        source_manifest_path=None,
        source_marker_path=None,
        created_at=None,
        timeframe=None,
        start_ts=None,
        end_ts=None,
        strategy_name=None,
        portfolio_name=None,
        allocator_name=None,
        alpha_model_name=None,
        regime_method=None,
        campaign_id=None,
        scenario_id=None,
        metrics_summary=None,
        qa_status=None,
        review_status=None,
        promotion_status=None,
        tags=[],
        source_files=[],
        metadata=metadata or {},
        validation=CatalogValidationStatus(
            catalog_status="valid",
            marker_status="present",
            manifest_status="missing",
            artifact_status="ok",
            qa_status=None,
            validation_errors=[],
            validation_warnings=[],
        ),
    )
