"""CI-safe M37 artifact-first evidence contracts example.

The example builds a tiny synthetic artifact tree in a temporary directory and
demonstrates the public M37 flow without live data, credentials, or repository
artifact mutation.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.catalog import (
    build_catalog,
    build_derived_index,
    build_lineage_export_for_workflow,
    load_catalog_records_with_source,
    resolve_canonical_record,
)


def run_m37_artifact_first_evidence_contracts_example() -> dict[str, Any]:
    """Return a compact summary of the M37 canonicality flow."""

    with TemporaryDirectory(prefix="stratlake_m37_") as tmp:
        repo_root = Path(tmp)
        artifacts_root = repo_root / "artifacts"
        index_path = artifacts_root / "_derived" / "catalog_index" / "catalog_index.sqlite"
        _write_fixture_artifacts(artifacts_root)

        direct_records = build_catalog(artifacts_root, repo_root=repo_root)
        index_metadata = build_derived_index(artifacts_root, index_path, repo_root=repo_root)
        indexed_load = load_catalog_records_with_source(
            artifacts_root,
            repo_root=repo_root,
            index_path=index_path,
            mode="auto",
        )
        lineage_export = build_lineage_export_for_workflow(
            "artifacts",
            repo_root=repo_root,
            index_path=index_path,
            index_mode="auto",
            selected_run_id="strategy_demo",
        )
        resolution = resolve_canonical_record(
            direct_records[0],
            artifacts_root=artifacts_root,
            repo_root=repo_root,
        )

        payload = {
            "direct_scan_is_canonical": len(direct_records) == 1,
            "derived_index_path": index_path.relative_to(repo_root).as_posix(),
            "derived_index_canonicality": index_metadata["canonicality"],
            "indexed_load_source": indexed_load.load_source,
            "lineage_load_source": lineage_export["load_source"],
            "lineage_canonicality": lineage_export["canonicality"],
            "resolver_status": resolution.resolution_status,
            "resolver_source_paths": resolution.source_paths,
            "resolver_source_fingerprint": resolution.source_fingerprint,
        }
        gc.collect()
        return payload


def _write_fixture_artifacts(artifacts_root: Path) -> None:
    run_root = artifacts_root / "strategies" / "strategy_demo"
    _write_json(run_root / "_SUCCESS.json", {"run_id": "strategy_demo", "status": "completed"})
    _write_json(run_root / "summary.json", {"run_id": "strategy_demo", "run_type": "strategy"})
    _write_json(
        run_root / "manifest.json",
        {
            "run_id": "strategy_demo",
            "run_type": "strategy",
            "artifacts": ["summary.json", "_SUCCESS.json"],
        },
    )
    _write_jsonl(
        artifacts_root / "strategies" / "registry.jsonl",
        [
            {
                "run_id": "strategy_demo",
                "run_type": "strategy",
                "artifact_dir": "artifacts/strategies/strategy_demo",
                "timestamp": "2026-01-01T00:00:00Z",
            }
        ],
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, payloads: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(payload, sort_keys=True) + "\n" for payload in payloads),
        encoding="utf-8",
    )


if __name__ == "__main__":
    print(json.dumps(run_m37_artifact_first_evidence_contracts_example(), indent=2, sort_keys=True))
