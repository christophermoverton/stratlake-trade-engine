from __future__ import annotations

from pathlib import Path

from docs.examples.scale_repro_benchmark_pack import run_example


def test_scale_repro_benchmark_pack_example_writes_reproducible_snapshots(tmp_path: Path) -> None:
    summary = run_example(output_root=tmp_path, reset_output=True)

    assert summary["partial"]["status"] == "partial"
    assert summary["resumed"]["status"] == "completed"
    assert summary["stable"]["status"] == "completed"
    assert summary["comparison"] is not None
    assert summary["comparison"]["matches"] is True

    snapshot_dir = tmp_path / "snapshots"
    assert (snapshot_dir / "partial_summary.json").exists()
    assert (snapshot_dir / "resumed_summary.json").exists()
    assert (snapshot_dir / "stable_summary.json").exists()
    assert (snapshot_dir / "stable_inventory.json").exists()
