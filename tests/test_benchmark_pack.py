from __future__ import annotations

from pathlib import Path

from docs.examples.scale_repro_benchmark_pack import _fake_campaign_runner
from src.config.benchmark_pack import load_benchmark_pack_config
from src.research.benchmark_pack import run_benchmark_pack


REPO_ROOT = Path(__file__).resolve().parents[1]
PACK_CONFIG_PATH = REPO_ROOT / "configs" / "benchmark_packs" / "m22_scale_repro.yml"


def test_benchmark_pack_supports_partial_resume_and_stable_reruns(tmp_path: Path) -> None:
    config = load_benchmark_pack_config(PACK_CONFIG_PATH)

    partial = run_benchmark_pack(
        config,
        output_root=tmp_path,
        stop_after_batches=1,
        campaign_runner=_fake_campaign_runner,
    )
    assert partial.status == "partial"
    assert partial.summary["batch_status_counts"] == {"completed": 1, "pending": 2}
    assert partial.summary["resume"]["pending_batch_ids"] == ["batch_0001", "batch_0002"]

    resumed = run_benchmark_pack(
        config,
        output_root=tmp_path,
        campaign_runner=_fake_campaign_runner,
    )
    assert resumed.status == "completed"
    assert resumed.summary["batch_status_counts"] == {"completed": 2, "reused": 1}
    assert resumed.summary["resume"]["reused_batch_ids"] == ["batch_0000"]
    assert resumed.summary["scenario_count"] == 7
    assert resumed.summary["benchmark_matrix"]["row_count"] == 7

    stable = run_benchmark_pack(
        config,
        output_root=tmp_path,
        compare_to=resumed.inventory_path,
        campaign_runner=_fake_campaign_runner,
    )
    assert stable.status == "completed"
    assert stable.summary["batch_status_counts"] == {"reused": 3}
    assert stable.summary["resume"]["reused_batch_ids"] == [
        "batch_0000",
        "batch_0001",
        "batch_0002",
    ]
    assert stable.comparison is not None
    assert stable.comparison["matches"] is True
    assert stable.inventory["aggregate_digest"] == resumed.inventory["aggregate_digest"]
