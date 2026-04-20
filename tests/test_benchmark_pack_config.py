from __future__ import annotations

from pathlib import Path

from src.config.benchmark_pack import load_benchmark_pack_config


REPO_ROOT = Path(__file__).resolve().parents[1]
PACK_CONFIG_PATH = REPO_ROOT / "configs" / "benchmark_packs" / "m22_scale_repro.yml"


def test_load_benchmark_pack_config_resolves_repo_paths() -> None:
    config = load_benchmark_pack_config(PACK_CONFIG_PATH)

    assert config.pack_id == "m22_scale_repro"
    assert config.dataset.generator == "synthetic_features_daily"
    assert config.dataset.features_root == "workspace"
    assert config.dataset.dataset_name == "features_daily"
    assert config.dataset.periods == 90
    assert config.batching.batch_size == 3
    assert config.campaign.config_path.endswith(
        "docs/examples/data/m22_7_campaign_configs/scale_repro_strategy_benchmark.yml"
    )
    assert config.outputs.inventory_excludes == (
        "checkpoint.json",
        "manifest.json",
        "summary.json",
        "comparisons/*",
        "inventory.json",
        "snapshots/*",
    )
