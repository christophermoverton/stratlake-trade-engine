# Milestone 22 Benchmark Packs

## Overview

Milestone 22 turns the repository's existing orchestration, checkpoint,
resume/reuse, and manifest patterns into one canonical benchmark-pack surface.

The new benchmark-pack runner does not replace `run_research_campaign`.
Instead, it sits above it and adds four operator-facing capabilities:

* deterministic scenario batching
* benchmark-level checkpoint and resume state
* stable benchmark-matrix and artifact-inventory outputs
* explicit cross-rerun inventory comparison

Use the canonical pack config at
`configs/benchmark_packs/m22_scale_repro.yml` when you want a milestone-grade
validation run that is larger than a single campaign, but still deterministic
enough for rerun comparison.

## What The Pack Produces

Every benchmark-pack run writes one top-level directory containing:

* `benchmark_pack_config.json`
* `dataset_summary.json`
* `batch_plan.json`
* `batch_plan.csv`
* `checkpoint.json`
* `manifest.json`
* `summary.json`
* `benchmark_matrix.csv`
* `benchmark_matrix.json`
* `inventory.json`

Each batch also writes its own native research-campaign orchestration artifacts
under `batches/<batch_id>/artifacts/research_campaigns/<orchestration_run_id>/`.

That means the benchmark pack preserves the existing campaign-level contracts
while adding one stable pack-level wrapper above them.

## Canonical Run

Run the pack directly:

```powershell
python -m src.cli.run_benchmark_pack --config configs/benchmark_packs/m22_scale_repro.yml
```

This pack:

* generates a deterministic synthetic `features_daily` dataset
* resolves the strategy-benchmark campaign config at
  `docs/examples/data/m22_7_campaign_configs/scale_repro_strategy_benchmark.yml`
* expands the scenario matrix into deterministic batches of three scenarios
* executes each batch through the existing research-campaign orchestration
* writes one global benchmark matrix that aggregates scenario results across
  every batch

## Simulating Interruption And Resume

Use `--stop-after-batches` to stop deterministically after a known number of
completed batches:

```powershell
python -m src.cli.run_benchmark_pack --config configs/benchmark_packs/m22_scale_repro.yml --stop-after-batches 1
```

That writes a partial benchmark checkpoint where:

* the completed batch is marked `completed`
* remaining batches stay `pending`
* completed batch artifacts are already persisted and reusable

Resume by rerunning the same command without `--stop-after-batches`:

```powershell
python -m src.cli.run_benchmark_pack --config configs/benchmark_packs/m22_scale_repro.yml
```

On resume:

* completed batches are marked `reused`
* pending batches are executed
* the benchmark matrix is rebuilt from the full set of batch outputs

This makes batch reuse explicit, inspectable, and deterministic.

## Comparing Reruns

Each run emits `inventory.json`, which records a stable file inventory and one
aggregate digest over the benchmark artifacts.

The inventory intentionally excludes the top-level benchmark `checkpoint.json`,
`manifest.json`, and `summary.json` because those files reflect operational
resume state such as `completed` vs `reused` batches. The digest is meant to
track stable comparison artifacts, not rerun bookkeeping.

Compare a later rerun to a saved inventory with:

```powershell
python -m src.cli.run_benchmark_pack --config configs/benchmark_packs/m22_scale_repro.yml --compare-to <prior-inventory.json>
```

Keep the reference inventory outside the active benchmark output directory, or
in a sibling location, so the reference file itself does not appear as a new
artifact in the next inventory pass.

That writes `comparisons/inventory_comparison.json` and answers:

* whether the rerun produced the same artifact inventory
* which files were added
* which files were removed
* which files changed

Successful reproducibility means:

* `comparison.matches == true`
* the benchmark matrix row count and ranking order are unchanged
* completed batches become `reused` on a stable identical rerun

## Canonical Example

For a checked-in milestone-validation walkthrough, run:

```powershell
python docs/examples/scale_repro_benchmark_pack.py
```

That example intentionally performs:

1. one partial benchmark pass
2. one resumed pass
3. one stable rerun compared against the resumed inventory

It writes snapshot copies under
`docs/examples/output/scale_repro_benchmark_pack/snapshots/` so the partial,
resumed, and stable states can be inspected side by side.

The example is intentionally lightweight and uses a deterministic benchmark
stub so milestone validation stays fast. The real campaign-backed benchmark
entrypoint is still `python -m src.cli.run_benchmark_pack`.
