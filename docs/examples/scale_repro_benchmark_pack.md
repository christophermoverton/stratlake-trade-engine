# Scale Repro Benchmark Pack Example

## Purpose

This example is the canonical Milestone 22 benchmark-pack validation flow.

It demonstrates:

* deterministic scenario batching
* safe partial-stop and resume behavior
* stable rerun comparison through artifact inventories
* one global benchmark matrix that stays comparable across reruns

The example uses a deterministic in-repo campaign stub so the resume and
comparison workflow stays fast and repeatable in tests and milestone
validation. Use `python -m src.cli.run_benchmark_pack` for the real
campaign-backed entrypoint.

## Run It

```powershell
python docs/examples/scale_repro_benchmark_pack.py
```

## What It Does

The example uses the benchmark-pack config at
`configs/benchmark_packs/m22_scale_repro.yml` and executes three passes:

1. a partial pass stopped after one batch
2. a resumed pass that reuses the completed batch and finishes the remaining ones
3. a stable rerun that compares its inventory against the resumed pass

## Expected Outputs

The example writes:

* `summary.json`
* `checkpoint.json`
* `manifest.json`
* `benchmark_matrix.csv`
* `benchmark_matrix.json`
* `inventory.json`
* `snapshots/partial_*.json`
* `snapshots/resumed_*.json`
* `snapshots/stable_*.json`

The most important success signals are:

* partial pass leaves pending batches in `checkpoint.json`
* resumed pass shows previously finished batches in `resume.reused_batch_ids`
* stable rerun reports `comparison.matches == true`
