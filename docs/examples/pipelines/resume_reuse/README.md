# Resume/Reuse Pipeline

## Purpose

Demonstrate deterministic campaign checkpointing semantics:

- partial interruption
- resumed retry behavior
- stable full reuse pass

## Components Exercised

- campaign checkpoint state transitions (`partial`, `completed`, `reused`)
- retry metadata and stage execution summaries
- reuse policy behavior across reruns

## Run

```powershell
python docs/examples/pipelines/resume_reuse/pipeline.py
```

## Configuration

`config.yml` points to the deterministic real-world resume case-study implementation used by this wrapper.

## Output Interpretation

`summary.json` in this folder gives the high-level state transitions; full checkpoint snapshots are in the wrapped case-study output directory.
