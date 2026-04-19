# Research Campaign Orchestration Pipeline

## Purpose

Demonstrate canonical campaign orchestration with multi-run grouping, scenario expansion, and orchestration-level aggregation outputs.

## Components Exercised

- scenario-enabled research campaign execution
- scenario catalog and matrix artifacts
- deterministic orchestration rerun behavior
- cross-scenario ranking and summary outputs

## Run

```powershell
python docs/examples/pipelines/research_campaign_orchestration/pipeline.py
```

## Configuration

`config.yml` points at the repository's real orchestration case-study implementation and captures output placement for this canonical pipeline wrapper.

## Output Interpretation

- wrapper `summary.json` records orchestration identity and status details
- source case study output directory contains scenario catalog, matrix, and orchestration manifests
