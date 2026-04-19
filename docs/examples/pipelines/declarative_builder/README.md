# Declarative Builder Pipeline

## Purpose

Demonstrate the Pipeline Helper / Declarative Builder path as a first-class canonical orchestration route.

## Components Exercised

- declarative config input (`config.yml`)
- builder rendering through `src.cli.build_pipeline`
- execution parity with equivalent imperative `PipelineBuilder` composition
- strategy plus portfolio stage composition in one generated M20 pipeline
- sweep plus downstream portfolio composition via top-ranked sweep variant selection (`--from-sweep-top-ranked`) when configured

## Run

```powershell
python docs/examples/pipelines/declarative_builder/pipeline.py
```

## Configuration

`config.yml` contains the full declarative strategy/signal/constructor/portfolio mapping.

## Output Interpretation

`summary.json` reports:

- generated pipeline run status
- execution order
- declarative-vs-imperative YAML/support-file equivalence checks
