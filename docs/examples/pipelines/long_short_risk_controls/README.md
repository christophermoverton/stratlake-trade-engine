# Long/Short and Risk Controls Pipeline

## Purpose

Demonstrate how directional asymmetry and constructor controls compose with strategy execution in Milestone 21.

## Components Exercised

- Signal semantics: `cross_section_rank`
- Constructor: `rank_dollar_neutral`
- Asymmetry controls:
  - `exclude_short: false`
  - `max_short_positions: 2`
  - `short_position_scale: 0.8`
- Strategy archetype: `cross_section_momentum`

## Run

```powershell
python docs/examples/pipelines/long_short_risk_controls/pipeline.py
```

## Configuration

The builder config in `config.yml` is declarative and mirrors production composition:

- strategy section
- signal section
- constructor section
- asymmetry section

## Output Interpretation

`summary.json` includes the selected asymmetry parameters and artifact references for traceability.
