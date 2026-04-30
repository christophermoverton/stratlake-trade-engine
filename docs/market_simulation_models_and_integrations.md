# Market Simulation Models and Integrations (Milestone 27)

## Overview

Milestone 27 adds a full market simulation stress-testing stack on top of the
Milestone 26 adaptive regime-policy research governance layer. Market simulation
stress testing uses synthetic market paths to probe how adaptive regime policies
behave under a wider range of market conditions than the observed historical
backtest window. It complements, but does not replace, the deterministic policy
stress transforms introduced in
[docs/regime_policy_stress_testing.md](regime_policy_stress_testing.md).

Simulation outputs are for distributional stress context and research
governance. They are not trading signals and do not forecast future returns.

---

## 1. Architecture Overview

The M27 market simulation stack follows this flow:

```text
config (scenario YAML)
  → scenario framework (deterministic artifact contracts and run IDs)
  → scenario-specific generator / replay / overlay
  → per-scenario path artifacts
  → simulation_metrics/
      ├── simulation_path_metrics.csv
      ├── simulation_summary.csv
      ├── simulation_leaderboard.csv
      ├── policy_failure_summary.json
      ├── simulation_metric_config.json
      └── manifest.json
  → M27 case-study stitching
      └── docs/examples/m27_market_simulation_case_study.py
  → optional M26 integration bridge
      └── market_simulation_stress block in M26 case-study config
```

### Artifact distinction

| Layer | Artifacts |
|---|---|
| Scenario framework | `scenario_catalog.json`, run-level `manifest.json`, `config.json` |
| Individual simulation model | per-scenario path CSVs, per-scenario episode/catalog files |
| Run-level simulation metrics | `simulation_metrics/simulation_path_metrics.csv`, `simulation_summary.csv`, `simulation_leaderboard.csv`, `policy_failure_summary.json` |
| M27 case-study stitching | stitched report artifacts under the case-study output root |
| M26 integration outputs | `market_simulation_stress_summary.json`, extended M26 case-study report |

The scenario framework owns deterministic run IDs and artifact contracts.
Simulation models own path-generation logic. The metrics layer consumes all
paths and emits normalized stress indicators. The case-study and M26
integration layers consume metrics artifacts without re-running generators.

---

## 2. Historical Episode Replay

### Purpose

Episode replay re-uses windows of observed historical return data as synthetic
stress paths. This preserves real market autocorrelation, drawdown structure,
and regime transitions that parametric models cannot reproduce. It answers
whether a policy that performed well in the backtest period also holds up when
exposed to other documented historical episodes (e.g., crisis windows,
volatility regimes).

### Input fixture columns

The replay engine reads a fixture CSV with at minimum:

- `date` — ISO 8601 date column
- return columns matching the strategy or policy being evaluated

Optional columns improve episode attribution:

- `regime_label` — for regime-conditional replay segmentation
- `confidence` — for confidence-weighted replay windows

### Episode windows

Episodes are defined in the scenario config as start/end date pairs or by
named episode tags. Windows must not overlap a policy's in-sample evaluation
window to avoid lookahead. Episode metadata is recorded in each scenario's
`episode_catalog.json`.

### Adaptive / static policy return handling

Each episode window is evaluated against both the adaptive policy and the
configured static baseline. Differences in per-episode drawdown, return, and
Sharpe are recorded as replay stress deltas.

### Generated artifacts

Per-scenario replay outputs under `artifacts/market_simulation/<run_id>/`:

- `episode_catalog.json` — episode window metadata
- `replay_paths.csv` — per-episode return series
- `replay_path_metrics.csv` — per-episode path metrics
- `manifest.json`

These feed into the run-level `simulation_metrics/` layer.

### When replay is useful

- stress-testing against crisis periods not in the backtest window
- verifying that policy behavior is stable across distinct historical regimes
- anchoring Monte Carlo calibration with observed stress episodes

### Limitations

- replay paths are bounded by available fixture history
- cross-asset correlation structure is preserved only within each episode window
- cannot generate out-of-distribution market conditions beyond the fixture range
- policy metrics are computed on observed returns, not fabricated returns

---

## 3. Market Shock Overlay Engine

### Purpose

The shock overlay engine applies systematic return and confidence perturbations
on top of an existing return series without generating entirely new paths. It
answers whether a policy degrades gracefully under configurable adverse market
shocks applied to known market conditions.

### Overlay source inputs

The overlay engine reads an existing return fixture or simulation output CSV and
applies shock transforms defined in the scenario config.

### Return-bps shocks

Each scenario specifies a `return_shock_bps` parameter. This is added to every
period's return in the overlay window. Negative values simulate adverse shocks;
positive values simulate tailwind scenarios.

### Confidence degradation

If the source fixture includes a `confidence` column, the overlay can apply a
`confidence_scale` multiplier (0.0–1.0) to simulate classifier confidence
deterioration under stress. Degraded confidence can trigger fallback activations
in adaptive policies.

### Entropy amplification

If the source fixture includes an `entropy` column, an `entropy_amplification`
factor can widen the entropy distribution to simulate increased regime
ambiguity.

### Overlay vs. new path generation

Overlays modify an existing series. They do not resample or synthesize new
path structure. Seasonality, autocorrelation, and cross-period dependency are
inherited from the source fixture. This makes overlays faster and more
interpretable than full bootstrap or Monte Carlo paths, but limits the range of
market structures they can represent.

### How overlay outputs feed simulation metrics

Each overlay produces `overlay_paths.csv` and `overlay_path_metrics.csv`. These
are consumed by the run-level metrics layer alongside bootstrap and Monte Carlo
paths, so leaderboard rankings reflect behavior under shocks as well as under
synthetically generated paths.

### Current limitations

- overlays are applied uniformly across the overlay window; intra-window
  heterogeneity requires multiple scenario configs
- file-backed overlay source is required; bootstrap-generated intermediary
  overlay is reserved for follow-up work
- entropy amplification requires a `entropy` column in the source fixture

---

## 4. Regime-Aware Block Bootstrap

### Purpose

The regime-aware block bootstrap generates synthetic return paths by resampling
contiguous blocks of historical returns while respecting regime boundaries.
Sampling blocks within the same regime prevents the bootstrap from stitching
incompatible market structures across a regime transition boundary.

### Source fixture sampling

The bootstrap reads a fixture CSV with `date`, return, and `regime_label`
columns. Returns are bucketed into regime-specific pools from which blocks are
drawn.

### Block length and path length

Config fields:

- `block_length` — number of periods per resampled block
- `path_length` — total periods in each synthetic path
- `num_paths` — number of synthetic paths to generate

### Regime-bucketed sampling

Blocks are drawn from the regime bucket matching the current synthetic path
position's regime. At each block boundary, the engine either continues in the
current regime or transitions to an adjacent regime based on the empirical
transition frequency from the fixture.

### Transition-window inclusion

The bootstrap can include cross-regime transition windows as special blocks to
ensure that synthetic paths capture realistic transition dynamics. This is
controlled by the `include_transition_windows` config flag.

### Deterministic seed behavior

All sampling is seeded by the `seed` field in the scenario config. Given the
same seed and fixture, the bootstrap produces the same paths across runs.

### Return path generation

Each synthetic path is a flat concatenation of resampled blocks. The engine
writes:

- `bootstrap_paths.csv` — one column per path, one row per period
- `path_catalog.json` — path metadata including seed, block assignments,
  and regime sequence

### Generated artifacts

Under the per-scenario output directory:

- `bootstrap_paths.csv`
- `path_catalog.json`
- `bootstrap_path_metrics.csv`
- `manifest.json`

### Interpretation limits

- bootstrap paths inherit the empirical return distribution of the fixture;
  they cannot produce returns outside the fixture range
- bootstrapped regime sequences may not match real forward regime distributions
- synthetic paths are for stress-testing context, not forecasting

---

## 5. Regime-Transition Monte Carlo

### Purpose

The regime-transition Monte Carlo simulates synthetic regime-label sequences
using an estimated or user-supplied transition matrix. It answers questions
about policy behavior under hypothetical regime dynamics that may differ from
any observed historical window.

### Why this model is regime-only

The Monte Carlo in this layer generates regime-label sequences only. It does
not fabricate return values, Sharpe ratios, drawdowns, or any other
policy-performance metric from synthetic data. Policy-performance metrics
require observed return series and are not extrapolated from regime sequences
alone. This design keeps the simulation layer honest: regime paths probe
transition dynamics without producing spurious performance claims.

### Transition matrix input or generation

The transition matrix can be:

- loaded from a user-supplied JSON file (field: `transition_matrix_path`)
- estimated from the regime labels in the fixture (field: `estimate_from_fixture: true`)

Each entry `T[i][j]` is the probability of transitioning from regime `i` to
regime `j`.

### Path count and path length

Config fields:

- `num_paths` — number of regime-label sequences to generate
- `path_length` — length of each sequence in periods

### Initial regime

The `initial_regime` field sets the starting regime for all paths. If omitted,
the most frequent regime in the fixture is used.

### Duration constraints

Optional `min_regime_duration` and `max_regime_duration` fields prevent
unrealistically short or infinitely persistent regimes.

### Sticky / stress-biased transition behavior

- `sticky_factor` increases the diagonal of the transition matrix to produce
  more persistent regimes
- `stress_bias` increases transition probability toward high-volatility or
  crisis regimes to generate adversarial sequence scenarios

### Deterministic seed behavior

All path generation is seeded by the `seed` field. Given the same seed and
transition matrix, the Monte Carlo produces identical regime sequences.

### Generated artifacts

Under the per-scenario output directory:

- `monte_carlo_regime_paths.csv` — one column per path, one row per period,
  values are regime labels
- `regime_path_catalog.json` — path metadata including transition matrix used,
  seed, and initial regime
- `manifest.json`

Note: `simulation_path_metrics.csv` entries for Monte Carlo paths have no
return-based columns. Return metric columns are `null` or absent for
regime-only paths. The metrics layer handles this by flagging
`return_metrics_available: false` for Monte Carlo path rows.

---

## 6. Simulation-Aware Stress Metrics

### Purpose

The simulation metrics layer consumes all per-scenario path outputs and
produces normalized, comparable stress indicators across replay, overlay,
bootstrap, and Monte Carlo scenarios.

### Path-level metrics

`simulation_path_metrics.csv` has one row per path per scenario:

| Column | Description |
|---|---|
| `scenario_id` | deterministic scenario identifier |
| `scenario_type` | `replay`, `overlay`, `bootstrap`, `monte_carlo` |
| `path_id` | per-scenario path index |
| `return_metrics_available` | `true` if return series are available for this path |
| `mean_return` | mean period return (null for Monte Carlo) |
| `cumulative_return` | path cumulative return (null for Monte Carlo) |
| `max_drawdown` | path maximum drawdown (null for Monte Carlo) |
| `sharpe_ratio` | path Sharpe ratio (null for Monte Carlo) |
| `regime_sequence_length` | path length in periods |
| `policy_failure` | `true` if any failure threshold was breached |
| `stress_score` | composite stress score (0–1, higher = more stress) |

### Scenario summaries

`simulation_summary.csv` has one row per scenario:

| Column | Description |
|---|---|
| `scenario_id` | deterministic scenario identifier |
| `scenario_type` | model type |
| `path_count` | number of paths evaluated |
| `return_metrics_available` | `true` if any path has return metrics |
| `mean_stress_score` | mean stress score across paths |
| `pct_paths_failing` | fraction of paths with `policy_failure = true` |
| `tail_p05_return` | 5th-percentile cumulative return across paths |
| `tail_p10_return` | 10th-percentile cumulative return across paths |
| `tail_p05_drawdown` | 5th-percentile max drawdown across paths |
| `mean_max_drawdown` | mean max drawdown across return-available paths |
| `mean_sharpe_ratio` | mean Sharpe ratio across return-available paths |

### Quantile-aware tail-risk columns

The `tail_p05_return` and `tail_p05_drawdown` columns represent the worst-case
percentile tail from the simulated path distribution. These are the primary
stress indicators for governance review: a policy that shows acceptable
aggregate behavior but a thin tail may be fragile under rare but plausible
market conditions.

### Policy failure thresholds

Failure thresholds are defined in `simulation_metric_config.json`:

- `max_drawdown_threshold` — paths with drawdown worse than this are marked
  failing
- `min_sharpe_threshold` — paths with Sharpe below this are marked failing
- `min_cumulative_return_threshold` — paths with return below this are failing

Thresholds apply only to paths with `return_metrics_available: true`. Monte
Carlo paths with no return series are never marked as failing on return
criteria; they may be marked failing on regime-sequence criteria if configured.

### Stress score interpretation

`stress_score` is a composite of normalized failure severity across failure
criteria. A score near 0 means no failure threshold was breached and the path
showed favorable characteristics. A score near 1 means multiple criteria were
badly breached. The formula is configurable and documented in
`simulation_metric_config.json`.

### Deterministic leaderboard ranking

`simulation_leaderboard.csv` ranks policies by their worst-case tail behavior
across all scenarios. Ranking is deterministic given the same paths and metric
config.

| Column | Description |
|---|---|
| `policy_name` | policy identifier |
| `rank` | leaderboard rank (1 = most resilient) |
| `mean_stress_score` | mean stress score across all scenarios |
| `pct_scenarios_failing` | fraction of scenarios where ≥1 path failed |
| `tail_p05_return` | worst 5th-percentile return across scenarios |
| `tail_p05_drawdown` | worst 5th-percentile drawdown across scenarios |
| `market_simulation_available` | `true` when M27 artifacts were present |

### `market_simulation_available` and metric availability flags

The `market_simulation_available` column is `true` when M27 simulation
artifacts were loaded and `false` when the M26 integration bridge was
configured but M27 artifacts were unavailable. An empty but schema-valid
leaderboard is written when `market_simulation_available` is `false`, so
downstream consumers always see a consistent schema regardless of artifact
availability.

### Empty / schema-only behavior

When M27 artifacts are unavailable and the bridge is in optional mode, the
metrics layer writes:

- `simulation_leaderboard.csv` with correct column headers and zero data rows
- `policy_failure_summary.json` with `market_simulation_available: false`
- `simulation_summary.csv` with correct headers and zero data rows

This ensures report consumers receive a stable schema contract even when M27
evidence is absent.

### Key artifact names

```text
simulation_path_metrics.csv
simulation_summary.csv
simulation_leaderboard.csv
policy_failure_summary.json
simulation_metric_config.json
manifest.json
```

---

## 7. M27 Market Simulation Case Study

### How to run

```bash
python docs/examples/m27_market_simulation_case_study.py
```

The script:

1. runs the scenario framework with the checked-in M27 config
2. runs episode replay, shock overlay, block bootstrap, and Monte Carlo
   scenarios as configured
3. computes simulation metrics from all scenario outputs
4. stitches a case-study report from simulation metrics and scenario summaries

### Config

The checked-in scenario config is at:

```text
configs/regime_stress_tests/m27_market_simulation_case_study.yml
```

The simulation metrics config is at:

```text
configs/regime_stress_tests/m27_simulation_metrics.yml
```

### Expected output root

```text
docs/examples/output/m27_market_simulation_case_study/
```

### Source simulation artifacts vs. stitched report artifacts

Source artifacts are written under a timestamped run directory:

```text
artifacts/market_simulation/<run_id>/
```

Stitched report artifacts are written under the case-study output root:

```text
docs/examples/output/m27_market_simulation_case_study/
  simulation_summary.json
  simulation_leaderboard.csv
  policy_failure_summary.json
  scenario_summaries/
  manifest.json
  report.md
```

### Generated report structure

`report.md` under the output root includes:

- scenario coverage summary
- leaderboard table
- per-scenario stress highlights
- tail-risk distribution summaries
- interpretation guardrails

### Interpretation guardrails

- simulation paths are fixture-backed synthetic paths, not live market data
- leaderboard rankings reflect behavior under the configured stress scenarios,
  not unconditional performance rankings
- Monte Carlo rows in the leaderboard have no return metrics; drawdown and
  Sharpe columns are null
- case-study outputs should not be used as trading signals

### How fixture-backed outputs should be used

The case-study script uses deterministic fixture data to ensure reproducibility
across environments without requiring full local real-data coverage. Use the
output artifacts for:

- research governance review
- stress-scenario comparison across policy variants
- calibrating failure thresholds before applying to real data
- documenting M27 evidence as part of M26 governance packages

---

## 8. M26 Integration Bridge

### Purpose

The M26 integration bridge allows the M26 full-year regime-policy benchmark
case study to optionally include M27 market simulation evidence without
requiring M27 to run first. When M27 artifacts are available they are stitched
into the M26 case-study report; when they are absent the bridge silently skips
them.

### Config block

In the M26 case-study config (or via CLI override), add:

```yaml
market_simulation_stress:
  enabled: true
  mode: existing_artifacts        # or run_config
  simulation_metrics_dir: artifacts/market_simulation/<run_id>/simulation_metrics
  config_path: configs/regime_stress_tests/m27_market_simulation_case_study.yml
  include_in_policy_stress_summary: true
  include_in_case_study_report: true
```

### `mode: existing_artifacts`

The bridge reads pre-computed M27 simulation metrics from `simulation_metrics_dir`.
This is the recommended mode for combining M27 evidence with M26 workflows:
run the M27 case study first, then point the M26 bridge at the output
`simulation_metrics/` directory.

### `mode: run_config`

The bridge reads the scenario config at `config_path` and runs the M27
simulation pipeline inline before stitching. This is slower but enables a
single-command M26+M27 run for CI or scheduled workflows.

### `include_in_policy_stress_summary`

When `true`, simulation leaderboard data is appended to the M26
`policy_stress_summary.json` under a `market_simulation_stress` key. This
allows reviewers to compare deterministic stress results (M26 Issue 5) and
simulation stress results (M27) in one artifact.

### `include_in_case_study_report`

When `true`, a simulation-stress section is added to the M26 case-study
`final_interpretation.md`. This section surfaces the simulation leaderboard,
policy failure rates, and tail-risk percentiles from M27 evidence.

### Default no-op behavior

If the `market_simulation_stress` block is absent or `enabled: false`, the M26
case study runs exactly as before. No M27 artifacts are loaded and no
simulation sections are added to the report.

### Unavailable evidence summary

If `enabled: true` but the configured `simulation_metrics_dir` does not exist
or does not contain the expected artifacts, the bridge writes an unavailable
evidence summary:

- `market_simulation_stress_summary.json` with `market_simulation_available: false`
- an empty `simulation_leaderboard.csv` with correct schema headers
- a note in `final_interpretation.md` explaining that M27 evidence was
  configured but unavailable

### Schema-only empty leaderboard

The empty leaderboard ensures downstream code and reports that consume
`simulation_leaderboard.csv` receive a consistent schema regardless of whether
M27 evidence is present.

### `--require-market-simulation-stress`

Passing `--require-market-simulation-stress` to the M26 case-study CLI
changes optional/unavailable behavior to a hard failure:

```powershell
python docs/examples/full_year_regime_policy_benchmark_case_study.py `
  --require-market-simulation-stress
```

When this flag is set and M27 artifacts are missing or invalid, the script
exits with a non-zero status and reports the missing artifacts instead of
writing an empty leaderboard. Use this flag in CI pipelines where M27 evidence
is required for governance sign-off.

### Relationship to M26 deterministic stress tests

M27 market simulation evidence complements M26 Issue 5 deterministic policy
stress tests. They are different layers:

| Layer | Method | Artifact |
|---|---|---|
| M26 Issue 5 | deterministic synthetic stress transforms | `stress_leaderboard.csv` |
| M27 market simulation | synthetic path generation (replay/overlay/bootstrap/Monte Carlo) | `simulation_leaderboard.csv` |

Neither layer replaces the other. Governance review should consider both
deterministic and simulation-based stress evidence together.

---

## 9. Troubleshooting

### Missing M27 metrics artifacts

**Symptom:** M26 bridge reports `market_simulation_available: false` or
simulation leaderboard is empty.

**Fix:** Run the M27 case study first:

```bash
python docs/examples/m27_market_simulation_case_study.py
```

Then verify that
`artifacts/market_simulation/<run_id>/simulation_metrics/simulation_leaderboard.csv`
exists and configure the M26 bridge `simulation_metrics_dir` to point to that
directory.

### When to run the M27 case study first

For `mode: existing_artifacts`, always run `m27_market_simulation_case_study.py`
before running the M26 full-year case study with the bridge enabled. For
`mode: run_config`, M27 runs inline and the case study handles ordering
automatically.

### Strict mode failure behavior

When `--require-market-simulation-stress` is set and M27 artifacts are missing,
the M26 case study exits with a non-zero status. The error output names the
missing paths. Fix by running the M27 case study and verifying the output root.

### Path / relative path expectations

All config paths and CLI paths are interpreted relative to the current working
directory. Run commands from the repository root:

```bash
cd /path/to/stratlake-trade-engine
python docs/examples/m27_market_simulation_case_study.py
```

Absolute paths in configs, docs, or scripts will be rejected by the docs-path
lint check.

### Empty leaderboard behavior

An empty `simulation_leaderboard.csv` (zero data rows) is the expected output
when M27 evidence is configured but unavailable. It is not an error in optional
mode. In strict mode it causes a failure. Inspect
`policy_failure_summary.json` for `market_simulation_available: false` to
confirm the cause.

### Monte Carlo has no return metrics

Monte Carlo paths contain only regime-label sequences. Return-metric columns
(`mean_return`, `cumulative_return`, `max_drawdown`, `sharpe_ratio`) are `null`
or absent for Monte Carlo path rows in `simulation_path_metrics.csv`. This is
expected and correct. Use bootstrap or replay paths when return-metric
comparisons are needed.

### Generated artifacts ignored or too deeply nested

The metrics layer expects all per-scenario artifacts under:

```text
artifacts/market_simulation/<run_id>/<scenario_id>/
```

If scenario output directories are nested more deeply (e.g., due to a
non-standard output root), the metrics aggregation step will not find them.
Verify that the `output_root` in the scenario config resolves to a directory
one level above the per-scenario folders.

### Line-ending and deterministic artifact reminders

The repository uses UTF-8 with LF line endings. If replay or overlay path CSVs
are written with CRLF on Windows, metric computation may produce mismatched
row counts. Verify `.gitattributes` line-ending rules are applied before
committing fixture-backed outputs. Deterministic artifacts committed to the
repository should match character-for-character across platforms.

---

## Limitations and Non-Goals

- simulation models do not produce live market data or real-time signals
- simulation outputs must not be used as trading signals or return forecasts
- Monte Carlo regime paths do not imply any particular return distribution
- bootstrap block length and path length are not fitted to any market
  microstructure model; treat synthetic paths as scenario probes, not
  calibrated predictions
- M27 simulation evidence supplements governance review; it does not replace
  observed backtest evidence, promotion gates, or review-pack decisions
- new simulation models, artifact schema changes, and new live-data dependencies
  are out of scope for this documentation release
