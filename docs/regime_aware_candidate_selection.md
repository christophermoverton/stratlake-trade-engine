# Regime-Aware Candidate Selection

Regime-aware candidate selection is an opt-in workflow that connects governed regime review packs back into candidate selection. It does not replace the existing global candidate-selection pipeline; it adds a separate selection surface for candidates that are useful because they are strong globally, specialized in a regime, resilient near regime transitions, or defensive during unstable/high-volatility regimes.

## Inputs

The workflow consumes:

- a source regime review-pack directory, used for review and benchmark provenance
- a candidate metrics CSV or JSON file
- optional alpha, strategy, or portfolio registry paths reserved for later registry-backed expansion

The first implementation is file-backed. The candidate metrics file must include `candidate_id`, `candidate_name`, and `artifact_type`. Other metrics are optional and are recorded as warnings when absent.

## Config

Example:

```yaml
selection_name: m26_regime_aware_candidate_selection
source_review_pack: configs/candidate_selection/fixtures/regime_review_pack
source_candidate_universe:
  candidate_metrics_path: configs/candidate_selection/fixtures/regime_candidate_metrics.csv
regime_context:
  regime_source: calibrated_taxonomy
  allow_gmm_confidence_overlay: true
  min_regime_confidence: 0.55
  transition_window_bars: 5
selection_categories:
  global_performer:
    enabled: true
    max_candidates: 5
    min_global_score: 0.60
  regime_specialist:
    enabled: true
    max_candidates_per_regime: 3
    min_regime_score: 0.65
    min_regime_observations: 20
redundancy:
  enabled: true
  max_pairwise_correlation: 0.85
allocation_hints:
  write_category_weight_hints: true
output_root: artifacts/candidate_selection
```

## Categories

`global_performer` selects candidates with `global_score >= min_global_score`, ranked by global score.

`regime_specialist` selects candidates with `regime_score >= min_regime_score`, subject to `min_regime_observations` when an observation count is supplied. Limits are applied per regime.

`transition_resilient` selects candidates with `transition_score >= min_transition_score` and transition-window drawdown no worse than `max_transition_drawdown` when supplied.

`defensive_fallback` selects candidates with strong defensive score or high-volatility drawdown containment, and respects `max_correlation_to_selected` when `correlation_to_selected` is supplied.

## Scoring

Explicit score columns are used when present. Missing scores are computed deterministically from raw metrics:

- global: Sharpe, total return, max drawdown
- regime: regime Sharpe, return, IC/rank IC, max drawdown
- transition: transition-window return and drawdown
- defensive: high-volatility drawdown, volatility, correlation to selected

Fallback scores are clipped to `0.0` through `1.0`.

## Redundancy

When `redundancy_group` is supplied, the workflow keeps the highest-scoring candidate per group within each category. When `correlation_to_selected` is supplied, candidates above `max_pairwise_correlation` can be pruned. If no redundancy data is available, selection continues and the limitation is written to the summary and manifest.

## Artifacts

Artifacts are written under:

```text
artifacts/candidate_selection/<selection_run_id>/
```

Expected outputs:

- `candidate_selection.csv` and `.json`
- `regime_candidate_scores.csv` and `.json`
- `category_assignments.csv`
- `transition_resilience.csv`
- `defensive_fallback_candidates.csv`
- `redundancy_report.csv`
- `allocation_hints.json`
- `selection_summary.json`
- `config.json`
- `manifest.json`
- `source_regime_review_pack.json`
- `candidate_input_inventory.json`

Allocation hints are advisory only. The workflow does not construct a final portfolio.

## CLI

```powershell
.\.venv\Scripts\python.exe -m src.cli.run_regime_aware_candidate_selection `
  --config configs\candidate_selection\m26_regime_aware_candidate_selection.yml
```

Useful overrides:

```powershell
--source-review-pack artifacts\regime_reviews\<review_run_id>
--candidate-metrics-path path\to\candidate_metrics.csv
--output-root artifacts\candidate_selection
```

## Non-Goals

This workflow does not rerun benchmark packs, promotion gates, or review packs. It does not mutate upstream governance artifacts, alter regime taxonomy or classifier behavior, run stress tests, or build a final portfolio.
