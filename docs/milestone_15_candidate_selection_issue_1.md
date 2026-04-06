# Milestone 15 — Candidate Selection & Portfolio Allocation Governance
## Issue 1: Core Deterministic Candidate Selection Engine

### Implementation Summary

This issue establishes the foundational candidate selection pipeline for Milestone 15. The implementation provides a complete, deterministic workflow for loading candidate alpha sleeves from the registry, ranking them using explicit evaluation metrics, selecting subsets, validating data quality, and persisting reproducible artifacts.

**Status**: ✅ Complete and tested

---

## Architecture Overview

### Module Structure

```
src/research/candidate_selection/
├── __init__.py           # Public API and end-to-end pipeline
├── schema.py             # CandidateRecord canonical schema
├── loader.py             # Registry loading with filters
├── ranker.py             # Deterministic ranking logic
├── persistence.py        # Artifact writing and manifest generation
└── validation.py         # Data quality checks
```

### Key Components

1. **CandidateRecord** (schema.py)
   - Immutable frozen dataclass representing a single candidate
   - Captures: provenance (run_id, artifact_path), evaluation context (dataset, timeframe, horizon), forecast metrics (ic_ir, mean_ic, etc.), sleeve metrics (sharpe, turnover), promotion/review status, and selection rank
   - Includes CSV serialization support

2. **load_candidate_universe()** (loader.py)
   - Loads candidates from the alpha evaluation registry (`artifacts/alpha/registry.jsonl`)
   - Supports optional filters: alpha_name, dataset, timeframe, evaluation_horizon, mapping_name
   - Normalizes registry entries into CandidateRecord objects
   - Validates required fields and handles missing metrics gracefully

3. **rank_candidates()** (ranker.py)
   - Deterministically ranks unordered candidates using primary metric + tiebreakers
   - Default metric: `ic_ir` (Information Ratio of IC)
   - Alternative metrics: `mean_ic`, `mean_rank_ic`, `rank_ic_ir`
   - Tiebreaker chain: primary_metric → secondary metrics → mapping_name → alpha_name → alpha_run_id
   - Handles None values consistently (sort last)
   - Returns candidates with 1-indexed selection_rank assigned

4. **select_top_candidates()** (ranker.py)
   - Selects top N candidates from ranked list
   - Preserves selection_rank from ranking
   - Supports None max_count (select all)

5. **write_candidate_selection_artifacts()** (persistence.py)
   - Persists artifacts to `artifacts/candidate_selection/<run_id>/`:
     - `candidate_universe.csv`: All ranked candidates
     - `selected_candidates.csv`: Selected subset
     - `selection_summary.json`: Metadata and counts
     - `manifest.json`: Run manifest with config and provenance
   - Deterministic run_id derived from filters + candidate_ids + primary_metric
   - All outputs are ordered and reproducible

6. **Validation** (validation.py)
   - `validate_candidate_universe()`: Checks for duplicate IDs, required fields, finite metrics
   - `validate_ranked_universe()`: Ensures sequential 1-indexed ranks
   - Raises `CandidateValidationError` with clear messages

---

## Ranking Logic

### Multi-Level Sort Key

The ranking uses a deterministic tuple sort:

```
(metric1_is_null, metric1_value, metric2_is_null, metric2_value, ..., 
 mapping_name_is_null, mapping_name, alpha_name, alpha_run_id)
```

### Metric Direction Handling

- **Higher-is-better (DESC)**: ic_ir, mean_ic, mean_rank_ic, rank_ic_ir, sharpe_ratio, annualized_return, total_return
  - Negated in sort key for descending order
- **Lower-is-better (ASC)**: max_drawdown, average_turnover
  - Used as-is in sort key for ascending order
- **None values**: Always sort last, regardless of direction

### Example: Three Candidates

```
Candidate A: ic_ir=1.5, mapping_name="rank_long_short", alpha_name="Alpha1"
Candidate B: ic_ir=1.5, mapping_name=None, alpha_name="Alpha2"            
Candidate C: ic_ir=1.2, mapping_name="rank_long_short", alpha_name="Alpha3"

Ranking by ic_ir (descending):
1. A: (False, -1.5, False, "rank_long_short", "Alpha1", ...)
2. B: (False, -1.5, True, "", "Alpha2", ...)                  ← mapping_name None sorts later
3. C: (False, -1.2, False, "rank_long_short", "Alpha3", ...)
```

### Determinism Guarantees

- Identical inputs (same candidates, filters, metric) always produce identical ranks
- Run IDs are deterministic (SHA256 hash of sorted candidate_ids + filters + metric)
- Repeated runs preserve artifact order and CSV row order
- No randomness in any stage

---

## CLI Usage

### Basic Command

```bash
python -m src.cli.run_candidate_selection
```

### With Filters

```bash
python -m src.cli.run_candidate_selection \
  --dataset features_daily \
  --timeframe daily \
  --metric ic_ir \
  --max-candidates 10
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--artifacts-root` | path | artifacts/alpha | Alpha evaluation artifacts root |
| `--alpha-name` | str | None | Filter by alpha name |  
| `--dataset` | str | None | Filter by dataset |
| `--timeframe` | str | None | Filter by timeframe |
| `--evaluation-horizon` | int | None | Filter by horizon (bars) |
| `--mapping-name` | str | None | Filter by signal mapping name |
| `--metric` | choice | ic_ir | Ranking metric (ic_ir, mean_ic, mean_rank_ic, rank_ic_ir) |
| `--max-candidates` | int | None | Select top N candidates |
| `--output-path` | path | artifacts/candidate_selection | Artifact output root |

### Output

```
Candidate Selection Run: candidate_selection_bd83df4d0cac
  Universe count: 3
  Selected count: 3
  Primary metric: ic_ir

  Universe CSV: artifacts\candidate_selection\candidate_selection_bd83df4d0cac\candidate_universe.csv
  Selected CSV: artifacts\candidate_selection\candidate_selection_bd83df4d0cac\selected_candidates.csv
  Summary JSON: artifacts\candidate_selection\candidate_selection_bd83df4d0cac\selection_summary.json
```

---

## Artifact Structure

### Directory Layout

```
artifacts/candidate_selection/
└── candidate_selection_<run_id_hash>/
    ├── candidate_universe.csv          # All ranked candidates (ranked by ic_ir)
    ├── selected_candidates.csv         # Selected subset (if max-candidates applied)
    ├── selection_summary.json          # Metadata, counts, filters used
    └── manifest.json                   # Full run manifest (deterministic)
```

### candidate_universe.csv

Columns (in order):
```
selection_rank, candidate_id, alpha_name, alpha_run_id, sleeve_run_id, 
mapping_name, dataset, timeframe, evaluation_horizon, 
mean_ic, ic_ir, mean_rank_ic, rank_ic_ir, n_periods,
sharpe_ratio, annualized_return, total_return, max_drawdown, average_turnover,
promotion_status, review_status, artifact_path
```

### selection_summary.json

```json
{
  "run_id": "candidate_selection_bd83df4d0cac",
  "universe_count": 3,
  "selected_count": 3,
  "primary_metric": "ic_ir",
  "filters": {
    "alpha_name": null,
    "dataset": null,
    "timeframe": null,
    "evaluation_horizon": null,
    "mapping_name": null
  }
}
```

### manifest.json

```json
{
  "run_id": "candidate_selection_bd83df4d0cac",
  "artifact_dir": "artifacts/candidate_selection/candidate_selection_bd83df4d0cac",
  "created": "candidate_selection",
  "candidate_universe_csv": "...",
  "selected_candidates_csv": "...",
  "selection_summary_json": "...",
  "universe_count": 3,
  "selected_count": 3,
  "primary_metric": "ic_ir",
  "filters": {...}
}
```

---

## Validation & Testing

### Test Coverage (26 tests, 100% pass)

#### Unit Tests (19)
- **Schema**: Creation, immutability, dict conversion (3)
- **Ranking**: Basic ranking, alternative metrics, deterministic tiebreakers, empty lists, invalid metrics (5)
- **Selection**: Top-N selection, all candidates, oversized max_count (3)
- **Validation**: Valid universe, duplicate detection, rank validation, empty lists (4)
- **Persistence**: Deterministic run IDs, different inputs differ (2)
- **Loaders**: Mapping name extraction, registry entry normalization (2)

#### Integration Tests (2)
- End-to-end rank + select workflow
- Artifact persistence validation

#### Determinism Tests (2)
- Repeated rankings produce identical order
- Run IDs are deterministic

### Test Execution

```bash
pytest tests/test_candidate_selection.py -v
# 26 passed in 0.88s
```

### End-to-End Verification

Successfully executed with actual alpha registry data:
- Loaded 3 real candidates from registry
- Ranked deterministically by ic_ir
- Selected top 2 with `--max-candidates 2`
- Generated valid CSV and JSON artifacts
- All manifest and summary fields correct

---

## Design Patterns & Conventions

### Reused from Existing Codebase
1. **Frozen dataclasses** for determinism: `@dataclass(frozen=True)`
2. **Registry pattern**: Load JSONL-based registry (like alpha evaluations)
3. **Artifact persistence**: Deterministic run_id from hash, structured JSON manifests
4. **Multi-level sorting**: Same pattern as alpha comparison leaderboard
5. **CLI conventions**: argparse, structured result dataclass, summary printer
6. **Error handling**: Domain-specific exception classes (CandidateSelectionError, etc.)

### Schema Design
- **Minimal required fields**: Only what's available in existing artifacts
- **All metrics optional**: Handle missing forecasts gracefully
- **Provenance tracking**: Full alpha_run_id and artifact_path for traceability
- **Status fields**: promotion_status and review_status from registry

### Deterministic Ordering
- **Explicit sort keys**: No implicit Python ordering
- **Null handling**: Consistent (False, 0.0) for comparability
- **Metric direction**: Explicit per-metric direction handling
- **Stable tiebreakers**: Alphabetic (names) and identity (IDs) fields

---

## Extension Points for Future Issues

The design explicitly supports Milestone 15 extensions:

### For Issue 2: Eligibility Filtering
- Add `eligibility_filter_stage()` function in new `filters.py`
- Apply before ranking: `filtered_universe = eligibility_filter_stage(universe, rules)`
- No changes needed to ranking/persistence

### For Issue 3: Correlation Pruning
- Add `correlation_pruning_stage()` in new `pruning.py`
- Apply after ranking: `pruned = correlation_pruning_stage(ranked_universe, config)`
- Preserve selection_rank from ranking for traceability

### For Issue 4: Allocation Governance
- Add `allocation_stage()` in new `allocation.py`
- Take selected_candidates + allocation_rules, produce allocation weights
- Can reference selection_rank for priority weighting

### For Issue 5: Review & Explainability
- Add `generate_review_report()` in new `reporting.py`
- Reference manifest, selection_rank, and artifact_paths for provenance
- Can cross-reference with promotion gate results

### Clean Extension Architecture
- Each stage operates on: `list[CandidateRecord]` (read-only)
- Each stage returns: modified `list[CandidateRecord]` or new output structure
- All stages log to structured summary JSONs in artifact directory
- Registry can record candidate selection runs as new artifact type

---

## Known Limitations (By Design)

1. **No correlation matrix**: Correlation filtering is Issue 3 scope
2. **No advanced allocation**: Allocation rules are Issue 4 scope
3. **No review scoring**: Candidate-specific review is Issue 5 scope
4. **No cross-sleeve constraints**: Multi-sleeve optimization deferred
5. **Single ranking metric**: Per-run metric choice (not multi-objective in this issue)

These are intentional deferred design decisions documented for future extension.

---

## Files Changed/Added

### New Files (6 modules + 1 CLI + 1 test)
- `src/research/candidate_selection/__init__.py` — Module public API
- `src/research/candidate_selection/schema.py` — CandidateRecord schema
- `src/research/candidate_selection/loader.py` — Registry loading + filtering
- `src/research/candidate_selection/ranker.py` — Ranking logic
- `src/research/candidate_selection/persistence.py` — Artifact writing
- `src/research/candidate_selection/validation.py` — Data quality checks
- `src/cli/run_candidate_selection.py` — CLI entry point
- `tests/test_candidate_selection.py` — 26 comprehensive tests
- `artifacts/candidate_selection/` — Artifact output directory (created at runtime)

### Modified Files
None. Implementation is pure addition with no refactoring of existing code.

---

## Backward Compatibility

✅ **No regressions** — Implementation is feature-additive only:
- No changes to alpha evaluation, comparison, portfolio, or review modules
- All imports are new (no existing imports modified)
- No shared utilities refactored
- Can be deployed independently

---

## Next Steps (Future Issues)

1. **Issue 2**: Implement eligibility filtering (quality gates, minimum metrics)
2. **Issue 3**: Implement correlation/redundancy pruning (principal component filtering)
3. **Issue 4**: Implement allocation governance (weight assignment, constraints)
4. **Issue 5**: Implement review & contribution explainability (feature importance, backtesting)

This foundation is designed to be extended cleanly without rewrites.

---

## Usage Examples

### Python API

```python
from src.research.candidate_selection import (
    load_candidate_universe,
    rank_candidates,
    select_top_candidates,
    write_candidate_selection_artifacts,
)

# Load candidates with filters
universe = load_candidate_universe(
    dataset="features_daily",
    evaluation_horizon=5,
)

# Rank deterministically
ranked = rank_candidates(universe, primary_metric="ic_ir")

# Select top 10
selected = select_top_candidates(ranked, max_count=10)

# Persist artifacts
run_id, universe_csv, selected_csv, summary_json, manifest = write_candidate_selection_artifacts(
    universe=ranked,
    selected=selected,
    filters={"dataset": "features_daily"},
    primary_metric="ic_ir",
)
```

### CLI Usage

```bash
# Load all candidates from registry
python -m src.cli.run_candidate_selection

# Filter by dataset and select top 5
python -m src.cli.run_candidate_selection \
  --dataset features_daily \
  --max-candidates 5

# Filter by alpha and metric
python -m src.cli.run_candidate_selection \
  --alpha-name my_alpha \
  --metric mean_ic
```

---

## Summary

The candidate selection engine is a complete, tested, deterministic implementation that:

✅ Loads candidates from existing alpha registry with optional filtering  
✅ Ranks deterministically using explicit multi-level sort keys  
✅ Selects subsets with configurable top-N logic  
✅ Validates data quality and detects inconsistencies  
✅ Persists reproducible artifacts with full provenance  
✅ Provides CLI with flexible filtering and output options  
✅ Includes 26 comprehensive unit, integration, and determinism tests  
✅ Follows existing StratLake patterns (dataclasses, registries, artifacts)  
✅ Designed for clean extension by future Milestone 15 issues  

**Ready for deployment and integration with downstream workflows.**
