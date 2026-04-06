# StratLake Trade Engine - Candidate Selection Implementation Patterns

## Executive Summary

The StratLake codebase demonstrates well-established patterns for ranking, filtering, and selecting artifacts across alpha evaluations and portfolio workflows. This document captures the key architectural patterns that should inform candidate selection implementation.

---

## 1. ARTIFACT DIRECTORY STRUCTURE AND NAMING CONVENTIONS

### Alpha Evaluation Artifacts

**Directory Structure:**
```
artifacts/
  alpha/
    registry.jsonl                                    # Master registry of all alpha runs
    {alpha_name}_alpha_eval_{run_id_hash}/          # Individual run directory
      manifest.json                                   # Complete metadata
      alpha_metrics.json                             # Forecast quality metrics
      sleeve_returns.csv                             # Backtested sleeve returns
      sleeve_equity_curve.csv                        # Compounded equity curve
      sleeve_metrics.json                            # Implementation/tradability metrics
      predictions.parquet                            # Model predictions by symbol/ts
      signals.parquet                                # Mapped trading signals
      signal_mapping.json                            # Mapping config and metadata
      qa_summary.json                                # Data quality and signal diagnostics
      training_summary.json                          # Training metadata
      coefficients.json                              # Feature importance/model weights
      cross_section_diagnostics.json                 # Cross-sectional analysis
      ic_timeseries.csv                              # IC per timestamp
      promotion_gates.json                           # Promotion gate evaluation results
```

### Portfolio Artifacts

**Directory Structure:**
```
artifacts/
  portfolios/
    registry.jsonl                                   # Master registry of all portfolio runs
    {portfolio_name}_{run_id_hash}/                 # Individual portfolio run
      config.json                                    # Portfolio construction config
      components.json                                # Component definitions
      weights.csv                                    # Time-series weights
      portfolio_returns.csv                          # Portfolio return stream
      portfolio_equity_curve.csv                     # Portfolio equity curve
      metrics.json                                   # Portfolio-level metrics
      qa_summary.json                                # Portfolio QA summary
      manifest.json                                  # Complete metadata
      promotion_gates.json                           # Promotion gate results
```

### Naming Conventions

**Run IDs:** Deterministic hash based on configuration
- Alpha: `rank_composite_momentum_alpha_eval_33ff16458d15`
- Portfolio: `momentum_meanrev_equal_abc123def456`
- Hash: 12-char SHA256 of canonical config JSON

**Manifest Keys:** Consistent across both alpha and portfolio layers
```json
{
  "alpha_name": "rank_composite_momentum",
  "run_id": "rank_composite_momentum_alpha_eval_33ff16458d15",
  "artifact_paths": {
    "coefficients": "coefficients.json",
    "metrics": "alpha_metrics.json",
    "qa_summary": "qa_summary.json"
  },
  "artifact_groups": {
    "alpha_evaluation": [...],
    "sleeve": [...]
  }
}
```

---

## 2. KEY METRICS AVAILABLE IN ALPHA EVALUATION OUTPUTS

### Forecast Quality Metrics (Alpha Signals)

These metrics measure how well the alpha signals predict forward returns:

| Metric | Source | Direction | Description |
|--------|--------|-----------|-------------|
| `mean_ic` | alpha_metrics.json | DESC ↓ | Average information coefficient |
| `ic_ir` | alpha_metrics.json | DESC ↓ | IC information ratio (**DEFAULT ranking**) |
| `mean_rank_ic` | alpha_metrics.json | DESC ↓ | Mean rank IC (percentile form) |
| `rank_ic_ir` | alpha_metrics.json | DESC ↓ | Rank IC information ratio |
| `n_periods` | qa_summary.json | DESC ↓ | Number of valid evaluation periods |
| `ic_positive_rate` | manifest → metric_summary | DESC ↓ | Rate of positive IC periods |

**Location:** `manifest.json → metric_summary` or `alpha_metrics.json`

### Sleeve/Implementation Metrics (Tradability)

These measure how the alpha performs as a backtested trading sleeve:

| Metric | Direction | Description |
|--------|-----------|-------------|
| `sharpe_ratio` | DESC ↓ | Risk-adjusted return (**DEFAULT for sleeve ranking**) |
| `annualized_return` | DESC ↓ | Annualized return over backtest period |
| `total_return` | DESC ↓ | Cumulative return over period |
| `cumulative_return` | DESC ↓ | Same as total_return |
| `max_drawdown` | ASC ↑ | Maximum loss from peak (lower is better) |
| `turnover` / `average_turnover` | ASC ↑ | Portfolio turnover (lower is better) |
| `total_turnover` | ASC ↑ | Cumulative turnover |
| `win_rate` | DESC ↓ | % of positive periods |
| `hit_rate` | DESC ↓ | Similar to win_rate |
| `profit_factor` | DESC ↓ | Ratio of gross profit to gross loss |
| `trade_count` | DESC ↓ | Number of trades executed |
| `exposure_pct` | DESC ↓ | % of time portfolio is invested |
| `total_transaction_cost` | ASC ↑ | Cumulative transaction friction |
| `total_slippage_cost` | ASC ↑ | Cumulative slippage friction |
| `total_execution_friction` | ASC ↑ | Total execution costs |

**Location:** `sleeve_metrics.json` or `manifest.json → sleeve → metric_summary`

### Quality Assurance Metrics

**Location:** `qa_summary.json`

```json
{
  "forecast": {
    "valid_timestamps": 19,
    "ic_positive_rate": 1.0
  },
  "cross_section": {
    "mean_valid_cross_section_size": 145.8,
    "min_cross_section_size": 125
  },
  "nulls": {
    "prediction_null_rate": 0.001,
    "forward_return_null_rate": 0.002
  },
  "signals": {
    "mean_turnover": 0.234,
    "max_single_name_abs_share": 0.15,
    "max_abs_net_exposure_share": 0.18
  }
}
```

---

## 3. MANIFEST AND REGISTRY PATTERNS

### Manifest Structure (Complete Configuration + Metadata)

**Key sections in manifest.json:**

```json
{
  "alpha_name": "...",
  "artifact_files": [list of all persisted files],
  "artifact_groups": {
    "alpha_evaluation": [...],
    "sleeve": [...]
  },
  "artifact_paths": {
    "metrics": "alpha_metrics.json",
    "predictions": "predictions.parquet",
    ...
  },
  "model": {
    "feature_columns": [...],
    "hyperparameters": {...}
  },
  "signal_mapping": {
    "policy": "top_bottom_quantile",
    "quantile": 0.2,
    "metadata": {
      "name": "top_bottom_quantile_q20",
      "case_study": "q1_2026_real_features_daily"
    }
  },
  "metric_summary": {
    "mean_ic": 0.254,
    "ic_ir": 1.920,
    ...
  },
  "sleeve": {
    "metric_summary": {
      "sharpe_ratio": 1.45,
      "total_return": 0.087,
      ...
    }
  }
}
```

### Registry Entry Structure (JSONL Format)

**File:** `artifacts/alpha/registry.jsonl` (one JSON object per line)

```json
{
  "run_id": "rank_composite_momentum_2026_q1_alpha_eval_33ff16458d15",
  "run_type": "alpha_evaluation",
  "timestamp": "2026-03-26T04:00:00Z",
  "alpha_name": "rank_composite_momentum_2026_q1",
  "dataset": "features_daily",
  "timeframe": "1D",
  "evaluation_horizon": 5,
  "metrics_summary": {
    "mean_ic": 0.254,
    "ic_ir": 1.920,
    "mean_rank_ic": 0.225,
    "rank_ic_ir": 3.083,
    "n_periods": 19
  },
  "artifact_path": "artifacts/alpha/rank_composite_momentum_2026_q1_alpha_eval_33ff16458d15",
  "artifact_paths": {...},
  "promotion_status": "eligible",
  "review_status": "candidate",
  "review_metadata": {
    "schema_version": 1,
    "status": "candidate",
    "promotion_status": "eligible",
    "decision_reason": "...",
    "decision_source": "..."
  },
  "promotion_gate_summary": {
    "configured": true,
    "evaluation_status": "all_passed",
    "promotion_status": "eligible",
    "gate_count": 8,
    "passed_gate_count": 8,
    "failed_gate_count": 0,
    "missing_gate_count": 0
  },
  "config": {...},
  "manifest": {...}
}
```

### Review and Promotion Status Values

**Review Status (mutually exclusive):**
- `candidate` - New run, under consideration
- `needs_review` - Flagged for human review
- `promoted` - Approved and in production
- `rejected` - Explicitly rejected

**Promotion Status:**
- `eligible` - Passed all promotion gates
- `blocked` - Failed one or more promotion gates
- Gate-specific values from promotion gate configuration

### Registry Loading Pattern

```python
from src.research.alpha_eval.registry import load_alpha_evaluation_registry

# Load all entries from registry.jsonl
entries = load_alpha_evaluation_registry(artifacts_root="artifacts/alpha")
# Returns: list[dict[str, Any]]

# Parse into structured objects
for entry in entries:
    run_id = entry["run_id"]
    alpha_name = entry["alpha_name"]
    promotion_status = entry["promotion_status"]
    metrics = entry["metrics_summary"]
    config = entry["config"]
    artifact_path = entry["artifact_path"]
```

---

## 4. CLI COMMAND STRUCTURE AND CONVENTIONS

### Alpha Evaluation CLI Pattern

**File:** `src/cli/run_alpha_evaluation.py`

```python
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="YAML config path")
    parser.add_argument("--alpha-name", help="Built-in alpha name")
    parser.add_argument("--alpha-model", help="Registered model name")
    parser.add_argument("--dataset", help="Feature dataset")
    parser.add_argument("--promotion-gates", help="Gate config override")
    parser.add_argument("--signal-policy", choices=["rank_long_short", "zscore_continuous", ...])
    parser.add_argument("--artifacts-root", help="Output directory")
    return parser.parse_args(argv)

def run_cli(argv: Sequence[str] | None = None) -> AlphaEvaluationRunResult:
    args = parse_args(argv)
    # Execute workflow and return structured result
    return AlphaEvaluationRunResult(...)
```

### Alpha Comparison CLI Pattern

**File:** `src/cli/compare_alpha.py`

```python
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-registry", action="store_true", required=True)
    parser.add_argument("--view", choices=["forecast", "sleeve", "combined"], 
                       default="forecast")
    parser.add_argument("--metric", default="ic_ir")
    parser.add_argument("--sleeve-metric", default="sharpe_ratio")
    parser.add_argument("--alpha-name", help="Optional filter")
    parser.add_argument("--dataset", help="Optional filter")
    return parser.parse_args(argv)

def run_cli(argv: Sequence[str] | None = None) -> AlphaEvaluationComparisonResult:
    args = parse_args(argv)
    result = compare_alpha_evaluation_runs(
        metric=args.metric,
        view=args.view,
        alpha_name=args.alpha_name,
        dataset=args.dataset,
        # ... more filters
    )
    return result
```

### Portfolio Construction CLI Pattern

**File:** `src/cli/run_portfolio.py`

```python
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--portfolio-config", help="YAML/JSON portfolio config")
    parser.add_argument("--portfolio-name", help="Portfolio name from config")
    parser.add_argument("--run-ids", nargs="+", help="Explicit run IDs or alpha names")
    parser.add_argument("--from-registry", action="store_true")
    parser.add_argument("--timeframe", required=True, choices=["1D", "1Min"])
    parser.add_argument("--optimizer-method", choices=["equal_weight", "minimum_variance", ...])
    parser.add_argument("--output-dir", help="Output directory")
    return parser.parse_args(argv)

def run_cli(argv: Sequence[str] | None = None) -> PortfolioRunResult:
    args = parse_args(argv)
    # Load components by run_id from registry or explicit list
    # Construct portfolio with specified allocator
    return PortfolioRunResult(...)
```

### Config Loading Convention

```python
def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML/JSON, extract nested section if present."""
    resolved_path = Path(path)
    with resolved_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    
    # Looks for section like "alpha_evaluation:" or "portfolios:"
    nested = payload.get("alpha_evaluation")  # or "portfolios", etc.
    return nested if nested is not None else payload
```

---

## 5. KEY DATA STRUCTURE PATTERNS

### Immutable Result Dataclasses

All result types use `@dataclass(frozen=True)` for deterministic outputs:

```python
@dataclass(frozen=True)
class AlphaEvaluationRunResult:
    alpha_name: str
    run_id: str
    artifact_dir: Path
    loaded_frame: pd.DataFrame
    trained_model: TrainedAlphaModel
    prediction_result: AlphaPredictionResult
    signal_mapping_result: AlphaSignalMappingResult | None
    aligned_frame: pd.DataFrame
    evaluation_result: AlphaEvaluationResult
    manifest: dict[str, Any]
    resolved_config: dict[str, Any]
```

### Leaderboard Entry Dataclass

```python
@dataclass(frozen=True)
class AlphaEvaluationLeaderboardEntry:
    rank: int
    alpha_name: str
    run_id: str
    dataset: str
    timeframe: str
    evaluation_horizon: int
    mapping_name: str | None
    # Forecast metrics (all optional float)
    mean_ic: float | None
    ic_ir: float | None
    mean_rank_ic: float | None
    rank_ic_ir: float | None
    n_periods: int | None
    # Sleeve metrics (all optional float)
    sharpe_ratio: float | None
    annualized_return: float | None
    total_return: float | None
    cumulative_return: float | None
    max_drawdown: float | None
    turnover: float | None
    average_turnover: float | None
    total_turnover: float | None
    win_rate: float | None
    hit_rate: float | None
    profit_factor: float | None
    trade_count: float | None
    exposure_pct: float | None
    total_transaction_cost: float | None
    total_slippage_cost: float | None
    total_execution_friction: float | None
    artifact_path: str
```

### Comparison Result

```python
@dataclass(frozen=True)
class AlphaEvaluationComparisonResult:
    comparison_id: str
    view: str
    metric: str
    forecast_metric: str
    sleeve_metric: str
    filters: dict[str, Any]
    leaderboard: list[AlphaEvaluationLeaderboardEntry]
    csv_path: Path
    json_path: Path
```

### Portfolio Component Reference

```python
# In portfolio config.json or run config:
{
  "portfolio_name": "my_portfolio",
  "allocator": "equal_weight",
  "components": [
    {
      "strategy_name": "rank_composite_momentum",
      "run_id": "rank_composite_momentum_alpha_eval_33ff16458d15",
      "artifact_type": "alpha_sleeve"
    }
  ]
}
```

### Portfolio Result

```python
@dataclass(frozen=True)
class PortfolioRunResult:
    portfolio_name: str
    run_id: str
    allocator_name: str
    timeframe: str
    component_count: int
    metrics: dict[str, float | None]
    experiment_dir: Path
    portfolio_output: pd.DataFrame
    config: dict[str, Any]
    components: list[dict[str, Any]]
    simulation_result: SimulationRunResult | None = None
```

---

## 6. EXISTING RANKING AND SORTING LOGIC

### Ranking Algorithm (from `alpha_eval/compare.py`)

```python
def _rank_rows(
    rows: list[AlphaEvaluationLeaderboardEntry],
    *,
    view: str,
    forecast_metric: str,
    sleeve_metric: str,
) -> list[AlphaEvaluationLeaderboardEntry]:
    """Sort runs by metrics and assign 1-based rank."""
    
    ranked = sorted(rows, key=lambda row: _sort_key(row, ...))
    return [
        AlphaEvaluationLeaderboardEntry(rank=index, ...)
        for index, row in enumerate(ranked, start=1)
    ]

def _sort_key(
    row: AlphaEvaluationLeaderboardEntry,
    *,
    view: str,
    forecast_metric: str,
    sleeve_metric: str,
) -> tuple[Any, ...]:
    """Build multi-level sort key with tiebreakers."""
    
    if view == "forecast":
        # Primary: specified metric, then others in fixed order
        ordered_metrics = [
            forecast_metric,
            "ic_ir", "mean_ic", "mean_rank_ic", "rank_ic_ir", "n_periods"
        ]
    elif view == "sleeve":
        ordered_metrics = [
            sleeve_metric,
            "sharpe_ratio", "total_return", "annualized_return", "max_drawdown", "turnover"
        ]
    else:  # combined
        ordered_metrics = [forecast_metric, sleeve_metric, ...]
    
    sort_parts = []
    for metric_name in ordered_metrics:
        sort_parts.extend(_metric_sort_components(getattr(row, metric_name), metric_name))
    
    # Tiebreakers
    sort_parts.extend([
        row.mapping_name is None,      # None values sort last
        "" if row.mapping_name is None else row.mapping_name,
        row.alpha_name,
        row.run_id,
    ])
    return tuple(sort_parts)

def _metric_sort_components(value: float | int | None, *, metric_name: str) -> tuple[bool, float]:
    """Convert metric to sortable tuple: (is_null, direction_adjusted_value)."""
    
    direction = _metric_direction(metric_name)
    if value is None:
        return (True, 0.0)  # None values sort last
    
    normalized = float(value)
    if direction == "desc":
        return (False, -normalized)  # Higher is better: negate for ascending sort
    return (False, normalized)       # Lower is better: use as-is
```

### Metric Direction Map

**Forecast Metrics:** ALL descending (higher better)
```python
_FORECAST_METRICS = {
    "mean_ic": "desc",
    "ic_ir": "desc",
    "mean_rank_ic": "desc",
    "rank_ic_ir": "desc",
    "n_periods": "desc",
}
```

**Sleeve Metrics:** Mixed
```python
_SLEEVE_METRICS = {
    # Better when higher
    "sharpe_ratio": "desc",
    "annualized_return": "desc",
    "total_return": "desc",
    "cumulative_return": "desc",
    "win_rate": "desc",
    "hit_rate": "desc",
    "profit_factor": "desc",
    "trade_count": "desc",
    "exposure_pct": "desc",
    
    # Better when lower
    "max_drawdown": "asc",
    "turnover": "asc",
    "average_turnover": "asc",
    "total_turnover": "asc",
    "total_transaction_cost": "asc",
    "total_slippage_cost": "asc",
    "total_execution_friction": "asc",
}
```

### Filtering Logic

```python
def _filter_rows(
    rows: list[AlphaEvaluationLeaderboardEntry],
    filters: dict[str, Any],
) -> list[AlphaEvaluationLeaderboardEntry]:
    """Apply sequential filters by each attribute."""
    
    filtered = rows
    if filters["alpha_name"] is not None:
        filtered = [row for row in filtered if row.alpha_name == filters["alpha_name"]]
    if filters["dataset"] is not None:
        filtered = [row for row in filtered if row.dataset == filters["dataset"]]
    if filters["timeframe"] is not None:
        filtered = [row for row in filtered if row.timeframe == filters["timeframe"]]
    if filters["evaluation_horizon"] is not None:
        filtered = [row for row in filtered if row.evaluation_horizon == filters["evaluation_horizon"]]
    if filters["mapping_name"] is not None:
        filtered = [row for row in filtered if row.mapping_name == filters["mapping_name"]]
    return filtered
```

---

## 7. HOW RUNS ARE PERSISTED AND VERSIONED

### Deterministic Run ID Generation

```python
# Run ID = {name}_{hash}
# Hash = first 12 chars of SHA256(canonical_config_json)

from src.research.registry import serialize_canonical_json
import hashlib

def generate_run_id(name: str, config: dict[str, Any]) -> str:
    canonical = serialize_canonical_json(config)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return f"{name}_{digest}"
```

### Manifest Registration Pattern

```python
from src.research.alpha_eval import (
    write_alpha_evaluation_artifacts,
    register_alpha_evaluation_run,
)

# Write artifacts to disk
manifest = write_alpha_evaluation_artifacts(
    output_dir=artifact_dir,
    result=evaluation_result,
    run_id=run_id,
    alpha_name=alpha_name,
    promotion_gate_config=promotion_config,
    effective_config=config,
)

# Register in JSONL registry
register_alpha_evaluation_run(
    run_id=run_id,
    alpha_name=alpha_name,
    effective_config=config,
    evaluation_result=evaluation_result,
    artifact_dir=artifact_dir,
    manifest=manifest,
    registry_path="artifacts/alpha/registry.jsonl",
)
```

### Registry Entry Creation

```python
def build_alpha_evaluation_registry_entry(
    *,
    run_id: str,
    alpha_name: str,
    effective_config: Mapping[str, Any],
    evaluation_result: AlphaEvaluationResult,
    artifact_dir: str | Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Create deterministic registry entry."""
    
    return {
        "run_id": run_id,
        "run_type": "alpha_evaluation",
        "timestamp": stable_timestamp_from_run_id(run_id),
        "alpha_name": alpha_name,
        "dataset": effective_config.get("dataset"),
        "timeframe": manifest.get("timeframe"),
        "evaluation_horizon": effective_config.get("alpha_horizon"),
        "metrics_summary": {key: evaluation_result.summary.get(key) for key in _SUMMARY_KEYS},
        "artifact_path": str(Path(artifact_dir)),
        "artifact_paths": manifest.get("artifact_paths", {}),
        "promotion_status": manifest.get("promotion_gate_summary", {}).get("promotion_status"),
        "review_status": review_metadata["status"],
        "review_metadata": review_metadata,
        "promotion_gate_summary": manifest.get("promotion_gate_summary"),
        "config": canonicalize_value(dict(effective_config)),
        "manifest": canonicalize_value(dict(manifest)),
    }
```

### Registry Format Guarantees

- **Deterministic:** Same config → same run ID → same entry
- **Immutable:** Once written, entries are append-only in JSONL
- **Queryable:** Each line is valid JSON, can be streamed/indexed
- **Auditable:** Contains full config and promotion gate results

### Artifact Persistence Pattern

```python
from pathlib import Path
import json

def write_alpha_evaluation_artifacts(
    output_dir: str | Path,
    result: AlphaEvaluationResult,
    **kwargs,
) -> dict[str, Any]:
    """Persist deterministic artifacts and return manifest."""
    
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Write data files
    _write_csv(resolved_output_dir / "ic_timeseries.csv", result.ic_timeseries)
    _write_json(resolved_output_dir / "alpha_metrics.json", metrics_payload)
    _write_parquet(resolved_output_dir / "predictions.parquet", predictions_frame)
    
    # 2. Write promotion gate results
    write_promotion_gate_artifact(resolved_output_dir, promotion_evaluation)
    
    # 3. Build and write manifest
    manifest = {
        "alpha_name": ...,
        "artifact_files": [...],
        "artifact_paths": {...},
        "metric_summary": {...},
        "promotion_gate_summary": promotion_evaluation.to_payload(),
    }
    _write_json(resolved_output_dir / "manifest.json", manifest)
    
    return manifest
```

---

## 8. PROMOTION GATES AND REVIEW PATTERNS

### Promotion Gate Definition

```json
{
  "gate_id": "min_valid_timestamps",
  "source": "qa_summary",
  "metric_path": "forecast.valid_timestamps",
  "comparator": "gte",
  "threshold": 20,
  "statistic": "value",
  "missing_behavior": "fail",
  "description": "Require enough usable timestamps for stable alpha read."
}
```

### Gate Evaluation Result

```json
{
  "gate_id": "min_valid_timestamps",
  "status": "passed",
  "source": "qa_summary",
  "metric_path": "forecast.valid_timestamps",
  "statistic": "value",
  "comparator": "gte",
  "threshold": 20,
  "actual_value": 19,
  "missing_behavior": "fail",
  "reason": "Value 19 is not >= 20"
}
```

### Promotion Gate Summary

```json
{
  "configured": true,
  "run_type": "alpha_evaluation",
  "evaluation_status": "all_passed",
  "promotion_status": "eligible",
  "status_on_pass": "eligible",
  "status_on_fail": "blocked",
  "gate_count": 8,
  "passed_gate_count": 8,
  "failed_gate_count": 0,
  "missing_gate_count": 0,
  "definitions": [...],
  "results": [...]
}
```

### Review Metadata

```json
{
  "schema_version": 1,
  "status": "candidate",
  "promotion_status": "eligible",
  "decision_reason": "Run passed all promotion gates and is eligible for promotion.",
  "decision_source": "promotion_gates"
}
```

---

## 9. SLEEVE LOADING AND PORTFOLIO INTEGRATION

### Alpha Sleeve Component Loading

```python
from src.portfolio.loaders import load_portfolio_component_returns

# Load alpha sleeve returns for portfolio construction
returns_df = load_portfolio_component_returns(
    run_dir="artifacts/alpha/rank_composite_momentum_2026_q1_alpha_eval_33ff16458d15",
    artifact_type="alpha_sleeve",
    strategy_name="rank_composite_momentum",
)

# Returns DataFrame with columns: ts_utc, strategy_name, strategy_return, run_id
```

### Portfolio Component Definition

```yaml
portfolios:
  my_portfolio:
    allocator: equal_weight
    components:
      - strategy_name: rank_composite_momentum
        run_id: "rank_composite_momentum_2026_q1_alpha_eval_33ff16458d15"
        artifact_type: "alpha_sleeve"
```

### Supported Artifact Types

- `"strategy"` → loads from `equity_curve.csv`
- `"alpha_sleeve"` → loads from `sleeve_returns.csv` in alpha eval directory

### Allocation Patterns

```python
from src.portfolio.allocators import EqualWeightAllocator, OptimizerAllocator

# Equal weight
allocator = EqualWeightAllocator()
weights = allocator.allocate(returns_wide_df)

# Optimizer-based (minimum variance, risk parity, etc.)
allocator = OptimizerAllocator(
    optimizer_config={"method": "minimum_variance"},
)
weights = allocator.allocate(returns_wide_df)
```

---

## 10. IMPLEMENTATION RECOMMENDATIONS FOR CANDIDATE SELECTION

### Suggested Data Structures

```python
@dataclass(frozen=True)
class CandidateSelectionCriteria:
    """Selection rules for alpha or portfolio candidates."""
    view: str  # "forecast", "sleeve", "combined"
    primary_metric: str  # e.g., "ic_ir", "sharpe_ratio"
    filters: dict[str, Any]  # alpha_name, dataset, evaluation_horizon, etc.
    promotion_gate_requirement: str  # "all_passed", "any_passed", None
    review_status_requirement: str | None  # None, "candidate", "promoted"
    max_candidates: int | None = None
    tiebreak_metrics: list[str] = dataclass.field(default_factory=list)

@dataclass(frozen=True)
class SelectedCandidate:
    """One selected candidate from comparison."""
    rank: int
    run_id: str
    alpha_name: str
    primary_metric_value: float | None
    promotion_status: str
    review_status: str
    artifact_path: str
    all_metrics: AlphaEvaluationLeaderboardEntry
```

### Suggested Implementation Approach

1. **Use existing registry loading:**
   ```python
   entries = load_alpha_evaluation_registry(artifacts_root)
   ```

2. **Convert to leaderboard entries:**
   ```python
   leaderboard_entries = [_normalize_registry_entry(entry) for entry in entries]
   ```

3. **Apply filters:**
   ```python
   filtered = _filter_rows(leaderboard_entries, filters)
   ```

4. **Rank by criteria:**
   ```python
   ranked = _rank_rows(filtered, view=view, forecast_metric=metric, sleeve_metric=sleeve_metric)
   ```

5. **Select top N with promotion gate check:**
   ```python
   candidates = [
       entry for entry in ranked[:max_candidates]
       if _promotion_status_qualifies(entry, requirement)
   ]
   ```

6. **Return structured selection result:**
   ```python
   return CandidateSelectionResult(
       criteria=criteria,
       selected_candidates=candidates,
       total_eligible=len(ranked),
       timestamp=datetime.utcnow(),
   )
   ```

### Key Patterns to Follow

- ✅ Use `@dataclass(frozen=True)` for immutable results
- ✅ Store deterministic run IDs, not file paths
- ✅ Reference metrics by name, not raw values
- ✅ Support filtering by dataset, timeframe, evaluation_horizon
- ✅ Respect promotion gates and review status
- ✅ Persist selection results with comparison ID/timestamp
- ✅ Output as JSONL-compatible dict or CSV leaderboard
- ✅ Use canonical JSON serialization for deterministic IDs

---

## Summary Table

| Pattern | Location | Key Learning |
|---------|----------|--------------|
| **Artifact Structure** | `artifacts/alpha/`, `artifacts/portfolios/` | Run dirs named `{name}_{hash}`, manifest.json contains all metadata |
| **Metrics Available** | `alpha_metrics.json`, `sleeve_metrics.json` | Forecast & sleeve metrics tracked separately, distinct sort directions |
| **Registry Format** | `registry.jsonl` — one entry per line | Deterministic, queryable, append-only, includes full config |
| **Manifest** | `manifest.json` per run | Complete capture of inputs, outputs, and promotion results |
| **Ranking Logic** | `alpha_eval/compare.py` | Multi-level sort: primary metric, tiebreakers, null handling |
| **CLI Conventions** | `src/cli/*.py` | parse_args → run_cli → Result dataclass |
| **Data Structures** | Frozen dataclasses | Immutable, JSONL-serializable results |
| **Run Persistence** | Manifest → Registry | Write artifacts, build manifest, register deterministically |
| **Promotion Gates** | `promotion_gates.json` | Evaluates against multiple sources (metrics, qa_summary, config) |
| **Portfolio Integration** | `portfolio/loaders.py` | Load alpha sleeves by run_id, select components by artifact_type |

