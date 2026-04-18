# StratLake Codebase Exploration Summary

**Date**: April 18, 2026  
**Scope**: Structure, Strategy Implementations, Signal Semantics (M21.1), Position Constructors (M21.2), Pipeline/CLI, and Artifact System

---

## 1. STRATEGY IMPLEMENTATIONS

### 1.1 Strategy Base Contract

**File**: [src/research/strategy_base.py](src/research/strategy_base.py)

Abstract base class defining the strategy interface:

```python
class BaseStrategy(ABC):
    name: str
    dataset: str
    required_input_columns: tuple[str, ...] = ()
    requires_return_column: bool = False
    signal_type: str = "target_weight"
    signal_params: dict[str, object] = {}
    position_constructor_name: str | None = "identity_weights"
    position_constructor_params: dict[str, object] = {}
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Returns {-1, 0, 1} signals or any finite numeric exposure"""
```

### 1.2 Built-in Strategy Implementations

**Directory**: [src/research/strategies/](src/research/strategies/)

**Core Implementations**:
- `MomentumStrategy` (name: `momentum_v1`) - [src/research/strategies/builtins.py](src/research/strategies/builtins.py)
- `MeanReversionStrategy` (name: `mean_reversion_v1`) - [src/research/strategies/builtins.py](src/research/strategies/builtins.py)
- `BuyAndHoldStrategy` - [src/research/strategies/baselines.py](src/research/strategies/baselines.py)
- `SMACrossoverStrategy` - [src/research/strategies/baselines.py](src/research/strategies/baselines.py)
- `SeededRandomStrategy` - [src/research/strategies/baselines.py](src/research/strategies/baselines.py)

**Strategy Registry**: [src/research/strategies/registry.py](src/research/strategies/registry.py)
- `build_strategy(name: str, config: dict) -> BaseStrategy`
- `STRATEGY_BUILDERS: dict[str, callable]` - factory mapping

### 1.3 Strategy Configuration

**File**: [configs/strategies.yml](configs/strategies.yml)

Example structure:
```yaml
momentum_v1:
  dataset: features_daily
  signal_type: ternary_quantile
  signal_params: {}
  position_constructor:
    name: identity_weights
    params: {}
  parameters:
    lookback_short: 5
    lookback_long: 20
```

**Key Fields**:
- `dataset`: Feature dataset to load
- `signal_type`: Semantic signal type (e.g., `ternary_quantile`, `binary_signal`)
- `signal_params`: Signal transformation parameters
- `position_constructor`: Name and params for position construction
- `parameters`: Strategy-specific hyperparameters

---

## 2. SIGNAL SEMANTICS LAYER (M21.1)

### 2.1 Core Types and Definitions

**File**: [src/research/signal_semantics.py](src/research/signal_semantics.py)

#### Signal Container
```python
@dataclass(frozen=True)
class Signal:
    signal_type: str
    version: str
    data: pd.DataFrame
    value_column: str = "signal"
    metadata: dict[str, Any] = field(default_factory=dict)
```

#### Signal Type Definition
```python
@dataclass(frozen=True)
class SignalTypeDefinition:
    signal_type_id: str
    version: str
    status: str
    domain: str
    codomain: str
    description: str
    validation_rules: dict[str, Any]
    transformation_policies: dict[str, Any]
    compatible_position_constructors: tuple[str, ...]
    directional: bool
    ordinal: bool
    probabilistic: bool
    executable: bool
    required_columns: tuple[str, ...] = ("symbol", "ts_utc")
    cross_sectional: bool = False
```

### 2.2 Canonical Signal Types Registry

**File**: [artifacts/registry/signal_types.jsonl](artifacts/registry/signal_types.jsonl)

Registered signal types (as of M21):
- `prediction_score` - Raw model outputs
- `cross_section_rank` - Integer ranks within symbol/timestamp
- `cross_section_percentile` - Percentile scores [0-1]
- `signed_zscore` - Standardized scores, typically [-3, +3]
- `ternary_quantile` - {-1, 0, 1} bucketed by quantile
- `binary_signal` - {0, 1} long-only signals
- `spread_zscore` - Z-scores for spread strategies
- `target_weight` - Direct portfolio weights

**Registry Path**: `DEFAULT_SIGNAL_TYPES_REGISTRY = REPO_ROOT / "artifacts" / "registry" / "signal_types.jsonl"`

### 2.3 Signal Generation Pipeline

**File**: [src/research/signal_engine.py](src/research/signal_engine.py)

```python
def generate_signals(df: pd.DataFrame, strategy: BaseStrategy) -> pd.DataFrame:
    """
    Apply strategy → validate → attach metadata → compute diagnostics
    
    Returns: DataFrame with 'signal' column + signal_semantics metadata
    """
```

**Process**:
1. Validate strategy input columns and timeframe
2. Call `strategy.generate_signals(df)` → Series
3. Create typed Signal via `create_signal()` with metadata
4. Ensure signal/position constructor compatibility
5. Compute signal diagnostics (turnover, exposure, flags)
6. Attach to result as `df.attrs["signal_diagnostics"]`

### 2.4 Signal Metadata Payload

**Attached to DataFrame as**: `df.attrs["signal_semantics"]`

Contains:
```python
{
    "signal_type": "ternary_quantile",
    "version": "1.0.0",
    "value_column": "signal",
    "source": {
        "layer": "strategy",
        "strategy_name": "momentum_v1",
        "strategy_class": "MomentumStrategy",
        "dataset": "features_daily"
    },
    "parameters": {},
    "timestamp_normalization": "UTC",
    "transformation_history": [],
    "definition": {
        "directional": bool,
        "ordinal": bool,
        "probabilistic": bool,
        "executable": bool,
        "compatible_position_constructors": [...]
    },
    "position_constructor": {
        "constructor_id": "identity_weights",
        "constructor_params": {}
    }
}
```

### 2.5 Validation Rules

**Function**: `validate_signal_frame(df, signal_type, ...)` 

Enforces deterministic fail-fast validation:
- Required columns present (`symbol`, `ts_utc`, value column)
- Symbol values non-empty
- `ts_utc` parses as UTC timestamps
- `(symbol, ts_utc)` keys unique (duplicate detection)
- Rows deterministically sorted
- Signal values numeric, finite, non-null
- Per-type range constraints (e.g., percentiles in [0,1])
- Cross-sectional minimum universe size requirements

**File**: [tests/test_signal_semantics.py](tests/test_signal_semantics.py)

---

## 3. POSITION CONSTRUCTORS (M21.2)

### 3.1 Position Constructor Base Contract

**File**: [src/research/position_constructors/base.py](src/research/position_constructors/base.py)

```python
@dataclass(frozen=True)
class PositionConstructor(ABC):
    constructor_id: str
    params: dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    
    @abstractmethod
    def construct(self, signal: Signal) -> pd.DataFrame:
        """
        Transform typed signal into executable positions.
        Returns: DataFrame with columns [symbol, ts_utc, position]
        """
```

### 3.2 Implemented Position Constructors

**Registry**: [src/research/position_constructors/registry.py](src/research/position_constructors/registry.py)  
**Definition Path**: [artifacts/registry/position_constructors.jsonl](artifacts/registry/position_constructors.jsonl)

| Constructor | File | Formula | Inputs | Notes |
|---|---|---|---|---|
| `identity_weights` | [src/research/position_constructors/identity.py](src/research/position_constructors/identity.py) | $w_{i,t} = s_{i,t}$ | target_weight | Use when signal already fully specified |
| `top_bottom_equal_weight` | [src/research/position_constructors/top_bottom_equal_weight.py](src/research/position_constructors/top_bottom_equal_weight.py) | Equal allocation to long/short | ternary_quantile | Handles empty long/short sets |
| `rank_dollar_neutral` | [src/research/position_constructors/rank_dollar_neutral.py](src/research/position_constructors/rank_dollar_neutral.py) | Normalized by sign + book | cross_section_rank | Gross long/short books separately |
| `softmax_long_only` | [src/research/position_constructors/softmax_long_only.py](src/research/position_constructors/softmax_long_only.py) | Softmax allocation | cross_section_rank | Temperature-based smoothing |
| `zscore_clip_scale` | [src/research/position_constructors/zscore_clip_scale.py](src/research/position_constructors/zscore_clip_scale.py) | Clip + normalize | signed_zscore | Handles extreme values |

### 3.3 Constructor Resolution

**Function**: `resolve_constructor(name: str, params: dict) -> PositionConstructor`

Maps:
- Constructor name → Python class
- Validates required parameters from registry
- Instantiates with validated params

**File**: [src/research/position_constructors/registry.py](src/research/position_constructors/registry.py)

```python
_IMPLEMENTATIONS: dict[str, type[PositionConstructor]] = {
    "identity_weights": IdentityWeightsPositionConstructor,
    "rank_dollar_neutral": RankDollarNeutralPositionConstructor,
    "top_bottom_equal_weight": TopBottomEqualWeightPositionConstructor,
    "softmax_long_only": SoftmaxLongOnlyPositionConstructor,
    "zscore_clip_scale": ZScoreClipScalePositionConstructor,
}
```

### 3.4 Signal-Constructor Compatibility

Each signal type declares compatible constructors in registry:

```json
{
    "signal_type_id": "cross_section_rank",
    "compatible_position_constructors": [
        "rank_dollar_neutral",
        "softmax_long_only"
    ]
}
```

**Validation**: `ensure_signal_type_compatible(signal_type, position_constructor)`

---

## 4. PIPELINE AND BACKTEST INTEGRATION

### 4.1 Research Pipeline Flow

[docs/strategy_evaluation_workflow.md](docs/strategy_evaluation_workflow.md)

```
feature dataset
    ↓ load_features()
signals
    ↓ generate_signals() [BaseStrategy.generate_signals()]
signal_frame + signal_semantics metadata
    ↓ run_backtest()
backtest results + equity curve
    ↓ compute_metrics()
metrics dict
    ↓ save_experiment()
artifacts/strategies/<run_id>/
```

### 4.2 Backtest Runner

**File**: [src/research/backtest_runner.py](src/research/backtest_runner.py)

```python
def run_backtest(df: pd.DataFrame, execution_config: ExecutionConfig | None = None) -> pd.DataFrame:
    """
    Convert signals → executed positions → strategy returns → equity curve
    
    Pipeline:
    1. Resolve/validate signal column
    2. Extract managed signal (typed Signal object) if present
    3. Apply position constructor if signal is typed
    4. Compute lagged executed signal (execution_delay bars)
    5. Calculate strategy returns: signal.shift(1) * asset_return
    6. Compound equity curve: (1 + returns).cumprod()
    7. Apply execution costs if configured
    """
```

**Key Functions**:
- `_construct_positions(signal: Signal) -> pd.Series` - Calls position constructor
- `_compute_executed_signal()` - Applies execution delay
- `_execution_cost()` / `_slippage_cost()` - Friction modeling

**Configuration**: [src/config/execution.py](src/config/execution.py)
- `execution_delay: int` (bars to shift)
- `transaction_cost_bps: float`
- `slippage_bps: float`
- `enabled: bool`

### 4.3 Walk-Forward Evaluation

**File**: [src/research/walk_forward.py](src/research/walk_forward.py)

```python
def execute_split(
    strategy: BaseStrategy,
    dataset: pd.DataFrame,
    split: EvaluationSplit,
    ...
) -> SplitExecutionResult:
    """Run full pipeline for one train/test split"""
```

**Configuration**: [configs/evaluation.yml](configs/evaluation.yml)

Example:
```yaml
splits:
  - split_id: train_2022_test_2023
    train_start: "2022-01-01"
    train_end: "2023-01-01"
    test_start: "2023-01-01"
    test_end: "2024-01-01"
```

### 4.4 Pipeline Runner (General Purpose)

**File**: [src/pipeline/pipeline_runner.py](src/pipeline/pipeline_runner.py)

Deterministic DAG execution framework:

```python
class PipelineRunner:
    def __init__(self, spec: PipelineSpec) -> None:
        self._spec = spec
    
    def run(self) -> PipelineRunResult:
        """Execute steps in dependency order"""
```

**Spec Structure** [src/pipeline/pipeline_runner.py](src/pipeline/pipeline_runner.py):
```python
@dataclass
class PipelineSpec:
    pipeline_id: str
    steps: tuple[PipelineStepSpec, ...]

@dataclass
class PipelineStepSpec:
    id: str
    adapter: str  # "python_module"
    module: str   # module path
    argv: list[str]
    depends_on: tuple[str, ...] = ()
```

**Execution**:
1. Topologically sort by dependencies
2. Run each step sequentially
3. Write artifacts, metrics, lineage
4. Register run to pipeline registry

---

## 5. ARTIFACT AND MANIFEST SYSTEM (M8-M10)

### 5.1 Strategy Artifact Layout

**Root**: `artifacts/strategies/<run_id>/`

**Standard Artifacts**:
```
config.json                 - Strategy config + parameters
metrics.json                - Performance metrics
signal_diagnostics.json     - Signal distribution + flags
qa_summary.json            - Data quality checks
promotion_gates.json       - Gate evaluation (if applicable)
equity_curve.csv           - Daily returns/equity curve
signals.parquet            - Full signal frame
equity_curve.parquet       - Parquet version
trades.parquet             - Per-trade analysis
manifest.json              - Run metadata + artifact inventory
```

**Walk-Forward Additional**:
```
metrics_by_split.csv       - Per-split summary metrics
splits/<split_id>/
  split.json               - Split definition
  metrics.json             - Split metrics
  equity_curve.csv
  equity_curve.parquet
  signals.parquet
```

### 5.2 Manifest Structure

**File**: [src/research/experiment_tracker.py](src/research/experiment_tracker.py) → `_build_manifest()`

```json
{
    "run_id": "momentum_v1_single_abc123def456",
    "timestamp": "2026-04-18T12:34:56Z",
    "strategy_name": "momentum_v1",
    "evaluation_mode": "single",
    "evaluation_config_path": null,
    "split_count": null,
    "strict_mode": false,
    "artifact_files": [
        "config.json",
        "metrics.json",
        "equity_curve.csv",
        "signals.parquet",
        "manifest.json"
    ],
    "signal_type": "ternary_quantile",
    "signal_version": "1.0.0",
    "signal_semantics_path": "signal_semantics.json",
    "constructor_id": "identity_weights",
    "constructor_params": {},
    "primary_metric": "sharpe_ratio",
    "metric_summary": {
        "cumulative_return": 0.0567,
        "sharpe_ratio": 1.23,
        ...
    }
}
```

### 5.3 Registry (JSONL)

**File**: `artifacts/strategies/registry.jsonl`

One JSON line per deterministic run. Each row:
```json
{
    "run_id": "momentum_v1_single_abc123def456",
    "timestamp": "2026-04-18T12:34:56Z",
    "strategy_name": "momentum_v1",
    "dataset": "features_daily",
    "strategy_params": {"lookback_short": 5, "lookback_long": 20},
    "evaluation_mode": "single",
    "evaluation_config_path": null,
    "data_range": {"start": "2020-01-01", "end": "2026-04-18"},
    "timeframe": "1D",
    "metrics_summary": {...},
    "artifact_path": "artifacts/strategies/momentum_v1_single_abc123def456"
}
```

**Functions**: 
- `load_registry(path)` - Read JSONL into list[dict]
- `upsert_registry_entry(path, entry)` - Update or append
- `default_registry_path(root)` - Compute registry path

### 5.4 Artifact Saving

**File**: [src/research/experiment_tracker.py](src/research/experiment_tracker.py)

Main entry point:
```python
def save_experiment(
    strategy_name: str,
    results_df: pd.DataFrame,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> Path:
    """
    1. Generate deterministic run_id from inputs
    2. Create artifact directory
    3. Write config.json, metrics.json, equity_curve.csv, signals.parquet, etc.
    4. Build and write manifest.json
    5. Validate artifact consistency
    6. Append to registry.jsonl
    7. Return artifact_dir Path
    """
```

Related functions:
- `save_walk_forward_experiment()` - For walk-forward runs
- `save_robustness_experiment()` - For parameter sweeps
- `_write_run_outputs()` - Write individual artifacts
- `_write_manifest()` - Write manifest.json
- `_write_registry_entry()` - Append to registry.jsonl

### 5.5 Pipeline Artifacts

**Root**: `artifacts/pipelines/<pipeline_run_id>/`

```
manifest.json          - Pipeline run summary
pipeline_metrics.json  - Step-level timing + status
lineage.json          - DAG edges: steps → artifacts
state.json            - Pipeline state snapshot
```

**Lineage Format**:
```json
{
    "pipeline_run_id": "test_pipeline_abc123",
    "run_type": "pipeline_lineage",
    "schema_version": 1,
    "edges": [
        {"from": "step:prepare", "to": "artifact:path/to/output", "type": "produces"},
        {"from": "step:prepare", "to": "step:evaluate", "type": "depends_on"}
    ]
}
```

---

## 6. CLI STRUCTURE

### 6.1 CLI Modules

**Directory**: [src/cli/](src/cli/)

| Module | Purpose | Entry Point |
|---|---|---|
| [src/cli/run_strategy.py](src/cli/run_strategy.py) | Run single strategy or walk-forward | `python -m src.cli.run_strategy --strategy <name>` |
| [src/cli/compare_strategies.py](src/cli/compare_strategies.py) | Compare multiple strategies | `python -m src.cli.compare_strategies --strategies <list>` |
| [src/cli/run_portfolio.py](src/cli/run_portfolio.py) | Portfolio construction | `python -m src.cli.run_portfolio --portfolio <name>` |
| [src/cli/run_pipeline.py](src/cli/run_pipeline.py) | General DAG pipeline execution | `python -m src.cli.run_pipeline --config <yaml>` |
| [src/cli/run_alpha.py](src/cli/run_alpha.py) | Alpha model training/prediction | `python -m src.cli.run_alpha` |
| [src/cli/run_candidate_selection.py](src/cli/run_candidate_selection.py) | Select portfolio candidates | `python -m src.cli.run_candidate_selection` |
| [src/cli/plot_strategy_run.py](src/cli/plot_strategy_run.py) | Generate visualizations | `python -m src.cli.plot_strategy_run --run-dir <path>` |
| [src/cli/generate_report.py](src/cli/generate_report.py) | Generate markdown reports | `python -m src.cli.generate_report` |
| [src/cli/generate_milestone_report.py](src/cli/generate_milestone_report.py) | Milestone reporting | `python -m src.cli.generate_milestone_report` |

### 6.2 CLI Argument Resolution

**Pattern**: Each module defines:
```python
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments"""

def run_cli(
    argv: Sequence[str] | None = None,
    *,
    state: dict[str, Any] | None = None,
    pipeline_context: dict[str, Any] | None = None,
) -> <ResultType>:
    """Execute from parsed arguments, return structured result"""

def main() -> None:
    """CLI entrypoint for direct module execution"""
    try:
        run_cli()
    except (...errors...) as exc:
        print(f"error message", file=sys.stderr)
        raise SystemExit(1)

if __name__ == "__main__":
    main()
```

### 6.3 Strategy Runner CLI (Detailed)

**File**: [src/cli/run_strategy.py](src/cli/run_strategy.py)

Arguments:
```
--strategy <name>              Strategy from configs/strategies.yml
--start <date>                 Inclusive start (YYYY-MM-DD)
--end <date>                   Exclusive end (YYYY-MM-DD)
--evaluation [<path>]          Enable walk-forward (optional path)
--robustness [<path>]          Run parameter sweep
--execution-delay <int>        Override delay
--transaction-cost-bps <float> Override transaction costs
--slippage-bps <float>         Override slippage
--execution-enabled            Force enable frictions
--disable-execution-model      Force disable frictions
--strict                       Strict validation mode
```

**Result Types**:
```python
@dataclass
class StrategyRunResult:
    strategy_name: str
    run_id: str
    metrics: dict[str, Any]
    experiment_dir: Path
    results_df: pd.DataFrame
    signal_diagnostics: dict[str, Any]
    qa_summary: dict[str, Any]
    simulation_result: SimulationRunResult | None
```

Plus:
- `WalkForwardRunResult` for evaluation mode
- `RobustnessRunResult` for parameter sweeps

### 6.4 Pipeline Runner CLI

**File**: [src/cli/run_pipeline.py](src/cli/run_pipeline.py)

```python
def load_pipeline_spec(yaml_path: str | Path) -> PipelineSpec:
    """Load and validate pipeline YAML"""

def run_cli(argv: Sequence[str] | None = None) -> PipelineRunResult:
    """Execute pipeline from YAML config"""
```

**Example Config** [configs/test_pipeline.yml](configs/test_pipeline.yml):
```yaml
pipeline_id: test_pipeline
steps:
  - id: prepare
    adapter: python_module
    module: "my_module"
    argv: ["--stage", "prepare"]
  - id: evaluate
    adapter: python_module
    module: "my_module"
    argv: ["--stage", "evaluate"]
    depends_on: ["prepare"]
```

### 6.5 Configuration Loading

**Configs Root**: [configs/](configs/)

Key files:
- [configs/strategies.yml](configs/strategies.yml) - Strategy registry
- [configs/evaluation.yml](configs/evaluation.yml) - Walk-forward splits
- [configs/robustness.yml](configs/robustness.yml) - Parameter sweeps
- [configs/execution.yml](configs/execution.yml) - Execution model (costs, delays)
- [configs/portfolios.yml](configs/portfolios.yml) - Portfolio configs
- [configs/alphas.yml](configs/alphas.yml) - Alpha model registry

**Loading Pattern**:
```python
def get_strategy_config(strategy_name: str) -> dict[str, Any]:
    """Load from YAML by name"""
    strategies = yaml.safe_load(STRATEGIES_CONFIG.read_text())
    if strategy_name not in strategies:
        raise ValueError(f"Strategy {strategy_name} not found")
    return strategies[strategy_name]
```

---

## 7. KEY INTEGRATION POINTS

### 7.1 Strategy → Signal → Position → Backtest

```python
# Step 1: Load strategy config
config = get_strategy_config("momentum_v1")  # from configs/strategies.yml

# Step 2: Build strategy instance
strategy = build_strategy("momentum_v1", config)  # Constructs MomentumStrategy

# Step 3: Load feature dataset
dataset = load_features(strategy.dataset)  # loads features_daily

# Step 4: Generate signals with semantics
signal_frame = generate_signals(dataset, strategy)
# - Calls strategy.generate_signals(dataset)
# - Wraps in Signal with metadata
# - Validates against signal_types registry
# - Ensures position_constructor compatibility

# Step 5: Run backtest with execution model
backtest_result = run_backtest(signal_frame)
# - Extracts typed Signal from metadata
# - Calls position_constructor.construct(signal)
# - Applies execution delay
# - Computes strategy returns
# - Calculates costs/slippage
# - Compounds equity curve

# Step 6: Compute metrics
metrics = compute_metrics(backtest_result)

# Step 7: Save artifacts
experiment_dir = save_experiment("momentum_v1", backtest_result, metrics, config)
# - Creates deterministic run_id
# - Writes config.json, metrics.json, equity_curve.csv, signals.parquet
# - Writes manifest.json with all metadata
# - Appends to registry.jsonl
```

### 7.2 Pipeline Adapter Interface

**File**: [src/pipeline/cli_adapter.py](src/pipeline/cli_adapter.py)

Bridges research CLI into pipeline DAG:
```python
def build_pipeline_cli_result(cli_result: object) -> dict[str, object]:
    """Convert StrategyRunResult → pipeline-compatible dict"""
```

---

## 8. CURRENT CONFIG STRUCTURE

### 8.1 Strategy Config Example

[configs/strategies.yml](configs/strategies.yml):
```yaml
momentum_v1:
  dataset: features_daily
  signal_type: ternary_quantile
  signal_params: {}
  position_constructor:
    name: identity_weights
    params: {}
  parameters:
    lookback_short: 5
    lookback_long: 20

mean_reversion_v1:
  dataset: features_daily
  signal_type: ternary_quantile
  signal_params: {}
  position_constructor:
    name: identity_weights
    params: {}
  parameters:
    lookback: 20
    threshold: 1.0
```

### 8.2 Execution Config

[configs/execution.yml](configs/execution.yml):
```yaml
execution_delay: 1              # bars to shift signals
transaction_cost_bps: 0         # transaction costs in bps
slippage_bps: 0                 # slippage in bps
enabled: false                  # enable frictions
```

### 8.3 Evaluation (Walk-Forward) Config

[configs/evaluation.yml](configs/evaluation.yml):
```yaml
splits:
  - split_id: "2022_2023"
    train_start: "2022-01-01"
    train_end: "2023-01-01"
    test_start: "2023-01-01"
    test_end: "2024-01-01"
```

---

## 9. KEY FILE REFERENCES

### Core Strategies
- Strategy base: [src/research/strategy_base.py](src/research/strategy_base.py)
- Built-in strategies: [src/research/strategies/builtins.py](src/research/strategies/builtins.py)
- Strategy registry: [src/research/strategies/registry.py](src/research/strategies/registry.py)

### Signal Semantics (M21.1)
- Signal layer: [src/research/signal_semantics.py](src/research/signal_semantics.py)
- Signal engine: [src/research/signal_engine.py](src/research/signal_engine.py)
- Signal registry: [artifacts/registry/signal_types.jsonl](artifacts/registry/signal_types.jsonl)
- Validation: [tests/test_signal_semantics.py](tests/test_signal_semantics.py)

### Position Constructors (M21.2)
- Base class: [src/research/position_constructors/base.py](src/research/position_constructors/base.py)
- Registry: [src/research/position_constructors/registry.py](src/research/position_constructors/registry.py)
- Implementations:
  - Identity: [src/research/position_constructors/identity.py](src/research/position_constructors/identity.py)
  - Rank Dollar Neutral: [src/research/position_constructors/rank_dollar_neutral.py](src/research/position_constructors/rank_dollar_neutral.py)
  - Top-Bottom Equal Weight: [src/research/position_constructors/top_bottom_equal_weight.py](src/research/position_constructors/top_bottom_equal_weight.py)
  - Softmax Long Only: [src/research/position_constructors/softmax_long_only.py](src/research/position_constructors/softmax_long_only.py)
  - ZScore Clip Scale: [src/research/position_constructors/zscore_clip_scale.py](src/research/position_constructors/zscore_clip_scale.py)
- Definition registry: [artifacts/registry/position_constructors.jsonl](artifacts/registry/position_constructors.jsonl)

### Backtest & Execution
- Backtest runner: [src/research/backtest_runner.py](src/research/backtest_runner.py)
- Walk-forward: [src/research/walk_forward.py](src/research/walk_forward.py)
- Execution config: [src/config/execution.py](src/config/execution.py)

### Artifacts (M8-M10)
- Experiment tracker: [src/research/experiment_tracker.py](src/research/experiment_tracker.py)
- Artifact logging docs: [docs/experiment_artifact_logging.md](docs/experiment_artifact_logging.md)
- Portfolio artifacts: [docs/portfolio_artifact_logging.md](docs/portfolio_artifact_logging.md)
- Manifest reporting: [docs/milestone_reporting_artifacts.md](docs/milestone_reporting_artifacts.md)

### Pipeline
- Pipeline runner: [src/pipeline/pipeline_runner.py](src/pipeline/pipeline_runner.py)
- Pipeline CLI: [src/cli/run_pipeline.py](src/cli/run_pipeline.py)
- Pipeline docs: [docs/backtest_runner.md](docs/backtest_runner.md)

### CLI
- Strategy runner: [src/cli/run_strategy.py](src/cli/run_strategy.py)
- Strategy comparison: [src/cli/compare_strategies.py](src/cli/compare_strategies.py)
- Portfolio runner: [src/cli/run_portfolio.py](src/cli/run_portfolio.py)
- CLI adapter: [src/pipeline/cli_adapter.py](src/pipeline/cli_adapter.py)

---

## 10. QUICK REFERENCE: RUNNING STRATEGIES

### Single-Run Strategy
```powershell
python -m src.cli.run_strategy --strategy momentum_v1
```

### With Date Bounds
```powershell
python -m src.cli.run_strategy --strategy momentum_v1 --start 2022-01-01 --end 2023-01-01
```

### Walk-Forward Evaluation
```powershell
python -m src.cli.run_strategy --strategy momentum_v1 --evaluation
```

### With Custom Execution Config
```powershell
python -m src.cli.run_strategy --strategy momentum_v1 --execution-delay 2 --transaction-cost-bps 5
```

### Compare Strategies
```powershell
python -m src.cli.compare_strategies --strategies momentum_v1,mean_reversion_v1
```

### Run Pipeline
```powershell
python -m src.cli.run_pipeline --config configs/test_pipeline.yml
```

---

## 11. ARTIFACT QUERY PATTERNS

### Loading Run Results
```python
from src.research.experiment_tracker import ARTIFACTS_ROOT
from src.research.reporting import load_run_artifacts

run_dir = ARTIFACTS_ROOT / "momentum_v1_single_abc123"
artifacts = load_run_artifacts(run_dir)

# artifacts.metrics, artifacts.config, artifacts.equity_curve_df, etc.
```

### Querying Registry
```python
from src.research.registry import load_registry, default_registry_path

registry_path = default_registry_path(ARTIFACTS_ROOT)
entries = load_registry(registry_path)

# Filter by strategy name or metric
filtered = [e for e in entries if e["strategy_name"] == "momentum_v1"]
```

### Inspecting Manifest
```python
import json

run_dir = ARTIFACTS_ROOT / "momentum_v1_single_abc123"
manifest = json.loads((run_dir / "manifest.json").read_text())

print(manifest["run_id"])
print(manifest["artifact_files"])
print(manifest["metric_summary"])
```

---

**End of Exploration Summary**
