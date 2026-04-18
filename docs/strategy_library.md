# Strategy Archetype Library

**Version:** 1.0.0  
**Status:** Production-Ready  
**Last Updated:** April 2026

---

## Overview

The **Strategy Archetype Library** formalizes a canonical set of research-grade strategies within StratLake. Each strategy is:

- **First-class research primitive** with stable identity (`strategy_id + version`)
- **Formally defined** with mathematical specification and research hypothesis
- **Typed and composable** - emits explicitly-typed signals (M21.1 Signal Semantics Layer)
- **Deterministic and reproducible** - identical inputs produce identical outputs
- **Registry-driven** - discoverable and resolvable declaratively
- **Production-ready** - suitable for backtesting, walk-forward validation, and live trading

---

## Canonical Strategy Archetypes

The library implements **6 core strategy archetypes**, representing distinct research hypotheses:

### 1. Time Series Momentum

**Strategy ID:** `time_series_momentum` (v1.0.0)

**Archetype:** Trend-Following | Time-Series | Price-Based

**Research Hypothesis:**
Assets exhibit return autocorrelation. Recent outperformers tend to continue higher due to:
- Behavioral momentum (herding)
- Trend persistence in risky assets
- Slow information dissemination

**Mathematical Definition:**

For each symbol independently:

1. Compute period returns: $r_t = \ln(P_t / P_{t-\text{lookback}})$
2. Compute short-term average: $\text{ma}_{\text{short}} = \frac{1}{n_s} \sum_{i=0}^{n_s-1} r_{t-i}$
3. Compute long-term average: $\text{ma}_{\text{long}} = \frac{1}{n_l} \sum_{i=0}^{n_l-1} r_{t-i}$
4. Generate ternary signal:
   - Signal = +1 if $\text{ma}_{\text{short}} > \text{ma}_{\text{long}}$
   - Signal = -1 if $\text{ma}_{\text{short}} < \text{ma}_{\text{long}}$
   - Signal = 0 otherwise

**Required Inputs:**
- `ts_utc` (timestamp)
- `symbol` (security identifier)
- `close` (closing price)

**Output Signal Type:** `ternary_quantile` (values: -1, 0, +1)

**Configuration Parameters:**
- `lookback_short` (int): Short-term window in periods (typical: 20-63)
- `lookback_long` (int): Long-term window in periods (typical: 100-250)

**Example Configuration:**
```yaml
strategy:
  name: time_series_momentum
  version: 1.0.0
  dataset: features_daily
  signal_type: ternary_quantile
  position_constructor: identity_weights
  parameters:
    lookback_short: 20
    lookback_long: 200
```

**Known Failure Modes:**

| Mode | Description | Regime | Mitigation |
|------|-------------|--------|-----------|
| **Regime Reversal** | Whipsaws in mean-reverting environments | Low trend persistence | Add trend filter; increase lookback_long |
| **Liquidity Constraint** | Poor execution in illiquid names | Low liquidity | Universe filtering; position sizing |

**Assumptions:**
- Return time series is stationary within rebalance period
- No structural breaks or regime shifts
- No transaction costs (accounted for separately)

**Regime Dependencies:**
- Trend-following works best in trending markets
- Fails in choppy, range-bound regimes
- Performance declines with increases in mean reversion frequency

---

### 2. Cross-Section Momentum

**Strategy ID:** `cross_section_momentum` (v1.0.0)

**Archetype:** Relative Strength | Cross-Sectional Rank | Factor-Based

**Research Hypothesis:**
Within a universe, relative momentum (relative strength) predicts relative returns:
- Factor persistence: momentum factors continue across time horizons
- Crowding dynamics: popular winners attract capital
- Risk compensation: investors demand premium for holding winners

**Mathematical Definition:**

At each timestamp $t$, for all securities in the universe:

1. Compute returns: $r_{i,t} = \ln(P_{i,t} / P_{i,t-\tau})$
2. Rank securities deterministically:
   - Primary: descending return magnitude
   - Tie-breaker: ascending symbol name (stable ordering)
3. Normalize rank to continuous scale:

$$\text{signal}_{i,t} = 1 - \frac{2(\text{rank}_{i,t} - 1)}{N_t - 1}$$

where $N_t$ = number of securities at timestamp $t$

**Required Inputs:**
- `ts_utc` (timestamp)
- `symbol` (security identifier)  
- `close` (closing price)
- Minimum 5 securities per timestamp
- Time series of at least 1 period

**Output Signal Type:** `cross_section_rank` (values: [-1.0, +1.0], continuous)

**Configuration Parameters:**
- `lookback_days` (int): Periods for return computation (default: 1)

**Example Configuration:**
```yaml
strategy:
  name: cross_section_momentum
  version: 1.0.0
  dataset: features_daily
  signal_type: cross_section_rank
  position_constructor: rank_dollar_neutral
  parameters:
    lookback_days: 1
```

**Known Failure Modes:**

| Mode | Description | Regime | Mitigation |
|------|-------------|--------|-----------|
| **Factor Crowding** | Extreme positions lead to slippage | High momentum crowding | Position sizing; vol scaling |
| **Reversal at Extremes** | Highest momentum reverses sharply | Momentum reversion | Reduce extreme quantile exposure |
| **Insufficient Cross-Section** | Rankings unreliable with few securities | Low universe size | Minimum size filters |

**Assumptions:**
- Cross-section sufficiently large (≥5 securities)
- Deterministic tie-breaking via symbol ascending
- Returns computed from closing prices

**Regime Dependencies:**
- Works best in persistent momentum regimes
- Fails when factor reverses (momentum → mean reversion)
- Sensitive to universe composition changes

---

### 3. Mean Reversion

**Strategy ID:** `mean_reversion` (v1.0.0)

**Archetype:** Contrarian | Z-Score | Stationary

**Research Hypothesis:**
Asset prices oscillate around a statistical equilibrium:
- Oversold assets revert upward
- Overbought assets revert downward
- Mean reversion is faster than drift in certain regimes

**Mathematical Definition:**

For each symbol independently:

1. Compute rolling statistics over lookback window $w$:
   - $\mu_t = \frac{1}{w} \sum_{i=0}^{w-1} P_{t-i}$
   - $\sigma_t = \sqrt{\frac{1}{w} \sum_{i=0}^{w-1} (P_{t-i} - \mu_t)^2}$

2. Compute normalized deviation:
   $$Z_t = \frac{P_t - \mu_t}{\sigma_t + \epsilon}$$
   where $\epsilon$ prevents division by zero

3. Generate ternary signal:
   - Signal = +1 if $Z_t < -\theta$ (oversold, buy)
   - Signal = -1 if $Z_t > +\theta$ (overbought, sell)
   - Signal = 0 otherwise

**Required Inputs:**
- `ts_utc` (timestamp)
- `symbol` (security identifier)
- `close` (closing price)
- Minimum 2 lookback periods

**Output Signal Type:** `ternary_quantile` (values: -1, 0, +1)

**Configuration Parameters:**
- `lookback` (int): Rolling window size (typical: 20-63)
- `threshold` (float): Z-score threshold magnitude (typical: 1.0-2.5)

**Example Configuration:**
```yaml
strategy:
  name: mean_reversion
  version: 1.0.0
  dataset: features_daily
  signal_type: ternary_quantile
  position_constructor: identity_weights
  parameters:
    lookback: 20
    threshold: 2.0
```

**Known Failure Modes:**

| Mode | Description | Regime | Mitigation |
|------|-------------|--------|-----------|
| **Trending Market Whipsaw** | Many losing trades in trends | Strong trend | Add trend filter |
| **Volatility Spike** | False extremes from vol spikes | Vol spike | Adaptive threshold; robust std |
| **Structural Shift** | Permanent level change | Structural break | Regime detection |

**Assumptions:**
- Close prices are covariance-stationary
- Mean and volatility stable over rolling window
- No structural breaks

**Regime Dependencies:**
- Works best in mean-reverting regimes with low drift
- Fails in strongly trending environments
- Sensitive to volatility regime

---

### 4. Breakout

**Strategy ID:** `breakout` (v1.0.0)

**Archetype:** Momentum | Technical | Price-Extreme

**Research Hypothesis:**
Price breaks beyond recent extremes signal potential trend initiation:
- Breakouts generate momentum due to stop-loss cascades
- Technical levels act as support/resistance
- Breakouts indicate decision-maker conviction

**Mathematical Definition:**

For each symbol independently:

1. Compute rolling high and low over lookback window:
   - $H_t = \max(P_{t}, P_{t-1}, ..., P_{t-w+1})$
   - $L_t = \min(P_{t}, P_{t-1}, ..., P_{t-w+1})$

2. Generate ternary signal:
   - Signal = +1 if $P_t \geq H_t$ (breakout above high)
   - Signal = -1 if $P_t \leq L_t$ (breakout below low)
   - Signal = 0 otherwise (trapped between)

**Required Inputs:**
- `ts_utc` (timestamp)
- `symbol` (security identifier)
- `high` (period high)
- `low` (period low)
- `close` (closing price)
- Minimum 2 lookback periods

**Output Signal Type:** `ternary_quantile` (values: -1, 0, +1)

**Configuration Parameters:**
- `lookback` (int): Window for identifying extremes (default: 20, typical: 10-50)

**Example Configuration:**
```yaml
strategy:
  name: breakout
  version: 1.0.0
  dataset: features_daily
  signal_type: ternary_quantile
  position_constructor: identity_weights
  parameters:
    lookback: 20
```

**Known Failure Modes:**

| Mode | Description | Regime | Mitigation |
|------|-------------|--------|-----------|
| **False Breakout** | Quick reversal on noise breakouts | Low persistence | Volume confirmation |
| **Trend Exhaustion** | Breakouts at trend peaks | Trend reversal | Vol filter |

**Assumptions:**
- High/low prices reflect true period extremes
- Rolling window captures relevant history

**Regime Dependencies:**
- Momentum continuation preferred
- Fails when breakouts immediately reverse

---

### 5. Pairs Trading

**Strategy ID:** `pairs_trading` (v1.0.0)

**Archetype:** Market-Neutral | Spread | Cointegrated

**Research Hypothesis:**
Co-integrated security pairs show temporary deviations that revert to long-term equilibrium:
- Long-term relationship persists despite short-term separation
- Spreads are mean-reverting
- Paired long/short hedge systematic risk

**Mathematical Definition:**

For two paired securities:

1. Compute spread: $S_t = P_{1,t} - \beta P_{2,t}$

2. Compute rolling statistics:
   - $\mu_S = \frac{1}{w} \sum_{i=0}^{w-1} S_{t-i}$
   - $\sigma_S = \sqrt{\frac{1}{w} \sum_{i=0}^{w-1} (S_{t-i} - \mu_S)^2}$

3. Compute spread z-score: $Z_t = \frac{S_t - \mu_S}{\sigma_S}$

4. Signal on z-score extremes:
   - Long spread if $Z_t < -\theta$
   - Short spread if $Z_t > +\theta$
   - Flat otherwise

**Required Inputs:**
- `ts_utc` (timestamp)
- `symbol` (paired security identifier)
- `close` (closing price)
- Minimum 2 distinct securities per timestamp
- Time series of at least 1 period

**Output Signal Type:** `spread_zscore` (values: continuous real)

**Configuration Parameters:**
- `lookback` (int): Window for spread statistics (default: 63)
- `threshold` (float): Z-score threshold (default: 2.0)

**Example Configuration:**
```yaml
strategy:
  name: pairs_trading
  version: 1.0.0
  dataset: features_daily
  signal_type: spread_zscore
  position_constructor: identity_weights
  parameters:
    lookback: 63
    threshold: 2.0
```

**Known Failure Modes:**

| Mode | Description | Regime | Mitigation |
|------|-------------|--------|-----------|
| **Cointegration Breakdown** | Long-term relationship decays | Structural change | Dynamic pair reselection |
| **Hedge Ratio Drift** | Beta relationship changes | Regime shift | Rolling beta estimation |

**Assumptions:**
- Pairs are cointegrated
- Hedge ratio stable within rebalance period
- Spread is mean-reverting

**Regime Dependencies:**
- Works in stable macro environments
- Fails when cointegrated relationship breaks

---

### 6. Residual Momentum

**Strategy ID:** `residual_momentum` (v1.0.0)

**Archetype:** Factor-Adjusted | Alpha | Cross-Sectional

**Research Hypothesis:**
Idiosyncratic (residual) momentum outperforms raw momentum:
- Removes systematic factor returns
- Isolates true alpha signal
- Beta-neutral alpha generation

**Mathematical Definition:**

At each timestamp $t$, for all securities:

1. Compute returns: $r_{i,t} = \ln(P_{i,t} / P_{i,t-\tau})$

2. Estimate factor exposure:
   $$r_{i,t} = \alpha_i + \beta_i r_{m,t} + \varepsilon_{i,t}$$
   where $r_{m,t}$ = market return, $\varepsilon_{i,t}$ = residual

3. Residual (alpha) return: $\alpha_i = r_{i,t} - \beta_i r_{m,t}$

4. Cross-sectional ranking of residuals:
   $$\text{signal}_{i,t} = 1 - \frac{2(\text{rank}_{\alpha_i} - 1)}{N_t - 1}$$

**Required Inputs:**
- `ts_utc` (timestamp)
- `symbol` (security identifier)
- `close` (closing price)
- `market_return` (systematic/market return)
- Minimum 5 securities per timestamp
- Time series of at least 1 period

**Output Signal Type:** `cross_section_rank` (values: [-1.0, +1.0])

**Configuration Parameters:**
- `lookback_days` (int): Return computation window (default: 20)

**Example Configuration:**
```yaml
strategy:
  name: residual_momentum
  version: 1.0.0
  dataset: features_daily
  signal_type: cross_section_rank
  position_constructor: rank_dollar_neutral
  parameters:
    lookback_days: 20
```

**Known Failure Modes:**

| Mode | Description | Regime | Mitigation |
|------|-------------|--------|-----------|
| **Factor Model Instability** | Betas change across regimes | Regime shift | Rolling beta; factor stability |
| **Estimation Error** | Residuals contain noise | Low signal-to-noise | Larger sample; robust regression |

**Assumptions:**
- Factor model correctly specified
- Betas stable within rebalance period
- Market return accurately represents systematic risk

**Regime Dependencies:**
- Works when idiosyncratic momentum distinct from factor momentum
- Fails when factors dominate all variation

---

## Usage Guide

### 1. Configuration

All strategies are configured via YAML:

```yaml
strategy:
  name: <strategy_id>
  version: 1.0.0
  dataset: <dataset_name>
  signal_type: <signal_type>
  position_constructor: <constructor_name>
  parameters:
    <param_name>: <value>
    ...
```

### 2. Python API

```python
from src.research.strategies.registry import build_strategy

config = {
    "dataset": "features_daily",
    "signal_type": "cross_section_rank",
    "position_constructor": {
        "name": "rank_dollar_neutral",
        "params": {}
    },
    "parameters": {
        "lookback_days": 1
    }
}

strategy = build_strategy("cross_section_momentum", config)
signals = strategy.generate_signals(df)
```

### 3. Validation

```python
from src.research.strategies.validation import (
    validate_strategy_config,
    validate_signal_output,
    verify_determinism
)

# Validate configuration
validate_strategy_config("cross_section_momentum", config)

# Validate output signal
validate_signal_output("cross_section_momentum", signals)

# Verify determinism
verify_determinism("cross_section_momentum", config, df)
```

### 4. Registry Inspection

```python
from src.research.strategies.validation import (
    load_strategies_registry,
    list_available_strategies,
    get_strategy_by_archetype
)

# List all available strategies
all_strategies = list_available_strategies()

# Get strategies by archetype
momentum_strategies = get_strategy_by_archetype("momentum")

# Load and inspect registry
registry = load_strategies_registry()
```

---

## Signal Semantics Integration (M21.1)

All strategies emit **explicitly typed signals** compatible with the Signal Semantics Layer:

### Signal Types

| Strategy | Output Signal Type | Values | Executable |
|----------|-------------------|--------|-----------|
| Time Series Momentum | `ternary_quantile` | {-1, 0, 1} | ✓ |
| Cross-Section Momentum | `cross_section_rank` | [-1.0, 1.0] | ✓ |
| Mean Reversion | `ternary_quantile` | {-1, 0, 1} | ✓ |
| Breakout | `ternary_quantile` | {-1, 0, 1} | ✓ |
| Pairs Trading | `spread_zscore` | ℝ (continuous) | ✓ |
| Residual Momentum | `cross_section_rank` | [-1.0, 1.0] | ✓ |

### Position Constructor Compatibility

Strategies support the following position constructors (M21.2):

- `identity_weights`: Pass signal directly as position
- `rank_dollar_neutral`: Separate long/short books, scale by sign
- `zscore_clip_scale`: Clip and normalize by z-score
- `top_bottom_equal_weight`: Equal allocation to top/bottom quantiles
- `softmax_long_only`: Softmax temperature scaling

---

## Determinism and Reproducibility

All strategies are **deterministic**:
- Identical inputs always produce identical outputs
- No randomness (seeding not applicable)
- Stable ordering (deterministic tie-breaking, e.g., symbol ascending)
- Suitable for reproducible research pipelines

### Verification

```python
from src.research.strategies.validation import verify_determinism

verify_determinism("cross_section_momentum", config, df, repetitions=5)
```

---

## Testing and Validation

### Determinism Tests

All strategies pass determinism verification:
```python
for i in range(10):
    signal = strategy.generate_signals(df)
    # All runs produce identical output
```

### Schema Validation Tests

Configurations are validated against registry schemas:
```python
validate_strategy_config("cross_section_momentum", config)
```

### Signal Validation Tests

Output signals conform to declared types:
```python
validate_signal_output("cross_section_momentum", signals)
```

### Integration Tests

Strategies integrate with:
- **Backtest Runner** (M20): Deterministic historical simulation
- **Pipeline Runner**: Declarative DAG execution
- **Position Constructors** (M21.2): Composable construction
- **Signal Semantics** (M21.1): Typed signals with metadata

---

## Performance Considerations

### Computational Complexity

| Strategy | Complexity | Notes |
|----------|-----------|-------|
| Time Series Momentum | O(N) | Per-symbol rolling aggregation |
| Cross-Section Momentum | O(N log N) | Cross-sectional ranking at each timestamp |
| Mean Reversion | O(N) | Per-symbol rolling statistics |
| Breakout | O(N) | Per-symbol rolling extrema |
| Pairs Trading | O(N) | Per-symbol pairing + z-score |
| Residual Momentum | O(N log N) | Factor regression + ranking |

### Memory Usage

All strategies operate on in-memory DataFrames. For large universes:
- Cross-section strategies: O(N × T) where N = num securities, T = num timestamps
- Time-series strategies: O(N × w) where w = window size

### Optimization Tips

1. **Filter universes** to relevant securities
2. **Use smaller lookback windows** for faster computation
3. **Parallelize** per-symbol computations where possible
4. **Cache** intermediate results in long pipelines

---

## Known Limitations and Biases

### Forward-Looking Bias

Strategies compute statistics using closing prices. Execution typically occurs on next open. Configure `execution_lag` appropriately.

### Survivorship Bias

Strategies applied to historical data should account for delisted securities and corporate actions.

### Look-Ahead Bias

Ensure rebalance frequency aligns with decision-making frequency. Daily signals should not use intraday data.

### Crowding Effects

Strategies may suffer when widely adopted. Monitor factor crowding and adjust position sizing.

---

## Future Extensions

### Planned Enhancements

1. **Adaptive parameters**: Dynamic threshold/lookback adjustment
2. **Regime detection**: Automatic strategy switching
3. **Multi-strategy ensembles**: Composite signal generation
4. **Statistical factor models**: Enhanced residual momentum
5. **Volatility scaling**: Normalized position sizing

### Extension Points

Implement new strategies by:
1. Extending `ArchetypeStrategy`
2. Defining `strategy_definition` class attribute
3. Implementing `generate_signals(df)` method
4. Registering in `STRATEGY_BUILDERS`
5. Adding entry to `strategies.jsonl` registry

---

## Bibliography and References

### Foundational Papers

- Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013). "Value and momentum everywhere."
- Blitz, D., Hanauer, M. X., Vidojevic, M., & Zaremba, A. (2021). "Residual momentum."
- Jegadeesh, N., & Titman, S. (1993). "Returns to buying winners and selling losers: Implications for stock market efficiency."

### Practical Guides

- "Algorithmic Trading: Winning Strategies and Their Rationale" - Ernest P. Chan
- "Evidence-Based Technical Analysis" - David Aronson

---

## Support and Troubleshooting

### Common Issues

**Q: Strategy produces NaN signals**  
A: Check that lookback periods are sufficient. Ensure min_periods are met.

**Q: Signal values outside expected range**  
A: Validate output against schema using `validate_signal_output()`.

**Q: Non-deterministic behavior**  
A: Verify no randomness in computation. Check for floating-point precision issues.

**Q: Poor backtest performance**  
A: Review failure modes for current market regime. Consider adding filters or position constraints.

### Debugging

```python
# Enable detailed validation
from src.research.strategies.validation import validate_strategy_config, validate_signal_output

validate_strategy_config(strategy_id, config)
signal = strategy.generate_signals(df)
validate_signal_output(strategy_id, signal)

# Inspect intermediate computations
import pandas as pd
print(f"Signal min: {signal.min()}, max: {signal.max()}, mean: {signal.mean()}")
print(f"NaN count: {signal.isna().sum()}")
```

---

## Version History

### v1.0.0 (April 2026)

- Initial release with 6 canonical archetypes
- Full signal semantics integration
- Registry-driven discovery
- Deterministic outputs
- Comprehensive documentation

---

**Last Updated:** April 18, 2026  
**Maintained By:** StratLake Research Team  
**Status:** Production
