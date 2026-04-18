# Long/Short Asymmetry - Quick Start Guide

## What's Implemented ✅

1. **Directional Execution Costs** - Different costs for long vs short positions
2. **Portfolio Long/Short Optimization** - Support for long/short allocation
3. **Directional Constraints** - Separate limits for long/short exposures
4. **Asymmetry Diagnostics** - Long/short performance decomposition
5. **Backward Compatible** - All existing configs still work unchanged

---

## 5-Minute Setup

### 1. Enable Directional Costs in Config

**Edit `configs/execution.yml`:**
```yaml
execution:
  enabled: true
  execution_delay: 1
  transaction_cost_bps: 3.0      # Baseline cost
  slippage_bps: 1.0
  # NEW: Override for specific sides
  long_transaction_cost_bps: 3.0   # Optional: if not set, uses transaction_cost_bps
  short_transaction_cost_bps: 5.0  # Higher cost for shorts (more realistic)
  short_slippage_multiplier: 1.25  # 25% higher slippage for shorts
  short_borrow_cost_bps: 25.0      # Daily borrow cost
```

### 2. Enable Long/Short Portfolio

**In your portfolio config:**
```python
optimizer_config = {
    "method": "equal_weight",
    "long_only": False,  # NEW: Enable long/short
}
```

### 3. Add Constraints (Optional)

```python
from src.portfolio import DirectionalPortfolioConstraints

constraints = DirectionalPortfolioConstraints(
    max_long_weight_sum=0.60,      # Max 60% long
    max_short_weight_sum=0.40,     # Max 40% short
    max_long_positions=50,
    max_short_positions=20,
    max_gross_exposure=1.20,       # Max 120% leverage
)

optimizer_config = {
    "method": "equal_weight",
    "long_only": False,
    "directional_constraints": constraints,
}
```

---

## Using Directional Features

### Backtest with Asymmetric Costs

```python
from src.config.execution import ExecutionConfig
from src.research.backtest_runner import backtest_signal

config = ExecutionConfig(
    enabled=True,
    transaction_cost_bps=3.0,
    short_transaction_cost_bps=5.0,    # 2 bps higher for shorts
    short_slippage_multiplier=1.25,    # 25% more slippage
    short_borrow_cost_bps=25.0,        # Borrow cost
)

result = backtest_signal(signal_df, execution_config=config)

# New columns available:
# - long_execution_cost: costs for positive positions
# - short_execution_cost: costs for negative positions
# - execution_friction: total costs
```

### Portfolio Optimization with Long/Short

```python
from src.portfolio import optimize_portfolio, DirectionalPortfolioConstraints

constraints = DirectionalPortfolioConstraints(
    max_long_weight_sum=0.60,
    max_short_weight_sum=0.40,
)

result = optimize_portfolio(
    returns_wide,
    optimizer_config={
        "method": "equal_weight",
        "long_only": False,
        "directional_constraints": constraints,
    }
)

# result.weights can now be negative (short positions)
```

### Compute Long/Short Metrics

```python
from src.portfolio import compute_long_short_directional_metrics

metrics = compute_long_short_directional_metrics(
    strategy_returns,
    positions,  # Can have negative values for shorts
)

# Metrics include:
# - long_total_return, short_total_return
# - long_volatility, short_volatility
# - long_hit_rate, short_hit_rate
# - long_return_contribution_pct, short_return_contribution_pct
```

---

## Configuration Examples

### Conservative Long/Short (60/40)

```yaml
execution:
  enabled: true
  transaction_cost_bps: 3.0
  slippage_bps: 1.0
  short_transaction_cost_bps: 4.0      # 1 bp premium
  short_slippage_multiplier: 1.0       # Same slippage
  short_borrow_cost_bps: 10.0          # Low borrow cost
```

### Aggressive Long/Short with Leverage

```yaml
execution:
  enabled: true
  transaction_cost_bps: 2.0
  slippage_bps: 0.5
  short_transaction_cost_bps: 3.0      # 1 bp premium
  short_slippage_multiplier: 1.5       # 50% more slippage
  short_borrow_cost_bps: 50.0          # Higher borrow cost
```

### Short-Heavy Strategy (Pairs Trading)

```yaml
execution:
  enabled: true
  transaction_cost_bps: 2.0
  long_transaction_cost_bps: 3.0       # 1 bp penalty for longs
  short_transaction_cost_bps: 2.0      # Preferred side
  short_slippage_multiplier: 0.8       # Better execution
  short_borrow_cost_bps: 5.0           # Low cost
```

---

## What Gets Calculated Automatically

When you enable directional features:

✅ **Costs**:
- Long transaction costs applied to positive position changes
- Short transaction costs applied to negative position changes  
- Short slippage calculated separately
- Borrow costs accumulated daily for short positions

✅ **Metrics**:
- Long/short return decomposition
- Side-specific volatility
- Position count by side
- Contribution to total return by side

✅ **Validation**:
- Constraint satisfaction checking
- Position/exposure limit enforcement
- Detailed violation reporting

---

## Backward Compatibility

**Existing configs work unchanged:**
```yaml
# This still works exactly as before
execution:
  enabled: true
  transaction_cost_bps: 3.0
  slippage_bps: 1.0
  # No directional fields = symmetric costs applied
```

**Existing optimizer configs work unchanged:**
```python
# This still works exactly as before
config = {"method": "equal_weight", "long_only": True}
```

---

## Common Patterns

### 1. Identify if Asymmetry is Configured
```python
from src.config.execution import ExecutionConfig

config = ExecutionConfig(...)
if config.has_directional_asymmetry:
    print("Asymmetric costs are enabled")
```

### 2. Get Effective Costs (with Fallback)
```python
# These automatically fall back to symmetric costs if directional not set
long_cost = config.get_long_transaction_cost_bps()
short_cost = config.get_short_transaction_cost_bps()
```

### 3. Validate Portfolio Meets Constraints
```python
from src.portfolio import validate_directional_constraints

result = validate_directional_constraints(weights, constraints)
if result["valid"]:
    print(f"Gross exposure: {result['metrics']['gross_exposure']:.2%}")
else:
    print("Violations:", result["violations"])
```

### 4. Decompose Returns by Side
```python
from src.portfolio import compute_long_short_directional_metrics

metrics = compute_long_short_directional_metrics(returns, positions)
long_pct = metrics.get("long_return_contribution_pct", 0)
short_pct = metrics.get("short_return_contribution_pct", 0)
print(f"Return attribution: Long {long_pct:.1f}%, Short {short_pct:.1f}%")
```

---

## Troubleshooting

### "Portfolio optimizer currently supports only long_only=True"
**Solution**: You're using an older version. This error has been removed. Make sure you have the latest code.

### Constraints not being enforced
**Solution**: Constraints are informational - you must call `validate_directional_constraints()` explicitly to check them. They don't automatically reject invalid portfolios.

### Output doesn't show long/short costs separately
**Solution**: Make sure `config.has_directional_asymmetry` is True. Check that you're using the `long_execution_cost` and `short_execution_cost` columns, not the placeholder `transaction_cost`/`slippage_cost` columns.

### Backward compatibility broken
**Solution**: This shouldn't happen. All new fields are optional with defaults. If existing configs fail, check that you're not mixing old/new config sources. File an issue if you find compatibility problems.

---

## Next Steps

For more advanced features currently in development:

1. **Position Constructor Asymmetry** - Configure max long/short positions per constructor
2. **Short Availability Constraints** - Model hard-to-borrow restrictions
3. **Signal Semantics Integration** - Validate signal+asymmetry compatibility
4. **Artifact Traceability** - Full audit trail of asymmetry decisions

See `ASYMMETRY_IMPLEMENTATION_STATUS.md` for implementation roadmap.

---

## Questions?

Reference the implementation status document for:
- Detailed API documentation
- Architecture diagrams
- Extended examples
- Roadmap for remaining features
