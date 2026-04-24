# Real Q1 2026 Regime-Aware Interpretation

## Strategy Surface
- The strongest strategy regime was `volatility=high_volatility|trend=sideways|drawdown_recovery=recovery|stress=normal_stress` with `total_return` of `0.009944287812721608`.
- The weakest strategy regime was `volatility=high_volatility|trend=sideways|drawdown_recovery=underwater|stress=normal_stress` with `total_return` of `-0.0033326061805848406`.
- Strategy fragility flag: `true`. Reason: most positive contribution comes from one dominant regime.

## Alpha Surface
- Across the compared Q1 2026 alpha runs, `ml_cross_sectional_lgbm_2026_q1` led the largest number of regime slices (`winner_count=7`).
- Robust regime leadership was observed for: ml_cross_sectional_lgbm_2026_q1.
- Candidate selection carried the following alpha runs into the portfolio path: ml_cross_sectional_xgb_2026_q1, rank_composite_momentum_2026_q1.

## Portfolio Surface
- The strongest portfolio regime was `volatility=low_volatility|trend=downtrend|drawdown_recovery=underwater|stress=normal_stress` with `total_return` of `0.015174656275256826`.
- The weakest portfolio regime was `volatility=low_volatility|trend=downtrend|drawdown_recovery=recovery|stress=normal_stress` with `total_return` of `-0.003646818401148222`.
- The most adverse portfolio transition category was `trend_breakdown` with metric value `0.001337844704837221`.

## Caution Boundaries
- These are descriptive regime slices over repository-available Q1 2026 research outputs, not causal claims about why performance changed.
- The classifier uses a fixed deterministic market basket from `features_daily`, so the regime labels should be read as a project-level context surface rather than a claim about the whole market.
- Sparse and empty regime slices remain visible and should limit confidence in any regime-specific interpretation.
