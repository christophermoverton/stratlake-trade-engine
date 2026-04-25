# Full-Year 2025 Regime Calibration Case Study

## Static Baseline
- The static portfolio baseline recorded total_return=0.022285812055291343 and sharpe=0.5210832301794449 before any regime-aware overlay.
- Raw M24 attribution identified `volatility=high_volatility|trend=downtrend|drawdown_recovery=recovery|stress=correlation_stress` as the strongest portfolio regime and `volatility=high_volatility|trend=uptrend|drawdown_recovery=recovery|stress=normal_stress` as the weakest.

## Calibration And Confidence
- Downstream calibration used the `baseline` profile; portfolio best-regime interpretation changed=true.
- The calibrated portfolio surface now highlights `volatility=high_volatility|trend=downtrend|drawdown_recovery=recovery|stress=dispersion_stress` as the strongest regime.
- GMM confidence flagged 0 low-confidence rows and 3 shift events, which were treated as a diagnostic layer rather than a taxonomy replacement.

## Adaptive Portfolio
- The adaptive overlay degraded the static baseline on total return by -0.005876084715474406 while moving Sharpe from 0.5210832301794449 to 0.4057155049839299.
- Static remained competitive when the fallback path was active on 23 rows, which limited how aggressively the adaptive policy could respond.

## Follow-Up Questions
- Should downstream policy use a more conservative calibration profile when GMM low-confidence share rises?
- Would symbol-level confidence or portfolio-component confidence improve adaptation timing versus market-level confidence only?
- Are the most adverse transition categories better handled by risk caps, allocation scaling, or explicit rebalance suppression?
