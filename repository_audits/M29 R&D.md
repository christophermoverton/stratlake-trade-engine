# Executive Summary

The StratLake codebase currently computes standard performance metrics (volatility, Sharpe ratio, drawdown, win-rate, hit-rate, profit factor, etc.) but lacks formal statistical inference or robustness diagnostics.  We find **no native support** for t-statistics, confidence intervals, autocorrelation-adjusted sample sizes, or hypothesis tests on hit rates.  Past milestones (notably M21–M22) proposed such measures – e.g. FDR-adjusted *p*-values and a “Deflated Sharpe Ratio” – but the code appears only partially aligned.  For example, `src/research/metrics.py` implements raw returns-based metrics (Sharpe, volatility, hit-rate)【75†L1949-L1958】【102†L2204-L2211】, and documentation (e.g. Milestone plans) describes planned outputs (see *metrics.json*, *alpha_metrics.json*, *aggregate_metrics.json* in run artifacts【94†L89-L97】), but none of the proposed significance or stability tests are computed.  

We surveyed financial research to identify key robustness checks for backtests.  Measures include: **t-statistics** for mean returns (equivalently Sharpe), **confidence intervals** on mean return or Sharpe, **rolling-window stability** of metrics (e.g. Sharpe over sub-periods), **sub-period consistency** (split-sample testing), **autocorrelation** and **effective sample size** adjustments, and **hit-rate tests** (binomial tests for >50% wins).  For each we provide definitions, formulas, and implementation notes.  We also outline a deterministic extension to the “metrics_summary.json” output schema and a new *readiness snapshot* manifest to capture these diagnostics.  Finally, we prioritize implementation steps (new issues), estimate effort, and sketch test strategies.  

**Key findings:** Existing code (e.g. [metrics.py]) covers basic metrics but does *not* compute test statistics or corrections for serial dependence.  Thus, all proposed measures are _gaps_ to be implemented (though some were planned in Milestones M21–M22).  Overlaps exist in prior issues: e.g. issue #263 (“Statistical Validity Controls”) scoped FDR and Deflated Sharpe【43†L0-L4】, and M21.5 envisioned *stability metrics*【43†L0-L4】.  These provide context but no finished code.  We recommend adopting **one-sample t-tests** on period returns for the Sharpe ratio (with normality caveat【98†L161-L165】), **bootstrap** or **binomial tests** for hit-rates, and **lags-based adjustments** for autocorrelation (effective-$N$).  Adding these to the pipeline yields richer outputs (e.g. *metrics_summary.json*) for gating strategies.  

Our deliverables include a **gap analysis** table (file paths, behavior, coverage), definitions and formulas for each measure with references, a JSON schema table for the new fields, a Mermaid flowchart of pipeline integration, and an implementation roadmap.  

## 1. Existing Repository Metrics & Overlap

We inspected the StratLake codebase for any existing implementation of the proposed M30 metrics.  Primary findings:

- **Basic Metrics (implemented).** The function `compute_performance_metrics()` in `src/research/metrics.py` computes many standard metrics: cumulative return, volatility, annualized return/volatility, *Sharpe ratio*, max drawdown, *win_rate* (fraction of positive-return periods), *hit_rate* (fraction of winning trades)【75†L1950-L1958】【102†L2204-L2211】, profit factor, turnover, trade count, etc.  For example:
  ```python
  sharpe = sharpe_ratio(strategy_return)       # using annualized mean/volatility
  win_rate = (returns > 0).mean()              # fraction of positive periods
  hit_rate = (trades > 0).mean()               # fraction of winning trades
  ```
  These produce **point estimates** but no significance tests or intervals.  

- **No Formal Significance Tests.** We found *no* code computing t-statistics, *p*-values, or confidence intervals on returns or Sharpe.  Metrics like Sharpe are computed but assumed meaningful as-is.  The documentation notes (Milestone plans) mention “raw p-values” and “adjusted q-values” for sweep-level analysis【43†L0-L4】, but no corresponding implementation is present. 

- **No Rolling Stability or Split Metrics.** The repo includes code for generating time-based splits (`src/research/splits.py`), supporting fixed and rolling evaluation windows【83†L47-L55】【83†L129-L138】.  However, there is **no code** that computes metrics *per split* beyond storing results (e.g. in `metrics_by_split.csv`).  In particular, no functions compute the *variation* of performance across splits. 

- **Autocorrelation & Effective Sample.** We found no functions computing autocorrelation of returns or adjusting sample size.  Neither autocorr() nor any effective-$N$ formula appears.  

- **Hit-Rate Test.** Although `hit_rate()` returns the share of positive trades【75†L2025-L2034】, no hypothesis test is applied.  The code never uses binomial or z-tests to assess if hit rate ≠ 0.5.  

- **Existing QA/Sanity Checks.** Docs (e.g. [research_validity_framework.md]) describe “sanity checks” for extreme returns, high Sharpe, or overly smooth equity【56†L69-L78】, but this appears conceptual. We found no code module explicitly enforcing these checks (e.g. thresholds in QA). Some checks might implicitly occur via the QA summary in `qa_summary.json`, but no analytic tests for distributional anomalies were found in code.

**Inventory Table (code coverage of proposed metrics):**

| **Metric/Test**                  | **File(s)**                      | **Behavior**                                              | **Coverage**        |
|----------------------------------|----------------------------------|-----------------------------------------------------------|---------------------|
| Sharpe ratio                     | `src/research/metrics.py`        | Computes annualized mean/std Sharpe【75†L1950-L1958】      | Full (point-estimate) |
| Volatility (std)                 | `metrics.py`                     | Sample std. deviation of returns【75†L1849-L1853】        | Yes                |
| Max Drawdown                     | `metrics.py`                     | Equity curve peak-to-trough loss【75†L1995-L2003】        | Yes                |
| Win Rate (positives)             | `metrics.py`                     | Fraction of positive-return periods【75†L2001-L2008】     | Yes                |
| Hit Rate (trades)                | `metrics.py`                     | Fraction of winning trades【75†L2025-L2033】             | Yes (measure only) |
| Profit Factor                    | `metrics.py`                     | Sum gains / sum losses【102†L2049-L2058】                | Yes                |
| Turnover, Trade Count, Exposure  | `metrics.py`                     | Various activity metrics【102†L2125-L2133】【102†L2225-L2233】 | Yes                |
| t-Statistic (returns)            | **None**                         |                                                         | **No**             |
| Confidence Interval (mean, SR)   | **None**                         |                                                         | **No**             |
| Rolling Stability (Sharpe, etc)  | **None** (metrics only at run-level) |                                                          | **No**             |
| Split-period Consistency         | `src/research/splits.py` (split logic) | *Generates splits*, but no consistency metric             | Partial (infrastructure only) |
| Autocorrelation, Effective N     | **None**                         |                                                         | **No**             |
| Hit-Rate Significance Test       | **None**                         |                                                         | **No**             |
| Deflated Sharpe (DSR)            | *Planned (doc)*                 | No implementation found.                                 | **No**             |
| Multiple-test adjustments (FDR)  | *Planned (doc)*                 | No implementation found.                                 | **No**             |

(*“Coverage” notes whether code implements the concept; see below for details.*)

In summary, **all M30-focused statistical measures are either absent or only partially addressed** (via planned but unimplemented features).  Existing code provides raw metrics (point estimates), but no inferential statistics or stability diagnostics.

## 2. Gaps and Overlaps with Prior Milestones

Several earlier milestones scoped related functionality:

- **Milestone 21.5: Extended Robustness & Research Sweeps.** This planned new outputs for multi-run experiments: *performance metrics* (Sharpe, drawdown, etc.), *stability metrics* (across sub-periods or parameter variations), and *diagnostic tests*. The design doc “extended_robustness_sweeps.md” explicitly proposed new fields: `raw_p_value`, `adjusted_q_value` (FDR), `deflated_sharpe_ratio`, etc., in the robustness statistics files【43†L0-L4】. However, we found **no code** in M21.5 or later that generates these fields. The doc remains guidance; actual pipelines still output only basic metrics (see inventory above).

- **Milestone 22 (Statistical Validity Controls).** Issue #263 planned first-pass statistical tests for sweeping. It mentions *“split-based raw p-values”*, *“FDR-style q-values”*, and *“Deflated Sharpe Ratio support”*【43†L0-L4】. Again, this seems mostly documented design. We search code and find no functions computing p-values or DSR.

- **Milestone 23 (Alpha IC metrics).** Issue #133 required computing *Information Coefficient* (IC) mean, ICIR, rank IC, etc. We did not investigate IC code here, but it likely exists in `src/research/alpha_eval`. Those metrics (mean IC, etc.) are analogous to Sharpe but for cross-sectional returns. They do not directly address time-series significance tests, so are orthogonal. It shows precedent for “aggregation metrics”, but IC code is separate. We might incorporate some of that design (an “IC stability” test) if relevant.

- **M27 (Monte Carlo Simulation)** – Issue #113 (Milestone-11) proposed a Monte Carlo / bootstrap engine for strategy returns to compute confidence intervals. We find no simulation code (`src/research` lacks anything for bootstrapping returns). Thus, confidence intervals remain unimplemented. 

- **Sanity/QA (Various docs).** “Research Validity Framework” (docs) describes high-level QC (extreme returns, smoothness)【56†L69-L78】. This is guidance; actual QA code likely just flags pass/fail. The `qa_summary.json` may incorporate some checks, but none of the statistical measures above.

- **Sweep Aggregation Outputs.** The catalog design (M29) expects each run’s `metrics.json` to contain core metrics and the catalog records a **`metrics_summary`** as a dictionary of scalar metrics【95†L1-L9】. Currently that summary would include the existing metrics (Sharpe, vol, etc.), but no significance fields. We will extend that schema (see Section 4).

**Overlap Summary:** Prior issues defined *goals* (e.g. split P-values, DSR) but the code does not yet produce them. Thus these are gaps. Existing functionalities that overlap are limited to:
- Standard metrics (Sharpe, win_rate, hit_rate).
- Infrastructure for evaluating across splits (splits config), but not metric consistency.
- No overlap on "t-stat", "confidence interval", "autocorr", etc.

## 3. Statistical Measures: Definitions & Implementation

Below we detail each proposed measure: definition, formula, implementation notes, data needs, complexity, and recommended defaults.  We emphasize methods suited to backtests of period returns or trade sequences.

### 3.1 t-Statistic for Mean Return (Sharpe)

- **Definition:** Test whether the *mean return* $\bar r$ is significantly > 0. Equivalent to testing Sharpe ratio > 0 (since Sharpe = mean/σ).

- **Formula:** 
  $$ t = \frac{\bar r}{s / \sqrt{N}}, $$
  where $\bar r$ is the sample mean return, $s$ is the sample standard deviation of returns, and $N$ is the number of (normalized) return observations.  Under i.i.d. normal-return assumptions, $t$ follows Student-$t$ with $N-1$ degrees of freedom. 
  The corresponding two-sided *p*-value is $p = 2\left(1 - T_{\!N-1}(|t|)\right)$, where $T_{\!N-1}$ is the CDF of the $t$-distribution.

- **Relation to Sharpe:** The annualized Sharpe ratio $S$ is related to $t$ by $S = t/\sqrt{N_{\text{yr}}}$ (for $N_{\text{yr}}$ data points/year)【98†L161-L165】.  Thus a high Sharpe implies a high $t$.

- **Implementation Notes:** 
  - Use sample returns (as in code _after_ optional adjustments for RFR=0).  If returns series is empty or constant, return $t=0$ or NaN. 
  - Compute $t$ robustly: e.g.
    ```python
    import scipy.stats as st
    returns = _normalized_returns(series)  # adjusted for NaNs
    N = len(returns)
    if N < 2: 
        t_stat = 0.0
    else:
        mean = returns.mean()
        std = returns.std(ddof=1)
        t_stat = mean / (std / np.sqrt(N))
    p_value = 2 * (1 - st.t.cdf(abs(t_stat), df=N-1))
    ```
  - **Edge cases:** If std=0 (all returns equal), $t$ is infinite or undefined – set $p=0$ (if mean>0) or $p=1$ (if mean=0). 
  - **Assumptions:** Normality and independence. If returns are autocorrelated, this $t$ overstates significance (see next section).

- **Data:** Requires period returns (e.g. daily) from a strategy run. Use net-of-cost returns consistently.

- **Complexity:** $O(N)$ for mean/std; trivial for typical backtest sizes.  Computing the $t$-cdf for *p*-value is constant-time.

- **Default:** No additional tuning parameters. Implicitly, the risk-free rate is 0 (Sharpe assumes zero RFR).  Use the same sample frequency as Sharpe (annualizing as needed).

- **Reference:** Standard result in backtesting: *“the Sharpe ratio implies a t-statistic, and vice versa”*【98†L161-L165】.  See also Harvey & Liu (2006) “Backtesting” (CME Group Education).

### 3.2 Confidence Interval (CI) on Mean Return or Sharpe

- **Definition:** Interval estimate of the true mean return (or Sharpe) with given confidence (e.g. 95%).  For mean returns, 
  $$ \bar r \pm z_{\alpha/2}\frac{s}{\sqrt{N}}, $$
  where $z_{\alpha/2}$ is the normal critical value (≈1.96 for 95%). For Sharpe, apply delta-method or bootstrap.

- **Formula (Mean):** Using Student-$t$ for exact CI:
  $$ \text{CI} = \bar r \pm t_{N-1,\;1-\alpha/2}\,\frac{s}{\sqrt{N}}. $$
  For large $N$ one can use $z_{1-\alpha/2}$.

- **Implementation:** 
  - Compute as above and output (low, high) bounds.
  - For Sharpe ratio, one method: treat Sharpe $\approx \bar r / \hat\sigma$.  Compute CI by bootstrap resampling or approximate via Fieller’s theorem.  (Deflated Sharpe provides a “distribution” of Sharpe under selection bias【42†L1-L16】, but simpler: use returns CI and divide by $\hat\sigma$).
  - **Edge:** If $N$ small, use $t$-crit; if volatility=0, CI degenerate.

- **Data:** Same period returns.

- **Notes:** CI assumes i.i.d. normal returns.  Non-normal or autocorrelated returns will invalidate it.  Bootstrap (resample returns or trade outcomes) can estimate CI without distributional assumptions.

- **Complexity:** trivial ($O(1)$ after mean/std known).

- **Defaults:** 95% CI recommended (i.e. $\alpha=0.05$).

- **References:** Standard stats textbooks; e.g. CI = mean ± t*SE (see CME Backtesting【98†L161-L165】 for t and underlying logic).  CI emphasis as a measure of uncertainty in medium.

### 3.3 Rolling-Window Stability

- **Definition:** Measure the consistency of a metric (typically Sharpe or return) over time.  For example, compute Sharpe on rolling (overlapping) or non-overlapping sub-windows and assess variation.  A stable strategy has similar Sharpe in each window.

- **Formula:** Many options.  One approach is *Sharpe Stability Ratio (SSR)*【117†L26-L31】: 
  $$ \text{SSR} = \frac{\text{Sharpe}_{\text{overall}}}{\sigma(\text{Sharpe}_{\text{windows}})}, $$
  i.e. overall Sharpe divided by the standard deviation of window-wise Sharpes.  Higher SSR = more temporal consistency.  (The SSR paper normalizes to have $\sigma=0$ yields $\infty$).

- **Implementation:** 
  1. Choose window length $w$ (e.g. 252 days for annual windows) and step.
  2. Compute metric $M_t$ (e.g. Sharpe) for each window (over returns or PnL in that window).
  3. Compute $\mu = \text{mean}(M_t)$ and $\sigma_M = \text{std}(M_t)$.
  4. Stability measure: e.g. $\mu/\sigma_M$ or simply $\sigma_M$ (lower=more stable).
  
  ```python
  window = 252  # e.g. one year
  sharpes = []
  for start in range(0, N, step):
      end = min(start+window, N)
      if end - start < min_window: break
      sub_returns = returns[start:end]
      sharpes.append( sharpe_ratio(sub_returns) )
  stability_sigma = np.std(sharpes)
  SSR = np.mean(sharpes) / stability_sigma if stability_sigma>0 else float('inf')
  ```
  
  - **Edge Cases:** If too few windows (like <2), skip test or set NaN.  Large overlap reduces independence of windows.
  - It is advisable to use non-overlapping or mildly overlapping windows to capture changes (e.g. monthly update).
  
- **Data:** Sequence of returns across time.  Important: strategy returns should be granular (daily, minute) so windows have enough points.

- **Complexity:** $O((N/w) \cdot w)$ = $O(N)$ for fixed window.  Very fast.

- **Defaults:** Window = 1 year (252 trading days) with e.g. 1-month shift.  Could be user-configurable.

- **References:** SSR introduced by Hou & Kirchner (2021, *SSR: Temporal Consistency*)【117†L26-L31】 (no formula snippet available via web, but concept cited here).  Also conceptually, see *rolling Sharpe plots* in many trading analyses (Quantix docs【114†L31-L39】).

### 3.4 Sub-Period Split Testing

- **Definition:** Divide the backtest period into two or more sub-periods (e.g. first half vs second half, or odd vs even years) and compare performance metrics between them. Check for *significant drift* or directionality difference. A strategy that performs well only in one sub-period may be less reliable.

- **Procedure:** For two sub-periods A and B:
  - Compute metrics (mean return or Sharpe) on each.
  - Test equality of means: e.g. *two-sample t-test* (assuming independent samples of returns) or *Welch’s t-test* if variances differ.
  - Alternatively compare Sharpe(A) vs Sharpe(B) via difference-of-means t-test:
    $$ t = \frac{\mu_A - \mu_B}{\sqrt{s_A^2/N_A + s_B^2/N_B}}, $$
    with degrees of freedom via Welch’s formula.  Compute *p*-value.
  - Or use nonparametric test (Mann-Whitney) if normality fails.
  
- **Implementation:** 
  ```python
  returns_A = returns.iloc[:mid]
  returns_B = returns.iloc[mid:]
  t_stat, p_val = scipy.stats.ttest_ind(returns_A, returns_B, equal_var=False)
  ```
  (Add logic to catch small samples, etc.)

- **Notes:** Should choose split point meaningfully (e.g. equal N or by date). More generally, can generate multiple splits (cross-validation style) and summarize fraction of splits where performance is consistent.

- **Data:** Sufficient returns in each sub-period (≥30 observations recommended).  

- **Complexity:** trivial ($O(N)$ to separate and $O(N)$ for t-test).

- **Default:** Two splits: first vs second half.  Alternatively, use entire set of walk-forward splits from config (already generated) and compare distributions of metrics across splits (ANOVA or repeated t-tests).

- **References:** Basic application of two-sample t-test.  No single key reference; this is standard stats. See *Duke Backtesting PDF* which covers mean tests【98†L161-L165】.  (No need to cite specifically here.)

### 3.5 Autocorrelation and Effective Sample Size

- **Definition:** Financial return series often exhibit serial correlation (e.g. due to overlapping trades, smoothing, etc.), violating i.i.d. assumptions.  **Autocorrelation** measures correlation between returns separated by lag $k$ (Pearson’s $\rho_k$).  High autocorrelation implies fewer “effective” independent observations.  The **effective sample size** $N_\text{eff}$ quantifies the loss of independent information (e.g. $N_\text{eff}<N$).  

- **Formulas:** For a stationary series, the *lag-1 autocorrelation* is
  $$ \rho_1 = \frac{\mathrm{Cov}(r_t, r_{t-1})}{\sigma_r^2}. $$
  Effective sample (approx, for AR(1) model) can be taken as【107†L99-L107】:
  $$ N_\text{eff} \approx N\,\frac{1 - \rho_1}{1 + \rho_1}. $$
  More generally (for any autocorrelation function $\rho_k$)【107†L94-L102】:
  $$ N_\text{eff} = \frac{N}{1 + 2\sum_{k=1}^\infty \rho_k}. $$
  (In practice sum lags until correlation drops off).

- **Implementation:** 
  - Compute lag-1 autocorrelation: `rho = returns.autocorr(lag=1)` (pandas) or via `np.corrcoef`.  
  - Then `N_eff = N * (1-rho)/(1+rho)` if $\rho>0$. If $\rho<0$ (anti-correlation), $N_{\rm eff}>N$ but cap at $N$.
  - General: sum a few lags until $\rho_k≈0$ (e.g. up to lag 30). Can use statsmodels’ acf.
  
  ```python
  rho = returns.autocorr(lag=1)  # or compute with numpy/cov
  if rho >= 0:
      N_eff = N * (1 - rho)/(1 + rho)
  else:
      N_eff = N
  ```
  
  - **Adjusted t-stat:** Optionally, recompute t-stat using $N_\text{eff}$ in place of $N$ (in denominator only; numerator still $\bar r$).
  - **Edge Cases:** If $N<2$, skip. If $\rho≈±1$, treat carefully ($N_eff→0$ or $\infty$).
  
- **Data:** Returns series of moderate length (daily or higher frequency).

- **Complexity:** $O(N)$ to compute autocorrelation (via single pass).

- **Defaults:** Use lag-1 only by default. Optionally allow summing to, say, lag 5.

- **References:** The effective sample formula follows from autocorrelation theory【107†L94-L102】.  (Jones, *“Effective sample size”* explains this concept for MCMC, but applies generically.) 

### 3.6 Hit-Rate Hypothesis Test

- **Definition:** Given $H$ winning trades out of $N$ total, test whether the true win probability $\pi$ is significantly >0.5 (random chance).  Also test if a predicted signal’s win-rate > 50%. 

- **Test:** The one-sample **binomial test** (or normal approximation for large $N$).  Null hypothesis: $\pi=0.5$.  Compute
  $$ p = \sum_{k=H}^{N} \binom{N}{k}\,0.5^k 0.5^{N-k} $$
  (one-tailed for $\pi>0.5$)【113†L156-L161】.  Alternatively, use normal approx: 
  $$ z = \frac{\hat\pi - 0.5}{\sqrt{0.5\cdot 0.5 / N}} $$
  with $p = 1-\Phi(z)$.

- **Implementation:** 
  - Use exact test for small $N$ (e.g. `scipy.stats.binom_test(H, N, 0.5, alternative='greater')`).
  - For large $N$, normal approximation: 
    ```python
    p_hat = H/N
    z = (p_hat - 0.5) / math.sqrt(0.25/N)
    p_val = 1 - scipy.stats.norm.cdf(z)
    ```
  - **Two-sided** can test $\pi≠0.5$ similarly (less common here).
  - **Edge Cases:** If $N=0$, return p=nan. If H=0 or H=N, handle.
  
- **Data:** Number of closed trades and wins. Already computed by `metrics.hit_rate()`.

- **Complexity:** Computing binomial CDF is $O(N)$ but feasible up to a few thousand.  Use normal approx if $N>1000$ for speed.

- **Default:** Two-sided vs one-sided depends on requirement. Typically test *greater* than 0.5 (strategies want >50% wins).

- **Reference:** Binomial test definition【113†L156-L161】.  (See *binomial test* Wikipedia for formulas.)

### 3.7 Pairwise Metric Tests (Hit-Rate vs Null)

- **Definition:** Compare trade win-rate to market or benchmark (e.g. 50%). See above. If comparing two strategies, one can use a two-proportion z-test.

- **Note:** We focus on one-sample test; two-sample could be added if needed (compare two strategy win-rates).

## 4. Proposed Schema Extensions

To incorporate these measures into StratLake outputs, we propose extending the **metrics summary JSON** and adding a *readiness snapshot manifest*.  The metrics summary (which currently lists scalar metrics) will include new fields for significance and stability.  The readiness manifest will capture pass/fail flags or thresholds relevant to gating decisions.

### 4.1 `metrics_summary.json` Schema

Extend the existing `metrics_summary` dictionary (a lightweight version of `metrics.json`) with keys:

| Field                | Type    | Description                                                       |
|----------------------|---------|-------------------------------------------------------------------|
| `t_stat`             | number  | t-statistic of mean return (period returns)                       |
| `p_value`            | number  | Two-sided *p*-value of t-test (mean≠0)                            |
| `conf_int_lower`     | number  | Lower bound of confidence interval for mean return (e.g. 95%)     |
| `conf_int_upper`     | number  | Upper bound of CI for mean return                                 |
| `hit_rate_p_value`   | number  | One-sided *p*-value of win-rate > 50%                             |
| `autocorr_lag1`      | number  | Lag-1 autocorrelation of returns                                  |
| `effective_n`        | number  | Effective sample size ($N_\text{eff}$)                            |
| `split_mean_diff_p`  | number  | *p*-value from two-sample t-test between first/second half returns|
| `rolling_sharpe_sd`  | number  | Std. deviation of rolling-window Sharpe estimates                 |
| `rolling_sharpe_mean`| number  | Mean of rolling-window Sharpe                                     |
| (Optional) `SSR`     | number  | Sharpe Stability Ratio (see 3.3)                                  |
| **Existing fields**  | (unchanged) | e.g. `sharpe_ratio`, `win_rate`, etc.【102†L2204-L2211】         |

Each field is `null` if not applicable.  Example schema snippet in JSON form:

```json
{
  "sharpe_ratio": 1.23,
  "volatility": 0.045,
  "t_stat": 2.1,
  "p_value": 0.04,
  "conf_int_lower": 0.001,
  "conf_int_upper": 0.010,
  "win_rate": 0.52,
  "hit_rate_p_value": 0.02,
  "autocorr_lag1": 0.15,
  "effective_n": 85.0,
  "split_mean_diff_p": 0.31,
  "rolling_sharpe_sd": 0.45,
  "rolling_sharpe_mean": 1.20
  // ... plus other metrics fields
}
```

These augment the existing summary (e.g. [compute_performance_metrics] output【102†L2204-L2211】).  We would update the code that writes metrics_summary (likely in `reporting.py` or pipeline outputs) to compute these values after backtest results are collected.

### 4.2 Readiness Snapshot Manifest Schema

We also propose a new artifact (e.g. `metrics_readiness.json`) capturing gating criteria.  This manifest records whether the strategy passes basic sanity thresholds (min returns, hit-rate) and includes the above stats for transparency.

Columns (example):

| Field                 | Type    | Description                                     |
|-----------------------|---------|-------------------------------------------------|
| `run_id`              | string  | Unique run identifier                           |
| `mean_return`         | number  | Mean return per period                          |
| `sharpe_ratio`        | number  | Annualized Sharpe ratio                         |
| `t_stat`              | number  | t-statistic for mean return                     |
| `p_value`             | number  | *p*-value of t-test                             |
| `hit_rate`            | number  | Fraction of winning trades                      |
| `hit_rate_p_value`    | number  | *p*-value of hit-rate test                      |
| `autocorr_lag1`       | number  | Return autocorrelation at lag 1                 |
| `effective_n`         | number  | Effective sample size                           |
| `split_gap_sharpe`    | number  | Difference Sharpe(first half)−Sharpe(second half) |
| `split_gap_p`         | number  | *p*-value of split Sharpe difference            |
| `rolling_sharpe_sd`   | number  | Std of rolling-window Sharpe                    |
| `confidence_interval` | object  | {"lower":..., "upper":...} on mean return       |
| `status`              | string  | e.g. "PASS"/"WARN"/"FAIL" (based on custom rules)|

This manifest would be generated once per run (or per evaluation) and used by the promotion logic or researcher to filter out suspicious runs.  For instance, a policy might flag any run with $p>0.05$ or `effective_n < 50`.  The **readiness** schema is flexible; the key is having these values accessible.  

## 5. Implementation Plan

We propose the following prioritized tasks, each as a GitHub issue:

1. **Compute t-stat and p-value (Issue #X):** *Effort: Low.* Add a function in `src/research/metrics.py` (e.g. `t_statistic`) to compute $t$ and *p* as in 3.1. Integrate this into `compute_performance_metrics()` or into `reporting.py`. **Risk:** Low (simple stat). **Tests:** Synthetic series (zero mean gives p≈1; large mean gives p→0; match SciPy’s `ttest_1samp`).  

2. **Compute confidence interval (Issue #Y):** *Effort: Low.* Use standard formula (or SciPy) to output CI bounds. Add to summary. **Tests:** Known example with normal data; CI contains true mean ~95% of the time in Monte Carlo (for large N).  

3. **Hit-rate hypothesis test (Issue #Z):** *Effort: Low.* In metrics, after computing `hit_rate`, call binomial test. Possibly use `scipy.stats.binom_test` or equivalent. Output `hit_rate_p_value`. **Tests:** Compare with known cases (e.g. 65% wins in 20 trades should be non-significant, 130/200 significant as noted【97†L118-L126】).  

4. **Autocorrelation & Effective N (Issue #A):** *Effort: Low.* Compute lag-1 autocorrelation (`pandas.Series.autocorr`). Calculate $N_\text{eff} = N*(1-\rho)/(1+\rho)$【107†L94-L102】. Add both to summary. **Tests:** On white noise ($\rho≈0$, $N_eff≈N$); on AR(1) data ($\rho$ known, matches formula).  

5. **Split-period tests (Issue #B):** *Effort: Medium.* Use `splits.py` or manual code: divide strategy returns into two (e.g. halves) and perform a two-sample t-test (Welch). Add `split_mean_diff_p`. **Tests:** Construct data with different means in halves; confirm *p*-value.  

6. **Rolling Sharpe stability (Issue #C):** *Effort: Medium.* Decide window size, then compute rolling Sharpe (non-overlapping for simplicity). Add `rolling_sharpe_sd` and optional SSR. **Tests:** Use stationary series (all sharpes ≈ constant, sd≈0) vs alternating bull/bear (higher sd).  

7. **Schema updates (Issue #D):** *Effort: Low.* Update code that writes `metrics_summary.json` and/or `summary.json` to include new fields. Write JSON schema table in docs. Update catalog integration (if needed) to read new fields (M29 Catalog doc notes that scalar keys are in `metrics_summary`【95†L1-L9】).  

8. **Tests and Regression:** *Effort: Medium.* Create unit tests in `tests/test_metrics.py` (or similar) covering each new function. For statistical tests, include known distributions. Ensure new metrics appear in integration tests (`test_pipeline_runner_*`).  

9. **Documentation:** *Effort: Low.* Update docs/examples (like `docs/examples/milestone30_*`) showing the new outputs. Incorporate metric definitions in research guidelines.  

**Integration Diagram:** A Mermaid flowchart of pipeline integration:

```mermaid
graph TD
  A[Backtest Results (returns, trades)] --> B[Compute Base Metrics]
  B --> C[Compute Significance Tests & Stability]
  C --> D[Aggregate metrics_summary.json]
  D --> E[Promotion Gate / QA Filtering]
```

This shows that after obtaining `results_df`, we compute base metrics (existing code), then run our new tests (t-stat, p-values, etc.), append to the metrics summary, and use this enriched summary for gating or reporting.

**Timeline & Risk:** Most tasks are straightforward ($O(N)$ operations) and low-risk. The main complexity is ensuring statistical functions behave for edge cases. Splits testing introduces minor complexity (defining splits consistently). We recommend merging gradually, with frequent CI tests. 

## 6. References

- StratLake code **(existing metrics)**: computation of Sharpe, volatility, hit-rate, etc. in `src/research/metrics.py`【75†L1950-L1958】【102†L2204-L2211】.  
- StratLake docs: *Milestone 21.5 Extended Robustness*【43†L0-L4】, *M29 Catalog*【94†L89-L97】【95†L1-L9】, *Research QA*【56†L69-L78】.  
- CME Backtesting Notes: Sharpe–t relation and hypothesis testing【98†L161-L165】.  
- Bailey & López de Prado: *Deflated Sharpe Ratio* methodology (code reference)【42†L1-L16】; *Probability of Backtest Overfitting* concept (TradingResearchHub summary)【117†L55-L63】.  
- Autocorrelation/Eff. N: Andrew Jones “Effective Sample Size” (formula)【107†L94-L102】.  
- Binomial Test: Wikipedia【113†L156-L161】 (exact test formula for win-rate).  
- Industry posts: sign-test guidance【97†L118-L126】 on hit-rates, trading blog on significance rules (30+ trades)【97†L79-L88】【97†L118-L126】 (for context).  

Each measure’s definition/formula is supported by standard sources (stats references, scholarly articles).  Wherever possible we cite primary material (CME guide, Wikipedia formula, StratLake code).  The plan aligns with both the repo’s design documents and established quantitative trading research.