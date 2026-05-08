# Executive Summary

Milestone 31 (“Statistical Readiness Promotion Policy Integration”) will integrate the diagnostic readiness artifacts from Milestone 30 into StratLake’s existing promotion workflow.  In M30 we added comprehensive statistical metrics (e.g. t‐stat, p‐value, autocorrelation, rolling Sharpe stability, etc.) and generated **metrics_readiness.json** manifests summarizing data quality (PASS/WARN/FAIL)【96†L7-L15】【96†L24-L32】.  M31 must **extend the promotion pipeline** so that strategies and portfolios are automatically classified as “promote,” “review,” “warn,” “reject,” or “block” based on those new diagnostics.  We will do this by enhancing the current promotion gate engine (which already evaluates numeric metrics via `src/research/promotion.py`) to incorporate readiness criteria.  This will primarily involve adding new promotion-gate rules (in `promotion_gates.json` configs) that reference the M30 metrics (e.g. `effective_n`, `p_value`, `autocorr_lag1`, etc.), without re‐implementing the whole gate engine.  The plan evaluates several architectures and recommends augmenting the existing gate definitions with readiness checks (“Option A” in the spec).  Key code components to modify include the promotion evaluator (`src/research/promotion.py`), the experiment output pipelines (`src/research/experiment_tracker.py`, `src/portfolio/artifacts.py`, `src/portfolio/walk_forward.py`, `src/research/alpha_eval/artifacts.py`), and relevant config files.  A detailed implementation roadmap is provided with task breakdowns, estimates, and a sprint timeline. Testing will cover unit and integration tests of the new gates (re‑using the existing `test_promotion_gates.py`) and validation in end‐to‐end backtests.  Risks include choosing appropriate thresholds, backwards compatibility, and ensuring deterministic outputs. Mitigations involve thorough testing, staged rollout of gate defaults, and clear documentation. 

**Sources:** The StratLake repository’s code and docs show the current promotion workflow (e.g. strategies write `metrics.json`, `metrics_readiness.json`, and evaluate promotion gates in `experiment_tracker.py`【76†L648-L654】【95†L644-L648】) and describe M30 deliverables【96†L7-L15】【96†L24-L32】.  

## Milestone 31 Scope and Requirements

Milestone 31 extends M30 by **integrating the readiness diagnostics into the promotion decision process**.  M30 produced advisory artifacts (`metrics_readiness.json`) alongside standard metrics (strategy/portfolio `metrics.json`)【96†L7-L15】【96†L24-L32】.  M31 must classify each candidate (strategy or portfolio run) into review categories (“should be promoted”, “review”, “warn”, “reject/block”) *and explain why*, using both the existing promotion rules and the new readiness data. Key points from the spec document include:

- **Context:**  The system already supports promotion rules via `promotion_gates.json` and evaluates them in the Experiment Tracker and portfolio runner. We must augment this without replacing it.  Promotion artifacts include a gated JSON (`promotion_gates.json`) that currently reflects simple metric thresholds. M31 adds statistically-informed criteria (e.g. sample size, p-value availability, autocorrelation) to those criteria.
- **Inputs:** M30 outputs, notably the metrics payloads and readiness manifests (fields like `effective_n`, `p_value`, `autocorr_lag1`, etc.)【96†L7-L15】【96†L24-L32】. Also the existing configuration and manifest infrastructure (e.g. `promotion_gates.json`, experiment `manifest.json`).
- **Acceptance Criteria:** A complete proposal should specify the updated milestone, its deliverables (new gate definitions, updated artifacts), criteria (e.g. no new nondeterminism, JSON-safe outputs, passing tests), and design notes. Implementation includes both code and documentation changes. All existing promotion workflows (strategy, portfolio, alpha) must produce correct gating outcomes that account for readiness metrics.
- **Non-Goals:** The spec explicitly excludes complex statistical corrections (FDR, etc.) in M31 – we only gate on readiness flags and underlying metric thresholds.  Also, the promotion architecture itself is reused, not rewritten.
- **Related Issues/PRs:** No explicit “M31” issue was found in the repository, but existing code references can be found in the release notes and docs (e.g. `docs/m30_release_notes.md`【96†L7-L15】【96†L24-L32】, `portfolio_artifact_logging.md`【91†L236-L252】). The example benchmarks and tests (e.g. `test_promotion_gates.py`) will need updates for new gates.
- **Design References:** The attached spec suggests multiple options (see “Design Options” below). The repository’s promotion code (`src/research/promotion.py`) forms the core integration point【112†L13-L22】【107†L254-L263】.

## Current Promotion and Readiness Architecture

StratLake’s promotion pipeline already *evaluates promotion rules on experiment outputs*.  In strategy runs (`run_strategy.py`), after backtesting the strategy, `experiment_tracker.save_experiment_outputs` writes `metrics.json`, `metrics_readiness.json`, and then calls `evaluate_promotion_gates` to produce `promotion_gates.json`【76†L648-L654】.  For walk-forward strategy runs and portfolio runs, similar logic applies: each split outputs `metrics.json` and `metrics_readiness.json`, then runs the gate evaluator on those metrics【95†L644-L648】【76†L648-L654】.  The `evaluate_promotion_gates` function (in `src/research/promotion.py`) loads a configured gate file and applies each gate to data in `sources` (e.g. the metrics dict, QA summaries)【112†L13-L22】【107†L322-L331】.  Gates are defined by (source, metric_path, comparator, threshold) and yield PASS/FAIL, producing an overall promotion status (eligible/blocked) based on whether any gate failed【105†L127-L137】【112†L13-L22】. 

Key code paths:

- **Strategy runs:** In `src/research/experiment_tracker.py`, `_prepare_run_outputs` appends a promotion evaluation to the payload if `config["promotion_gates"]` is set, then `write_run_outputs` writes `promotion_gates.json` via `write_promotion_gate_artifact`【76†L648-L654】【73†L730-L739】. The manifest and registry entries include the promotion summary【73†L780-L789】【76†L665-L673】.
- **Portfolio runs:** In `src/portfolio/artifacts.py`, after computing portfolio metrics and QA, `evaluate_promotion_gates` is invoked (run_type="portfolio") and the artifact is saved【63†L64-L72】. In walk-forward (`src/portfolio/walk_forward.py`), each split’s directory gets `metrics.json`, `metrics_readiness.json`, and `qa_summary.json`【95†L644-L648】, then after all splits the manifest includes links to these.
- **Alpha evaluations:** In `src/research/alpha_eval/artifacts.py`, after generating IC timeseries and QA, `evaluate_promotion_gates` is called with run_type="alpha_evaluation"【88†L97-L105】 and the result is written.

The **metrics_readiness** manifests are advisory summaries (status PASS/WARN/FAIL) derived from the metrics.  They are generated by `src/research/metrics.build_metrics_readiness_manifest` and written as JSON【101†L124-L133】. Example fields include `schema_version`, `status`, `diagnostics` (the raw metric values), and individual “checks” like “return_p_value_available”【101†L46-L55】【101†L72-L80】. By design, readiness status is advisory (see M30 notes: *“Readiness manifests are advisory review artifacts, not hard promotion gates.”*【96†L30-L34】).

Together, these components form the basis for adding M31 logic.  Importantly, **the existing gate engine only reads numeric values** from sources. It supports sources `{metrics, qa_summary, manifest, config, metadata, split_metrics, timeseries, aggregate_metrics}`【112†L16-L25】, and compares numeric metric values to thresholds. It does *not* read the `metrics_readiness.json` file; instead, we will feed readiness by referencing the underlying metrics themselves. 

## Integration Design Options

The spec considered several approaches to fusing readiness with promotion.  We evaluate each:

- **Option A: Extend existing promotion_gates.json with readiness checks.**  
  *Approach:* Add new gates in `promotion_gates.json` that directly reference M30 metrics (e.g. `"autocorr_lag1"`, `"effective_n"`, `"split_mean_diff_p"`, etc.), so that failing those gates triggers promotion failure or a “needs review” status. For example, a rule might require `effective_n >= 30` or `p_value <= 0.05`. Because the gate engine already handles numeric thresholds, this fits well. *Pros:* Reuses existing engine, one config file, clear artifact (promotion_gates.json) shows all gating. Easy to document as part of promotion rules. *Cons:* We must define appropriate numeric thresholds (policy decision), and ensure the logic correctly distinguishes “warn” vs “fail” (the engine only outputs PASS/FAIL per gate). We could encode a WARN by setting the gate’s `status_on_fail` to “needs_review” instead of “blocked”. This option does not output a separate “readiness” artifact; readiness is implicit in gate results.

- **Option B: Create a separate “promotion_readiness” artifact/decision.**  
  *Approach:* Run one set of gates for normal criteria, another for statistical readiness, then merge them. E.g. write a `promotion_readiness.json` alongside `promotion_gates.json`, then a combining step picks a final status. *Pros:* Clear separation between functional gates and statistical gates. *Cons:* Introduces complexity – duplicating the gating logic or running the engine twice, handling two outputs, merging logic. Hard to keep atomic and deterministic. Not aligned with “one execution, one artifact” principle. More complex testing and user confusion likely.

- **Option C: New config file for readiness thresholds.**  
  *Approach:* Define a separate YAML (e.g. `configs/statistical_readiness_policy.yml`) that lists the thresholds for metrics (independent of promotion_gates.json). The runtime would load both promotion_gates and readiness policy, then incorporate. *Pros:* Decouples policy (could be edited by quant researchers) from promotion rules. *Cons:* Similar complexity to B, plus integration logic. The current engine doesn’t support two configs. Would need code to merge or coordinate two configs, increasing risk of misalignment.

- **Option D: Review-level gating or campaign selection.**  
  *Approach:* Leave promotion engine unchanged, but enforce readiness rules at campaign selection or manual review time (e.g. a candidate is flagged if readiness FAILs and human must override). *Pros:* Minimal code change. *Cons:* Does not automate “should be warned or rejected” categories. Contradicts requirement to classify automatically “warn/reject”.

**Comparison:** Option A has the best architectural fit: it keeps the artifact-driven workflow and leverages existing gating code (supported sources include all metrics values). Options B/C duplicate infrastructure and break “single decisive artifact” principle. Option D fails to meet the automation goals. Therefore, we **recommend Option A**: augment the `promotion_gates.json` schema with new gates covering readiness metrics. In this approach, we might define certain gates whose failure maps to a “WARN” status instead of a hard “BLOCK”. For example, gates on autocorrelation or split stability could be `missing_behavior: skip` (ignore missing diagnostics) and `status_on_fail: "needs_review"`, so that failing only raises a review flag【103†L1086-L1093】【112†L28-L33】. 

We will document these gates and justify thresholds. For instance, a gate `effective_n >= 30` could produce a WARN if not met (since small sample size undermines statistical significance). A gate requiring `p_value <= 0.05` could be FAIL for promotion. Rolling Sharpe stability could be WARN if `rolling_sharpe_sd` is too high. Ultimately, the promotion artifact will still be one JSON with definitions of all gates (old and new)【112†L13-L22】【107†L251-L260】.

## Implementation Plan

We break down the work into epics and tasks. Below is a high-level task list with estimates:

| **Task**                                    | **Area**             | **Effort** | **Assignees/Roles** | **Dependencies/Notes**              |
|--------------------------------------------|----------------------|-----------:|---------------------|-------------------------------------|
| **1. Code Audit and Design**                | Architecture, Design | 2 days    | Architect, Dev      | Review current promotion code and decide gate set. (Using code above【76†L648-L654】【107†L322-L331】.) |
| 1.1 Review existing gate definitions and usage (strategy, portfolio, alpha) | Research / Dev    | 0.5d | Dev | Map where promotion gates are loaded and applied. |
| 1.2 Identify needed new metric fields and gate behaviors (pass/warn thresholds) | Research         | 1d   | Dev/Quant | Survey metrics from M30 (e.g. `effective_n`, `p_value`, `split_mean_diff_p`, `autocorr_lag1`, rolling Sharpe stats)【101†L46-L55】【103†L1086-L1093】. |
| 1.3 Design promotion gate config updates (policy decisions) | Policy / Dev    | 0.5d | PM/Lead | Draft new gate rules and status mappings. |
| **2. Update Promotion Gate Configs**        | Configuration        | 1 day     | Dev                | Create or update `promotion_gates.json` (e.g. default configs in `configs/`) to include new gates for readiness metrics. |
| 2.1 Add gates for `effective_n`, `autocorr_lag1` thresholds | Config          | 0.5d | Dev | Likely require `effective_n >= 30` (or WARN) and `|autocorr_lag1| < some` (no autocorrelation). |
| 2.2 Add gates for split-consistency: `split_mean_diff_p` | Config | 0.25d | Dev | Eg. require split mean diff not significant. |
| 2.3 Add gates for rolling Sharpe: `rolling_sharpe_sd`, `sharpe_stability_ratio` | Config | 0.25d | Dev | Eg. `rolling_sharpe_sd < X`. |
| 2.4 Review/update `status_on_fail/pass` to produce “eligible”/“needs_review” appropriately. | Config/Dev | 0.5d | Dev | Ensure WARN gates map to “needs_review” (by setting `status_on_fail` to “needs_review” for those gates) vs PASS. |
| **3. Code Changes (if needed)**             | Implementation       | 1–2 days  | Dev                | Mostly none expected, but possibly minor tweaks. |
| 3.1 Modify promotion evaluation logic (if needed) | Dev             | 0.5d | Dev | Possibly allow `missing_behavior: skip` by default for these gates (already supported)【107†L332-L341】; ensure gating on QA or metadata if needed. |
| 3.2 Update registry or metadata builder (likely none) | Dev        | 0d | Dev | The registry already records `promotion_status` and reasons【110†L89-L98】. No change unless new status labels. |
| **4. Documentation**                        | Docs, Examples      | 1–1.5 days | Dev/Tech Writer  |  |
| 4.1 Update docs (e.g. `docs/milestone_31_*.md`) | Docs        | 0.5d | Writer | Document new milestone plan, gate meanings. |
| 4.2 Update examples/campaigns with new gate use | Examples      | 0.5d | Dev | Add example config, show output of promotion_gates. |
| 4.3 Update `README` or workflows docs to mention new criteria | Docs | 0.5d | Writer | Likely small note in strategy/portfolio workflow docs. |
| **5. Testing**                             | QA                  | 2 days    | QA/Dev           |  |
| 5.1 Unit tests for new promotion gates (include failure and skip cases) | Testing    | 1d | QA | Extend `tests/test_promotion_gates.py` to include new metric cases (failing split, autocorr, etc.). |
| 5.2 Integration tests: run strategy and portfolio with known data to verify gating outcomes | Testing | 0.5d | QA | Could reuse existing backtest fixtures, modify metrics to trigger WARN/FAIL. |
| 5.3 End-to-end campaign test (e.g. real_alpha_workflow) to ensure new statuses appear in registry. | Testing | 0.5d | QA | Check registry entry includes `promotion_status` “eligible” or “needs_review” appropriately. |
| **6. Release Preparation**                  | Release Management  | 0.5–1 day | Dev/DevOps       | Merge to main branch, update changelog. |
| 6.1 Prepare release notes for M31, update version/tag. | Dev      | 0.5d | Dev | Summarize M31 features, citing changes. |
| 6.2 CI/CD: update pipelines if needed (none expected). | DevOps     | 0d   | DevOps | Possibly add any new test targets (already existing tests cover). |

**Effort Summary:** Approximately 7–9 person-days total. We plan this over 2 sprints: Sprint 1 for design, config and initial code changes; Sprint 2 for documentation and testing. If the team is larger, tasks can overlap. No strict deadlines given, but we assume “milestone 31” is the next release cycle.

## Dependencies and Constraints

- **Internal modules:** We depend on the existing **promotion** and **metrics** modules. No new libraries are needed; all gating logic reuses current code (which uses pandas, scipy, etc., already in the project). The gating config and metrics are JSON/YAML, so no format changes.
- **Data Feeds:** None new. The necessary metrics come from the same backtest data used in M30.
- **Version constraints:** We should ensure compatibility with the repository’s current versions of pandas (for groupby in walk-forward), SciPy (for p-values), etc. M30 code is already passing tests, so use the same versions. We must strictly maintain deterministic JSON output (`sort_keys=True, allow_nan=False`) as enforced in `experiment_tracker` and `metrics` writers【76†L648-L654】【101†L137-L145】.
- **Architectural:** Must respect “artifact-first” and one execution path. We will not call any external service or change data contracts. All new outputs (if any) will be JSON or CSV with sorted keys.

## Design Details

### New Promotion Gate Rules

We will design gates around the M30 diagnostics. For example:

- **Minimum Effective N (Sample Size):** Gate ID `min_effective_n`. Source: `metrics`, metric_path `"effective_n"`, comparator `"gte"`, threshold `30.0` (or configurable). If effective_n is below threshold, status_on_fail = `"needs_review"` (to indicate warning rather than block).  
- **Return Inference Availability:** Gate ID `return_p_value_available`. Source: `metrics`, metric_path `"p_value"`, comparator `"ne"`, threshold `null` (i.e. check existence). Or use missing_behavior `"skip"` with comparator `"eq" 0` (not ideal). Alternatively, require `p_value <= 0.05` as block.  
- **Autocorrelation Check:** Gate ID `autocorr_lag1_low`. Source: `metrics`, metric_path `"autocorr_lag1"`, comparator `"lte"`, threshold (some small value, e.g. `0.01`). If autocorr is high, likely violates independence.  
- **Split Consistency:** Gate ID `split_p_value_ok`. Source: `metrics` (or possibly `split_metrics` if available), metric_path `"split_mean_diff_p"`, comparator `"gte"`, threshold (e.g. `0.05` meaning no significant difference).  
- **Rolling Sharpe Stability:** Gate ID `rolling_sharpe_stable`. Source: `metrics`, metric_path `"rolling_sharpe_sd"`, comparator `"lte"`, threshold (e.g. `0.5` or relative to mean).  
- **Stability Ratio:** Gate ID `sharpe_stability_ratio`. Source: `metrics`, metric_path `"sharpe_stability_ratio"`, comparator `"gte"`, threshold `1.0`. (If stability ratio <1, warn).  
- **Hit Rate P-Value:** Gate ID `hit_rate_significance`. Source: `metrics`, metric_path `"hit_rate_p_value"`, comparator `"lte"`, threshold `0.05`. If fails, maybe block (if strategy’s trades seem random).  

These are examples; exact values would come from statistical policy decisions. Each gate’s `status_on_fail` can be set to either `"eligible"` or `"needs_review"` (the latter defined in code as a potential status by the registry)【110†L21-L22】【110†L85-L94】. The overall promotion status in the summary is then chosen based on whether any gate failed or skipped【105†L176-L184】.

### Code Changes Summary

No major new modules are required. Most changes are in configuration and documentation. However, we will:

- **Promotion Evaluator (`src/research/promotion.py`):** No code changes unless special handling is needed. We may verify that all new metric paths (`autocorr_lag1`, etc.) are accessible. For split-level checks, note that strategy runs include `split_metrics` source if walk-forward, so gates with source `"split_metrics"` could use those values【73†L730-L739】【107†L327-L336】. For simplicity, we can treat each split separately or aggregate. 
- **Experiment Tracker (`src/research/experiment_tracker.py`):** Already writes metrics and promotion output【76†L648-L654】. We may need to ensure the promotion gates config digest (for registry) includes our new rules. That code uses `promotion_gate_config_digest`【105†L214-L223】 which automatically captures all gate definitions. No code change.
- **Portfolio Artifacts (`src/portfolio/artifacts.py`, `walk_forward.py`):** Already calls `evaluate_promotion_gates` and writes the JSON【63†L64-L72】【95†L644-L648】. We should ensure CLI options for promotion gates are still respected (run_portfolio accepts `--promotion-gates` which loads a config【59†L335-L339】). No code change needed unless we want to add CLI docs.
- **Alpha Evaluation Artifacts:** Already integrated (as seen in [88] where it passes `promotion_gate_config` to writer). No code change expected.

**Code Snippet (Example):** Below is an illustrative example of adding a gate to `promotion_gates.json` (YAML) for minimum effective N:

```yaml
promotion_gates:
  status_on_pass: eligible
  status_on_fail: blocked
  gates:
    - gate_id: min_effective_n
      source: metrics
      metric_path: effective_n
      comparator: gte
      threshold: 30.0
      # On fail, set needs_review instead of blocking:
      missing_behavior: skip
      description: "Require at least 30 effective samples."
```

For a WARN gate, we set `status_on_fail: needs_review` at the config level or per gate (if supported). If per-gate mapping isn’t supported, we can simulate warns by having two promotion_gates configs (one default “eligible/blocked” and another “eligible/needs_review”) or by merging a list with different status flags. The cleanest is to add a second YAML that is identical except with `status_on_fail: needs_review`, and use that to annotate. (This could be an implementation detail decided during the code audit.)

## Risks and Mitigations

- **Threshold Selection:** Setting too-strict thresholds could block good strategies. *Mitigation:* Use statistical rationale (e.g. 95% CI). Possibly allow customization. Document assumptions.
- **Coverage of Metrics:** If some runs lack certain metrics (e.g. no `split` data), gates might be skipped. Use `missing_behavior: skip` for those gates【107†L332-L341】 so missing data does not falsely block a strategy.
- **Backward Compatibility:** Existing gate configs might not list readiness fields. Since we only add, old configs still work. The engine remains deterministic. Ensure new keys don’t break JSON sorting or serialization (they won’t). 
- **Complexity:** Introducing multiple gates may confuse users. *Mitigation:* Document each gate clearly in comments/descriptions.
- **Testing Robustness:** We rely on existing test suites. We must extend them to cover new gates. If not done carefully, we risk false positives/negatives. *Mitigation:* Write targeted tests for each new gate and combination (see Testing plan).
- **Implementation Gaps:** The code currently doesn’t natively interpret a “needs_review” status (it only knows pass/fail). However, the registry’s `build_review_metadata` has logic to map a promotion_status to a review `status` like `"needs_review"`【110†L85-L94】. We must ensure our status strings align (e.g. using `"needs_review"` exactly). The default `status_on_pass`/`on_fail` strings might need adjusting in config.
  
## Testing Strategy

- **Unit Tests:** Extend `tests/test_promotion_gates.py` (existing) to include cases for the new gates. For example, create a dummy `sources = {"metrics": {...}}` dict with values under/above thresholds and verify `evaluate_promotion_gates` yields expected gate statuses. Also test the case where values are missing (`None`) to exercise `missing_behavior`.
- **Integration Tests:** Use small synthetic strategies to trigger edge cases: e.g. a strategy with zero trades (hit_rate p-value missing), or with a short period (effective_n small). Verify the promotion summary in the manifest has `promotion_status` set to `"eligible"` or `"blocked"` or `"needs_review"` appropriately. Compare against expected gating logic.
- **CI Changes:** No major CI changes – the existing tests should cover strategy/portfolio runs and registry. We may add a quick check in `tests/test_experiment_tracker.py` to ensure `promotion_gates.json` contains our new fields in actual runs. We can re-run the entire suite with the added tests to see no regressions.
- **End-to-End:** Run a full campaign with an example strategy to ensure outputs. Check `registry.jsonl` entries (via `build_review_metadata`) include the new `promotion_status` and “decision_reason” fields reflecting the readiness check (the registry code will automatically include `promotion_status` from the promotion summary【110†L89-L98】).
- **Validation:** After implementation, validate with Milestone 30 example tests (the ones in `test_walk_forward.py`, etc.) to ensure we haven’t broken M30 functionality. The execution should remain deterministic (stable JSON, no nan).

## Timeline (Proposed Gantt)

```mermaid
gantt
    dateFormat  MM-DD
    section Design & Planning
    Review current gate system              :done,   des1, 05-01, 1d
    Define new gate criteria & thresholds   :active, des2, 05-02, 2d
    section Development
    Update gate config files                :         dev1, 05-04, 1d
    Code tweaks for new gate evaluation     :         dev2, 05-05, 1d
    section Documentation
    Write milestone spec and docs           :         doc1, 05-06, 1d
    Update examples & READMEs              :         doc2, 05-07, 1d
    section Testing
    Unit tests for promotion gates         :         test1, 05-08, 1d
    Integration tests (strategy/portfolio)  :         test2, 05-09, 1d
    section Release
    Final review and release prep           :         rel1, 05-10, 0.5d
    ```
*(Dates are illustrative, spanning roughly one week of work assuming start 2026-05-01. Actual timeline would align to your sprint schedule.)*

## Conclusions

Milestone 31 will involve **configuring** and **documenting** new promotion rules that incorporate the Milestone 30 statistical readiness diagnostics.  The existing code is already structured for promotion gating【112†L13-L22】【107†L322-L331】, so implementation chiefly means augmenting `promotion_gates.json` and ensuring the outputs flow through the manifest and registry correctly.  By following Option A (extending the current gate definitions) we keep the design simple and consistent. The tasks above outline the required code and config changes, testing, and timeline. With clear criteria and thorough validation, the system will classify candidates using the new statistical checks without reengineering the promotion framework.

**Sources:** Official repository code and docs (especially `experiment_tracker.py`, `promotion.py`, `metrics.py`, and release notes) were used to understand current workflows and M30 deliverables【76†L648-L654】【95†L644-L648】【101†L46-L55】【103†L1086-L1093】. These informed the design of M31 integration.