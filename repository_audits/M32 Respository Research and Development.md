# Executive Summary

Milestone 32 (M32) extends StratLake’s governance by building a deterministic observability layer on top of the existing promotion artifacts from M31. We first inventory the relevant artifacts in the `stratlake-trade-engine` repo (promotion files, registries, reviews, campaigns, portfolios), summarizing their locations, schemas, and example values. We then analyze gaps – missing files or format mismatches – that M32 components will need to address (e.g. normalizing status strings, parsing nested JSON fields). We propose a concrete module layout under `src/research/governance/`, including an *artifact loader*, *aggregator*, *validator*, and *report writer* (with a CLI entrypoint) that consumes artifacts in a deterministic order. We refine schemas for the new governance reports (`promotion_governance_summary.json`, `promotion_outcome_matrix.csv`, `consistency_validation.json`) to align with actual fields (e.g. using the `promotion_status` and run metadata in the repo). We also enumerate consistency-check rules with pseudocode referencing real field names. Finally, we suggest unit/integration tests and a small synthetic case study (with a mermaid diagram showing the data flow), and draft a list of prioritized GitHub issues to implement M32 in the repo. 

Key findings: *Promotion artifacts* (`promotion_gates.json` and its summary in manifests) are already well-defined by M31【31†L122-L130】【38†L831-L834】. The **registry** (`registry.jsonl`) is a JSON-lines table under `artifacts/strategies` (and `artifacts/portfolios`) with one entry per run, including fields like `run_id`, `run_type`, `timestamp`, plus *promotion* and *review* fields【58†L76-L84】【58†L148-L156】. The **review summary** (`review_summary.json`) is a JSON file under `artifacts/reviews/<review_id>/` containing keys like `review_id`, `filters`, and a list of run entries (fields shown in [79]). **Campaign** artifacts live under `artifacts/research_campaigns/<campaign_id>/` (e.g. `campaign_config.json`, `checkpoint.json`, `summary.json`, plus a `milestone_report/` folder) as shown in the example manifest【70†L0-L10】. **Portfolio** artifacts live under `artifacts/portfolios/<run_id>/` (see table below). 

Many required fields already exist (e.g. `promotion_status` in manifests/registries, `promotion_gate_summary` in run manifests【81†L831-L834】, M30 metrics in `metrics.json`) so no new data collection is needed. Gaps include: no standalone **promotion_gate_summary.json** file (it’s only in manifests/registry), no explicit inventory of campaign runs, and some inconsistent naming (e.g. registry `promotion_status` vs. manifest summary). M32 will normalize statuses (e.g. map legacy vs. severity labels), parse nested fields (like manifest lists), and consolidate data across artifacts. 

The proposed implementation will live in `src/research/governance/`, with modules like `loader.py` (loading JSON/CSV artifacts), `aggregator.py` (combining data), `validator.py` (checks), and `writer.py` (emitting reports). We define schemas for the governance reports: for example, `promotion_governance_summary.json` might include total runs and counts by status per run type, `promotion_outcome_matrix.csv` is a cross-tab of run type vs. promotion status, and `consistency_validation.json` lists each check (e.g. unique run IDs, matching counts). We provide example outputs matching actual fields (using `run_id`, `run_type`, `promotion_status`, etc.). Deterministic ordering rules (sort keys, stable CSV ordering by run type) are specified. 

We enumerate consistency rules such as “every run in registry appears in at most one review” or “promotion status in manifest matches registry’s summary” with SQL-like pseudocode on real field names. For testing, we propose fixtures (small JSON and CSV examples) and a mermaid diagram illustrating the dataflow (runs → aggregator → reports). Finally, we list candidate GitHub issues (titles, descriptions, labels, effort) for implementing each M32 subtask. 

**Sources:** All details are drawn from the StratLake repo code and docs. Key files: `src/research/promotion.py` (promotion_gates schema)【31†L122-L130】【32†L206-L215】, `src/research/experiment_tracker.py` (manifest and registry writing)【81†L831-L834】【58†L146-L154】, `src/research/review.py` (review summary schema)【76†L315-L324】【79†L72-L80】, and example outputs (e.g. campaign manifest)【70†L0-L10】. 

## Inventory of Existing Promotion and Related Artifacts

We list each artifact, its repository path, schema fields, and sample values (when available). 

- **promotion_gates.json** – Emitted by strategy, portfolio, and review workflows【27†L64-L72】.  
  - *Path:* `<artifacts_root>/strategies/<run_id>/promotion_gates.json` (or similarly under `artifacts/portfolios/`).  
  - *Schema:* Defined by `PromotionGateEvaluation.to_payload()`. Fields include:  
    - `artifact_filename` (string, e.g. `"promotion_gates.json"`)  
    - `configured` (bool)  
    - `run_type` (string, e.g. `"strategy"`)  
    - `evaluation_status` (string, `"pass"` if no failures)  
    - `promotion_status` (string or null, final status based on severity)  
    - `status_on_pass`, `status_on_fail` (strings)  
    - `gate_count`, `passed_gate_count`, `failed_gate_count`, `missing_gate_count` (ints)  
    - `highest_severity` (string or null)  
    - `severity_counts` (dict: counts of each severity level)  
    - `warning_gate_count`, `review_gate_count`, `rejected_gate_count`, `blocked_gate_count` (ints)  
    - `decision_reason_codes` (list of strings)  
    - `definitions` (list of gate definitions) and `results` (list of gate results) *[excluded from summary]*.  
  - *Example:* (from Milestone 13 example【7†L0-L8】)  
    ```
    {
      "artifact_filename": "promotion_gates.json",
      "configured": true,
      "run_type": "review",
      "evaluation_status": "pass",
      "promotion_status": "review_ready",
      "status_on_pass": "eligible",
      "status_on_fail": "needs_work",
      "gate_count": 2,
      "passed_gate_count": 2,
      "failed_gate_count": 0,
      "missing_gate_count": 0,
      "highest_severity": null,
      "severity_counts": {"warn": 0, "review": 0, "reject": 0, "block": 0},
      "warning_gate_count": 0,
      "review_gate_count": 0,
      "rejected_gate_count": 0,
      "blocked_gate_count": 0,
      "decision_reason_codes": [],
      "definitions": [...],
      "results": [...]
    }
    ```  
  (See full schema in [31] and [32]【31†L122-L130】【32†L216-L225】.)  

- **promotion_gate_summary** – Not a separate file; it is the summary object saved in each run’s `manifest.json` or in registry entries.  
  - *Path:* `<artifacts_root>/strategies/<run_id>/manifest.json` contains key `"promotion_gate_summary"`, and `registry.jsonl` entries also have a `promotion_gate_summary` field. See [81] showing the manifest payload includes `"promotion_gate_summary": promotion_evaluation.summary()`【81†L828-L834】.  
  - *Schema:* Same keys as above *except* it omits the full `definitions` and `results` lists. From `PromotionGateEvaluation.summary()`:  
    ```
    {
      "configured": bool,
      "evaluation_status": str,
      "promotion_status": str | null,
      "status_on_pass": str | null,
      "status_on_fail": str | null,
      "gate_count": int,
      "passed_gate_count": int,
      "failed_gate_count": int,
      "missing_gate_count": int,
      "highest_severity": str | null,
      "severity_counts": { "warn": int, "review": int, "reject": int, "block": int },
      "warning_gate_count": int,
      "review_gate_count": int,
      "rejected_gate_count": int,
      "blocked_gate_count": int,
      "decision_reason_codes": [str],
      "artifact_filename": str
    }
    ```  
  - *Example:* (Hypothetical) `{ "configured": true, "evaluation_status": "fail", "promotion_status": "blocked", "status_on_pass": "eligible", "status_on_fail": "blocked", "gate_count": 3, ... }`. These fields flow into review metadata and the registry【81†L828-L834】.

- **registry.jsonl** – The **Run Registry**. This is a JSON-lines file at `artifacts/strategies/registry.jsonl` (and similarly `artifacts/portfolios/registry.jsonl` for portfolio runs). It has one JSON object per completed run.  
  - *Path:* `artifacts/strategies/registry.jsonl` (also `artifacts/portfolios/registry.jsonl`).  
  - *Schema:* Varies by run type, but common fields include:  
    - `run_id` (string, deterministic run identifier)  
    - `run_type` (string, e.g. `"strategy"`, `"portfolio"`, `"alpha_evaluation"`)  
    - `timestamp` (string, a pseudo-UTC time derived from run_id)  
    - Name fields: e.g. `strategy_name`, `alpha_name`, `portfolio_name` (depending on run_type)  
    - Experiment parameters: `dataset`, `timeframe`, `evaluation_horizon`, etc.  
    - *Artifact paths:* e.g. `artifact_path`, `manifest_path`, `artifact_paths` (listing key file paths)【58†L93-L101】.  
    - *Metrics paths:* e.g. `metrics_path`, `qa_summary_path`, etc【58†L95-L104】.  
    - *Promotion fields:* `promotion_status` (string), `promotion_gate_summary` (object with summary)【58†L148-L156】.  
    - *Review fields:* `review_status`, `review_metadata` (dict with status, reviewer, etc)【58†L152-L154】.  
    - `row_count`, `timestamp_count`, `symbol_count` (ints for data counts)【58†L153-L158】.  
    - `config` (normalized config dict)  
    - `manifest` (the full manifest JSON from the run)  
    - `metadata` (any additional metadata from the run).  

  - *Example entry:* (from `build_alpha_evaluation_registry_entry`【58†L77-L85】)  
    ```json
    {
      "run_id": "strategyX_single_3f2a1b9c",
      "run_type": "strategy",
      "timestamp": "2025-11-03T12:34:56Z",
      "strategy_name": "strategyX",
      "dataset": "SP500_daily",
      "timeframe": "1D",
      "artifact_path": "artifacts/strategies/strategyX_single_3f2a1b9c",
      "manifest_path": "artifacts/strategies/strategyX_single_3f2a1b9c/manifest.json",
      "promotion_status": "eligible",
      "promotion_gate_summary": { "evaluation_status": "pass", ... },
      "review_status": "candidate",
      "review_metadata": { "status": "candidate", "reviewer": "alice", ... },
      "row_count": 250,
      "config": { ... },
      "manifest": { ... },
      "metadata": { "ts_utc_start": "...", "ts_utc_end": "..." }
    }
    ```  
    See [58†L146-L154] for the keys of an alpha-entry, which are similar for strategy/portfolio (with appropriate fields).  

- **review_summary.json** – Unified Research Review summary. Written under `artifacts/reviews/<review_id>/review_summary.json`【76†L315-L324】.  
  - *Path:* `artifacts/reviews/<review_id>/review_summary.json`.  
  - *Schema:* As produced by `_review_summary_payload()`【76†L315-L324】. Fields include:  
    - `review_id` (string)  
    - `filters` (dict of filter criteria applied)  
    - `review_config` (dict of config options)  
    - `counts_by_run_type` (dict: number of entries per run type)  
    - `entry_count` (int: total entries)  
    - `entries` (list of objects): each entry is a run, with fields from `ResearchReviewEntry`【79†L72-L80】:  
      (`run_type, rank_within_type, entity_name, run_id, selected_metric_name, selected_metric_value, secondary_metric_name, secondary_metric_value, timeframe, evaluation_mode, promotion_status, passed_gate_count, gate_count, mapping_name, sleeve_metric_name, sleeve_metric_value, sleeve_secondary_metric_name, sleeve_secondary_metric_value, linked_portfolio_count, linked_portfolio_names, linked_portfolio_metric_name, linked_portfolio_metric_value, artifact_path`)  
    - `plot_paths` (mapping of generated plot filenames)  
    - `run_ids` (list of reviewed run IDs)  
    - `skipped_plots` (any disabled plots).  

  - *Example:* (simulated)  
    ```json
    {
      "review_id": "registry_review_ab12cd34ef",
      "filters": {"dataset": "SP500_daily", "run_types": ["strategy","portfolio"]},
      "review_config": {"ranking": {"strategy_primary_metric": "sharpe_ratio"}},
      "counts_by_run_type": {"strategy": 5, "portfolio": 3},
      "entry_count": 8,
      "entries": [
        {"run_type": "strategy", "rank_within_type": 1, "entity_name": "stratA", "run_id": "stratA_single_abc123", 
         "selected_metric_name": "sharpe_ratio", "selected_metric_value": 1.23, 
         "secondary_metric_name": "total_return", "secondary_metric_value": 0.05, 
         "timeframe": "1D", "evaluation_mode": "single", "promotion_status": "eligible", 
         "passed_gate_count": 2, "gate_count": 2, "mapping_name": null, 
         "sleeve_metric_name": null, "sleeve_metric_value": null, 
         "sleeve_secondary_metric_name": null, "sleeve_secondary_metric_value": null, 
         "linked_portfolio_count": 1, "linked_portfolio_names": "portX", 
         "linked_portfolio_metric_name": "sharpe_ratio", "linked_portfolio_metric_value": 1.5,
         "artifact_path": "artifacts/strategies/stratA_single_abc123"}
        // ... (other entries)
      ],
      "plot_paths": {}, "run_ids": ["stratA_single_abc123", "..."], "skipped_plots": {}
    }
    ```  
    (Schema from [76†L315-L324], entry fields from [79†L72-L80].)  

- **Campaign artifacts** – Generated by research campaigns. Under `artifacts/research_campaigns/<campaign_id>/`. Example files (per [70]):  
  - `campaign_config.json` – The campaign’s config (contains selection criteria, etc).  
  - `checkpoint.json` – Campaign checkpoint state (stage, fingerprint).  
  - `manifest.json` – Campaign manifest listing all output files and stats (example [70†L1-L10]).  
  - `preflight_summary.json` – Pre-run checks summary (exists if campaign has a preflight stage).  
  - `summary.json` – Campaign summary (includes final outcomes, stage statuses).  
  - `milestone_report/` subfolder with `report.md`, `summary.json`, `decision_log.json`, `manifest.json`.  
  - *Schema:* The manifest ([70]) and summary JSON schemas depend on the campaign logic; key fields include campaign run_id, selected run_ids (strategy, alpha, portfolio, review), stage statuses, etc. For example, in [70†L98-L106] we see `"selected_run_ids": {"alpha_run_ids": [...], "portfolio_run_id": "...", "candidate_selection_run_id": "...", "review_id": "..."}`.  
  - *Example:* The example manifest [70†L0-L10] shows the campaign’s artifact files. The campaign summary (not fully shown) would list final promotion outcomes by strategy/portfolio.  

- **Portfolio artifacts** – Outputs from portfolio construction. Under `artifacts/portfolios/<run_id>/` (single-run or walk-forward). From [62]:  
  - `config.json` – the normalized portfolio config (name, allocator, etc).  
  - `components.json` – list of component strategies/alphas used (see example schema in [62†L126-L134]).  
  - `weights.csv` – portfolio weight matrix over time.  
  - `portfolio_returns.csv`, `portfolio_equity_curve.csv` – return stream and equity curve.  
  - `metrics.json` – portfolio metrics (fields: total_return, sharpe_ratio, p_value, etc【62†L205-L213】).  
  - `metrics_readiness.json` – split-level readiness diagnostics (fields like `schema_version`, `status`, diagnostic groups)【62†L238-L247】.  
  - `qa_summary.json` – QA checks summary (like strategy runs).  
  - `manifest.json` – portal artifact manifest.  

  For **walk-forward portfolios**, there is also `metrics_by_split.csv`, `aggregate_metrics.json`, and under `splits/<split_id>/`: each split has its own `split.json`, weights, returns, equity, `metrics.json`, `metrics_readiness.json`, `qa_summary.json`.  

  - *Example (single-run):* See [62†L205-L213] for typical `metrics.json` fields (e.g. `sharpe_ratio`, `p_value`, etc).  

The table below summarizes key artifacts, paths, schemas, and sample values:

| **Artifact**             | **Path**                                 | **Schema (fields)**                                                                                                                                                                                                                                                                                                   | **Example values**                       |
|--------------------------|------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------|
| **promotion_gates.json** | `artifacts/strategies/<run_id>/promotion_gates.json` (or under `portfolios`) | see PromotionGateEvaluation (to_payload)【31†L122-L130】【32†L216-L225】:  
`artifact_filename`, `configured`, `run_type`, `evaluation_status`, `promotion_status`, `status_on_pass`, `status_on_fail`, `gate_count`, `passed_gate_count`, `failed_gate_count`, `missing_gate_count`, `highest_severity`, `severity_counts`, `warning_gate_count`, `review_gate_count`, `rejected_gate_count`, `blocked_gate_count`, `decision_reason_codes`, plus `definitions` and `results` lists. | `{ "artifact_filename":"promotion_gates.json", "configured":true, "run_type":"review", "evaluation_status":"pass", "promotion_status":"review_ready", "status_on_pass":"eligible", "status_on_fail":"needs_work", "gate_count":2, "passed_gate_count":2, "failed_gate_count":0, "missing_gate_count":0, "highest_severity":null, "severity_counts":{"warn":0,"review":0,"reject":0,"block":0}, "warning_gate_count":0,"review_gate_count":0,"rejected_gate_count":0,"blocked_gate_count":0,"decision_reason_codes":[],"definitions":[...],"results":[...]} |
| **promotion_gate_summary** | In each run’s manifest (e.g. `artifacts/strategies/<run_id>/manifest.json`) and in registry entries | Summary of above (see PromotionGateEvaluation.summary)【31†L147-L156】:  
`configured`, `evaluation_status`, `promotion_status`, `status_on_pass`, `status_on_fail`, `gate_count`, `passed_gate_count`, `failed_gate_count`, `missing_gate_count`, `highest_severity`, `severity_counts`, `warning_gate_count`, `review_gate_count`, `rejected_gate_count`, `blocked_gate_count`, `decision_reason_codes`, `artifact_filename`. | `{ "configured": true, "evaluation_status": "fail", "promotion_status": "blocked", "status_on_pass":"eligible", "status_on_fail":"blocked", "gate_count":3, "passed_gate_count": 0, "failed_gate_count":3, "missing_gate_count":0, "highest_severity":"block", "severity_counts":{"warn":0,"review":0,"reject":0,"block":3}, "warning_gate_count":0,"review_gate_count":0,"rejected_gate_count":0,"blocked_gate_count":3,"decision_reason_codes":["severity_block"],"artifact_filename":"promotion_gates.json"}` |
| **registry.jsonl**        | `artifacts/strategies/registry.jsonl` (and `portfolios/registry.jsonl`) | JSON-lines, one per run. Keys include:  
`run_id`, `run_type`, `timestamp`, plus fields specific to type.  
Common: `strategy_name`/`portfolio_name`/`alpha_name`, `dataset`, `timeframe`, etc【58†L80-L88】; `artifact_path`, `manifest_path`, `metrics_path`, `qa_summary_path`【58†L93-L101】; *Promotion:* `promotion_status`, `promotion_gate_summary`【58†L147-L154】; *Review:* `review_status`, `review_metadata`【58†L152-L154】; `row_count`, `timestamp_count`, `symbol_count`【58†L153-L158】; `config`, `manifest`, `metadata`. | (Strategy example) `{ "run_id":"stratX_single_ab12", "run_type":"strategy", "timestamp":"2026-01-15T10:20:30Z", "strategy_name":"stratX", "dataset":"market_data", "timeframe":"1D", "artifact_path":".../stratX_single_ab12", "manifest_path":".../stratX_single_ab12/manifest.json", "promotion_status":"eligible", "promotion_gate_summary":{...}, "review_status":"candidate", "review_metadata":{ "status":"candidate","reviewer":"alice" }, "row_count":500, "config":{...}, "manifest":{...}, "metadata":{...} }`. Fields per [58]. |
| **review_summary.json**  | `artifacts/reviews/<review_id>/review_summary.json`   | See [76†L315-L324]:  
`review_id`, `filters`, `review_config`, `counts_by_run_type`, `entry_count`, `entries` (list of run entries), `plot_paths`, `run_ids`, `skipped_plots`.  
Each entry has schema [79†L72-L80]: `run_type, rank_within_type, entity_name, run_id, selected_metric_name, selected_metric_value, secondary_metric_name, secondary_metric_value, timeframe, evaluation_mode, promotion_status, passed_gate_count, gate_count, mapping_name, sleeve_metric_name, sleeve_metric_value, sleeve_secondary_metric_name, sleeve_secondary_metric_value, linked_portfolio_count, linked_portfolio_names, linked_portfolio_metric_name, linked_portfolio_metric_value, artifact_path`. | `{ "review_id":"registry_review_d4e5f6g7", "filters":{"dataset":"SP500","run_types":["strategy"]}, "review_config":{...}, "counts_by_run_type":{"strategy":2}, "entry_count":2, "entries":[ {"run_type":"strategy","rank_within_type":1,"entity_name":"stratA","run_id":"stratA_single_xyz","selected_metric_name":"sharpe_ratio","selected_metric_value":1.50,"secondary_metric_name":"total_return","secondary_metric_value":0.10,"timeframe":"1D","evaluation_mode":"single","promotion_status":"eligible","passed_gate_count":2,"gate_count":2,"mapping_name":null,"sleeve_metric_name":null,"sleeve_metric_value":null,"sleeve_secondary_metric_name":null,"sleeve_secondary_metric_value":null,"linked_portfolio_count":0,"linked_portfolio_names":null,"linked_portfolio_metric_name":null,"linked_portfolio_metric_value":null,"artifact_path":"artifacts/strategies/stratA_single_xyz"} , ...], "plot_paths":{} , "run_ids":["stratA_single_xyz", ...], "skipped_plots":{} }` (fields per [76] and [79]). |
| **campaign_config.json**    | `artifacts/research_campaigns/<camp_id>/campaign_config.json` | Campaign run configuration (criteria for selection, mapping, etc). Schema varies by campaign logic.  
| e.g. `{ "campaign_name":"real_world_campaign", "selection":"top_sharpe", "target_run_count":3, ... }`. |
| **checkpoint.json**        | `artifacts/research_campaigns/<camp_id>/checkpoint.json`  | Checkpoint state of the campaign (stages completed). Example fields: `schema_version`, `stage_count`, per-stage status. In [70†L52-L60]: `"schema_version":2, "stage_count":7`. |
| **preflight_summary.json** | `artifacts/research_campaigns/<camp_id>/preflight_summary.json` | Preflight checks (e.g. data availability). Schema depends on campaign; likely fields like `status`, `issues`. |
| **summary.json** (campaign) | `artifacts/research_campaigns/<camp_id>/summary.json`      | Campaign summary. In [70†L1-L10] manifest lists `summary.json` under artifacts. It likely includes `stage_count`, `selected_run_ids`, final `promotion_status` of campaign, etc (see [70†L98-L106]). Example snippet:  
```json
{ 
  "campaign_run_id": "research_campaign_1b8cdf9211cf",
  "selected_run_ids": {
    "alpha_run_ids": ["..."],
    "candidate_selection_run_id": "candidate_selection_e35992810577",
    "portfolio_run_id": "real_world_portfolio_cc99ba561315",
    "strategy_run_ids": [],
    "review_id": "registry_review_63c30be06cb7"
  },
  "stage_statuses": {"candidate_selection":"completed","review":"completed", ...},
  "promotion_status": "eligible",
  "milestone_reports": {...}
}
```  
(In [70†L98-L106], `selected_run_ids` is shown.) |
| **portfolio config.json**     | `artifacts/portfolios/<run_id>/config.json`             | Portfolio configuration. Fields: `portfolio_name`, `allocator`, `initial_capital`, `timeframe`, `evaluation_config_path`, etc【62†L97-L105】.  
| e.g. `{ "portfolio_name":"PF1", "allocator":"equal", "initial_capital":1000000, "timeframe":"1D", "evaluation_config_path":null }`. |
| **components.json**       | `artifacts/portfolios/<run_id>/components.json`         | Components used (strategy/alpha runs). Schema:  
```json
{ "components": [
    { "artifact_type":"strategy", "strategy_name":"momentum_v1", "run_id":"momentum_v1_single_ab12", "source_artifact_path":".../artifacts/strategies/momentum_v1_single_ab12" }
] }
```  
(as in [62†L126-L134]). |
| **weights.csv**          | `artifacts/portfolios/<run_id>/weights.csv`             | CSV with columns `ts_utc`, `weight__<strategy>` for each component【62†L155-L164】. Sample rows (dates and weights). |
| **portfolio_returns.csv**   | `artifacts/portfolios/<run_id>/portfolio_returns.csv`   | Columns: `ts_utc`, `strategy_return__<strategy>` (each component’s return), `weight__<strategy>`, and `portfolio_return`【62†L171-L179】. |
| **portfolio_equity_curve.csv** | `artifacts/portfolios/<run_id>/portfolio_equity_curve.csv` | Columns: `ts_utc`, `portfolio_equity_curve` (compounded from `portfolio_return`)【62†L189-L198】. |
| **metrics.json**         | `artifacts/portfolios/<run_id>/metrics.json`            | Portfolio metrics (as in [62†L205-L213]): fields like  
`total_return, annualized_return, volatility, sharpe_ratio, t_stat, p_value, conf_int_lower, conf_int_upper, autocorr_lag1, effective_n, hit_rate, hit_rate_p_value, split_mean_diff, split_mean_diff_p, rolling_sharpe_mean, rolling_sharpe_sd, sharpe_stability_ratio, ...` | Example: `{ "total_return":0.12, "sharpe_ratio":1.50, "t_stat":3.1, "p_value":0.002, "conf_int_lower":0.07, "conf_int_upper":0.17, "autocorr_lag1":-0.02, "effective_n":250, ... }`. |
| **metrics_readiness.json** | `artifacts/portfolios/<run_id>/metrics_readiness.json`  | Readiness diagnostics (per [62†L238-L247]): keys like `schema_version`, `status` (pass/fail), `run_id`, `source_metrics_artifact`, and grouped checks (e.g. `return_variation`, `hit_rate_significance`, etc).  
| e.g. `{ "schema_version": 1, "status":"pass", "run_id":"PF1_2026_walk", "source_metrics_artifact":"metrics.json", "drift_checks":{...}, "hit_rate_checks":{...}, ... }`. |
| **qa_summary.json**      | `artifacts/portfolios/<run_id>/qa_summary.json`         | QA checks summary for portfolio (fields depend on QA rules). |
| **manifest.json**        | `artifacts/portfolios/<run_id>/manifest.json`           | Portfolio run manifest listing all artifact files. |

*Table: Artifact → Path → Schema → Example. Citations refer to repo code and docs (e.g. promotion schema【31†L122-L130】, manifest writing【81†L831-L834】, portfolio artifact docs【62†L126-L134】【62†L205-L213】).*

## Gap Analysis

Our inventory shows that most needed M31 artifacts exist, but some gaps or mismatches will require handling in M32:

- **Missing Artifacts:**  
  - There is *no* standalone `promotion_gate_summary.json` file – the summary only appears inside each run’s `manifest.json` (and in registry entries). M32 components must extract it from manifests or registry lines.  
  - Portfolio single-run artifacts in M31 did not always emit `metrics_readiness.json` (per [27†L82-L85]). M32 should note that absence (or generate a stub summary) if needed.  
  - Campaign outputs lack a formal “promotion matrix” or consolidated outcome file – M32 will need to produce those.

- **Schema Mismatches / Normalization Needs:**  
  - **Promotion status labeling:** The repo uses strings like `"eligible"`, `"needs_review"`, `"rejected"`, `"blocked"` in manifests/registry【27†L23-L31】. Ensure consistent normalization (e.g. mapping any legacy status_on_pass/fail labels to these canonical ones). For instance, review metadata maps `eligible -> candidate`, which might need mapping back to a unified “promotion_status”.  
  - **Severity fields:** M31 introduced per-gate `severity`. Registry `promotion_gate_summary` includes counts of `warn/review/reject/block`【27†L20-L23】, but review/registry often expose only the top-level `promotion_status`. M32 should parse both and normalize (e.g. treat any severity as a status label).  
  - **Date/time formats:** Run IDs encode timestamps (e.g. `_walk_forward_2026-05-06T...`). Ensure consistent parsing of these pseudo-timestamps (note functions like `_utc_timestamp_from_run_id`【36†L129-L138】).  
  - **Lists vs JSON structure:** Some fields (like registry `artifact_paths`) may be lists or dicts; our parser must handle either. The manifest uses lists for `artifact_files`, but registry stores some paths as dicts (see [58†L93-L101]).  
  - **Run type categories:** The governance reports should treat `alpha_evaluation`, `strategy`, `portfolio` runs distinctly. M32 should normalize run_type names (e.g. use `"strategy"`, `"portfolio"`, `"alpha_evaluation"` consistently).  
  - **Null/Missing values:** Some fields may be missing (`null`) if not applicable (e.g. `promotion_status` when no gates were configured【32†L217-L225】). M32 should handle nulls without crashing.

- **New Parsers Needed:**  
  - Parsing the campaign artifacts (`checkpoint.json`, `summary.json`) will likely require new code (these are not standard JSON schemas already parsed elsewhere).  
  - A **campaign registry** may be needed to list completed campaigns (similar to strategy registry). If so, implement reading of `artifacts/research_campaigns/registry.jsonl` (if it exists) or scanning campaign dirs.

- **Data to Normalize:**  
  - Map statuses from promotions to unified labels (e.g. `"needs_work"`, `"needs_review"` → `"needs_review"`).  
  - Convert `run_id` strings to canonical values (e.g. remove evaluation config hash differences) to compare across artifacts.  
  - Aggregate metrics: e.g. `counts_by_run_type` in review summary must align with registry counts.  
  - Ensure field name consistency: e.g. registry uses `"promotion_status"` vs manifest `"promotion_status"` (same), but campaign may use `"promotion_status"` or `"promotion_outcome"` – confirm names.  

In summary, M32 will not need new data fields (all gating outcomes exist), but must parse nested JSON fields, unify status labels, and fill in missing reports. Key fields to watch: **run_id**, **run_type**, **promotion_status**, **evaluation_status**, and the counts in **promotion_gate_summary** (severity vs binary status).

## Proposed M32 Design (Code Layout and Snippets)

We propose a new module tree under `src/research/governance/`:

```
src/research/governance/
├── __init__.py
├── loader.py          # Functions to load artifacts (JSONL, JSON, CSV)
├── aggregator.py     # Aggregate data from loaded artifacts
├── validator.py      # Consistency checks on aggregated data
├── writer.py         # Generate final reports (JSON/CSV)
├── cli.py            # CLI entrypoint (e.g. `stratlake governance run`)
└── models.py         # (optional) data classes for core concepts
```

- **loader.py:** Functions like `load_registry(artifacts_root) -> List[dict]` and `load_review_summary(review_id)`, etc. Use repo code for parsing (e.g. reuse `load_registry` from `src/research/registry`). It should read all relevant JSON and CSV into Python structures in a deterministic sort order (e.g. sort registry entries by run_id). 

- **aggregator.py:** E.g. `aggregate_promotion_data(registry_entries, review_summaries, campaign_summaries)`. This might produce combined data needed for reports (counts by status, cross-run comparisons). It should join data by run_id or run_type. Pseudocode:
  ```python
  def aggregate_promotion_data(registry, reviews, campaigns):
      # Flatten registry entries with run_id, run_type, promotion_status
      runs = []
      for entry in registry:
          runs.append({
              "run_id": entry["run_id"],
              "run_type": entry["run_type"],
              "promotion_status": entry.get("promotion_status"),
              # optionally include other fields like run label, rank, metrics
          })
      # Possibly include review entries if they refine promotion_status
      # Combine campaigns info if needed (which runs were selected)
      # Return as a DataFrame or dicts.
      return runs_df
  ```
  It should enforce deterministic ordering (e.g. sort by run_type then run_id).

- **validator.py:** Implement consistency checks. For example:
  - *Unique run_id:* ensure no duplicates across runs.
    ```python
    assert len(runs_df["run_id"]) == len(set(runs_df["run_id"]))
    ```
  - *Registry vs run folders:* For each run_id in registry, check existence of `manifest.json` in artifacts.
  - *Status alignment:* e.g. `manifest_promotion_status == registry_promotion_status`.  
  - *Counts match:* Sum of eligible + review + rejected + blocked = total runs etc.
  
  The module may produce a JSON result like:
  ```json
  {
    "unique_run_id_check": {"status": "pass", "count": 50},
    "registry_manifest_sync": {"status": "pass", "missing_manifests": []},
    "promotion_status_consistency": {"status": "pass", "mismatches": []}
    // ...
  }
  ```
  
- **writer.py:** Functions to write:
  - `write_promotion_governance_summary(path, data)`: emits `promotion_governance_summary.json`. Schema might be:
    ```json
    {
      "generated_at": "...",
      "total_runs": 120,
      "counts_by_run_type": {
         "strategy": 50,
         "portfolio": 30,
         "alpha_evaluation": 40
      },
      "counts_by_promotion_status": {
         "eligible": 80,
         "needs_review": 20,
         "rejected": 10,
         "blocked": 10
      },
      "counts_by_run_type_and_status": {
         "strategy": {"eligible": 30, "needs_review":10, ...},
         "portfolio": {...}, "alpha_evaluation": {...}
      }
    }
    ```
    Example values use actual fields `run_type` and `promotion_status` from the repo. 
  - `write_promotion_outcome_matrix(path, matrix_df)`: CSV with columns `run_type,promotion_status,count`. E.g.:
    ```
    run_type,promotion_status,count
    strategy,eligible,30
    strategy,needs_review,10
    portfolio,eligible,20
    ...
    ```
  - `write_consistency_validation(path, checks)`: JSON of each check result (see above).
  
  These writers must sort rows and keys deterministically (e.g. sort run types alphabetically, status by severity order).  

- **CLI (cli.py):** Provide a command, e.g. `stratlake governance run [--artifacts ROOT] [--output_dir OUT]`, which loads data, runs aggregator and validator, and writes reports. Use `argparse` to handle options.

**Deterministic ordering:** Always sort runs by `(run_type, run_id)`, sort status categories by a fixed order (e.g. `eligible,needs_review,rejected,blocked`), and use JSON dumps with `sort_keys=True` for output. The `serialize_canonical_json` in registry ensures stable JSON order【58†L158-L160】; do likewise for new outputs.

**File Tree Example:**
```
src/research/governance/
├── cli.py
├── loader.py
├── aggregator.py
├── validator.py
├── writer.py
├── models.py
└── __init__.py
```

**Pseudocode snippet (aggregation + writer):**
```python
# In aggregator.py
def aggregate_data(registry_entries, review_entries):
    df = pd.DataFrame(registry_entries)
    # Merge review data if needed, align on run_id...
    return df

# In writer.py
def write_reports(df, output_dir):
    summary = {
        "total_runs": len(df),
        "counts_by_run_type": df["run_type"].value_counts().to_dict(),
        "counts_by_promotion_status": df["promotion_status"].value_counts().to_dict(),
        "counts_by_run_type_and_status": {
            rt: df[df.run_type==rt]["promotion_status"].value_counts().to_dict()
            for rt in sorted(df.run_type.unique())
        }
    }
    with open(output_dir/"promotion_governance_summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    # Build outcome matrix CSV
    matrix_rows = []
    for (rt, status), count in df.groupby(["run_type","promotion_status"]).size().items():
        matrix_rows.append((rt, status, count))
    matrix_df = pd.DataFrame(matrix_rows, columns=["run_type","promotion_status","count"])
    matrix_df.to_csv(output_dir/"promotion_outcome_matrix.csv", index=False)
```

(*See also the repository’s existing patterns: e.g. the unified review uses `json.dump(..., sort_keys=True)`【52†L1021-L1026】 for manifest.*)

## Proposed Report Schemas

We define schemas for the new M32 reports, aligning with actual field names:

### 1. `promotion_governance_summary.json`

A JSON summary of promotion outcomes across all runs. Proposed fields:
```json
{
  "total_runs": 120,
  "counts_by_run_type": {
    "alpha_evaluation": 40,
    "portfolio": 30,
    "strategy": 50
  },
  "counts_by_promotion_status": {
    "eligible": 80,
    "needs_review": 20,
    "rejected": 10,
    "blocked": 10
  },
  "counts_by_run_type_and_status": {
    "alpha_evaluation": { "eligible": 25, "needs_review": 10, "rejected": 5, "blocked": 0 },
    "portfolio":         { "eligible": 20, "needs_review": 5, "rejected": 3, "blocked": 2 },
    "strategy":          { "eligible": 35, "needs_review": 5, "rejected": 2, "blocked": 8 }
  }
}
```
- Keys `run_type` and `promotion_status` match those in registry/manifest.  
- Values are counts of runs.  
- Example JSON above references `promotion_status` labels (as in repository) and run types (`strategy`, `portfolio`, `alpha_evaluation`). 

### 2. `promotion_outcome_matrix.csv`

A CSV matrix (long form) summarizing counts of runs by run_type and promotion_status. Columns: `run_type,promotion_status,count`. Example rows:
```
run_type,promotion_status,count
alpha_evaluation,eligible,25
alpha_evaluation,needs_review,10
alpha_evaluation,rejected,5
alpha_evaluation,blocked,0
portfolio,eligible,20
portfolio,needs_review,5
portfolio,rejected,3
portfolio,blocked,2
strategy,eligible,35
strategy,needs_review,5
strategy,rejected,2
strategy,blocked,8
```
Columns correspond exactly to the repo’s field names (`run_type`, `promotion_status`). Rows should be sorted (e.g. alphabetically by run_type, then status by severity order).

### 3. `consistency_validation.json`

A JSON object listing each consistency check and its result. Example schema:
```json
{
  "unique_run_id": { "status": "pass", "found": 100 },
  "registry_vs_manifest": {
    "status": "pass",
    "mismatches": []
  },
  "promotion_status_consistency": {
    "status": "fail",
    "details": [
      {"run_id": "stratX_single_ab12", "registry_status": "eligible", "manifest_status": "blocked"}
    ]
  },
  "review_coverage": {
    "status": "pass",
    "runs_without_review": []
  }
}
```
- Keys are check names.  
- Each has `status` ("pass"/"fail") and optionally details or counts.  
- Check names reflect the check (e.g. ensuring no duplicate IDs, matching counts, consistent fields). 

Fields like `run_id`, `registry_status`, `manifest_status` refer to real fields from the repo. Each check can be documented in code with pseudocode (see below).

## Consistency Validation Rules

We enumerate concrete validation rules, referencing actual repo fields, in pseudocode or SQL-like form:

1. **Unique Run IDs:** Ensure no duplicate `run_id` across registry entries.  
   ```python
   duplicate_ids = df.groupby('run_id').size()[lambda x: x>1]
   assert duplicate_ids.empty, f"Duplicate run_ids: {duplicate_ids.index.tolist()}"
   ```
2. **Registry vs Manifests:** Each `run_id` in `registry.jsonl` should have a corresponding `manifest.json` file at the expected artifact path.  
   SQL-like: 
   ```
   SELECT run_id FROM registry
   WHERE run_id NOT IN (SELECT manifest.run_id FROM manifest_files);
   ```
3. **Registry vs Artifacts Count:** The number of runs per `run_type` in `registry.jsonl` should match the number of subdirectories under `artifacts/strategies/` (and `artifacts/portfolios/`).  
   Pseudocode: compare `COUNT(*) FROM registry WHERE run_type='strategy'` to filesystem count of `artifacts/strategies/*`.
4. **Promotion Status Consistency:** For each run, check `registry.promotion_status` equals the `manifest.promotion_gate_summary.promotion_status`.  
   ```python
   for entry in registry:
       manifest_summary = entry['promotion_gate_summary']
       assert entry['promotion_status'] == manifest_summary.get('promotion_status'), f"Mismatch for {entry['run_id']}"
   ```
5. **Review Status Mapping:** Check that registry `review_metadata.status` is consistent with `promotion_status`. (E.g. mapping `eligible`→`candidate`, `needs_review`→`needs_review`, etc【53†L23-L27】.)  
6. **Run Count in Review:** In `review_summary.json`, `entry_count` should equal the number of entries listed and should match sum of corresponding registry entries for that review filter.  
7. **Campaign Selection Integrity:** In a campaign `summary.json`, every `run_id` in `selected_run_ids` should appear exactly once among strategy/portfolio/alpha runs in registry.  
8. **Milestone Report:** If present, ensure `milestone_report/summary.json` decisions count matches its `decision_log.json` entries.

Each rule failure should record details (e.g. offending `run_id`). These checks form the rows of `consistency_validation.json` output.

## Tests, Examples, and Dataflow Diagram

To ensure determinism and correctness, we propose:

- **Unit tests:**  
  - **Loader tests:** e.g. feed a small set of synthetic `registry.jsonl`, `promotion_gates.json`, and check parsed fields.  
  - **Aggregator tests:** on toy dataframes (with known counts), verify the summary and matrix outputs match expectations.  
  - **Validator tests:** ensure each rule flags known inconsistencies.  
  - **Writer tests:** compare generated JSON/CSV (with sort_keys) to expected string (snapshot test).  

- **Integration test:** Use a small synthetic case: e.g. create two fake strategy runs and one portfolio, with simple promotion gates. Run the governance CLI to produce reports; verify them against hand-crafted expected output. (These fixtures could mimic the docs example runs.)

- **Case study:** We can reuse the docs examples as a mini-case-study:
  - Run `docs/examples/m31_readiness_gated_promotion_case_study.py` to generate output artifacts (which include promotion_gates, registry, review)【53†L37-L46】.
  - Feed those artifacts to the new M32 code and verify reports.
  - This ensures the M32 layer works on real structured output.

- **Deterministic fixtures:** Use fixed config and fixed data so that runs always produce the same `run_id` (the repo’s use of hashed IDs ensures reproducibility【40†L348-L356】).

- **Flowchart (Mermaid):** Below is a conceptual dataflow from artifacts to reports:

```mermaid
flowchart LR
  A[Experiments / Portfolios / Alphas] -->|Write artifacts| B((Promotion artifacts))
  B --> C[Registry (registry.jsonl)]
  B --> D[Manifests (manifest.json)]
  D --> E[promotion_gate_summary]
  C --> G[load registry entries]
  D --> H[load promotion summaries]
  I[Unified Review] -->|Writes| F((review_summary.json))
  F --> G
  G --> J[Aggregator]
  H --> J
  J --> K[Validator]
  K --> L[Report Writer]
  L --> M[promotion_governance_summary.json]
  L --> N[promotion_outcome_matrix.csv]
  L --> O[consistency_validation.json]
  style B fill:#E3F6FF,stroke:#004A8F
  style C fill:#E3F6FF,stroke:#004A8F
  style D fill:#E3F6FF,stroke:#004A8F
  style F fill:#E3F6FF,stroke:#004A8F
  style J fill:#FFF2CC,stroke:#7A4F01
  style K fill:#FFF2CC,stroke:#7A4F01
  style L fill:#D5E8D4,stroke:#82B366
```

This illustrates how promotion gating outputs feed into the governance logic: artifacts (A) generate promotion files (B), which feed the registry (C) and manifests (D/E), along with review outputs (F). The loader collects (`G`, `H`), `aggregator` (`J`) collates them, `validator` (`K`) checks consistency, and `writer` (`L`) produces final reports (`M, N, O`).

## Proposed GitHub Issues (Prioritized)

To implement M32 in the `stratlake-trade-engine` repo, we suggest the following issues (labels and effort estimates):

1. **Issue: "Implement governance observability module"** (label: enhancement, size: **Large**)  
   *Description:* Create `src/research/governance/` modules (`loader.py`, `aggregator.py`, `validator.py`, `writer.py`, CLI) as above. Write code to load registry, manifests, and reviews, and emit the new reports (`promotion_governance_summary.json`, `promotion_outcome_matrix.csv`, `consistency_validation.json`) according to the defined schemas. Ensure deterministic ordering in all outputs.  
   *Rationale:* Core M32 work.

2. **Issue: "Add parser for campaign artifacts (preflight, checkpoint, summary)"** (label: enhancement, size: **Medium**)  
   *Description:* Extend loader to read campaign outputs under `artifacts/research_campaigns/`. Support `campaign_config.json`, `checkpoint.json`, and `summary.json`. Extract relevant fields (e.g. selected run IDs, final status) for governance reports.  
   *Rationale:* Campaign outputs are not currently parsed by any code. Needed to incorporate campaign-level outcomes into governance.

3. **Issue: "Normalize promotion_status across components"** (label: bug/enhancement, size: **Small**)  
   *Description:* Ensure that promotion status labels from various sources are unified (e.g. map `needs_work` to `needs_review`). This may involve mapping registry/review status to a canonical set of {"eligible","needs_review","rejected","blocked"} as described in [53†L23-L27]. Update any checks or reports to use the canonical string.  
   *Rationale:* Avoid confusion due to synonyms.

4. **Issue: "Consistent reading of registry vs manifest keys"** (label: bug, size: **Small**)  
   *Description:* Fix any mismatches where the same concept is named differently. For example, registry stores `promotion_status`, manifest stores `promotion_gate_summary.promotion_status`. Ensure the loader consolidates these correctly. Similar for `artifact_paths` vs. `artifact_files`.  
   *Rationale:* Eliminate inconsistencies in field usage.

5. **Issue: "Unit tests for governance components"** (label: test, size: **Medium**)  
   *Description:* Add pytest unit tests for the governance module. Include fixtures: small JSON samples for registry entries, promotion_gates, review_summary, and expected report outputs. Test all consistency checks with pass and fail cases.  
   *Rationale:* Ensure correctness and prevent regressions.  

6. **Issue: "Integration test with docs examples"** (label: test, size: **Medium**)  
   *Description:* Use the existing docs example outputs (e.g. `docs/examples/output/m31_readiness_gated_promotion_case_study/`) as test fixtures. After running the governance CLI on these artifacts, compare the produced JSON/CSV to reference snapshots.  
   *Rationale:* Verifies the full pipeline on realistic data.

7. **Issue: "Add candidate selection to registry builder"** (label: enhancement, size: **Small**)  
   *Description:* If not present, include candidate_selection runs in the run registry (currently, registry covers strategy/portfolio/alpha). This enables governance to see which candidate was chosen.  
   *Rationale:* Ensure all research-run types are tracked.

8. **Issue: "Document promotion governance schema and CLI"** (label: documentation, size: **Small**)  
   *Description:* Update README/docs to describe the new governance reports, their schemas, and how to invoke the CLI. Provide example outputs.  
   *Rationale:* Essential for user guidance and clarity.  

Each issue can be created in the `stratlake-trade-engine` repository. Labels like `enhancement`, `bug`, `test`, and `documentation` help categorize them.

## Sources

- **Repository code**: `src/research/promotion.py` (gate evaluation schema)【31†L122-L130】【32†L216-L225】, `src/research/experiment_tracker.py` (manifest & registry writing)【81†L831-L834】【58†L146-L154】, `src/research/review.py` (review summary schema)【76†L315-L324】【79†L72-L80】, `src/portfolio/artifacts.py` and docs (portfolio file contracts)【62†L126-L134】【62†L205-L213】.  
- **Docs/examples**: example promotion output【7†L0-L8】, campaign manifest【70†L0-L10】.  
- **Release notes**: M31 overview of promotion artifacts and new fields【53†L21-L29】.  
- (No web search needed beyond repo; all references are in the StratLake repo.)