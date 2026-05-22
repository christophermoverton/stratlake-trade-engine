# M40 Release Notes - Corporate Actions Dividend Evidence

Milestone title: `M40 - Corporate Actions Dividend Evidence`

M40 branch:
`feature/m40-corporate-actions-dividend-evidence`

Candidate milestone release tag:
`v0.40.0-corporate-actions-dividend-event-evidence`

## Milestone Principle

Corporate-action data should enter StratLake as explicit event evidence with
deterministic contracts and provenance, not as hidden adjustments to OHLCV bars
or research results.

## Summary

M40 adds a local-file corporate-actions dividend evidence stack. StratLake can
consume upstream dividend Parquet and metadata JSON artifacts, map them into
the `corporate_actions.dividends.v1` contract, validate and QA the resulting
events, write a curated partitioned dividend event dataset, write deterministic
import artifacts, and expose the evidence through the existing read-only
catalog.

The milestone preserves the artifact-first boundary. Dividend events are
explicit evidence. They are separate from price bars, adjusted prices, strategy
outputs, alpha outputs, portfolio outputs, promotion decisions, and backtest
returns.

## Scope Summary

M40 covers:

* dividend event contract validation
* local upstream artifact import
* deterministic curated event dataset writing
* import summaries, QA summaries, schema contracts, duplicate/invalid row
  reports, and source provenance artifacts
* source fingerprint and import-config fingerprint capture
* catalog direct-scan discovery for dividend evidence
* CLI/API parity for the import wrapper
* notebook and pipeline integration examples that remain explicit and
  credential-free
* cross-repo local-file handoff documentation for `fintech-market-ingestion`

M40 does not add live ingestion, credential handling, price adjustment,
adjusted-price reconstruction, dividend reinvestment, a scheduler, a dashboard,
a server, a remote metadata service, a graph store, or automatic mutation of
research workflows.

## Key Files And Docs

Implementation entry points:

* `src/corporate_actions/dividend_contract.py`
* `src/corporate_actions/dividend_importer.py`
* `src/corporate_actions/dividends.py`
* `src/cli/import_corporate_actions_dividends.py`

Documentation:

* [Corporate Actions Dividend Evidence](corporate_actions_dividend_evidence.md)
* [Corporate Actions Event Contracts](corporate_actions_event_contracts.md)
* [M40 Cross-Repo Q1 Dividend Smoke Workflow](m40_cross_repo_q1_dividend_smoke_workflow.md)
* [Catalog Indexer](catalog_indexer.md)
* [Catalog Evidence Explorer](catalog_evidence_explorer.md)
* [Notebook Execution API](notebook_execution_api.md)
* [Pipeline Integration](pipeline_integration.md)
* [Runtime Profiles](runtime_profiles.md)
* [M40 Release Validation Checklist](m40_release_validation_checklist.md)

Examples:

* `docs/examples/m40_dividend_evidence_import_example.py`
* `docs/examples/m40_dividend_pipeline_step_example.py`
* `docs/examples/m40_cross_repo_q1_dividend_smoke_workflow.py`

## Usage Notes

CLI:

```bash
python -m src.cli.import_corporate_actions_dividends \
  --source-data data/external/corporate_actions/dividends/dividends.parquet \
  --source-metadata data/external/corporate_actions/dividends/metadata.json \
  --output-root data/curated/events/dividends \
  --artifact-root artifacts/corporate_actions \
  --start 2024-01-01 \
  --end 2025-01-01 \
  --strict
```

Python:

```python
from src.corporate_actions import import_dividend_events, load_dividend_events

result = import_dividend_events(
    source_data_path="data/external/corporate_actions/dividends/dividends.parquet",
    source_metadata_path="data/external/corporate_actions/dividends/metadata.json",
    output_root="data/curated/events/dividends",
    artifact_root="artifacts/corporate_actions",
    start="2024-01-01",
    end="2025-01-01",
    strict=True,
)
events = load_dividend_events("data/curated/events/dividends")
```

Read from the dataset root, not individual partition files, so Hive-style
`symbol=<SYMBOL>` and `year=<YYYY>` partition columns are reconstructed.

## Validation Commands

Focused M40 tests:

```bash
pytest tests/test_dividend_event_contract.py tests/test_dividend_event_schema_validation.py tests/test_dividend_importer.py tests/test_dividend_import_artifacts.py tests/test_dividend_catalog_registration.py tests/test_catalog_direct_scan_dividend_evidence.py tests/test_dividend_evidence_classification.py tests/test_dividend_api_cli_parity.py tests/test_dividend_cli_smoke.py tests/test_dividend_duplicate_detection.py tests/test_dividend_path_portability.py tests/test_dividend_qa_summary.py tests/test_dividend_source_provenance.py tests/test_dividend_writer_determinism.py tests/test_dividend_example_smoke.py tests/test_m40_cross_repo_dividend_smoke_workflow.py
```

Release-facing checks:

```bash
python -m ruff check src tests docs/examples
pytest
python -m src.cli.run_docs_path_lint
python -m build
```

## Boundaries Preserved

* canonical artifacts remain the source of truth
* direct catalog scan remains available and canonical
* derived indexes and review views remain disposable and non-authoritative
* provenance is evidence context, not a second source of truth
* examples are deterministic and CI-safe
* generated outputs remain ignored
* repository docs use relative paths
* no credentials, secrets, live market data, network access, or external
  services are required for StratLake M40 validation

## Draft GitHub Release Notes

Title:
`M40 - Corporate Actions Dividend Evidence`

Tag:
`v0.40.0-corporate-actions-dividend-event-evidence`

Branch:
`feature/m40-corporate-actions-dividend-evidence`

Summary:
M40 adds deterministic local corporate-actions dividend evidence. StratLake
imports local upstream dividend artifacts into the
`corporate_actions.dividends.v1` event contract, writes curated partitioned
event data and deterministic QA/provenance artifacts, and exposes dividend
evidence through the read-only catalog without live ingestion or hidden price
adjustment.

Highlights:

* Added dividend event schema contracts and validation.
* Added local Parquet/metadata import with deterministic output ordering.
* Added import summaries, QA summaries, duplicate/invalid reports, schema
  contracts, and source provenance artifacts.
* Added read-only catalog discovery for `dividend_events`.
* Added CLI/API parity and CI-safe examples.
* Documented the local-file handoff from upstream systems such as
  `fintech-market-ingestion`.

Validation:

* Focused Ruff: `<record result>`
* Focused M40 pytest: `<record result>`
* Full pytest: `<record result>`
* Docs/path lint: `<record result>`
* Package build: `<record result>`
* Hosted GitHub Actions: `<record result>`

Known boundaries:
M40 does not add live Alpaca calls, credentials, network access, hidden OHLCV
adjustment, adjusted-price reconstruction, backtest dividend reinvestment,
schedulers, dashboards, servers, remote metadata services, graph stores, or
automatic strategy, alpha, portfolio, promotion, or backtest mutation.
