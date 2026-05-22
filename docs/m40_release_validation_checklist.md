# M40 Release Validation Checklist

This checklist documents release-readiness checks for Milestone 40. It does not
replace existing CI, milestone validation, or release automation.

Milestone title: `M40 - Corporate Actions Dividend Evidence`

M40 branch:
`feature/m40-corporate-actions-dividend-evidence`

Candidate milestone release tag:
`v0.40.0-corporate-actions-dividend-evidence`

## Milestone Principle

Corporate-action data should enter StratLake as explicit event evidence with
deterministic contracts and provenance, not as hidden adjustments to OHLCV bars
or research results.

## Pre-Merge Validation

Run the focused M40 validation slice:

```bash
pytest tests/test_dividend_event_contract.py tests/test_dividend_event_schema_validation.py tests/test_dividend_importer.py tests/test_dividend_import_artifacts.py tests/test_dividend_catalog_registration.py tests/test_catalog_direct_scan_dividend_evidence.py tests/test_dividend_evidence_classification.py tests/test_dividend_api_cli_parity.py tests/test_dividend_cli_smoke.py tests/test_dividend_duplicate_detection.py tests/test_dividend_path_portability.py tests/test_dividend_qa_summary.py tests/test_dividend_source_provenance.py tests/test_dividend_writer_determinism.py tests/test_dividend_example_smoke.py tests/test_m40_cross_repo_dividend_smoke_workflow.py
```

Run Ruff over release-facing source, tests, and examples:

```bash
python -m ruff check src tests docs/examples
```

Run full pytest when practical:

```bash
pytest
```

Run docs/path lint:

```bash
python -m src.cli.run_docs_path_lint
```

Run package build validation:

```bash
python -m build
```

Run the CI-safe dividend evidence example:

```bash
python docs/examples/m40_dividend_evidence_import_example.py
```

Run the CI-safe pipeline-step example:

```bash
python docs/examples/m40_dividend_pipeline_step_example.py
```

## Validation Coverage

The focused M40 pytest slice covers:

* dividend event contract fields and schema serialization
* schema validation for required columns, nullable columns, dates, supported
  event types, schema versions, and duplicate primary-key rows
* local upstream artifact reading and normalization
* half-open `ex_date` import-window filtering
* deterministic partitioned Parquet writing and dataset loading
* duplicate and invalid event reporting
* QA summary counts and strict/advisory status behavior
* source provenance, source fingerprints, import-config fingerprints, redacted
  secret-like metadata, and explicit no-network/no-credential flags
* portable path output
* catalog direct-scan discovery and dividend evidence classification facets
* API/CLI parity
* CI-safe examples and cross-repo smoke helper behavior

## Manual Documentation Review

Before merging or tagging, inspect the M40 docs for:

* no local absolute paths
* no `file://` links
* no generated local output committed
* no credentials, secrets, tokens, API keys, or passwords
* no claim that StratLake performs live ingestion in M40
* no claim that dividend events adjust prices or returns automatically
* no claim that dividend evidence mutates strategy, alpha, portfolio, promotion,
  or backtest outputs
* repository-relative documentation links only

## API And CLI Parity Checks

Confirm that:

* `src.corporate_actions.import_dividend_events` and
  `src.cli.import_corporate_actions_dividends` write the same curated dataset
  and artifact bundle for the same local inputs
* `load_dividend_events` reads from the dataset root and reconstructs partition
  columns
* CLI stdout remains deterministic JSON
* CLI documentation says local upstream artifacts only, no Alpaca calls, and no
  adjusted price data

## Catalog And Evidence Checks

Confirm that:

* direct scan discovers `artifacts/corporate_actions/<run_id>/`
* no separate corporate-actions registry is required
* dividend records use `record_family: corporate_action_event_dataset`
* query facets include `evidence_type: dividend_events`,
  `source_domain: corporate_actions`, and `event_domain: dividends`
* catalog records point to the canonical dataset root and import artifact files
* derived catalog indexes and evidence views remain disposable and
  non-authoritative

## Generated Output Hygiene

Generated M40 example output should remain under:

```text
docs/examples/output/m40_dividend_events/
```

That directory is ignored by default. Do not commit generated dividend datasets,
generated import artifacts, local cross-repo smoke outputs, package build
outputs, docs/path lint reports, or machine-specific validation reports.

Before merging:

* confirm `git status --short` contains only intentional source/doc changes
* confirm no `data/curated/events/dividends/` outputs are staged
* confirm no `artifacts/corporate_actions/` outputs are staged
* confirm no `dist/`, `build/`, or `*.egg-info/` outputs are staged
* confirm no local absolute paths appear in M40 docs

## Architecture Checks

Confirm M40 keeps the artifact-first boundary intact:

* canonical artifacts remain the source of truth
* direct scan remains available and canonical
* derived outputs remain disposable and non-authoritative
* provenance metadata remains evidence context, not a second source of truth
* no hidden execution behavior or implicit global state is introduced
* no live market data, credentials, network access, or external services are
  required by StratLake
* no OHLCV bars are mutated
* no adjusted prices are reconstructed
* dividend events do not adjust returns automatically
* strategy, alpha, portfolio, promotion, and backtest outputs remain unchanged

## Non-Goals Confirmed

M40 does not implement:

* a new live dividend ingestion engine
* live Alpaca API calls
* credential handling
* a scheduler
* a dashboard, web server, remote metadata service, or graph store
* adjusted-price reconstruction
* dividend reinvestment in backtests
* automatic strategy, alpha, portfolio, promotion, or backtest mutation
* generated artifacts committed to source control
* TestPyPI publication as an ordinary CI requirement

## Post-Merge Validation On Main

After merge:

* checkout and update `main`
* rerun the focused M40 validation slice
* rerun docs/path lint
* rerun package build validation
* run the full test suite or milestone validation bundle when practical
* confirm hosted GitHub Actions are green
* confirm no generated machine-specific files were committed
* confirm the release tag candidate is still appropriate
* create the release tag:
  `v0.40.0-corporate-actions-dividend-evidence`
* prepare GitHub Release notes from [M40 Release Notes](m40_release_notes.md)
