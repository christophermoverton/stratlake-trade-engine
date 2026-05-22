# Corporate Actions Dividend Evidence

M40 adds deterministic corporate-actions dividend evidence to StratLake as
explicit local artifacts. Dividend records are event evidence: they preserve
what an upstream source reported about dividend events, when StratLake imported
that local evidence, and how the import was validated.

The M40 boundary is intentionally narrow. StratLake consumes local upstream
files, writes a curated dividend event dataset and an import artifact bundle,
and exposes those artifacts through the existing read-only catalog. It does not
perform live Alpaca or market-data ingestion, require credentials, require
network access, adjust OHLCV bars, reconstruct adjusted prices, model dividend
reinvestment, or automatically alter strategy, alpha, portfolio, promotion, or
backtest results.

## Architecture Boundary

The supported handoff is:

```text
local upstream dividend artifacts
  -> src.corporate_actions import API or thin CLI
  -> data/curated/events/dividends/
  -> artifacts/corporate_actions/<run_id>/
  -> read-only catalog and evidence discovery
```

The local upstream artifacts may be produced by another project such as
`fintech-market-ingestion`, but M40 treats those files as the integration
boundary. StratLake does not import the upstream package, shell out to upstream
CLIs, call upstream APIs, or read upstream credentials.

Dividend evidence stays separate from:

* OHLCV price bars
* adjusted-price datasets
* strategy and alpha outputs
* portfolio and backtest results
* promotion, governance, and release decisions

Those workflows can review dividend evidence explicitly, but M40 does not
silently mutate their inputs or outputs.

## Dataset Contract

The StratLake schema name is `corporate_actions.dividends.v1`; the contract
schema version is `1.0.0`. The implementation lives in
`src/corporate_actions/dividend_contract.py` and is summarized in
[Corporate Actions Event Contracts](corporate_actions_event_contracts.md).

Required non-null columns:

| Column | Semantics |
| --- | --- |
| `symbol` | Security symbol, normalized to uppercase during import. |
| `event_type` | `cash_dividend` or `stock_dividend`. |
| `ex_date` | Event date used for import-window filtering. |
| `process_date` | Upstream process date for the record. |
| `source` | Upstream source system identifier. |
| `source_event_id` | Upstream event identifier, mapped from `corporate_action_id`. |
| `source_payload_fingerprint` | Stable upstream payload fingerprint, mapped from `source_payload_hash`. |
| `as_of_date` | StratLake evidence as-of date; currently mapped from `process_date` unless supplied. |
| `schema_version` | Contract version, currently `1.0.0`. |

Required nullable columns:

| Column | Semantics |
| --- | --- |
| `declaration_date` | Declaration date when supplied upstream. |
| `record_date` | Record date when supplied upstream. |
| `payable_date` | Payable date when supplied upstream. |
| `cash_amount` | Cash dividend amount when cash-based. |
| `stock_amount` | Stock dividend amount when stock-based. |
| `currency` | Currency for cash dividends when supplied. |
| `raw_payload` | Deterministically serialized upstream payload context. |

The primary key is:

```text
symbol
event_type
source_event_id
ex_date
process_date
source
```

The importer sorts deterministic output by:

```text
symbol
ex_date
event_type
process_date
source_event_id
source
```

Import windows are half-open on `ex_date`:

```text
start <= ex_date < end
```

Curated dividend events are written as Hive-style partitioned Parquet:

```text
data/curated/events/dividends/symbol=<SYMBOL>/year=<YYYY>/part-0.parquet
```

Use `load_dividend_events("data/curated/events/dividends")` on the dataset root
so partition columns such as `symbol` and `year` are reconstructed. Individual
partition files are not the supported downstream access pattern.

## Upstream Handoff

The expected upstream local files are:

```text
data/external/corporate_actions/dividends/dividends.parquet
data/external/corporate_actions/dividends/metadata.json
```

or another explicit local path with the same upstream column semantics. For the
`fintech-market-ingestion` handoff, the upstream fields map as documented in
[Corporate Actions Event Contracts](corporate_actions_event_contracts.md).

The upstream metadata should include explicit source context when available,
such as `source_vendor`, `upstream_package_name`, `upstream_package_version`,
`upstream_project`, and `upstream_source_repository`. StratLake stores this as
provenance context only. The curated dataset and import artifacts remain the
StratLake evidence outputs.

Live credentials, secrets, API calls, and network access are outside the
StratLake M40 scope.

## Local Import Workflow

CLI import:

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

Python API import:

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

Notebook-friendly review:

```python
from src.catalog import CatalogQuery, build_catalog, query_catalog
from src.corporate_actions import load_dividend_events

events = load_dividend_events("data/curated/events/dividends")
records = build_catalog("artifacts", repo_root=".")
dividend_records = query_catalog(records, CatalogQuery(evidence_type="dividend_events"))
```

Pipeline-friendly wrapper:

```python
from src.corporate_actions import import_dividend_events


def import_dividend_evidence_step() -> dict[str, object]:
    result = import_dividend_events(
        source_data_path="data/external/corporate_actions/dividends/dividends.parquet",
        source_metadata_path="data/external/corporate_actions/dividends/metadata.json",
        output_root="data/curated/events/dividends",
        artifact_root="artifacts/corporate_actions",
        start="2024-01-01",
        end="2025-01-01",
        strict=True,
    )
    return result.to_dict()
```

The wrapper should remain an explicit step. It should not be hidden inside
strategy, alpha, portfolio, or backtest execution.

## Provenance Model

Each import writes:

```text
artifacts/corporate_actions/<run_id>/manifest.json
artifacts/corporate_actions/<run_id>/summary.json
artifacts/corporate_actions/<run_id>/qa_summary.json
artifacts/corporate_actions/<run_id>/schema_contract.json
artifacts/corporate_actions/<run_id>/source_provenance.json
artifacts/corporate_actions/<run_id>/duplicate_events.csv
artifacts/corporate_actions/<run_id>/invalid_events.csv
artifacts/corporate_actions/<run_id>/import_config.json
```

`source_provenance.json` records:

* `source_data_path` and `source_metadata_path`
* `source_dataset_fingerprint`
* `import_config_fingerprint`
* upstream package and source metadata when supplied
* `live_network_used: false`
* `credentials_used: false`
* the repository-relative path policy

Metadata keys that look like secrets, tokens, credentials, API keys, or
passwords are redacted before being copied into provenance. Provenance is audit
context for the evidence handoff; it is not a second source of truth and does
not replace the source dataset, curated dataset, or import artifact bundle.

## QA And Data Quality

The importer and contract validator check:

* required column presence
* required nullable column presence
* non-null required values
* supported `event_type` values
* schema version
* required and nullable date parseability
* half-open `ex_date` import-window filtering
* duplicate primary-key rows
* negative `cash_amount` and `stock_amount` counts
* missing `currency` for cash dividends
* deterministic sorting and partition writing
* portable artifact paths

Strict mode rejects duplicate primary-key rows and invalid contract rows after
writing the import artifact bundle. Advisory mode reports invalid rows and
duplicates, excludes invalid rows, and keeps the first deterministic duplicate
occurrence in curated output.

QA outputs are written to `qa_summary.json`, `duplicate_events.csv`, and
`invalid_events.csv`. These files explain what happened during import; they do
not adjust prices, alter returns, or replay research decisions.

## Catalog Discovery

Dividend imports are discovered by direct scan under
`artifacts/corporate_actions/<run_id>/`. They do not require a separate
corporate-actions registry.

Catalog records use these facets:

```text
record_family: corporate_action_event_dataset
run_type: corporate_action_event_dataset
artifact_type: corporate_action_event_dataset
evidence_type: dividend_events
source_domain: corporate_actions
event_domain: dividends
schema_version: corporate_actions.dividends.v1
canonicality: canonical_import_artifact
```

The catalog stores portable references to the canonical dataset root and import
artifacts. Direct scan remains canonical for discovery. Derived catalog indexes,
evidence views, and review outputs remain rebuildable, disposable, and
non-authoritative.

## Non-Goals

M40 does not add:

* live Alpaca calls
* hidden OHLCV adjustment
* adjusted-price reconstruction
* dividend reinvestment in backtests
* remote metadata services
* dashboards or servers
* schedulers
* graph stores
* credential requirements
* generated outputs committed to source control
* local absolute paths in docs or artifacts
* ordinary CI dependence on TestPyPI
* automatic strategy, alpha, portfolio, promotion, or backtest mutation

## Further Reading

* [Corporate Actions Event Contracts](corporate_actions_event_contracts.md)
* [M40 Cross-Repo Q1 Dividend Smoke Workflow](m40_cross_repo_q1_dividend_smoke_workflow.md)
* [Catalog Indexer](catalog_indexer.md)
* [Catalog Evidence Explorer](catalog_evidence_explorer.md)
* [Notebook Execution API](notebook_execution_api.md)
* [Pipeline Integration](pipeline_integration.md)
* [M40 Release Notes](m40_release_notes.md)
* [M40 Release Validation Checklist](m40_release_validation_checklist.md)
