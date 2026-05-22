# Corporate Actions Event Contracts

M40 introduces corporate-action evidence as explicit downstream research events.
These contracts define the StratLake-facing event surface only. They do not
ingest live data, adjust prices, register catalog evidence, or change strategy,
alpha, portfolio, or backtest behavior.

## Dividend Events

Dividend events use the schema name `corporate_actions.dividends.v1` and schema
version `1.0.0`. The implementation lives in
`src/corporate_actions/dividend_contract.py`.

Required non-null fields:

| Field | Meaning |
| --- | --- |
| `symbol` | Security symbol from the upstream corporate-action record. |
| `event_type` | Dividend event type. Supported values are `cash_dividend` and `stock_dividend`. |
| `ex_date` | Ex-dividend date used for event timing evidence. |
| `process_date` | Upstream process date for the record. |
| `source` | Upstream source system identifier. |
| `source_event_id` | Upstream event identifier, mapped from `corporate_action_id`. |
| `source_payload_fingerprint` | Stable upstream payload fingerprint, mapped from `source_payload_hash`. |
| `as_of_date` | StratLake as-of date for the event evidence. Initially mapped from `process_date`. |
| `schema_version` | StratLake contract version. |

Required nullable fields must be present as columns but may contain null values:

| Field | Meaning |
| --- | --- |
| `declaration_date` | Declared date when supplied upstream. |
| `record_date` | Record date when supplied upstream. |
| `payable_date` | Payable date when supplied upstream. |
| `cash_amount` | Cash dividend amount when the event is cash-based. Required as a column even when null. |
| `stock_amount` | Stock dividend amount when the event is stock-based. Required as a column even when null. |
| `currency` | Currency for `cash_amount` when supplied upstream. |
| `raw_payload` | Upstream raw payload retained as evidence context, not as a second source of truth. |

## Keys

Recommended primary key:

```text
symbol
event_type
source_event_id
ex_date
process_date
source
```

The primary key uses the upstream event identifier when available. Validation
requires these columns and rejects duplicate primary-key rows in dataframe-like
inputs.

Recommended fallback key for later ingestion work when `source_event_id` is not
available upstream:

```text
symbol
event_type
ex_date
process_date
cash_amount
stock_amount
currency
source_payload_fingerprint
```

M40.1 defines the fallback-key semantics only. It does not relax the current
contract's required `source_event_id` field or implement fallback-key import
logic.

## Upstream Mapping

The companion `fintech-market-ingestion` output maps into StratLake fields as
follows:

| StratLake field | Upstream field |
| --- | --- |
| `symbol` | `symbol` |
| `event_type` | `corporate_action_type` |
| `ex_date` | `ex_date` |
| `process_date` | `process_date` |
| `source` | `source` |
| `source_event_id` | `corporate_action_id` |
| `source_payload_fingerprint` | `source_payload_hash` |
| `as_of_date` | `process_date` |
| `schema_version` | StratLake constant |
| `declaration_date` | `declaration_date` |
| `record_date` | `record_date` |
| `payable_date` | `payable_date` |
| `cash_amount` | `cash_amount` |
| `stock_amount` | `stock_amount` |
| `currency` | `currency` |
| `raw_payload` | `raw` |

## Non-Goals

The contract is the StratLake-facing event surface. It does not:

* call Alpaca or any external service;
* read live credentials;
* adjust OHLCV bars;
* create adjusted price datasets;
* reconstruct total returns;
* model dividend reinvestment;
* mutate strategy, alpha, portfolio, promotion, or backtest behavior;
* make provenance metadata a second source of truth.

## Local Dividend Importer

M40.2 adds a deterministic local-artifact importer in
`src/corporate_actions/dividend_importer.py`. It reads upstream
`dividends.parquet` and `metadata.json` files from the companion ingestion
project's local output, maps fields through the
`corporate_actions.dividends.v1` contract, validates the normalized rows, and
writes a curated event dataset under:

```text
data/curated/events/dividends/symbol=<SYMBOL>/year=<YYYY>/part-0.parquet
```

Import windows use half-open event-date semantics:

```text
start <= ex_date < end
```

The importer trims string fields, uppercases `symbol`, normalizes date fields to
`YYYY-MM-DD`, preserves nullable amount/date fields, serializes object-like
`raw_payload` values deterministically, and sorts output rows by:

```text
symbol
ex_date
event_type
process_date
source_event_id
source
```

Strict mode rejects invalid contract rows and duplicate primary-key rows after
writing the import artifact bundle. Advisory mode reports invalid rows, reports
duplicate primary-key row counts, drops invalid rows, and keeps the first
deterministic duplicate occurrence for the curated output.

The importer is local-file only. It does not import the upstream package, shell
out to upstream CLIs, call Alpaca, read credentials, use network access, adjust
OHLCV bars, create adjusted price datasets, reconstruct total returns, or
mutate strategy, alpha, portfolio, promotion, or backtest results.

## Import Artifacts

M40.3 adds a deterministic import artifact bundle for each dividend import run.
By default artifacts are written under:

```text
artifacts/corporate_actions/<run_id>/
```

The importer writes:

```text
manifest.json
summary.json
qa_summary.json
schema_contract.json
source_provenance.json
duplicate_events.csv
invalid_events.csv
import_config.json
```

JSON artifacts use sorted-key deterministic formatting. CSV artifacts use
deterministic row ordering. Persisted paths are portable POSIX-style relative
references and must not contain absolute local roots, credentials, tokens, or
machine-local usernames.

`qa_summary.json` reports import counts, duplicate and invalid row counts,
required-column checks, invalid event type counts, date parseability counts,
required-null counts, negative amount counts, missing cash-dividend currency
counts, rows outside the half-open import window, strict mode, QA status, and
the advisory duplicate policy. In advisory mode duplicate primary-key rows are
reported in `duplicate_events.csv` and the curated dataset keeps the first
deterministic occurrence. Invalid rows are reported in `invalid_events.csv` and
excluded from advisory-mode output.

`source_provenance.json` records local source-file fingerprints, upstream
package/source metadata when supplied, the schema name/version, import-config
fingerprint, path policy, and explicit `live_network_used: false` and
`credentials_used: false` flags. Provenance preserves source context for audit;
it does not create a second source of truth for upstream records.

`schema_contract.json` is the deterministic
`corporate_actions.dividends.v1` contract representation from
`src/corporate_actions/dividend_contract.py`.

For the release-facing architecture, workflow, provenance, QA, catalog, and
non-goal overview, see
[`docs/corporate_actions_dividend_evidence.md`](corporate_actions_dividend_evidence.md).

Fallback-key operationalization remains inactive for M40. `source_event_id`
stays required for imported records. Fallback-key definitions remain documented
future compatibility and should become active only through an explicit later
issue if a supported upstream source lacks event IDs.

## Catalog Discovery

M40.4 makes dividend import artifacts discoverable through the existing
read-only catalog direct scan. Dividend imports are not registered in a separate
corporate-actions registry, and catalog records are not authoritative. The
canonical sources remain the curated dividend event dataset and the import
artifact bundle.

Catalog records for dividend imports use the evidence family:

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

The catalog evidence metadata points to the canonical dataset root and import
artifacts:

```text
data/curated/events/dividends/
artifacts/corporate_actions/<run_id>/manifest.json
artifacts/corporate_actions/<run_id>/qa_summary.json
artifacts/corporate_actions/<run_id>/schema_contract.json
artifacts/corporate_actions/<run_id>/source_provenance.json
artifacts/corporate_actions/<run_id>/summary.json
```

Catalog query helpers can filter dividend evidence by `artifact_type`,
`evidence_type`, `source_domain`, `event_domain`, and `schema_version`. Derived
catalog/query/review surfaces remain disposable read models over direct-scan
source artifacts. Catalog validation can warn about missing dividend import
artifacts or non-portable metadata paths, but those warnings do not alter
promotion, governance, strategy, alpha, portfolio, backtest, OHLCV, or adjusted
price behavior.

## API, CLI, and Examples

The public Python API is the primary M40 entry point:

```python
from src.corporate_actions import import_dividend_events, load_dividend_events
```

`load_dividend_events(dataset_root)` reads from the partitioned dataset root so
Hive-style `symbol=<SYMBOL>/year=<YYYY>/` partition columns are reconstructed.
Individual partition files are not the supported downstream access pattern and
may not contain `symbol` as a physical file column.

The CLI is a thin wrapper around the same Python API:

```bash
python -m src.cli.import_corporate_actions_dividends \
  --source-data docs/examples/output/m40_dividend_events/fixtures/corporate_actions/dividends.parquet \
  --source-metadata docs/examples/output/m40_dividend_events/fixtures/corporate_actions/metadata.json \
  --output-root docs/examples/output/m40_dividend_events/data \
  --artifact-root docs/examples/output/m40_dividend_events/artifacts \
  --start 2024-01-01 \
  --end 2025-01-01 \
  --strict
```

CI-safe examples live in:

```text
docs/examples/m40_dividend_evidence_import_example.py
docs/examples/m40_dividend_pipeline_step_example.py
```

Both examples generate tiny synthetic local fixtures and write only under the
ignored `docs/examples/output/m40_dividend_events/` directory. They do not use
live market data, credentials, network access, adjusted prices, total-return
features, or dividend reinvestment logic.

## Cross-Repo Smoke Workflow

M40.7 documents an optional manual smoke workflow for the companion
`fintech-market-ingestion` repository handoff:

```text
docs/m40_cross_repo_q1_dividend_smoke_workflow.md
```

That workflow keeps upstream live ingestion outside StratLake. StratLake
consumes only explicit local `dividends.parquet` and `metadata.json` artifacts,
then writes the existing curated dividend event dataset and deterministic import
artifacts. The helper
`docs/examples/m40_cross_repo_q1_dividend_smoke_workflow.py` validates only the
StratLake side with explicit paths and is tested with synthetic fixtures.
