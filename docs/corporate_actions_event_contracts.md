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

M40.1 is contract-only. It does not:

* call Alpaca or any external service;
* read live credentials;
* implement the dividend importer;
* write import artifacts, QA summaries, provenance files, or catalog records;
* adjust OHLCV bars;
* create adjusted price datasets;
* reconstruct total returns;
* model dividend reinvestment;
* mutate strategy, alpha, portfolio, or backtest behavior.

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

Strict mode rejects invalid contract rows and duplicate primary-key rows.
Advisory mode reports invalid rows, reports duplicate primary-key row counts,
drops invalid rows, and keeps the first deterministic duplicate occurrence for
the curated output. Full QA and provenance artifacts are deferred to later M40
issues.

The importer is local-file only. It does not import the upstream package, shell
out to upstream CLIs, call Alpaca, read credentials, use network access, adjust
OHLCV bars, create adjusted price datasets, reconstruct total returns, or
register catalog evidence.
