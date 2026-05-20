# M40 Cross-Repo Q1 Dividend Smoke Workflow

This optional smoke workflow checks the manual handoff from the companion
`fintech-market-ingestion` repository into StratLake's M40 dividend event
evidence stack. It is an integration check for local workstations, not a new
canonical ingestion engine inside StratLake.

The handoff remains:

```text
fintech-market-ingestion
  -> local dividends.parquet + metadata.json
  -> stratlake-trade-engine M40 dividend evidence import
  -> curated dividend event dataset + import artifacts + catalog evidence
```

Ordinary StratLake CI remains synthetic, deterministic, credential-free, and
network-free. Do not commit downloaded live dividend data or generated live
import artifacts.

## Prerequisites

* A local checkout of `christophermoverton/fintech-market-ingestion`.
* Local Alpaca credentials configured only for the upstream ingestion step.
* A local checkout of `christophermoverton/stratlake-trade-engine`.
* StratLake dependencies installed locally.

StratLake consumes only local upstream artifacts. It does not call Alpaca, read
credentials, require network access, import `fintech-market-ingestion`, or shell
out to `fintech-ingest-corporate-actions`.

## Upstream Producer

Run this from `fintech-market-ingestion` with local credentials configured:

```bash
fintech-ingest-corporate-actions \
  --symbols AAPL \
  --start 2024-01-01 \
  --end 2024-04-01 \
  --types cash_dividend stock_dividend \
  --output-root data/curated/corporate_actions/dividends
```

Expected local upstream outputs:

```text
data/curated/corporate_actions/dividends/dividends.parquet
data/curated/corporate_actions/dividends/metadata.json
```

## StratLake Import

Run this from `stratlake-trade-engine`, replacing the upstream checkout
placeholder with your explicit local path:

```bash
python -m src.cli.import_corporate_actions_dividends \
  --source-data <path-to-fintech-market-ingestion>/data/curated/corporate_actions/dividends/dividends.parquet \
  --source-metadata <path-to-fintech-market-ingestion>/data/curated/corporate_actions/dividends/metadata.json \
  --output-root data/curated/events/dividends \
  --artifact-root artifacts/corporate_actions \
  --start 2024-01-01 \
  --end 2024-04-01 \
  --strict
```

The importer applies half-open event-date semantics:

```text
start <= ex_date < end
```

It writes curated dividend event data under:

```text
data/curated/events/dividends/symbol=<SYMBOL>/year=<YYYY>/part-0.parquet
```

and deterministic import artifacts under:

```text
artifacts/corporate_actions/<run_id>/
```

## Python API Equivalent

```python
from src.corporate_actions import import_dividend_events, load_dividend_events

result = import_dividend_events(
    source_data_path="<path-to-fintech-market-ingestion>/data/curated/corporate_actions/dividends/dividends.parquet",
    source_metadata_path="<path-to-fintech-market-ingestion>/data/curated/corporate_actions/dividends/metadata.json",
    output_root="data/curated/events/dividends",
    artifact_root="artifacts/corporate_actions",
    start="2024-01-01",
    end="2024-04-01",
    strict=True,
)

events = load_dividend_events("data/curated/events/dividends")
print(result.to_dict())
print(events.head())
```

Read curated events from the dataset root so Hive-style `symbol=<SYMBOL>` and
`year=<YYYY>` partition columns are reconstructed. Individual partition files
are not the supported downstream access pattern.

## Catalog Verification

```python
from src.catalog import CatalogQuery, build_catalog, query_catalog

records = build_catalog("artifacts", repo_root=".")
dividend_records = query_catalog(records, CatalogQuery(evidence_type="dividend_events"))

for record in dividend_records:
    print(record.run_id, record.qa_status, record.metadata.get("evidence", {}).get("canonical_dataset_root"))
```

Dividend evidence appears as `corporate_action_event_dataset` records with
`evidence_type: dividend_events`, `source_domain: corporate_actions`, and
`event_domain: dividends`. Catalog records are read-only discovery views; the
curated dataset and import artifact bundle remain the source of truth.

## Optional StratLake-Side Helper

The helper below validates the StratLake side of the handoff using explicit
local artifact paths. It does not run the upstream command or require upstream
dependencies.

```bash
python docs/examples/m40_cross_repo_q1_dividend_smoke_workflow.py \
  --source-data <path-to-fintech-market-ingestion>/data/curated/corporate_actions/dividends/dividends.parquet \
  --source-metadata <path-to-fintech-market-ingestion>/data/curated/corporate_actions/dividends/metadata.json \
  --output-root data/curated/events/dividends \
  --artifact-root artifacts/corporate_actions \
  --start 2024-01-01 \
  --end 2024-04-01 \
  --symbol AAPL
```

The helper prints deterministic JSON with the StratLake `run_id`, written row
count, QA status, artifact path, dataset root, and catalog discovery result.

## Manual Validation Checklist

Record only these fields in any local smoke notes:

```text
upstream command used
symbol/date window
local upstream artifact paths
StratLake run_id
written_row_count
qa_status
artifact_path
catalog discovery result
```

Do not commit live upstream files, curated live dividend datasets, generated
live import artifacts, local absolute paths, credentials, tokens, or local
machine-specific values.

## Evidence Boundary

Dividend records remain explicit event evidence. This workflow does not adjust
OHLCV bars, create adjusted price datasets, reconstruct total returns, model
dividend reinvestment, mutate backtest cash flows, or alter strategy, alpha,
portfolio, governance, or promotion behavior.
