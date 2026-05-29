from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import duckdb
import pandas as pd
import yaml

from src.artifacts.safety import atomic_write_json
from src.data.catalog import build_where_clause, parquet_scan_sql
from src.data.contract_validation import BarsContract

SUPPORTED_TIMEFRAME_DATASETS: dict[str, str] = {
    "1D": "bars_daily",
    "1Min": "bars_1m",
}
REQUIRED_BARS_COLUMNS = (
    "symbol",
    "ts_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "source",
    "timeframe",
    "date",
)
ARCHIVE_PACK_MARKERS = (
    "manifest.json",
    "archive_index.json",
    "checksums.json",
    "restore_plan.json",
)


@dataclass(frozen=True)
class HandoffCheck:
    name: str
    status: str
    severity: str
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "name": self.name,
            "status": self.status,
            "severity": self.severity,
            "message": self.message,
        }
        if self.details:
            data["details"] = _stable_jsonable(self.details)
        return data


@dataclass(frozen=True)
class MarketLakeHandoffValidationResult:
    schema_version: int
    status: str
    validated: bool
    authoritative: bool
    root: str
    marketlake_root: str
    universe: str
    paths_config: str | None
    dataset_name: str
    timeframe: str
    start: str
    end: str
    checks: tuple[HandoffCheck, ...]
    symbols: dict[str, Any]
    coverage: dict[str, Any]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "validated": self.validated,
            "authoritative": self.authoritative,
            "root": self.root,
            "marketlake_root": self.marketlake_root,
            "universe": self.universe,
            "paths_config": self.paths_config,
            "dataset_name": self.dataset_name,
            "timeframe": self.timeframe,
            "start": self.start,
            "end": self.end,
            "checks": [check.to_dict() for check in self.checks],
            "symbols": _stable_jsonable(self.symbols),
            "coverage": _stable_jsonable(self.coverage),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class UniverseSelection:
    symbols: tuple[str, ...]
    source: str
    source_path: str | None
    payload: dict[str, Any]


def validate_marketlake_handoff(
    *,
    root: str | Path,
    marketlake_root: str | Path,
    universe: str | Path,
    start: str,
    end: str,
    timeframe: str,
    output: str | Path | None = None,
) -> MarketLakeHandoffValidationResult:
    session_root = Path(root).expanduser().resolve()
    curated_root = Path(marketlake_root).expanduser().resolve()
    universe_path = Path(universe).expanduser().resolve()
    dataset_name = _dataset_name_for_timeframe(timeframe)
    paths_config_path = session_root / "configs" / "paths.yml"

    checks: list[HandoffCheck] = []
    warnings: list[str] = []
    errors: list[str] = []

    checks.extend(_validate_session_root(session_root))
    checks.extend(_validate_marketlake_root(curated_root))

    universe_selection = _load_universe_selection(universe_path, session_root)
    checks.append(
        HandoffCheck(
            name="universe_load",
            status="pass" if universe_selection.symbols else "fail",
            severity="error" if universe_selection.symbols else "error",
            message=(
                f"Loaded {len(universe_selection.symbols)} requested symbol(s) from "
                f"{universe_selection.source_path or universe_selection.source}."
            )
            if universe_selection.symbols
            else "Universe config did not define any requested symbols or ticker-file reference.",
            details={
                "source": universe_selection.source,
                "source_path": universe_selection.source_path,
                "symbol_count": len(universe_selection.symbols),
            },
        )
    )

    requested_symbols = list(universe_selection.symbols)
    dataset_root = curated_root / dataset_name
    dataset_files = sorted(dataset_root.rglob("*.parquet")) if dataset_root.exists() else []
    if not dataset_root.exists():
        checks.append(
            HandoffCheck(
                name="dataset_root_exists",
                status="fail",
                severity="error",
                message=f"Expected dataset root is missing: {dataset_root.as_posix()}",
                details={"dataset_root": dataset_root.as_posix()},
            )
        )
    elif not dataset_root.is_dir():
        checks.append(
            HandoffCheck(
                name="dataset_root_directory",
                status="fail",
                severity="error",
                message=f"Expected dataset root is not a directory: {dataset_root.as_posix()}",
                details={"dataset_root": dataset_root.as_posix()},
            )
        )
    elif not dataset_files:
        checks.append(
            HandoffCheck(
                name="dataset_files_present",
                status="fail",
                severity="error",
                message=f"No parquet files were found under {dataset_root.as_posix()}.",
                details={"dataset_root": dataset_root.as_posix()},
            )
        )
    else:
        checks.append(
            HandoffCheck(
                name="dataset_files_present",
                status="pass",
                severity="info",
                message=f"Found {len(dataset_files)} parquet file(s) under {dataset_root.as_posix()}.",
                details={
                    "dataset_root": dataset_root.as_posix(),
                    "parquet_file_count": len(dataset_files),
                },
            )
        )

    if dataset_files:
        schema_check = _validate_schema_and_load(
            dataset_root=dataset_root,
            dataset_name=dataset_name,
            timeframe=timeframe,
            symbols=requested_symbols,
            start=start,
            end=end,
        )
        checks.extend(schema_check.checks)
        warnings.extend(schema_check.warnings)
        errors.extend(schema_check.errors)
        available_symbols = schema_check.available_symbols
        overall_stats = schema_check.overall_stats
        window_rows = schema_check.window_rows
        symbol_rows = schema_check.symbol_rows
    else:
        available_symbols = ()
        overall_stats = {}
        window_rows = pd.DataFrame()
        symbol_rows = []

    paths_config_check = _validate_paths_config_alignment(
        paths_config_path=paths_config_path,
        session_root=session_root,
        marketlake_root=curated_root,
    )
    checks.append(paths_config_check)
    if paths_config_check.status == "warn":
        warnings.append(paths_config_check.message)
    elif paths_config_check.status == "fail":
        errors.append(paths_config_check.message)

    requested_sorted = sorted({symbol.upper() for symbol in requested_symbols})
    available_sorted = sorted({symbol.upper() for symbol in available_symbols})
    missing_symbols = [symbol for symbol in requested_sorted if symbol not in set(available_sorted)]

    symbols_by_name: dict[str, dict[str, Any]] = {}
    for symbol in requested_sorted:
        row = next((item for item in symbol_rows if item["symbol"] == symbol), None)
        if row is None:
            symbols_by_name[symbol] = {
                "available": False,
                "dataset_row_count": 0,
                "window_row_count": 0,
                "first_date": None,
                "last_date": None,
                "window_first_date": None,
                "window_last_date": None,
            }
            continue
        symbols_by_name[symbol] = row

    coverage_pct = 1.0
    if requested_sorted:
        coverage_pct = round(
            (len(requested_sorted) - len(missing_symbols)) / len(requested_sorted), 6
        )

    symbol_coverage_pass = not missing_symbols
    checks.append(
        HandoffCheck(
            name="symbol_coverage",
            status="pass" if symbol_coverage_pass else "fail",
            severity="error",
            message=(
                f"All {len(requested_sorted)} requested symbol(s) are present in the curated dataset."
                if symbol_coverage_pass
                else f"Missing requested symbol(s): {', '.join(missing_symbols)}."
            ),
            details={
                "requested_symbol_count": len(requested_sorted),
                "available_symbol_count": len(available_sorted),
                "missing_symbols": missing_symbols,
                "coverage_pct": coverage_pct,
            },
        )
    )

    window_coverage_failures = [
        symbol
        for symbol, row in symbols_by_name.items()
        if row["dataset_row_count"] > 0 and row["window_row_count"] == 0
    ]
    if requested_sorted:
        checks.append(
            HandoffCheck(
                name="window_coverage",
                status="pass" if not window_coverage_failures and not missing_symbols else "fail",
                severity="error",
                message=(
                    f"All requested symbol(s) have rows in the requested half-open window [{start}, {end})."
                    if not window_coverage_failures and not missing_symbols
                    else f"Symbols with no rows in requested window [{start}, {end}): {', '.join(window_coverage_failures or missing_symbols)}."
                ),
                details={
                    "window_start": start,
                    "window_end": end,
                    "failures": window_coverage_failures,
                },
            )
        )

    if dataset_files:
        checks.append(
            HandoffCheck(
                name="date_range_coverage",
                status="pass" if overall_stats else "fail",
                severity="info" if overall_stats else "error",
                message=(
                    f"Dataset spans {overall_stats.get('dataset_min_date')} through {overall_stats.get('dataset_max_date')}."
                    if overall_stats
                    else "Unable to determine dataset date coverage.",
                ),
                details=overall_stats,
            )
        )

    status = _overall_status(checks)
    if status == "warn":
        warnings.append("Handoff validation completed with warnings.")
    elif status == "fail":
        errors.append("Handoff validation failed.")

    result = MarketLakeHandoffValidationResult(
        schema_version=1,
        status=status,
        validated=True,
        authoritative=False,
        root=session_root.as_posix(),
        marketlake_root=curated_root.as_posix(),
        universe=universe_path.as_posix(),
        paths_config=paths_config_path.as_posix() if paths_config_path.exists() else None,
        dataset_name=dataset_name,
        timeframe=timeframe,
        start=start,
        end=end,
        checks=tuple(checks),
        symbols={
            "requested": requested_sorted,
            "available": available_sorted,
            "missing": missing_symbols,
            "by_symbol": [symbols_by_name[symbol] for symbol in requested_sorted],
        },
        coverage={
            "requested_window": {
                "start": start,
                "end": end,
                "semantics": "half-open",
            },
            "coverage_pct": coverage_pct,
            "coverage_threshold": 1.0,
            "dataset_min_date": overall_stats.get("dataset_min_date"),
            "dataset_max_date": overall_stats.get("dataset_max_date"),
            "symbol_count": len(requested_sorted),
            "available_symbol_count": len(available_sorted),
            "window_symbol_count": sum(
                1 for row in symbols_by_name.values() if row["window_row_count"] > 0
            ),
            "window_row_count": int(len(window_rows.index)) if not window_rows.empty else 0,
        },
        errors=tuple(
            dict.fromkeys(errors + [check.message for check in checks if check.status == "fail"])
        ),
        warnings=tuple(
            dict.fromkeys(warnings + [check.message for check in checks if check.status == "warn"])
        ),
    )

    if output is not None:
        output_path = Path(output)
        _validate_output_path(output_path=output_path, marketlake_root=curated_root)
        atomic_write_json(output_path, result.to_dict(), sort_keys=True)
    return result


def write_marketlake_handoff_report(
    result: MarketLakeHandoffValidationResult, output: str | Path
) -> Path:
    return atomic_write_json(output, result.to_dict(), sort_keys=True)


def _validate_output_path(*, output_path: Path, marketlake_root: Path) -> None:
    resolved_output = output_path.expanduser().resolve()
    resolved_marketlake_root = marketlake_root.expanduser().resolve()
    try:
        resolved_output.relative_to(resolved_marketlake_root)
        inside_marketlake_root = True
    except ValueError:
        inside_marketlake_root = False
    if resolved_output == resolved_marketlake_root or inside_marketlake_root:
        raise ValueError("Refusing to write handoff validation report inside MarketLake root.")


def _validate_session_root(session_root: Path) -> list[HandoffCheck]:
    checks: list[HandoffCheck] = []
    if not session_root.exists():
        checks.append(
            HandoffCheck(
                name="session_root_exists",
                status="fail",
                severity="error",
                message=f"StratLake root does not exist: {session_root.as_posix()}",
                details={"root": session_root.as_posix()},
            )
        )
        return checks
    if not session_root.is_dir():
        checks.append(
            HandoffCheck(
                name="session_root_directory",
                status="fail",
                severity="error",
                message=f"StratLake root is not a directory: {session_root.as_posix()}",
                details={"root": session_root.as_posix()},
            )
        )
        return checks
    try:
        next(session_root.iterdir())
        readable = True
    except StopIteration:
        readable = True
    except OSError as exc:
        readable = False
        checks.append(
            HandoffCheck(
                name="session_root_readable",
                status="fail",
                severity="error",
                message=f"StratLake root is not readable: {exc}",
                details={"root": session_root.as_posix()},
            )
        )
    if readable:
        checks.append(
            HandoffCheck(
                name="session_root_readable",
                status="pass",
                severity="info",
                message=f"StratLake root is readable: {session_root.as_posix()}",
                details={"root": session_root.as_posix()},
            )
        )
    return checks


def _validate_marketlake_root(curated_root: Path) -> list[HandoffCheck]:
    checks: list[HandoffCheck] = []
    if not curated_root.exists():
        checks.append(
            HandoffCheck(
                name="marketlake_root_exists",
                status="fail",
                severity="error",
                message=f"MarketLake root is missing: {curated_root.as_posix()}",
                details={"marketlake_root": curated_root.as_posix()},
            )
        )
        return checks
    if not curated_root.is_dir():
        checks.append(
            HandoffCheck(
                name="marketlake_root_directory",
                status="fail",
                severity="error",
                message=f"MarketLake root is not a directory: {curated_root.as_posix()}",
                details={"marketlake_root": curated_root.as_posix()},
            )
        )
        return checks
    if any((curated_root / marker).exists() for marker in ARCHIVE_PACK_MARKERS):
        checks.append(
            HandoffCheck(
                name="marketlake_root_not_archive_pack",
                status="fail",
                severity="error",
                message=(
                    "Drive archive-pack directories are not canonical MarketLake roots; "
                    f"found archive-pack markers under {curated_root.as_posix()}."
                ),
                details={"marketlake_root": curated_root.as_posix()},
            )
        )
    else:
        checks.append(
            HandoffCheck(
                name="marketlake_root_not_archive_pack",
                status="pass",
                severity="info",
                message=f"MarketLake root is not an archive pack: {curated_root.as_posix()}",
                details={"marketlake_root": curated_root.as_posix()},
            )
        )
    try:
        next(curated_root.iterdir())
        readable = True
    except StopIteration:
        readable = True
    except OSError as exc:
        readable = False
        checks.append(
            HandoffCheck(
                name="marketlake_root_readable",
                status="fail",
                severity="error",
                message=f"MarketLake root is not readable: {exc}",
                details={"marketlake_root": curated_root.as_posix()},
            )
        )
    if readable:
        checks.append(
            HandoffCheck(
                name="marketlake_root_readable",
                status="pass",
                severity="info",
                message=f"MarketLake root is readable: {curated_root.as_posix()}",
                details={"marketlake_root": curated_root.as_posix()},
            )
        )
    return checks


@dataclass(frozen=True)
class _DatasetValidationResult:
    checks: tuple[HandoffCheck, ...]
    warnings: list[str]
    errors: list[str]
    available_symbols: tuple[str, ...]
    overall_stats: dict[str, Any]
    window_rows: pd.DataFrame
    symbol_rows: list[dict[str, Any]]


def _validate_schema_and_load(
    *,
    dataset_root: Path,
    dataset_name: str,
    timeframe: str,
    symbols: Sequence[str],
    start: str,
    end: str,
) -> _DatasetValidationResult:
    checks: list[HandoffCheck] = []
    warnings: list[str] = []
    errors: list[str] = []
    dataset_glob = (dataset_root / "**" / "*.parquet").as_posix()

    columns = _describe_parquet_columns(dataset_glob)
    schema_columns = [column[0] for column in columns]
    missing_columns = [column for column in REQUIRED_BARS_COLUMNS if column not in schema_columns]
    if missing_columns:
        message = f"Dataset schema is missing required column(s): {', '.join(missing_columns)}."
        checks.append(
            HandoffCheck(
                name="dataset_schema",
                status="fail",
                severity="error",
                message=message,
                details={"missing_columns": missing_columns, "columns": schema_columns},
            )
        )
        errors.append(message)
    else:
        checks.append(
            HandoffCheck(
                name="dataset_schema",
                status="pass",
                severity="info",
                message=f"Dataset schema includes required columns for {dataset_name}.",
                details={"columns": schema_columns},
            )
        )

    if missing_columns:
        return _DatasetValidationResult(
            checks=tuple(checks),
            warnings=warnings,
            errors=errors,
            available_symbols=(),
            overall_stats={},
            window_rows=pd.DataFrame(),
            symbol_rows=[],
        )

    connection = duckdb.connect(database=":memory:")
    try:
        select_sql = f"SELECT * FROM {parquet_scan_sql(dataset_glob)}"
        timeframe_where_sql, params = build_where_clause(
            symbols=symbols, start_date=start, end_date=end
        )
        params = dict(params)
        params["timeframe"] = timeframe
        where_sql = "WHERE timeframe = $timeframe"
        if timeframe_where_sql:
            where_sql = f"{where_sql} AND {timeframe_where_sql.removeprefix('WHERE ')}"

        window_rows = connection.execute(
            f"{select_sql} {where_sql} ORDER BY symbol, ts_utc, timeframe, date",
            params,
        ).df()
        full_rows = connection.execute(
            f"{select_sql} WHERE timeframe = $timeframe ORDER BY symbol, ts_utc, timeframe, date",
            {"timeframe": timeframe},
        ).df()
    finally:
        connection.close()

    if not window_rows.empty:
        window_rows["symbol"] = window_rows["symbol"].astype(str).str.upper()
        window_rows["timeframe"] = window_rows["timeframe"].astype(str)
        window_rows["date"] = window_rows["date"].astype(str)
    if not full_rows.empty:
        full_rows["symbol"] = full_rows["symbol"].astype(str).str.upper()
        full_rows["timeframe"] = full_rows["timeframe"].astype(str)
        full_rows["date"] = full_rows["date"].astype(str)

    contract = BarsContract()
    try:
        contract.validate(window_rows.copy(), strict=True, normalize_ts_utc=True)
        if "date" in window_rows.columns and window_rows["date"].isna().any():
            raise ValueError("date column contains null values.")
        checks.append(
            HandoffCheck(
                name="dataset_contract",
                status="pass",
                severity="info",
                message=f"Dataset rows satisfy the curated bars contract for timeframe {timeframe}.",
                details={"row_count": len(window_rows.index)},
            )
        )
    except Exception as exc:
        message = f"Dataset schema is incompatible with downstream feature building: {exc}"
        checks.append(
            HandoffCheck(
                name="dataset_contract",
                status="fail",
                severity="error",
                message=message,
                details={"row_count": len(window_rows.index)},
            )
        )
        errors.append(message)

    if not full_rows.empty:
        distinct_timeframes = sorted(
            {str(value) for value in full_rows["timeframe"].dropna().tolist()}
        )
        if distinct_timeframes == [timeframe]:
            checks.append(
                HandoffCheck(
                    name="timeframe_alignment",
                    status="pass",
                    severity="info",
                    message=f"All dataset rows use timeframe {timeframe}.",
                    details={"timeframes": distinct_timeframes},
                )
            )
        else:
            message = (
                f"Dataset contains unexpected timeframe value(s): {', '.join(distinct_timeframes)}."
            )
            checks.append(
                HandoffCheck(
                    name="timeframe_alignment",
                    status="fail",
                    severity="error",
                    message=message,
                    details={"timeframes": distinct_timeframes},
                )
            )
            errors.append(message)

    available_symbols = tuple(
        sorted(
            {
                str(symbol).upper()
                for symbol in full_rows.get("symbol", pd.Series(dtype=str)).dropna().tolist()
            }
        )
    )
    overall_stats: dict[str, Any] = {}
    if not full_rows.empty and "date" in full_rows.columns:
        per_symbol: list[dict[str, Any]] = []
        for symbol in sorted(
            {str(symbol).upper() for symbol in full_rows["symbol"].dropna().tolist()}
        ):
            symbol_full = full_rows[full_rows["symbol"].astype(str).str.upper() == symbol].copy()
            symbol_window = (
                window_rows[window_rows["symbol"].astype(str).str.upper() == symbol].copy()
                if not window_rows.empty
                else symbol_full.iloc[0:0]
            )
            row = {
                "symbol": symbol,
                "available": True,
                "dataset_row_count": int(len(symbol_full.index)),
                "window_row_count": int(len(symbol_window.index)),
                "first_date": _min_date_string(symbol_full["date"]),
                "last_date": _max_date_string(symbol_full["date"]),
                "window_first_date": _min_date_string(symbol_window["date"]),
                "window_last_date": _max_date_string(symbol_window["date"]),
            }
            per_symbol.append(row)
        overall_stats = {
            "dataset_min_date": _min_date_string(full_rows["date"]),
            "dataset_max_date": _max_date_string(full_rows["date"]),
            "available_symbol_count": len(available_symbols),
            "dataset_row_count": int(len(full_rows.index)),
            "per_symbol": per_symbol,
        }
    return _DatasetValidationResult(
        checks=tuple(checks),
        warnings=warnings,
        errors=errors,
        available_symbols=available_symbols,
        overall_stats=overall_stats,
        window_rows=window_rows,
        symbol_rows=overall_stats.get("per_symbol", []),
    )


def _describe_parquet_columns(dataset_glob: str) -> list[tuple[str, str]]:
    connection = duckdb.connect(database=":memory:")
    try:
        query = f"DESCRIBE SELECT * FROM {parquet_scan_sql(dataset_glob)}"
        rows = connection.execute(query).fetchall()
    finally:
        connection.close()
    return [(str(row[0]), str(row[1])) for row in rows]


def _load_universe_selection(universe_path: Path, session_root: Path) -> UniverseSelection:
    payload = _load_yaml(universe_path)
    source_path: str | None = universe_path.as_posix()
    source = "symbols"
    symbols = _extract_symbols(payload, session_root=session_root, universe_path=universe_path)
    if (
        not _has_explicit_symbol_list(payload)
        and _resolve_config_path(
            payload,
            keys=("tickers_file", "tickers_path", "tickers"),
            session_root=session_root,
            universe_path=universe_path,
        )
        is not None
    ):
        source = "tickers_file"
        source_path = _resolve_config_path(
            payload,
            keys=("tickers_file", "tickers_path", "tickers"),
            session_root=session_root,
            universe_path=universe_path,
        ).as_posix()
    return UniverseSelection(
        symbols=tuple(symbols),
        source=source,
        source_path=source_path,
        payload=payload,
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Universe config not found: {path.as_posix()}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Universe config must be a mapping: {path.as_posix()}")
    return payload


def _extract_symbols(
    payload: Mapping[str, Any],
    *,
    session_root: Path,
    universe_path: Path,
) -> tuple[str, ...]:
    symbol_candidates: list[str] = []
    for key in ("symbols", "tickers", "universe_symbols"):
        value = payload.get(key)
        if isinstance(value, list):
            symbol_candidates.extend(
                str(item).strip().upper() for item in value if str(item).strip()
            )
        elif isinstance(value, str) and value.strip():
            symbol_candidates.append(value.strip().upper())

    nested = payload.get("universe")
    if isinstance(nested, Mapping):
        nested_symbols = nested.get("symbols")
        if isinstance(nested_symbols, list):
            symbol_candidates.extend(
                str(item).strip().upper() for item in nested_symbols if str(item).strip()
            )

    if symbol_candidates:
        return tuple(dict.fromkeys(sorted(set(symbol_candidates))))

    file_path = _resolve_config_path(
        payload,
        keys=("tickers_file", "tickers_path", "tickers"),
        session_root=session_root,
        universe_path=universe_path,
    )
    if file_path is None:
        return ()
    return tuple(_load_ticker_file(file_path))


def _has_explicit_symbol_list(payload: Mapping[str, Any]) -> bool:
    for key in ("symbols", "tickers", "universe_symbols"):
        value = payload.get(key)
        if isinstance(value, list) and value:
            return True
        if isinstance(value, str) and value.strip():
            return True
    nested = payload.get("universe")
    if isinstance(nested, Mapping):
        nested_symbols = nested.get("symbols")
        if isinstance(nested_symbols, list) and nested_symbols:
            return True
    return False


def _resolve_config_path(
    payload: Mapping[str, Any],
    *,
    keys: Sequence[str],
    session_root: Path,
    universe_path: Path,
) -> Path | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            candidate = Path(value).expanduser()
            if candidate.is_absolute():
                return candidate.resolve()
            return (session_root / candidate).resolve()
    return None


def _load_ticker_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Ticker file referenced by universe config not found: {path.as_posix()}"
        )
    symbols = [
        line.strip().upper()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return list(dict.fromkeys(symbols))


def _validate_paths_config_alignment(
    *,
    paths_config_path: Path,
    session_root: Path,
    marketlake_root: Path,
) -> HandoffCheck:
    if not paths_config_path.exists():
        return HandoffCheck(
            name="paths_config_alignment",
            status="warn",
            severity="warning",
            message=f"Path config not found at {paths_config_path.as_posix()}.",
            details={"paths_config": paths_config_path.as_posix()},
        )

    payload = _load_yaml(paths_config_path)
    expected_root_values = {
        "project_root": ".",
        "configs_root": "configs",
    }
    for key, expected_value in expected_root_values.items():
        value = payload.get(key)
        if value is None:
            continue
        if str(value) != expected_value:
            return HandoffCheck(
                name="paths_config_alignment",
                status="fail",
                severity="error",
                message=(
                    f"Path config mismatch for {key}: expected {expected_value!r} but found {value!r}."
                ),
                details={"key": key, "expected": expected_value, "observed": value},
            )

    marketlake_value = payload.get("marketlake_root")
    if isinstance(marketlake_value, str) and marketlake_value.strip():
        candidate = Path(marketlake_value).expanduser()
        resolved = (
            candidate.resolve() if candidate.is_absolute() else (session_root / candidate).resolve()
        )
        if resolved != marketlake_root:
            return HandoffCheck(
                name="paths_config_alignment",
                status="fail",
                severity="error",
                message=(
                    "Path config marketlake_root does not match the requested MarketLake root."
                ),
                details={
                    "expected_marketlake_root": marketlake_root.as_posix(),
                    "observed_marketlake_root": resolved.as_posix(),
                },
            )

    return HandoffCheck(
        name="paths_config_alignment",
        status="pass",
        severity="info",
        message="Path config aligns with the requested notebook/session roots.",
        details={"paths_config": paths_config_path.as_posix()},
    )


def _dataset_name_for_timeframe(timeframe: str) -> str:
    try:
        return SUPPORTED_TIMEFRAME_DATASETS[timeframe]
    except KeyError as exc:
        expected = ", ".join(sorted(SUPPORTED_TIMEFRAME_DATASETS))
        raise ValueError(
            f"Unsupported timeframe: {timeframe!r}. Expected one of: {expected}."
        ) from exc


def _overall_status(checks: Sequence[HandoffCheck]) -> str:
    if any(check.status == "fail" for check in checks):
        return "fail"
    if any(check.status == "warn" for check in checks):
        return "warn"
    return "pass"


def _stable_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _stable_jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_stable_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_stable_jsonable(item) for item in value]
    if isinstance(value, set):
        return [_stable_jsonable(item) for item in sorted(value)]
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is not None:
            return value.tz_convert("UTC").isoformat()
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, pd.Series):
        return _stable_jsonable(value.tolist())
    if isinstance(value, pd.DataFrame):
        return [_stable_jsonable(row) for row in value.to_dict(orient="records")]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _min_date_string(series: pd.Series) -> str | None:
    if series.empty:
        return None
    return str(pd.to_datetime(series, errors="coerce").min().date())


def _max_date_string(series: pd.Series) -> str | None:
    if series.empty:
        return None
    return str(pd.to_datetime(series, errors="coerce").max().date())
