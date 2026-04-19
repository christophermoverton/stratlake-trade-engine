from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from types import ModuleType
from typing import Any, Iterator

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]


def ensure_output_root(output_root: Path, *, reset_output: bool) -> Path:
    if reset_output and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_output_path(path_value: str | Path, *, base_dir: Path) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def relative_to_output(path: Path, output_root: Path) -> str:
    try:
        return path.resolve().relative_to(output_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _relativize_text(text: str, *, output_root: Path, repo_root: Path) -> str:
    normalized = text.replace("\\", "/")
    output_root_text = output_root.resolve().as_posix()
    repo_root_text = repo_root.resolve().as_posix()
    if normalized == repo_root_text:
        return "."
    if normalized.startswith(output_root_text + "/"):
        return normalized[len(output_root_text) + 1 :]
    if normalized == output_root_text:
        return "."
    if output_root_text in normalized:
        return normalized.replace(output_root_text + "/", "").replace(output_root_text, ".")
    if repo_root_text in normalized:
        return normalized.replace(repo_root_text, ".")
    return normalized


def _relativize_payload(payload: Any, *, output_root: Path, repo_root: Path) -> Any:
    if isinstance(payload, dict):
        return {
            key: _relativize_payload(value, output_root=output_root, repo_root=repo_root)
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [_relativize_payload(value, output_root=output_root, repo_root=repo_root) for value in payload]
    if isinstance(payload, str):
        return _relativize_text(payload, output_root=output_root, repo_root=repo_root)
    return payload


def sanitize_json_outputs(output_root: Path) -> None:
    for path in sorted(output_root.rglob("*.json")):
        payload = read_json(path)
        sanitized = _relativize_payload(payload, output_root=output_root, repo_root=REPO_ROOT)
        write_json(path, sanitized)


@contextmanager
def example_environment(workspace_root: Path) -> Iterator[None]:
    previous_features_root = os.environ.get("FEATURES_ROOT")
    previous_cwd = Path.cwd()
    os.environ["FEATURES_ROOT"] = str(workspace_root / "data")
    os.chdir(workspace_root)
    try:
        yield
    finally:
        os.chdir(previous_cwd)
        if previous_features_root is None:
            os.environ.pop("FEATURES_ROOT", None)
        else:
            os.environ["FEATURES_ROOT"] = previous_features_root


def write_cross_section_dataset(root: Path, *, periods: int = 55) -> None:
    timestamps = pd.date_range("2025-01-01", periods=periods, freq="D", tz="UTC")
    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    for symbol_index, symbol in enumerate(symbols):
        closes: list[float] = []
        level = 100.0 + float(symbol_index * 6)
        for index in range(periods):
            drift = (symbol_index - 2) * 0.5 + (0.35 if index % 4 else -0.25)
            level += drift
            closes.append(level)
        close_series = pd.Series(closes, dtype="float64")
        frame = pd.DataFrame(
            {
                "symbol": pd.Series([symbol] * periods, dtype="string"),
                "ts_utc": timestamps,
                "timeframe": pd.Series(["1D"] * periods, dtype="string"),
                "date": pd.Series(timestamps.strftime("%Y-%m-%d"), dtype="string"),
                "close": close_series,
                "feature_ret_1d": close_series.div(close_series.shift(1)).sub(1.0).fillna(0.0),
                "high": close_series + 1.0,
                "low": close_series - 1.0,
                "market_return": pd.Series([0.001] * periods, dtype="float64"),
            }
        )
        dataset_dir = root / "data" / "curated" / "features_daily" / f"symbol={symbol}" / "year=2025"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(dataset_dir / "part-0.parquet", index=False)


def load_module_from_repo(relative_path: str, module_name: str) -> ModuleType:
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path.as_posix()}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
