from __future__ import annotations

import logging

import pandas as pd

from src.pipeline.feature_pipeline import (
    run_daily_feature_pipeline,
    run_minute_feature_pipeline,
)


def _daily_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": pd.Series(["AAPL"], dtype="string"),
            "ts_utc": pd.to_datetime(["2025-01-02T21:00:00Z"], utc=True),
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1000],
            "source": pd.Series(["alpaca_iex"], dtype="string"),
            "timeframe": pd.Series(["1D"], dtype="string"),
            "date": pd.Series(["2025-01-02"], dtype="string"),
        }
    )


def _minute_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": pd.Series(["AAPL"], dtype="string"),
            "ts_utc": pd.to_datetime(["2025-01-02T14:30:00Z"], utc=True),
            "open": [100.0],
            "high": [100.5],
            "low": [99.8],
            "close": [100.2],
            "volume": [250],
            "source": pd.Series(["alpaca_iex"], dtype="string"),
            "timeframe": pd.Series(["1Min"], dtype="string"),
            "date": pd.Series(["2025-01-02"], dtype="string"),
        }
    )


def _daily_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": pd.Series(["AAPL"], dtype="string"),
            "ts_utc": pd.to_datetime(["2025-01-02T21:00:00Z"], utc=True),
            "timeframe": pd.Series(["1D"], dtype="string"),
            "date": pd.Series(["2025-01-02"], dtype="string"),
            "feature_ret_1d": [0.01],
        }
    )


def _minute_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": pd.Series(["AAPL"], dtype="string"),
            "ts_utc": pd.to_datetime(["2025-01-02T14:30:00Z"], utc=True),
            "timeframe": pd.Series(["1Min"], dtype="string"),
            "date": pd.Series(["2025-01-02"], dtype="string"),
            "feature_ret_1m": [0.001],
        }
    )


def test_run_daily_feature_pipeline_orchestrates(monkeypatch) -> None:
    bars = _daily_bars()
    features = _daily_features()
    calls: dict[str, object] = {}

    def fake_load(symbols, **kwargs):
        calls["load"] = {"symbols": symbols, **kwargs}
        return bars

    def fake_compute(frame, *, cfg=None):
        calls["compute"] = {"frame": frame, "cfg": cfg}
        return features

    def fake_write(frame, timeframe):
        calls["write"] = {"frame": frame, "timeframe": timeframe}

    def fake_qa(frame, *, timeframe, expected_symbols=None, qa_root=None):
        calls["qa"] = {
            "frame": frame,
            "timeframe": timeframe,
            "expected_symbols": expected_symbols,
            "qa_root": qa_root,
        }

    def fake_metadata():
        calls["metadata"] = True

    monkeypatch.setattr("src.pipeline.feature_pipeline.load_bars_daily", fake_load)
    monkeypatch.setattr("src.pipeline.feature_pipeline.compute_daily_features_v1", fake_compute)
    monkeypatch.setattr("src.pipeline.feature_pipeline.write_features", fake_write)
    monkeypatch.setattr("src.pipeline.feature_pipeline.write_feature_qa_artifacts", fake_qa)
    monkeypatch.setattr("src.pipeline.feature_pipeline.export_feature_metadata", fake_metadata)

    result = run_daily_feature_pipeline(["AAPL"], start_date="2025-01-01", end_date="2025-01-03")

    assert result is features
    assert calls["load"]["symbols"] == ["AAPL"]
    assert calls["load"]["start_date"] == "2025-01-01"
    assert calls["load"]["end_date"] == "2025-01-03"
    assert calls["compute"]["frame"] is bars
    assert calls["write"]["frame"] is features
    assert calls["write"]["timeframe"] == "1D"
    assert calls["qa"]["frame"] is features
    assert calls["qa"]["timeframe"] == "1D"
    assert calls["qa"]["expected_symbols"] == ["AAPL"]
    assert calls["metadata"] is True


def test_run_minute_feature_pipeline_orchestrates(monkeypatch) -> None:
    bars = _minute_bars()
    features = _minute_features()
    calls: dict[str, object] = {}

    def fake_load(symbols, **kwargs):
        calls["load"] = {"symbols": symbols, **kwargs}
        return bars

    def fake_compute(frame, *, cfg=None):
        calls["compute"] = {"frame": frame, "cfg": cfg}
        return features

    def fake_write(frame, timeframe):
        calls["write"] = {"frame": frame, "timeframe": timeframe}

    def fake_qa(frame, *, timeframe, expected_symbols=None, qa_root=None):
        calls["qa"] = {
            "frame": frame,
            "timeframe": timeframe,
            "expected_symbols": expected_symbols,
            "qa_root": qa_root,
        }

    def fake_metadata():
        calls["metadata"] = True

    monkeypatch.setattr("src.pipeline.feature_pipeline.load_bars_1m", fake_load)
    monkeypatch.setattr("src.pipeline.feature_pipeline.compute_minute_features_v1", fake_compute)
    monkeypatch.setattr("src.pipeline.feature_pipeline.write_features", fake_write)
    monkeypatch.setattr("src.pipeline.feature_pipeline.write_feature_qa_artifacts", fake_qa)
    monkeypatch.setattr("src.pipeline.feature_pipeline.export_feature_metadata", fake_metadata)

    result = run_minute_feature_pipeline(["AAPL"], start_date="2025-01-02", end_date="2025-01-03")

    assert result is features
    assert calls["load"]["symbols"] == ["AAPL"]
    assert calls["load"]["start_date"] == "2025-01-02"
    assert calls["load"]["end_date"] == "2025-01-03"
    assert calls["compute"]["frame"] is bars
    assert calls["write"]["frame"] is features
    assert calls["write"]["timeframe"] == "1Min"
    assert calls["qa"]["frame"] is features
    assert calls["qa"]["timeframe"] == "1Min"
    assert calls["qa"]["expected_symbols"] == ["AAPL"]
    assert calls["metadata"] is True


def test_run_daily_feature_pipeline_writes_empty_features(monkeypatch) -> None:
    empty_bars = _daily_bars().iloc[0:0]
    empty_features = _daily_features().iloc[0:0]
    calls: dict[str, object] = {}

    monkeypatch.setattr("src.pipeline.feature_pipeline.load_bars_daily", lambda symbols, **kwargs: empty_bars)
    monkeypatch.setattr("src.pipeline.feature_pipeline.compute_daily_features_v1", lambda frame, *, cfg=None: empty_features)
    monkeypatch.setattr(
        "src.pipeline.feature_pipeline.write_features",
        lambda frame, timeframe: calls.update({"frame": frame, "timeframe": timeframe}),
    )
    monkeypatch.setattr(
        "src.pipeline.feature_pipeline.write_feature_qa_artifacts",
        lambda frame, *, timeframe, expected_symbols=None, qa_root=None: calls.update(
            {
                "qa_frame": frame,
                "qa_timeframe": timeframe,
                "qa_expected_symbols": expected_symbols,
            }
        ),
    )
    monkeypatch.setattr("src.pipeline.feature_pipeline.export_feature_metadata", lambda: calls.update({"metadata": True}))

    result = run_daily_feature_pipeline(["AAPL"])

    assert result.empty
    assert calls["frame"] is empty_features
    assert calls["timeframe"] == "1D"
    assert calls["qa_frame"] is empty_features
    assert calls["qa_timeframe"] == "1D"
    assert calls["qa_expected_symbols"] == ["AAPL"]
    assert calls["metadata"] is True


def test_run_minute_feature_pipeline_writes_empty_features(monkeypatch) -> None:
    empty_bars = _minute_bars().iloc[0:0]
    empty_features = _minute_features().iloc[0:0]
    calls: dict[str, object] = {}

    monkeypatch.setattr("src.pipeline.feature_pipeline.load_bars_1m", lambda symbols, **kwargs: empty_bars)
    monkeypatch.setattr("src.pipeline.feature_pipeline.compute_minute_features_v1", lambda frame, *, cfg=None: empty_features)
    monkeypatch.setattr(
        "src.pipeline.feature_pipeline.write_features",
        lambda frame, timeframe: calls.update({"frame": frame, "timeframe": timeframe}),
    )
    monkeypatch.setattr(
        "src.pipeline.feature_pipeline.write_feature_qa_artifacts",
        lambda frame, *, timeframe, expected_symbols=None, qa_root=None: calls.update(
            {
                "qa_frame": frame,
                "qa_timeframe": timeframe,
                "qa_expected_symbols": expected_symbols,
            }
        ),
    )
    monkeypatch.setattr("src.pipeline.feature_pipeline.export_feature_metadata", lambda: calls.update({"metadata": True}))

    result = run_minute_feature_pipeline(["AAPL"])

    assert result.empty
    assert calls["frame"] is empty_features
    assert calls["timeframe"] == "1Min"
    assert calls["qa_frame"] is empty_features
    assert calls["qa_timeframe"] == "1Min"
    assert calls["qa_expected_symbols"] == ["AAPL"]
    assert calls["metadata"] is True


def test_run_minute_feature_pipeline_warns_when_bars_empty(monkeypatch, caplog) -> None:
    empty_bars = _minute_bars().iloc[0:0]
    empty_features = _minute_features().iloc[0:0]

    monkeypatch.setattr("src.pipeline.feature_pipeline.load_bars_1m", lambda symbols, **kwargs: empty_bars)
    monkeypatch.setattr("src.pipeline.feature_pipeline.compute_minute_features_v1", lambda frame, *, cfg=None: empty_features)
    monkeypatch.setattr("src.pipeline.feature_pipeline.write_features", lambda frame, timeframe: None)
    monkeypatch.setattr(
        "src.pipeline.feature_pipeline.write_feature_qa_artifacts",
        lambda frame, *, timeframe, expected_symbols=None, qa_root=None: None,
    )
    monkeypatch.setattr("src.pipeline.feature_pipeline.export_feature_metadata", lambda: None)

    caplog.set_level(logging.WARNING)
    run_minute_feature_pipeline(
        ["AAPL"],
        start_date="2025-01-02",
        end_date="2025-01-03",
        paths=None,
    )

    assert "No bars loaded timeframe=1Min" in caplog.text
    assert "start=2025-01-02" in caplog.text
    assert "end=2025-01-03" in caplog.text
    assert "parquet_scan_path=" in caplog.text
