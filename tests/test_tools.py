from __future__ import annotations

import pandas as pd

from app import tools


def test_get_price_data_loads_private_bars_and_filters_range(monkeypatch):
    idx = pd.to_datetime(
        [
            "2026-01-02T09:30:00Z",
            "2026-01-02T09:31:00Z",
            "2026-01-02T09:32:00Z",
        ],
        utc=True,
    )
    bars = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1000, 1100, 1200],
        },
        index=idx,
    )

    def fake_private_loader(date: str, ticker: str) -> pd.DataFrame:
        assert date == "2026-01-02"
        assert ticker == "SPY"
        return bars

    monkeypatch.setattr(tools, "_load_bars_from_spx_repo", fake_private_loader)

    out = tools.get_price_data(
        "SPY",
        "2026-01-02T09:31:00Z",
        "2026-01-02T09:32:00Z",
    )

    assert out["available"] is True
    assert out["source"] == "spx-news-intraday"
    assert out["ticker"] == "SPY"
    assert out["n_rows"] == 2
    assert out["rows"][0]["ts"] == "2026-01-02T09:31:00+00:00"
    assert out["rows"][0]["close"] == 101.5
    assert out["rows"][1]["ts"] == "2026-01-02T09:32:00+00:00"


def test_get_price_data_returns_unavailable_when_sources_fail(monkeypatch):
    def private_missing(date: str, ticker: str) -> pd.DataFrame:
        raise FileNotFoundError("no cached bars")

    def yfinance_missing(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        raise RuntimeError("yfinance unavailable")

    monkeypatch.setattr(tools, "_load_bars_from_spx_repo", private_missing)
    monkeypatch.setattr(tools, "_load_bars_from_yfinance", yfinance_missing)

    out = tools.get_price_data("SPY", "2026-01-02", "2026-01-02")

    assert out["available"] is False
    assert out["ticker"] == "SPY"
    assert out["n_rows"] == 0
    assert "no price data" in out["reason"].lower()
