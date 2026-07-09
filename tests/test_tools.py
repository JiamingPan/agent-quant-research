from __future__ import annotations

import pytest
import pandas as pd
from fastapi.testclient import TestClient

from app import tools
from app.main import app


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


def test_run_event_study_fits_baseline_only_before_event(monkeypatch):
    def fake_prices(ticker: str, start: str, end: str) -> dict:
        return _price_payload(
            ticker,
            [
                ("2026-01-01T21:00:00+00:00", 100.0),
                ("2026-01-02T21:00:00+00:00", 101.0),
                ("2026-01-03T21:00:00+00:00", 102.01),
                ("2026-01-04T21:00:00+00:00", 153.015),
                ("2026-01-05T21:00:00+00:00", 229.5225),
            ],
        )

    monkeypatch.setattr(tools, "get_price_data", fake_prices)

    out = tools.run_event_study("SPY", "2026-01-04", window=1)

    assert out["available"] is True
    assert out["baseline"]["n_returns"] == 2
    assert out["baseline"]["return_dates"] == ["2026-01-02", "2026-01-03"]
    assert out["baseline"]["end"] == "2026-01-03"
    assert out["baseline"]["expected_return_bps"] == pytest.approx(100.0)
    assert out["leakage_check"]["baseline_uses_only_pre_event_data"] is True
    event_row = next(row for row in out["event_window"] if row["relative_day"] == 0)
    assert event_row["date"] == "2026-01-04"
    assert event_row["actual_return_bps"] == pytest.approx(5000.0)
    assert event_row["abnormal_return_bps"] == pytest.approx(4900.0)


def test_run_event_study_baseline_is_invariant_to_post_event_prices(monkeypatch):
    def fake_prices_with_post_close(post_close: float):
        def fake_prices(ticker: str, start: str, end: str) -> dict:
            return _price_payload(
                ticker,
                [
                    ("2026-01-01T21:00:00+00:00", 100.0),
                    ("2026-01-02T21:00:00+00:00", 101.0),
                    ("2026-01-03T21:00:00+00:00", 102.01),
                    ("2026-01-04T21:00:00+00:00", 153.015),
                    ("2026-01-05T21:00:00+00:00", post_close),
                ],
            )

        return fake_prices

    monkeypatch.setattr(tools, "get_price_data", fake_prices_with_post_close(200.0))
    baseline_a = tools.run_event_study("SPY", "2026-01-04", window=1)["baseline"]

    monkeypatch.setattr(tools, "get_price_data", fake_prices_with_post_close(1000.0))
    baseline_b = tools.run_event_study("SPY", "2026-01-04", window=1)["baseline"]

    assert baseline_a == baseline_b


def test_run_event_study_requires_pre_event_baseline(monkeypatch):
    def fake_prices(ticker: str, start: str, end: str) -> dict:
        return _price_payload(
            ticker,
            [
                ("2026-01-03T21:00:00+00:00", 100.0),
                ("2026-01-04T21:00:00+00:00", 101.0),
                ("2026-01-05T21:00:00+00:00", 102.0),
            ],
        )

    monkeypatch.setattr(tools, "get_price_data", fake_prices)

    out = tools.run_event_study("SPY", "2026-01-04", window=1)

    assert out["available"] is False
    assert "pre-event" in out["reason"].lower()
    assert out["leakage_check"]["baseline_uses_only_pre_event_data"] is True


def test_event_study_endpoint_calls_tool(monkeypatch):
    def fake_event_study(ticker: str, event_date: str, window: int) -> dict:
        return {
            "ticker": ticker,
            "event_date": event_date,
            "window": window,
            "available": True,
        }

    monkeypatch.setattr(tools, "run_event_study", fake_event_study)

    response = TestClient(app).post(
        "/event-study",
        json={"ticker": "SPY", "event_date": "2026-01-04", "window": 1},
    )

    assert response.status_code == 200
    assert response.json() == {
        "ticker": "SPY",
        "event_date": "2026-01-04",
        "window": 1,
        "available": True,
    }


def _price_payload(ticker: str, closes: list[tuple[str, float]]) -> dict:
    return {
        "ticker": ticker,
        "start": closes[0][0],
        "end": closes[-1][0],
        "available": True,
        "source": "test",
        "n_rows": len(closes),
        "columns": ["close"],
        "rows": [{"ts": ts, "close": close} for ts, close in closes],
        "truncated": False,
        "reason": None,
    }
