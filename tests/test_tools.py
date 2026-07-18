from __future__ import annotations

import numpy as np
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


def test_get_price_data_can_return_all_rows_for_internal_consumers(monkeypatch):
    idx = pd.date_range("2026-01-02T09:30:00Z", periods=2501, freq="min")
    bars = pd.DataFrame({"close": np.linspace(100.0, 110.0, len(idx))}, index=idx)

    def fake_range_loader(
        ticker: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        return bars

    monkeypatch.setattr(tools, "_load_range_from_private_loader", fake_range_loader)

    out = tools.get_price_data(
        "SPY",
        "2026-01-02T09:30:00Z",
        "2026-01-04T03:10:00Z",
        max_rows=None,
    )

    assert out["available"] is True
    assert out["n_rows"] == 2501
    assert len(out["rows"]) == 2501
    assert out["truncated"] is False


def test_normalize_bars_flattens_single_ticker_yfinance_columns():
    frame = pd.DataFrame(
        [[100.0, 101.0, 99.0, 100.5, 1000]],
        index=pd.to_datetime(["2026-01-02"]),
        columns=pd.MultiIndex.from_tuples(
            [
                ("Open", "SPY"),
                ("High", "SPY"),
                ("Low", "SPY"),
                ("Close", "SPY"),
                ("Volume", "SPY"),
            ]
        ),
    )

    normalized = tools._normalize_bars(frame)

    assert normalized.columns.tolist() == ["open", "high", "low", "close", "volume"]
    assert normalized.index.tz is not None
    assert normalized.iloc[0]["close"] == 100.5


def test_normalize_bars_rejects_duplicate_multi_ticker_fields():
    frame = pd.DataFrame(
        [[100.0, 200.0]],
        index=pd.to_datetime(["2026-01-02"]),
        columns=pd.MultiIndex.from_tuples(
            [("Close", "SPY"), ("Close", "AAPL")]
        ),
    )

    with pytest.raises(ValueError, match="duplicate OHLCV"):
        tools._normalize_bars(frame)


def test_run_event_study_returns_log_ar_car_and_bootstrap_ci(monkeypatch):
    def fake_prices(
        ticker: str,
        start: str,
        end: str,
        *,
        max_rows: int | None = 2000,
    ) -> dict:
        assert max_rows is None
        return _price_payload(
            ticker,
            [
                ("2026-01-02T21:00:00+00:00", 100.0),
                ("2026-01-05T21:00:00+00:00", 101.0),
                ("2026-01-06T21:00:00+00:00", 101.505),
                ("2026-01-07T21:00:00+00:00", 103.5351),
                ("2026-01-08T21:00:00+00:00", 103.0174245),
                ("2026-01-09T21:00:00+00:00", 108.168295725),
                ("2026-01-12T21:00:00+00:00", 107.08661276775),
            ],
        )

    monkeypatch.setattr(tools, "get_price_data", fake_prices)

    out = tools.run_event_study("SPY", "2026-01-09", window=1)

    assert out["available"] is True
    assert out["event_input"] == "2026-01-09"
    assert out["event_date"] == "2026-01-09"
    assert out["alignment"]["input_kind"] == "date"
    assert out["alignment"]["rule"] == "date_as_given"
    assert out["n_pre_obs"] == 3
    assert out["baseline"]["n_returns"] == 3
    assert out["baseline"]["return_dates"] == [
        "2026-01-05",
        "2026-01-06",
        "2026-01-07",
    ]
    assert out["baseline"]["cutoff_date"] == "2026-01-08"
    assert out["baseline"]["return_type"] == "log"
    expected_return_bps = np.mean([np.log(1.01), np.log(1.005), np.log(1.02)]) * 1e4
    assert out["baseline"]["expected_return_bps"] == pytest.approx(expected_return_bps)
    event_rows = out["event_window"]
    assert [row["relative_day"] for row in event_rows] == [-1, 0, 1]
    assert [row["date"] for row in event_rows] == [
        "2026-01-08",
        "2026-01-09",
        "2026-01-12",
    ]
    running_car = 0.0
    for row in event_rows:
        assert row["ar_bps"] == pytest.approx(
            row["actual_return_bps"] - row["expected_return_bps"], abs=1e-5
        )
        running_car += row["ar_bps"]
        assert row["car_bps"] == pytest.approx(running_car, abs=1e-5)

    assert out["summary"]["car_bps"] == pytest.approx(event_rows[-1]["car_bps"])
    ci = out["summary"]["bootstrap_car_ci_bps"]
    assert ci["method"] == "pre_event_residual_percentile"
    assert ci["samples"] == 1000
    assert ci["n_pre_obs"] == out["n_pre_obs"]
    assert ci["event_n"] == len(event_rows)
    pre_event_returns = np.array(
        [np.log(1.01), np.log(1.005), np.log(1.02)], dtype=float
    ) * 1e4
    centered = pre_event_returns - pre_event_returns.mean()
    rng = np.random.default_rng(0)
    expected_cars = out["summary"]["car_bps"] + rng.choice(
        centered,
        size=(1000, len(event_rows)),
        replace=True,
    ).sum(axis=1)
    expected_low, expected_high = np.percentile(expected_cars, [2.5, 97.5])
    assert ci["low"] == pytest.approx(expected_low)
    assert ci["high"] == pytest.approx(expected_high)

    leakage = out["leakage_check"]
    assert leakage["status"] == "passed"
    assert leakage["baseline_uses_only_pre_window_data"] is True
    assert leakage["bootstrap_uses_only_pre_window_data"] is True
    assert leakage["baseline_cutoff_date"] == "2026-01-08"
    assert leakage["max_baseline_date"] == "2026-01-07"
    assert leakage["error_estimate_uses_only_pre_event_data"] is True
    assert leakage["error_estimate_return_dates"] == out["baseline"]["return_dates"]


def test_before_close_timestamp_uses_same_trading_day(monkeypatch):
    monkeypatch.setattr(tools, "get_price_data", _event_price_loader)

    out = tools.run_event_study(
        "SPY",
        "2026-01-09T15:30:00-05:00",
        window=1,
    )

    assert out["available"] is True
    assert out["event_input"] == "2026-01-09T15:30:00-05:00"
    assert out["event_date"] == "2026-01-09"
    assert out["alignment"]["aligned_event_date"] == "2026-01-09"
    assert out["alignment"]["local_timestamp"] == "2026-01-09T15:30:00-05:00"
    assert out["alignment"]["rule"] == "before_close_same_trading_day"
    assert [row["date"] for row in out["event_window"]] == [
        "2026-01-08",
        "2026-01-09",
        "2026-01-12",
    ]


def test_after_close_timestamp_uses_next_trading_day(monkeypatch):
    monkeypatch.setattr(tools, "get_price_data", _event_price_loader)

    out = tools.run_event_study(
        "SPY",
        "2026-01-09T16:30:00-05:00",
        window=1,
    )

    assert out["available"] is True
    assert out["event_input"] == "2026-01-09T16:30:00-05:00"
    assert out["event_date"] == "2026-01-12"
    assert out["alignment"]["aligned_event_date"] == "2026-01-12"
    assert out["alignment"]["market_timezone"] == "America/New_York"
    assert out["alignment"]["market_close"] == "16:00"
    assert out["alignment"]["rule"] == "at_or_after_close_next_trading_day"
    assert [row["date"] for row in out["event_window"]] == [
        "2026-01-09",
        "2026-01-12",
        "2026-01-13",
    ]


def test_exact_market_close_uses_next_trading_day(monkeypatch):
    monkeypatch.setattr(tools, "get_price_data", _event_price_loader)

    out = tools.run_event_study(
        "SPY",
        "2026-01-09T16:00:00-05:00",
        window=1,
    )

    assert out["event_date"] == "2026-01-12"
    assert out["alignment"]["rule"] == "at_or_after_close_next_trading_day"


def test_utc_timestamp_converts_with_summer_daylight_saving_time():
    context = tools._parse_event_input("2026-07-10T20:30:00Z")
    trading_dates = pd.DatetimeIndex(
        [pd.Timestamp("2026-07-10", tz="UTC"), pd.Timestamp("2026-07-13", tz="UTC")]
    )

    event_date, alignment = tools._align_event_to_trading_day(context, trading_dates)

    assert alignment["local_timestamp"] == "2026-07-10T16:30:00-04:00"
    assert event_date == pd.Timestamp("2026-07-13", tz="UTC")


def test_missing_holiday_date_aligns_to_next_observed_trading_day():
    context = tools._parse_event_input("2026-07-03T12:00:00-04:00")
    trading_dates = pd.DatetimeIndex(
        [pd.Timestamp("2026-07-02", tz="UTC"), pd.Timestamp("2026-07-06", tz="UTC")]
    )

    event_date, alignment = tools._align_event_to_trading_day(context, trading_dates)

    assert event_date == pd.Timestamp("2026-07-06", tz="UTC")
    assert alignment["rule"] == "non_trading_day_next_trading_day"


def test_weekend_timestamp_uses_next_trading_day(monkeypatch):
    monkeypatch.setattr(tools, "get_price_data", _event_price_loader)

    out = tools.run_event_study(
        "SPY",
        "2026-01-10T12:00:00-05:00",
        window=1,
    )

    assert out["available"] is True
    assert out["event_date"] == "2026-01-12"
    assert out["alignment"]["rule"] == "non_trading_day_next_trading_day"


def test_unavailable_window_uses_aligned_event_cutoff(monkeypatch):
    def prices_missing_next_day(
        ticker: str,
        start: str,
        end: str,
        *,
        max_rows: int | None = 2000,
    ) -> dict:
        return _price_payload(
            ticker,
            [
                ("2026-01-02T21:00:00+00:00", 100.0),
                ("2026-01-05T21:00:00+00:00", 101.0),
                ("2026-01-06T21:00:00+00:00", 102.0),
                ("2026-01-07T21:00:00+00:00", 103.0),
                ("2026-01-08T21:00:00+00:00", 104.0),
                ("2026-01-09T21:00:00+00:00", 105.0),
                ("2026-01-12T21:00:00+00:00", 106.0),
            ],
        )

    monkeypatch.setattr(tools, "get_price_data", prices_missing_next_day)

    out = tools.run_event_study(
        "SPY",
        "2026-01-09T16:30:00-05:00",
        window=1,
    )

    assert out["available"] is False
    assert out["event_date"] == "2026-01-12"
    assert out["baseline"]["cutoff_date"] == "2026-01-09"
    assert out["leakage_check"]["event_date"] == "2026-01-12"


def test_price_unavailable_does_not_claim_an_aligned_date(monkeypatch):
    def unavailable_prices(
        ticker: str,
        start: str,
        end: str,
        *,
        max_rows: int | None = 2000,
    ) -> dict:
        return {"available": False, "reason": "cache unavailable"}

    monkeypatch.setattr(tools, "get_price_data", unavailable_prices)

    out = tools.run_event_study(
        "SPY",
        "2026-01-09T16:30:00-05:00",
        window=1,
    )

    assert out["available"] is False
    assert out["event_date"] is None
    assert out["baseline"]["cutoff_date"] is None
    assert out["alignment"]["aligned_event_date"] is None
    assert out["alignment"]["rule"] == "unresolved"
    assert out["leakage_check"]["status"] == "not_run"


def test_baseline_leakage_assertion_fires_at_window_start():
    cutoff = pd.Timestamp("2026-01-05", tz="UTC")
    leaking_dates = pd.DatetimeIndex(
        [pd.Timestamp("2026-01-04", tz="UTC"), pd.Timestamp("2026-01-05", tz="UTC")]
    )

    with pytest.raises(AssertionError, match="Baseline leakage"):
        tools._assert_no_baseline_leakage(leaking_dates, cutoff)


def test_run_event_study_requires_pre_event_baseline(monkeypatch):
    def fake_prices(
        ticker: str,
        start: str,
        end: str,
        *,
        max_rows: int | None = 2000,
    ) -> dict:
        assert max_rows is None
        return _price_payload(
            ticker,
            [
                ("2026-01-06T21:00:00+00:00", 100.0),
                ("2026-01-07T21:00:00+00:00", 101.0),
                ("2026-01-08T21:00:00+00:00", 102.0),
                ("2026-01-09T21:00:00+00:00", 103.0),
                ("2026-01-12T21:00:00+00:00", 104.0),
            ],
        )

    monkeypatch.setattr(tools, "get_price_data", fake_prices)

    out = tools.run_event_study("SPY", "2026-01-09", window=1)

    assert out["available"] is False
    assert "pre-event" in out["reason"].lower()
    assert out["n_pre_obs"] == 1
    assert out["leakage_check"]["status"] == "passed"
    assert out["leakage_check"]["baseline_uses_only_pre_window_data"] is True


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


def test_event_study_endpoint_rejects_timestamp_without_timezone(monkeypatch):
    def unavailable_prices(
        ticker: str,
        start: str,
        end: str,
        *,
        max_rows: int | None = 2000,
    ) -> dict:
        return {
            "available": False,
            "reason": "test should fail validation before loading prices",
        }

    monkeypatch.setattr(tools, "get_price_data", unavailable_prices)

    response = TestClient(app).post(
        "/event-study",
        json={
            "ticker": "SPY",
            "event_date": "2026-01-09T16:30:00",
            "window": 1,
        },
    )

    assert response.status_code == 422
    assert "timezone" in response.json()["detail"].lower()


def test_event_study_endpoint_does_not_map_internal_value_error_to_422(monkeypatch):
    def broken_event_study(ticker: str, event_date: str, window: int) -> dict:
        raise ValueError("internal price frame error")

    monkeypatch.setattr(tools, "run_event_study", broken_event_study)

    response = TestClient(app, raise_server_exceptions=False).post(
        "/event-study",
        json={"ticker": "SPY", "event_date": "2026-01-09", "window": 1},
    )

    assert response.status_code == 500


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


def _event_price_loader(
    ticker: str,
    start: str,
    end: str,
    *,
    max_rows: int | None = 2000,
) -> dict:
    assert max_rows is None
    return _price_payload(
        ticker,
        [
            ("2026-01-02T21:00:00+00:00", 100.0),
            ("2026-01-05T21:00:00+00:00", 101.0),
            ("2026-01-06T21:00:00+00:00", 101.505),
            ("2026-01-07T21:00:00+00:00", 103.5351),
            ("2026-01-08T21:00:00+00:00", 103.0174245),
            ("2026-01-09T21:00:00+00:00", 108.168295725),
            ("2026-01-12T21:00:00+00:00", 107.08661276775),
            ("2026-01-13T21:00:00+00:00", 108.1574788954275),
        ],
    )
