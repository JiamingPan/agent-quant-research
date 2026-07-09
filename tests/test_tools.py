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


def test_run_event_study_uses_log_returns_in_basis_points(monkeypatch):
    def fake_prices(ticker: str, start: str, end: str) -> dict:
        return _price_payload(
            ticker,
            [
                ("2026-01-01T21:00:00+00:00", 100.0),
                ("2026-01-02T21:00:00+00:00", 110.0),
                ("2026-01-03T21:00:00+00:00", 121.0),
                ("2026-01-04T21:00:00+00:00", 133.1),
            ],
        )

    monkeypatch.setattr(tools, "get_price_data", fake_prices)

    out = tools.run_event_study("SPY", "2026-01-04", window=0)

    expected_log_bps = np.log(1.10) * 1e4
    assert out["available"] is True
    assert out["baseline"]["return_type"] == "log"
    assert out["baseline"]["expected_return_bps"] == pytest.approx(expected_log_bps)
    assert out["event_window"][0]["actual_return_bps"] == pytest.approx(expected_log_bps)


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
    assert out["baseline"]["return_type"] == "log"
    expected_return_bps = np.log(1.01) * 1e4
    actual_event_return_bps = np.log(1.5) * 1e4
    assert out["baseline"]["expected_return_bps"] == pytest.approx(expected_return_bps)
    assert out["leakage_check"]["baseline_uses_only_pre_event_data"] is True
    event_row = next(row for row in out["event_window"] if row["relative_day"] == 0)
    assert event_row["date"] == "2026-01-04"
    assert event_row["actual_return_bps"] == pytest.approx(actual_event_return_bps)
    assert event_row["abnormal_return_bps"] == pytest.approx(
        actual_event_return_bps - expected_return_bps
    )


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


def test_run_event_study_reports_pre_event_hac_error_estimate(monkeypatch):
    def fake_prices(ticker: str, start: str, end: str) -> dict:
        return _price_payload(
            ticker,
            [
                ("2026-01-01T21:00:00+00:00", 100.0),
                ("2026-01-02T21:00:00+00:00", 101.0),
                ("2026-01-03T21:00:00+00:00", 101.505),
                ("2026-01-04T21:00:00+00:00", 103.5351),
                ("2026-01-05T21:00:00+00:00", 103.0174245),
                ("2026-01-06T21:00:00+00:00", 108.168295725),
                ("2026-01-07T21:00:00+00:00", 107.08661276775),
            ],
        )

    monkeypatch.setattr(tools, "get_price_data", fake_prices)

    out = tools.run_event_study("SPY", "2026-01-06", window=1)

    assert out["available"] is True
    assert "bootstrap_mean_abnormal_return_ci_bps" not in out["summary"]

    ci = out["summary"]["pre_event_hac_mean_abnormal_return_ci_bps"]
    assert ci["method"] == "pre_event_newey_west"
    assert ci["baseline_n"] == out["baseline"]["n_returns"]
    assert ci["event_n"] == out["summary"]["n_observations"]
    assert ci["lags"] >= 0
    assert ci["se"] > 0
    assert ci["low"] < out["summary"]["mean_abnormal_return_bps"] < ci["high"]

    leakage = out["leakage_check"]
    assert leakage["error_estimate_uses_only_pre_event_data"] is True
    assert leakage["error_estimate_return_dates"] == out["baseline"]["return_dates"]


def test_run_event_study_error_scale_is_invariant_to_post_event_prices(monkeypatch):
    def fake_prices_with_post_close(post_close: float):
        def fake_prices(ticker: str, start: str, end: str) -> dict:
            return _price_payload(
                ticker,
                [
                    ("2026-01-01T21:00:00+00:00", 100.0),
                    ("2026-01-02T21:00:00+00:00", 101.0),
                    ("2026-01-03T21:00:00+00:00", 101.505),
                    ("2026-01-04T21:00:00+00:00", 103.5351),
                    ("2026-01-05T21:00:00+00:00", 103.0174245),
                    ("2026-01-06T21:00:00+00:00", 108.168295725),
                    ("2026-01-07T21:00:00+00:00", post_close),
                ],
            )

        return fake_prices

    monkeypatch.setattr(tools, "get_price_data", fake_prices_with_post_close(107.0))
    out_a = tools.run_event_study("SPY", "2026-01-06", window=1)

    monkeypatch.setattr(tools, "get_price_data", fake_prices_with_post_close(130.0))
    out_b = tools.run_event_study("SPY", "2026-01-06", window=1)

    ci_a = out_a["summary"]["pre_event_hac_mean_abnormal_return_ci_bps"]
    ci_b = out_b["summary"]["pre_event_hac_mean_abnormal_return_ci_bps"]
    assert ci_a["se"] == ci_b["se"]
    assert ci_a["baseline_n"] == ci_b["baseline_n"]
    assert out_a["leakage_check"]["error_estimate_return_dates"] == out_b["leakage_check"][
        "error_estimate_return_dates"
    ]
    assert (
        out_a["summary"]["mean_abnormal_return_bps"]
        != out_b["summary"]["mean_abnormal_return_bps"]
    )


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
