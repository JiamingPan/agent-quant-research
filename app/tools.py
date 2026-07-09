"""
The 3 agent tools. Exactly three — resist adding more.

Day 1-2: search_docs is wired to the RAG core.
Day 3: get_price_data is wired as a thin, JSON-safe adapter over cached bars.
Day 3: run_event_study is leakage-checked with a pre-event-only error estimate.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import rag

SPX_ROOT_ENV = "SPX_NEWS_INTRADAY_ROOT"
BASELINE_LOOKBACK_DAYS = 30
MIN_BASELINE_RETURNS = 2
MAX_HAC_LAG = 5
HAC_Z_VALUE = 1.96


def search_docs(query: str, k: int = 4) -> dict:
    """RAG retrieval with citations + refusal. Returns a JSON-able dict for the agent."""
    passages, refused, reason = rag.search(query, k=k)
    return {
        "passages": [p.model_dump() for p in passages],
        "refused": refused,
        "reason": reason,
    }


def get_price_data(ticker: str, start: str, end: str) -> dict:
    """
    Return a price series for `ticker` in [start, end].

    Prefer cached 1-minute bars from a local spx-news-intraday checkout when present.
    Fall back to optional yfinance daily data only when the local loader is unavailable.
    """
    start_ts, end_ts = _parse_time_range(start, end)
    ticker = ticker.upper()

    errors: list[str] = []
    try:
        frame = _load_range_from_private_loader(ticker, start_ts, end_ts)
        source = "spx-news-intraday"
    except Exception as exc:
        errors.append(f"spx-news-intraday: {exc}")
        try:
            frame = _load_bars_from_yfinance(ticker, start_ts, end_ts)
            source = "yfinance"
        except Exception as yf_exc:
            errors.append(f"yfinance: {yf_exc}")
            return {
                "ticker": ticker,
                "start": start,
                "end": end,
                "available": False,
                "source": None,
                "n_rows": 0,
                "columns": [],
                "rows": [],
                "truncated": False,
                "reason": "No price data available. " + " | ".join(errors),
            }

    frame = _normalize_bars(frame)
    frame = frame[(frame.index >= start_ts) & (frame.index <= end_ts)].sort_index()
    if frame.empty:
        return {
            "ticker": ticker,
            "start": start,
            "end": end,
            "available": False,
            "source": source,
            "n_rows": 0,
            "columns": [],
            "rows": [],
            "truncated": False,
            "reason": "No price data in requested range.",
        }

    rows, truncated = _frame_to_records(frame)
    return {
        "ticker": ticker,
        "start": start,
        "end": end,
        "available": True,
        "source": source,
        "n_rows": int(len(frame)),
        "columns": list(frame.columns),
        "rows": rows,
        "truncated": truncated,
        "reason": None,
    }


def _parse_time_range(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_ts = _parse_bound(start, is_end=False)
    end_ts = _parse_bound(end, is_end=True)
    if end_ts < start_ts:
        raise ValueError("end must be >= start")
    return start_ts, end_ts


def _parse_bound(value: str, is_end: bool) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    has_time = "T" in value or " " in value
    if is_end and not has_time:
        ts = ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _load_range_from_private_loader(
    ticker: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for day in pd.date_range(start_ts.date(), end_ts.date(), freq="D"):
        date = day.date().isoformat()
        try:
            frames.append(_load_bars_from_spx_repo(date, ticker))
        except FileNotFoundError as exc:
            errors.append(str(exc))

    if not frames:
        detail = errors[-1] if errors else "no cached bars found"
        raise FileNotFoundError(detail)
    return pd.concat(frames).sort_index()


def _load_bars_from_spx_repo(date: str, ticker: str) -> pd.DataFrame:
    root = _spx_news_intraday_root()
    if root is None:
        raise FileNotFoundError(
            f"Set {SPX_ROOT_ENV} or keep spx-news-intraday at ~/spx-news-intraday."
        )

    root_str = str(root)
    inserted = False
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
        inserted = True
    try:
        from src.research.news_event_study import load_price_bars

        return load_price_bars(
            date=date,
            ticker=ticker,
            cache_only=True,
            data_root=root,
        )
    finally:
        if inserted:
            try:
                sys.path.remove(root_str)
            except ValueError:
                pass


def _spx_news_intraday_root() -> Optional[Path]:
    configured = os.getenv(SPX_ROOT_ENV, "").strip()
    candidates = [Path(configured)] if configured else []
    candidates.append(Path.home() / "spx-news-intraday")
    for candidate in candidates:
        if (candidate / "src" / "research" / "news_event_study.py").exists():
            return candidate
    return None


def _load_bars_from_yfinance(
    ticker: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except Exception as exc:
        raise RuntimeError("yfinance is not installed") from exc

    end_exclusive = (end_ts + pd.Timedelta(days=1)).date().isoformat()
    frame = yf.download(
        ticker,
        start=start_ts.date().isoformat(),
        end=end_exclusive,
        progress=False,
        auto_adjust=False,
    )
    if frame is None or frame.empty:
        raise RuntimeError("yfinance returned no rows")
    return frame


def _normalize_bars(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(c).lower() for c in frame.columns]
    if not isinstance(frame.index, pd.DatetimeIndex):
        if "ts" not in frame.columns:
            raise ValueError("price frame needs a DatetimeIndex or ts column")
        frame.index = pd.to_datetime(frame.pop("ts"), utc=True, errors="coerce")
    elif frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    else:
        frame.index = frame.index.tz_convert("UTC")
    frame.index.name = "ts"
    frame = frame[~frame.index.isna()].sort_index()
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in frame.columns]
    if not keep:
        raise ValueError("price frame has no OHLCV columns")
    frame = frame[keep]
    for col in keep:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna(axis=0, subset=["close"])


def _frame_to_records(
    frame: pd.DataFrame,
    max_rows: int = 2000,
) -> tuple[list[dict], bool]:
    truncated = len(frame) > max_rows
    limited = frame.iloc[:max_rows]
    rows: list[dict] = []
    for ts, row in limited.iterrows():
        record = {"ts": ts.isoformat()}
        for col, value in row.items():
            if pd.isna(value):
                record[col] = None
            elif col == "volume":
                record[col] = int(value)
            else:
                record[col] = float(value)
        rows.append(record)
    return rows, truncated


def run_event_study(ticker: str, event_date: str, window: int = 5) -> dict:
    """
    Compute close-to-close abnormal returns around an event date.

    The expected-return baseline is intentionally simple for the MVP: mean
    close-to-close return estimated only from dates strictly before event_date.
    """
    if window < 0:
        raise ValueError("window must be non-negative")

    ticker = ticker.upper()
    event_ts = pd.Timestamp(event_date).tz_localize("UTC").normalize()
    fetch_start = (event_ts - pd.Timedelta(days=BASELINE_LOOKBACK_DAYS)).date().isoformat()
    fetch_end = (event_ts + pd.Timedelta(days=window)).date().isoformat()
    prices = get_price_data(ticker, fetch_start, fetch_end)
    if not prices.get("available"):
        return _event_study_unavailable(
            ticker,
            event_date,
            window,
            prices.get("reason") or "price data unavailable",
            baseline_dates=[],
        )

    daily = _daily_close_returns(prices.get("rows", []))
    if daily.empty:
        return _event_study_unavailable(
            ticker,
            event_date,
            window,
            "price data has no daily close returns",
            baseline_dates=[],
        )

    baseline = daily.loc[daily.index < event_ts, "return_bps"].dropna()
    baseline_dates = [idx.date().isoformat() for idx in baseline.index]
    if len(baseline) < MIN_BASELINE_RETURNS:
        return _event_study_unavailable(
            ticker,
            event_date,
            window,
            f"Need at least {MIN_BASELINE_RETURNS} pre-event returns for baseline.",
            baseline_dates=baseline_dates,
        )

    expected_return_bps = float(baseline.mean())
    win_start = event_ts - pd.Timedelta(days=window)
    win_end = event_ts + pd.Timedelta(days=window)
    event_window = daily.loc[(daily.index >= win_start) & (daily.index <= win_end)].copy()
    event_window = event_window.dropna(axis=0, subset=["return_bps"])
    if event_window.empty:
        return _event_study_unavailable(
            ticker,
            event_date,
            window,
            "No return observations in event window.",
            baseline_dates=baseline_dates,
        )

    event_rows = _event_window_rows(event_window, event_ts, expected_return_bps)
    abnormal_values = [row["abnormal_return_bps"] for row in event_rows]
    mean_abnormal_return_bps = float(np.mean(abnormal_values))
    ci = _pre_event_hac_mean_ci(
        mean_abnormal_return_bps=mean_abnormal_return_bps,
        baseline_returns_bps=baseline,
        event_n=len(abnormal_values),
    )

    return {
        "ticker": ticker,
        "event_date": event_date,
        "window": window,
        "available": True,
        "price_source": prices.get("source"),
        "baseline": {
            "model": "mean_close_to_close_return",
            "lookback_days": BASELINE_LOOKBACK_DAYS,
            "start": baseline.index.min().date().isoformat(),
            "end": baseline.index.max().date().isoformat(),
            "n_returns": int(len(baseline)),
            "return_dates": baseline_dates,
            "expected_return_bps": _round_float(expected_return_bps),
        },
        "event_window": event_rows,
        "summary": {
            "n_observations": len(event_rows),
            "mean_abnormal_return_bps": _round_float(mean_abnormal_return_bps),
            "cumulative_abnormal_return_bps": _round_float(float(np.sum(abnormal_values))),
            "pre_event_hac_mean_abnormal_return_ci_bps": ci,
        },
        "leakage_check": _leakage_check(event_ts, baseline.index),
        "reason": None,
    }


def _daily_close_returns(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if "ts" not in frame or "close" not in frame:
        return pd.DataFrame()
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(axis=0, subset=["ts", "close"]).sort_values("ts")
    if frame.empty:
        return pd.DataFrame()
    frame["date"] = frame["ts"].dt.normalize()
    daily = frame.groupby("date", sort=True)["close"].last().to_frame()
    daily["return_bps"] = daily["close"].pct_change() * 1e4
    return daily


def _event_window_rows(
    event_window: pd.DataFrame,
    event_ts: pd.Timestamp,
    expected_return_bps: float,
) -> list[dict]:
    rows: list[dict] = []
    for idx, row in event_window.iterrows():
        actual_return_bps = float(row["return_bps"])
        abnormal_return_bps = actual_return_bps - expected_return_bps
        rows.append(
            {
                "date": idx.date().isoformat(),
                "relative_day": int((idx - event_ts).days),
                "close": _round_float(float(row["close"])),
                "actual_return_bps": _round_float(actual_return_bps),
                "expected_return_bps": _round_float(expected_return_bps),
                "abnormal_return_bps": _round_float(abnormal_return_bps),
            }
        )
    return rows


def _pre_event_hac_mean_ci(
    mean_abnormal_return_bps: float,
    baseline_returns_bps: pd.Series,
    event_n: int,
) -> dict:
    baseline = np.asarray(baseline_returns_bps.dropna(), dtype=float)
    baseline_n = len(baseline)
    if baseline_n < 2 or event_n < 1:
        return {
            "low": None,
            "high": None,
            "se": None,
            "method": "pre_event_newey_west",
            "z": HAC_Z_VALUE,
            "lags": 0,
            "baseline_n": int(baseline_n),
            "event_n": int(event_n),
            "long_run_variance": None,
        }

    residuals = baseline - float(np.mean(baseline))
    lags = min(MAX_HAC_LAG, baseline_n - 1)
    long_run_variance = _newey_west_long_run_variance(residuals, lags)
    se = float(np.sqrt(long_run_variance * (1.0 / event_n + 1.0 / baseline_n)))
    margin = HAC_Z_VALUE * se
    return {
        "low": _round_float(mean_abnormal_return_bps - margin),
        "high": _round_float(mean_abnormal_return_bps + margin),
        "se": _round_float(se),
        "method": "pre_event_newey_west",
        "z": HAC_Z_VALUE,
        "lags": int(lags),
        "baseline_n": int(baseline_n),
        "event_n": int(event_n),
        "long_run_variance": _round_float(long_run_variance),
    }


def _newey_west_long_run_variance(residuals: np.ndarray, lags: int) -> float:
    arr = np.asarray(residuals, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return 0.0

    lags = max(0, min(int(lags), n - 1))
    variance = float(np.dot(arr, arr) / n)
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1)
        autocovariance = float(np.dot(arr[lag:], arr[:-lag]) / n)
        variance += 2.0 * weight * autocovariance
    return max(variance, 0.0)


def _leakage_check(event_ts: pd.Timestamp, baseline_index: pd.DatetimeIndex) -> dict:
    dates = [idx.date().isoformat() for idx in baseline_index]
    uses_only_pre_event = all(idx < event_ts for idx in baseline_index)
    return {
        "baseline_uses_only_pre_event_data": bool(uses_only_pre_event),
        "error_estimate_uses_only_pre_event_data": bool(uses_only_pre_event),
        "event_date": event_ts.date().isoformat(),
        "baseline_return_dates": dates,
        "error_estimate_return_dates": dates,
        "max_baseline_date": dates[-1] if dates else None,
    }


def _event_study_unavailable(
    ticker: str,
    event_date: str,
    window: int,
    reason: str,
    baseline_dates: list[str],
) -> dict:
    event_ts = pd.Timestamp(event_date).tz_localize("UTC").normalize()
    baseline_index = pd.DatetimeIndex(
        [pd.Timestamp(date).tz_localize("UTC") for date in baseline_dates]
    )
    return {
        "ticker": ticker.upper(),
        "event_date": event_date,
        "window": window,
        "available": False,
        "baseline": {
            "model": "mean_close_to_close_return",
            "lookback_days": BASELINE_LOOKBACK_DAYS,
            "n_returns": len(baseline_dates),
            "return_dates": baseline_dates,
        },
        "event_window": [],
        "summary": {},
        "leakage_check": _leakage_check(event_ts, baseline_index),
        "reason": reason,
    }


def _round_float(value: float, digits: int = 6) -> float:
    return round(float(value), digits)
