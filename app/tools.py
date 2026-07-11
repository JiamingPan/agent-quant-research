"""
The 3 agent tools. Exactly three — resist adding more.

Day 1-2: search_docs is wired to the RAG core.
Day 3: get_price_data is wired as a thin, JSON-safe adapter over cached bars.
Day 3: run_event_study is leakage-checked with a pre-window bootstrap CI.
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
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 0
EVENT_FETCH_BUFFER_DAYS = 7


def search_docs(query: str, k: int = 4) -> dict:
    """RAG retrieval with citations + refusal. Returns a JSON-able dict for the agent."""
    passages, refused, reason = rag.search(query, k=k)
    return {
        "passages": [p.model_dump() for p in passages],
        "refused": refused,
        "reason": reason,
    }


def get_price_data(
    ticker: str,
    start: str,
    end: str,
    *,
    max_rows: Optional[int] = 2000,
) -> dict:
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

    rows, truncated = _frame_to_records(frame, max_rows=max_rows)
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
    max_rows: Optional[int] = 2000,
) -> tuple[list[dict], bool]:
    truncated = max_rows is not None and len(frame) > max_rows
    limited = frame if max_rows is None else frame.iloc[:max_rows]
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
    Compute close-to-close log abnormal returns around an event date.

    The expected-return baseline is the mean close-to-close log return from
    dates strictly before the event window starts.
    """
    if window < 0:
        raise ValueError("window must be non-negative")

    ticker = ticker.upper()
    event_ts = _parse_bound(event_date, is_end=False).normalize()
    baseline_cutoff = event_ts - pd.Timedelta(days=window)
    fetch_start = (event_ts - pd.Timedelta(days=BASELINE_LOOKBACK_DAYS)).date().isoformat()
    fetch_end = (
        event_ts + pd.Timedelta(days=2 * window + EVENT_FETCH_BUFFER_DAYS)
    ).date().isoformat()
    prices = get_price_data(ticker, fetch_start, fetch_end, max_rows=None)
    if not prices.get("available"):
        return _event_study_unavailable(
            ticker,
            event_date,
            window,
            prices.get("reason") or "price data unavailable",
            baseline_dates=[],
            baseline_cutoff=baseline_cutoff,
        )

    daily = _daily_close_returns(prices.get("rows", []))
    if daily.empty:
        return _event_study_unavailable(
            ticker,
            event_date,
            window,
            "price data has no daily close returns",
            baseline_dates=[],
            baseline_cutoff=baseline_cutoff,
        )

    if event_ts not in daily.index:
        return _event_study_unavailable(
            ticker,
            event_date,
            window,
            "Event date has no daily close observation.",
            baseline_dates=[],
            baseline_cutoff=baseline_cutoff,
        )

    event_position = int(daily.index.get_loc(event_ts))
    window_start_position = event_position - window
    window_end_position = event_position + window
    if window_start_position < 0 or window_end_position >= len(daily):
        return _event_study_unavailable(
            ticker,
            event_date,
            window,
            f"Need {window} trading observations on each side of the event date.",
            baseline_dates=[],
            baseline_cutoff=baseline_cutoff,
        )

    baseline_cutoff = daily.index[window_start_position]
    baseline = daily.iloc[:window_start_position]["return_bps"].dropna()
    _assert_no_baseline_leakage(baseline.index, baseline_cutoff)
    baseline_dates = [idx.date().isoformat() for idx in baseline.index]
    if len(baseline) < MIN_BASELINE_RETURNS:
        return _event_study_unavailable(
            ticker,
            event_date,
            window,
            f"Need at least {MIN_BASELINE_RETURNS} pre-event returns before "
            f"{baseline_cutoff.date().isoformat()} for baseline.",
            baseline_dates=baseline_dates,
            baseline_cutoff=baseline_cutoff,
        )

    expected_return_bps = float(baseline.mean())
    event_window = daily.iloc[window_start_position : window_end_position + 1].copy()
    event_window = event_window.dropna(axis=0, subset=["return_bps"])
    if len(event_window) != 2 * window + 1:
        return _event_study_unavailable(
            ticker,
            event_date,
            window,
            "Event window has missing daily return observations.",
            baseline_dates=baseline_dates,
            baseline_cutoff=baseline_cutoff,
        )

    event_window["relative_day"] = np.arange(-window, window + 1)
    event_rows = _event_window_rows(event_window, expected_return_bps)
    abnormal_values = (
        event_window["return_bps"].to_numpy(dtype=float) - expected_return_bps
    )
    mean_abnormal_return_bps = float(np.mean(abnormal_values))
    car_bps = float(np.sum(abnormal_values))
    ci = _bootstrap_car_ci(
        car_bps=car_bps,
        baseline_returns_bps=baseline,
        event_n=len(abnormal_values),
    )

    return {
        "ticker": ticker,
        "event_date": event_date,
        "window": window,
        "available": True,
        "price_source": prices.get("source"),
        "n_pre_obs": int(len(baseline)),
        "baseline": {
            "model": "mean_close_to_close_log_return",
            "return_type": "log",
            "lookback_days": BASELINE_LOOKBACK_DAYS,
            "cutoff_date": baseline_cutoff.date().isoformat(),
            "start": baseline.index.min().date().isoformat(),
            "end": baseline.index.max().date().isoformat(),
            "n_returns": int(len(baseline)),
            "return_dates": baseline_dates,
            "expected_return_bps": _round_float(expected_return_bps),
        },
        "event_window": event_rows,
        "summary": {
            "n_observations": len(event_rows),
            "n_pre_obs": int(len(baseline)),
            "mean_abnormal_return_bps": _round_float(mean_abnormal_return_bps),
            "car_bps": _round_float(car_bps),
            "cumulative_abnormal_return_bps": _round_float(car_bps),
            "bootstrap_car_ci_bps": ci,
        },
        "leakage_check": _leakage_check(event_ts, baseline_cutoff, baseline.index),
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
    daily["return_bps"] = np.log(daily["close"] / daily["close"].shift(1)) * 1e4
    return daily


def _event_window_rows(
    event_window: pd.DataFrame,
    expected_return_bps: float,
) -> list[dict]:
    rows: list[dict] = []
    car_bps = 0.0
    for idx, row in event_window.iterrows():
        actual_return_bps = float(row["return_bps"])
        abnormal_return_bps = actual_return_bps - expected_return_bps
        car_bps += abnormal_return_bps
        rows.append(
            {
                "date": idx.date().isoformat(),
                "relative_day": int(row["relative_day"]),
                "close": _round_float(float(row["close"])),
                "actual_return_bps": _round_float(actual_return_bps),
                "expected_return_bps": _round_float(expected_return_bps),
                "ar_bps": _round_float(abnormal_return_bps),
                "abnormal_return_bps": _round_float(abnormal_return_bps),
                "car_bps": _round_float(car_bps),
            }
        )
    return rows


def _bootstrap_car_ci(
    car_bps: float,
    baseline_returns_bps: pd.Series,
    event_n: int,
) -> dict:
    baseline = np.asarray(baseline_returns_bps.dropna(), dtype=float)
    n_pre_obs = len(baseline)
    if n_pre_obs < MIN_BASELINE_RETURNS or event_n < 1:
        return {
            "low": None,
            "high": None,
            "method": "pre_event_residual_percentile",
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "n_pre_obs": int(n_pre_obs),
            "event_n": int(event_n),
        }

    centered_pre_event_returns = baseline - float(np.mean(baseline))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled_residuals = rng.choice(
        centered_pre_event_returns,
        size=(BOOTSTRAP_SAMPLES, event_n),
        replace=True,
    )
    bootstrap_cars = car_bps + sampled_residuals.sum(axis=1)
    low, high = np.percentile(bootstrap_cars, [2.5, 97.5])
    return {
        "low": _round_float(low),
        "high": _round_float(high),
        "method": "pre_event_residual_percentile",
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "n_pre_obs": int(n_pre_obs),
        "event_n": int(event_n),
    }


def _assert_no_baseline_leakage(
    baseline_index: pd.DatetimeIndex,
    baseline_cutoff: pd.Timestamp,
) -> None:
    leaking_dates = baseline_index[baseline_index >= baseline_cutoff]
    assert len(leaking_dates) == 0, (
        "Baseline leakage: return dates must be strictly before "
        f"{baseline_cutoff.date().isoformat()}; got "
        f"{[idx.date().isoformat() for idx in leaking_dates]}"
    )


def _leakage_check(
    event_ts: pd.Timestamp,
    baseline_cutoff: pd.Timestamp,
    baseline_index: pd.DatetimeIndex,
) -> dict:
    dates = [idx.date().isoformat() for idx in baseline_index]
    uses_only_pre_window = all(idx < baseline_cutoff for idx in baseline_index)
    return {
        "status": "passed" if uses_only_pre_window else "failed",
        "baseline_uses_only_pre_window_data": bool(uses_only_pre_window),
        "bootstrap_uses_only_pre_window_data": bool(uses_only_pre_window),
        "baseline_uses_only_pre_event_data": bool(uses_only_pre_window),
        "event_date": event_ts.date().isoformat(),
        "baseline_cutoff_date": baseline_cutoff.date().isoformat(),
        "baseline_return_dates": dates,
        "bootstrap_return_dates": dates,
        "error_estimate_uses_only_pre_event_data": bool(uses_only_pre_window),
        "error_estimate_return_dates": dates,
        "max_baseline_date": dates[-1] if dates else None,
    }


def _event_study_unavailable(
    ticker: str,
    event_date: str,
    window: int,
    reason: str,
    baseline_dates: list[str],
    baseline_cutoff: pd.Timestamp,
) -> dict:
    event_ts = _parse_bound(event_date, is_end=False).normalize()
    baseline_index = pd.DatetimeIndex(
        [pd.Timestamp(date).tz_localize("UTC") for date in baseline_dates]
    )
    _assert_no_baseline_leakage(baseline_index, baseline_cutoff)
    return {
        "ticker": ticker.upper(),
        "event_date": event_date,
        "window": window,
        "available": False,
        "n_pre_obs": len(baseline_dates),
        "baseline": {
            "model": "mean_close_to_close_log_return",
            "return_type": "log",
            "lookback_days": BASELINE_LOOKBACK_DAYS,
            "cutoff_date": baseline_cutoff.date().isoformat(),
            "n_returns": len(baseline_dates),
            "return_dates": baseline_dates,
        },
        "event_window": [],
        "summary": {},
        "leakage_check": _leakage_check(event_ts, baseline_cutoff, baseline_index),
        "reason": reason,
    }


def _round_float(value: float, digits: int = 6) -> float:
    return round(float(value), digits)
