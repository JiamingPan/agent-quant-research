"""
The 3 agent tools. Exactly three — resist adding more.

Day 1-2: search_docs is wired to the RAG core.
Day 3: get_price_data is wired as a thin, JSON-safe adapter over cached bars.
run_event_study remains the next increment.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from . import rag

SPX_ROOT_ENV = "SPX_NEWS_INTRADAY_ROOT"


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
    TODO (Day 3): abnormal returns in [-window, +window] around event_date, with a bootstrap CI.
    LEAKAGE CHECK: the expected-return model must be estimated ONLY on pre-event data. Assert that
    no post-event observation feeds the baseline. This leakage check is the quant-rigor centerpiece.
    """
    raise NotImplementedError("run_event_study — Day 3")
