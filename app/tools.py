"""
The 3 agent tools. Exactly three — resist adding more.

Day 1-2: search_docs is wired to the RAG core.
Day 3 (YOUR increment): implement get_price_data + run_event_study with a bootstrap CI and an
explicit no-look-ahead (leakage) check. Stubs below raise NotImplementedError on purpose.
"""
from __future__ import annotations
from . import rag
from .models import Citation


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
    TODO (Day 3): return a daily price/return series for `ticker` in [start, end].
    Keep it reproducible (pin the source + a local cache). No look-ahead beyond `end`.
    """
    raise NotImplementedError("get_price_data — Day 3")


def run_event_study(ticker: str, event_date: str, window: int = 5) -> dict:
    """
    TODO (Day 3): abnormal returns in [-window, +window] around event_date, with a bootstrap CI.
    LEAKAGE CHECK: the expected-return model must be estimated ONLY on pre-event data. Assert that
    no post-event observation feeds the baseline. This leakage check is the quant-rigor centerpiece.
    """
    raise NotImplementedError("run_event_study — Day 3")
