"""
ReAct-style agent loop over the 3 tools.

Day 4 (YOUR increment). The skeleton below is the shape: reason -> pick a tool -> observe ->
repeat -> answer, and every claim in the final answer must carry a citation from search_docs
(or the agent refuses). Wire an LLM API where marked. Keep the loop bounded (max_steps).
"""
from __future__ import annotations
from . import tools

TOOLS = {
    "search_docs": tools.search_docs,
    "get_price_data": tools.get_price_data,
    "run_event_study": tools.run_event_study,
}

SYSTEM_PROMPT = """You are a careful quant-research assistant. Use tools to gather evidence.
Every factual claim in your answer MUST cite a retrieved passage (doc_id, chunk_id). If the
evidence is weak or missing, refuse and say what you'd need. Never invent numbers or citations."""


def run_agent(question: str, max_steps: int = 6) -> dict:
    """
    TODO (Day 4): implement the ReAct loop.
      1. Send SYSTEM_PROMPT + question + tool schemas to the LLM.
      2. Parse the tool call; dispatch via TOOLS; feed the observation back.
      3. Loop until the model answers or max_steps; enforce citations / refusal on the final answer.
    Returns {answer, citations, confidence, refused}.
    """
    raise NotImplementedError("run_agent — Day 4")
