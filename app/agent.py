"""Bounded ReAct-style agent loop over the three research tools."""
from __future__ import annotations

import json
import os
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import tools

TOOLS = {
    "search_docs": tools.search_docs,
    "get_price_data": tools.get_price_data,
    "run_event_study": tools.run_event_study,
}

SYSTEM_PROMPT = """You are a careful quant-research assistant. You may call exactly three
tools: search_docs, get_price_data, and run_event_study. Respond with one JSON object only.

Available tool actions and argument shapes:
{"type":"tool","name":"search_docs","arguments":{"query":"financial question","k":4}}
{"type":"tool","name":"get_price_data","arguments":{"ticker":"SPY","start":"2026-01-01","end":"2026-01-31"}}
{"type":"tool","name":"run_event_study","arguments":{"ticker":"SPY","event_date":"2026-01-15","window":5}}

Final action:
{"type":"final","answer":"Claim [doc_id::chunk_id].","citation_ids":["doc_id::chunk_id"],
"confidence":0.8,"refused":false}

Use tools to gather evidence. Every non-refused answer needs at least one citation returned by
search_docs, and every citation id must appear in the answer text. If evidence is weak or
missing, return a final action with refused=true, no citation ids, and explain what is needed.
Never invent numbers or citations."""


class AgentModel(Protocol):
    """Minimal model interface used by the provider-independent loop."""

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Return one JSON action for the current conversation."""


class AgentActionError(ValueError):
    """The model returned an action that does not satisfy the agent contract."""


class AgentConfigurationError(RuntimeError):
    """The external model adapter is not configured or installed."""


class OpenAICompatibleModel:
    """Small adapter over an OpenAI-compatible chat-completions endpoint."""

    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise AgentConfigurationError(
                "Install dependencies with pip install -r requirements.txt."
            ) from exc

        client_args = {"api_key": api_key}
        if base_url:
            client_args["base_url"] = base_url
        self._client = OpenAI(**client_args)
        self._model = model

    def complete(self, messages: list[dict[str, str]]) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0,
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("model returned empty content")
        return content


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolAction(_StrictModel):
    type: Literal["tool"]
    name: str
    arguments: dict[str, Any]


class FinalAction(_StrictModel):
    type: Literal["final"]
    answer: str
    citation_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    refused: bool


class SearchDocsArgs(_StrictModel):
    query: str
    k: int = Field(default=4, ge=1, le=20)


class GetPriceDataArgs(_StrictModel):
    ticker: str
    start: str
    end: str


class RunEventStudyArgs(_StrictModel):
    ticker: str
    event_date: str
    window: int = Field(default=5, ge=0)


TOOL_ARGUMENT_MODELS: dict[str, type[BaseModel]] = {
    "search_docs": SearchDocsArgs,
    "get_price_data": GetPriceDataArgs,
    "run_event_study": RunEventStudyArgs,
}


def _parse_action(raw: str) -> ToolAction | FinalAction:
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise AgentActionError(f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise AgentActionError("action must be a JSON object")

    action_type = payload.get("type")
    action_model: type[ToolAction] | type[FinalAction]
    if action_type == "tool":
        action_model = ToolAction
    elif action_type == "final":
        action_model = FinalAction
    else:
        raise AgentActionError("action type must be 'tool' or 'final'")

    try:
        return action_model.model_validate(payload)
    except ValidationError as exc:
        raise AgentActionError(str(exc)) from exc


def _dispatch_tool(action: ToolAction) -> dict:
    argument_model = TOOL_ARGUMENT_MODELS.get(action.name)
    tool = TOOLS.get(action.name)
    if argument_model is None or tool is None:
        return {
            "ok": False,
            "tool": action.name,
            "error": f"unknown tool: {action.name}",
        }

    try:
        arguments = argument_model.model_validate(action.arguments)
    except ValidationError as exc:
        return {
            "ok": False,
            "tool": action.name,
            "error": f"invalid arguments: {exc}",
        }

    try:
        result = tool(**arguments.model_dump())
    except Exception as exc:
        return {
            "ok": False,
            "tool": action.name,
            "error": f"tool failed: {type(exc).__name__}: {exc}",
        }
    return {"ok": True, "tool": action.name, "result": result}


def _register_citations(observation: dict, registry: dict[str, dict]) -> None:
    if not observation.get("ok") or observation.get("tool") != "search_docs":
        return
    result = observation.get("result") or {}
    for passage in result.get("passages", []):
        citation_id = str(passage.get("citation", "")).strip()
        if citation_id:
            registry[citation_id] = passage


def _refusal(reason: str) -> dict:
    return {
        "answer": reason,
        "citations": [],
        "confidence": 0.0,
        "refused": True,
    }


def _finalize(action: FinalAction, registry: dict[str, dict]) -> dict:
    if action.refused:
        return {
            "answer": action.answer,
            "citations": [],
            "confidence": action.confidence,
            "refused": True,
        }
    if not action.citation_ids:
        return _refusal("Refused: the answer did not cite retrieved evidence.")

    citations: list[dict] = []
    seen: set[str] = set()
    for citation_id in action.citation_ids:
        if citation_id not in registry:
            return _refusal(f"Refused: citation {citation_id!r} was not retrieved.")
        if citation_id not in action.answer:
            return _refusal(
                f"Refused: citation {citation_id!r} is missing from the answer text."
            )
        if citation_id not in seen:
            citations.append(registry[citation_id])
            seen.add(citation_id)

    return {
        "answer": action.answer,
        "citations": citations,
        "confidence": action.confidence,
        "refused": False,
    }


def _default_model() -> AgentModel:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("AGENT_MODEL", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    if not api_key or not model:
        raise AgentConfigurationError("Set OPENAI_API_KEY and AGENT_MODEL.")
    return OpenAICompatibleModel(api_key=api_key, model=model, base_url=base_url)


def run_agent(
    question: str,
    max_steps: int = 6,
    model: AgentModel | None = None,
) -> dict:
    """Run a bounded model-tool loop and enforce citation provenance."""
    if not question.strip():
        return _refusal("Refused: a non-empty research question is required.")
    if model is None:
        model = _default_model()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    citation_registry: dict[str, dict] = {}

    for _ in range(max_steps):
        raw_action = model.complete(messages)
        try:
            action = _parse_action(raw_action)
        except AgentActionError as exc:
            return _refusal(f"Refused: malformed model action: {exc}")

        messages.append({"role": "assistant", "content": raw_action})
        if isinstance(action, FinalAction):
            return _finalize(action, citation_registry)

        observation = _dispatch_tool(action)
        _register_citations(observation, citation_registry)
        messages.append(
            {
                "role": "user",
                "content": "TOOL_OBSERVATION\n" + json.dumps(observation, default=str),
            }
        )

    return _refusal(f"Refused: agent reached the maximum {max_steps} steps without an answer.")
