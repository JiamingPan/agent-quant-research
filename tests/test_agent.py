from __future__ import annotations

from fastapi.testclient import TestClient

from app import agent
from app.main import app


class ScriptedModel:
    def __init__(self, outputs: list[str]):
        self._outputs = iter(outputs)
        self.messages: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.messages.append([message.copy() for message in messages])
        return next(self._outputs)


def _citation(citation: str = "apple::2") -> dict:
    doc_id, chunk_id = citation.split("::")
    return {
        "doc_id": doc_id,
        "chunk_id": int(chunk_id),
        "citation": citation,
        "text": "Services revenue reached a record level.",
        "distance": 0.2,
        "score": 0.8,
        "score_kind": "cosine_similarity",
    }


def test_run_agent_retrieves_then_returns_grounded_answer(monkeypatch):
    monkeypatch.setitem(
        agent.TOOLS,
        "search_docs",
        lambda query, k=4: {
            "passages": [_citation()],
            "refused": False,
            "reason": None,
        },
    )
    model = ScriptedModel(
        [
            '{"type":"tool","name":"search_docs","arguments":'
            '{"query":"Apple services","k":4}}',
            '{"type":"final","answer":"Services reached a record [apple::2].",'
            '"citation_ids":["apple::2"],"confidence":0.9,"refused":false}',
        ]
    )

    result = agent.run_agent("What happened to services?", model=model)

    assert result["refused"] is False
    assert result["answer"] == "Services reached a record [apple::2]."
    assert result["confidence"] == 0.9
    assert result["citations"] == [_citation()]
    assert result["trace"] == [
        {
            "step": 1,
            "tool": "search_docs",
            "arguments": {"query": "Apple services", "k": 4},
            "ok": True,
            "error": None,
        }
    ]
    assert result["steps_used"] == 2


def test_model_can_call_price_then_search_before_answer(monkeypatch):
    calls: list[tuple[str, dict]] = []

    def fake_prices(ticker: str, start: str, end: str) -> dict:
        calls.append(("get_price_data", {"ticker": ticker, "start": start, "end": end}))
        return {"ticker": ticker, "available": True, "rows": [{"close": 200.0}]}

    def fake_search(query: str, k: int = 4) -> dict:
        calls.append(("search_docs", {"query": query, "k": k}))
        return {"passages": [_citation()], "refused": False, "reason": None}

    monkeypatch.setitem(agent.TOOLS, "get_price_data", fake_prices)
    monkeypatch.setitem(agent.TOOLS, "search_docs", fake_search)
    model = ScriptedModel(
        [
            '{"type":"tool","name":"get_price_data","arguments":'
            '{"ticker":"AAPL","start":"2026-01-01","end":"2026-01-05"}}',
            '{"type":"tool","name":"search_docs","arguments":'
            '{"query":"Apple services","k":2}}',
            '{"type":"final","answer":"The filing provides context [apple::2].",'
            '"citation_ids":["apple::2"],"confidence":0.75,"refused":false}',
        ]
    )

    result = agent.run_agent("Research Apple", model=model)

    assert result["refused"] is False
    assert calls == [
        (
            "get_price_data",
            {"ticker": "AAPL", "start": "2026-01-01", "end": "2026-01-05"},
        ),
        ("search_docs", {"query": "Apple services", "k": 2}),
    ]


def test_unknown_tool_is_observed_and_model_can_recover(monkeypatch):
    monkeypatch.setitem(
        agent.TOOLS,
        "search_docs",
        lambda query, k=4: {
            "passages": [_citation()],
            "refused": False,
            "reason": None,
        },
    )
    model = ScriptedModel(
        [
            '{"type":"tool","name":"trade_stock","arguments":{"ticker":"AAPL"}}',
            '{"type":"tool","name":"search_docs","arguments":{"query":"Apple"}}',
            '{"type":"final","answer":"Supported [apple::2].",'
            '"citation_ids":["apple::2"],"confidence":0.7,"refused":false}',
        ]
    )

    result = agent.run_agent("Research Apple", model=model)

    assert result["refused"] is False
    assert "unknown tool: trade_stock" in model.messages[1][-1]["content"]
    assert result["trace"] == [
        {
            "step": 1,
            "tool": "trade_stock",
            "arguments": {"ticker": "AAPL"},
            "ok": False,
            "error": "unknown tool: trade_stock",
        },
        {
            "step": 2,
            "tool": "search_docs",
            "arguments": {"query": "Apple", "k": 4},
            "ok": True,
            "error": None,
        },
    ]
    assert result["steps_used"] == 3


def test_invalid_arguments_are_observed_and_model_can_recover(monkeypatch):
    monkeypatch.setitem(
        agent.TOOLS,
        "search_docs",
        lambda query, k=4: {
            "passages": [_citation()],
            "refused": False,
            "reason": None,
        },
    )
    model = ScriptedModel(
        [
            '{"type":"tool","name":"search_docs","arguments":{"query":"Apple","k":0}}',
            '{"type":"tool","name":"search_docs","arguments":{"query":"Apple","k":4}}',
            '{"type":"final","answer":"Supported [apple::2].",'
            '"citation_ids":["apple::2"],"confidence":0.7,"refused":false}',
        ]
    )

    result = agent.run_agent("Research Apple", model=model)

    assert result["refused"] is False
    assert "invalid arguments" in model.messages[1][-1]["content"]


def test_retrieval_refusal_cannot_become_uncited_answer(monkeypatch):
    monkeypatch.setitem(
        agent.TOOLS,
        "search_docs",
        lambda query, k=4: {
            "passages": [],
            "refused": True,
            "reason": "weak retrieval",
        },
    )
    model = ScriptedModel(
        [
            '{"type":"tool","name":"search_docs","arguments":{"query":"Unknown"}}',
            '{"type":"final","answer":"I know the answer.","citation_ids":[],'
            '"confidence":0.9,"refused":false}',
        ]
    )

    result = agent.run_agent("Unknown question", model=model)

    assert result["refused"] is True
    assert result["confidence"] == 0.0
    assert "did not cite" in result["answer"]


def test_fabricated_citation_is_refused():
    model = ScriptedModel(
        [
            '{"type":"final","answer":"Invented [fake::9].",'
            '"citation_ids":["fake::9"],"confidence":0.9,"refused":false}'
        ]
    )

    result = agent.run_agent("Make something up", model=model)

    assert result["refused"] is True
    assert "was not retrieved" in result["answer"]


def test_citation_missing_from_answer_text_is_refused(monkeypatch):
    monkeypatch.setitem(
        agent.TOOLS,
        "search_docs",
        lambda query, k=4: {
            "passages": [_citation()],
            "refused": False,
            "reason": None,
        },
    )
    model = ScriptedModel(
        [
            '{"type":"tool","name":"search_docs","arguments":{"query":"Apple"}}',
            '{"type":"final","answer":"Services reached a record.",'
            '"citation_ids":["apple::2"],"confidence":0.8,"refused":false}',
        ]
    )

    result = agent.run_agent("Research Apple", model=model)

    assert result["refused"] is True
    assert "missing from the answer text" in result["answer"]


def test_malformed_action_is_refused():
    model = ScriptedModel(["not JSON"])

    result = agent.run_agent("Research Apple", model=model)

    assert result["refused"] is True
    assert result["confidence"] == 0.0
    assert "malformed model action" in result["answer"]


def test_max_steps_exhaustion_refuses(monkeypatch):
    monkeypatch.setitem(
        agent.TOOLS,
        "search_docs",
        lambda query, k=4: {
            "passages": [_citation()],
            "refused": False,
            "reason": None,
        },
    )
    action = '{"type":"tool","name":"search_docs","arguments":{"query":"Apple"}}'
    model = ScriptedModel([action, action])

    result = agent.run_agent("Research Apple", max_steps=2, model=model)

    assert result["refused"] is True
    assert result["confidence"] == 0.0
    assert "maximum 2 steps" in result["answer"]
    assert result["trace"] == [
        {
            "step": 1,
            "tool": "search_docs",
            "arguments": {"query": "Apple", "k": 4},
            "ok": True,
            "error": None,
        },
        {
            "step": 2,
            "tool": "search_docs",
            "arguments": {"query": "Apple", "k": 4},
            "ok": True,
            "error": None,
        },
    ]
    assert result["steps_used"] == 2


def test_research_endpoint_returns_agent_result(monkeypatch):
    monkeypatch.setattr(
        agent,
        "run_agent",
        lambda question: {
            "answer": "Grounded [apple::2].",
            "citations": [_citation()],
            "confidence": 0.9,
            "refused": False,
            "trace": [
                {
                    "step": 1,
                    "tool": "search_docs",
                    "arguments": {"query": "Apple services", "k": 4},
                    "ok": True,
                    "error": None,
                }
            ],
            "steps_used": 2,
        },
    )

    response = TestClient(app).post(
        "/research",
        json={"question": "What happened to Apple services?"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "question": "What happened to Apple services?",
        "answer": "Grounded [apple::2].",
        "citations": [_citation()],
        "confidence": 0.9,
        "refused": False,
        "trace": [
            {
                "step": 1,
                "tool": "search_docs",
                "arguments": {"query": "Apple services", "k": 4},
                "ok": True,
                "error": None,
            }
        ],
        "steps_used": 2,
    }


def test_research_endpoint_maps_missing_model_config_to_503(monkeypatch):
    def fail(question: str) -> dict:
        raise agent.AgentConfigurationError("Set OPENAI_API_KEY and AGENT_MODEL.")

    monkeypatch.setattr(agent, "run_agent", fail)

    response = TestClient(app).post(
        "/research",
        json={"question": "What happened?"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Set OPENAI_API_KEY and AGENT_MODEL."


def test_system_prompt_describes_arguments_for_all_three_tools():
    model = ScriptedModel(
        [
            '{"type":"final","answer":"Insufficient evidence.","citation_ids":[],'
            '"confidence":0.0,"refused":true}'
        ]
    )

    agent.run_agent("Research Apple", model=model)

    system_prompt = model.messages[0][0]["content"]
    assert '"name":"search_docs"' in system_prompt
    assert '"query":"financial question"' in system_prompt
    assert '"name":"get_price_data"' in system_prompt
    assert '"ticker":"SPY"' in system_prompt
    assert '"start":"2026-01-01"' in system_prompt
    assert '"end":"2026-01-31"' in system_prompt
    assert '"name":"run_event_study"' in system_prompt
    assert '"event_date":"2026-01-15"' in system_prompt
    assert '"window":5' in system_prompt


def test_system_prompt_distinguishes_raw_prices_from_event_study():
    model = ScriptedModel(
        [
            '{"type":"final","answer":"Insufficient evidence.","citation_ids":[],'
            '"confidence":0.0,"refused":true}'
        ]
    )

    agent.run_agent("Evaluate SPY around an event", model=model)

    system_prompt = model.messages[0][0]["content"]
    assert "get_price_data returns raw historical bars" in system_prompt
    assert "run_event_study loads its own price data" in system_prompt
    assert "do not call get_price_data first" in system_prompt
