from __future__ import annotations

import json
import sys

import pytest

from app import agent, live_eval
from app.live_eval import MAX_MODEL_CALLS, run_live_eval


class ScriptedModel:
    def __init__(self, outputs: list[str]):
        self._outputs = iter(outputs)
        self.n_calls = 0

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.n_calls += 1
        return next(self._outputs)


class LiveCollection:
    def add(self, **kwargs) -> None:
        return None

    def query(self, **kwargs) -> dict:
        return {
            "documents": [["Services revenue reached a record level."]],
            "metadatas": [[{"doc_id": "apple", "chunk_id": 0}]],
            "distances": [[0.1]],
        }


def _scripted_outputs() -> list[str]:
    return [
        '{"type":"tool","name":"search_docs","arguments":'
        '{"query":"Apple Services revenue","k":3}}',
        '{"type":"final","answer":"Supported [apple::0].",'
        '"citation_ids":["apple::0"],"confidence":0.8,"refused":false}',
        '{"type":"tool","name":"search_docs","arguments":'
        '{"query":"Federal Reserve inflation interest rates","k":3}}',
        '{"type":"tool","name":"run_event_study","arguments":'
        '{"ticker":"SPY","event_date":"2026-01-07","window":1}}',
        '{"type":"final","answer":"Supported [apple::0].",'
        '"citation_ids":["apple::0"],"confidence":0.8,"refused":false}',
    ]


def test_live_eval_is_capped_and_reports_only_routing_metadata():
    model = ScriptedModel(_scripted_outputs())

    result = run_live_eval(model=model, collection=LiveCollection())

    assert result["mode"] == "live_model_smoke"
    assert result["n_cases"] == 2
    assert result["max_model_calls"] == 6
    assert MAX_MODEL_CALLS == 6
    assert result["model_calls_used"] == 5
    assert result["ordered_trajectory_accuracy"] == 1.0
    assert result["tool_coverage"] == 1.0
    assert model.n_calls == 5
    assert "answer" not in json.dumps(result)
    assert "citations" not in json.dumps(result)


def test_live_eval_requires_model_configuration_only_when_run(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_MODEL", raising=False)

    with pytest.raises(agent.AgentConfigurationError):
        run_live_eval(collection=LiveCollection())


def test_live_eval_sanitizes_provider_failures():
    class FailingModel:
        def complete(self, messages: list[dict[str, str]]) -> str:
            raise ValueError("provider response contains sensitive details")

    with pytest.raises(
        RuntimeError,
        match="live model case search_only failed: ValueError",
    ) as exc_info:
        run_live_eval(model=FailingModel(), collection=LiveCollection())

    assert "sensitive details" not in str(exc_info.value)


def test_live_eval_cli_writes_the_printed_aggregate(tmp_path, monkeypatch, capsys):
    output = tmp_path / "live.json"
    result = {
        "mode": "live_model_smoke",
        "n_cases": 0,
        "max_model_calls": 6,
        "model_calls_used": 0,
        "ordered_trajectory_accuracy": 0.0,
        "tool_coverage": 0.0,
        "cases": [],
    }
    monkeypatch.setattr(live_eval, "run_live_eval", lambda: result)
    monkeypatch.setattr(sys, "argv", ["live_eval", "--output", str(output)])

    live_eval.main()

    expected = json.dumps(result, indent=2, sort_keys=True) + "\n"
    assert output.read_text(encoding="utf-8") == expected
    assert capsys.readouterr().out == expected
