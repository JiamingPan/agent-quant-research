from __future__ import annotations

import json

from app import agent, rag


class RecordingCollection:
    def __init__(self):
        self.add_calls: list[dict] = []
        self.query_calls: list[dict] = []

    def add(self, **kwargs) -> None:
        self.add_calls.append(kwargs)

    def query(self, **kwargs) -> dict:
        self.query_calls.append(kwargs)
        return {
            "documents": [["Services revenue reached a record level."]],
            "metadatas": [[{"doc_id": "apple", "chunk_id": 0}]],
            "distances": [[0.1]],
        }


class FailingCollection:
    def add(self, **kwargs) -> None:
        raise AssertionError("default collection must not receive eval ingest")

    def query(self, **kwargs) -> dict:
        raise AssertionError("default collection must not receive eval search")


class ScriptedModel:
    def __init__(self, outputs: list[str]):
        self._outputs = iter(outputs)

    def complete(self, messages: list[dict[str, str]]) -> str:
        return next(self._outputs)


def _search_result() -> dict:
    return {
        "passages": [
            {
                "doc_id": "apple",
                "chunk_id": 0,
                "citation": "apple::0",
                "text": "Services revenue reached a record level.",
                "distance": 0.1,
                "score": 0.9,
                "score_kind": "cosine_similarity",
            }
        ],
        "refused": False,
        "reason": None,
    }


def test_explicit_collection_isolates_eval_ingest_and_search(tmp_path, monkeypatch):
    document = tmp_path / "apple.txt"
    document.write_text("Services revenue reached a record level.", encoding="utf-8")
    explicit = RecordingCollection()
    monkeypatch.setattr(rag, "_collection", FailingCollection())

    doc_id, n_chunks = rag.ingest(str(document), "apple", collection=explicit)
    passages, refused, reason = rag.search(
        "record services revenue",
        k=1,
        collection=explicit,
    )

    assert (doc_id, n_chunks) == ("apple", 1)
    assert len(explicit.add_calls) == 1
    assert len(explicit.query_calls) == 1
    assert passages[0].citation == "apple::0"
    assert refused is False
    assert reason is None


def test_supplied_tool_registry_isolates_agent_dispatch(monkeypatch):
    def fail_global(query: str, k: int = 4) -> dict:
        raise AssertionError("global search tool must not receive eval call")

    monkeypatch.setitem(agent.TOOLS, "search_docs", fail_global)
    local_calls: list[tuple[str, int]] = []

    def local_search(query: str, k: int = 4) -> dict:
        local_calls.append((query, k))
        return _search_result()

    registry = {
        "search_docs": local_search,
        "get_price_data": agent.TOOLS["get_price_data"],
        "run_event_study": agent.TOOLS["run_event_study"],
    }
    model = ScriptedModel(
        [
            '{"type":"tool","name":"search_docs","arguments":'
            '{"query":"Apple services","k":3}}',
            '{"type":"final","answer":"Record services revenue [apple::0].",'
            '"citation_ids":["apple::0"],"confidence":1.0,"refused":false}',
        ]
    )

    result = agent.run_agent(
        "What happened to Apple services?",
        model=model,
        tool_registry=registry,
    )

    assert result["refused"] is False
    assert local_calls == [("Apple services", 3)]


def test_offline_eval_model_grounds_or_refuses_from_tool_observation():
    from app.eval_harness import OfflineEvalModel

    grounded = OfflineEvalModel("What happened to Apple services?")
    first_action = json.loads(grounded.complete([]))
    assert first_action == {
        "type": "tool",
        "name": "search_docs",
        "arguments": {
            "query": "What happened to Apple services?",
            "k": 3,
        },
    }
    success_observation = {
        "ok": True,
        "tool": "search_docs",
        "result": _search_result(),
    }
    final_action = json.loads(
        grounded.complete(
            [
                {
                    "role": "user",
                    "content": "TOOL_OBSERVATION\n"
                    + json.dumps(success_observation),
                }
            ]
        )
    )
    assert final_action["refused"] is False
    assert final_action["citation_ids"] == ["apple::0"]
    assert "apple::0" in final_action["answer"]
    assert grounded.call_records == [
        {
            "expected": "search_docs",
            "actual": "search_docs",
            "succeeded": True,
        }
    ]

    refusing = OfflineEvalModel("How do I bake bread?")
    refusing.complete([])
    refused_observation = {
        "ok": True,
        "tool": "search_docs",
        "result": {"passages": [], "refused": True, "reason": "weak"},
    }
    refused_action = json.loads(
        refusing.complete(
            [
                {
                    "role": "user",
                    "content": "TOOL_OBSERVATION\n"
                    + json.dumps(refused_observation),
                }
            ]
        )
    )
    assert refused_action["refused"] is True
    assert refused_action["citation_ids"] == []


class HarnessCollection(RecordingCollection):
    def query(self, **kwargs) -> dict:
        self.query_calls.append(kwargs)
        query = kwargs["query_texts"][0].lower()
        distance = 0.95 if "bread" in query else 0.1
        return {
            "documents": [["Services revenue reached a record level."]],
            "metadatas": [[{"doc_id": "apple", "chunk_id": 0}]],
            "distances": [[distance]],
        }


def test_run_offline_eval_aggregates_stable_contract_metrics(tmp_path):
    from app.eval_harness import run_offline_eval

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "apple.txt").write_text(
        "Services revenue reached a record level.",
        encoding="utf-8",
    )
    (corpus_dir / "manifest.json").write_text(
        json.dumps([{"doc_id": "apple", "filename": "apple.txt"}]),
        encoding="utf-8",
    )
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            [
                {
                    "question": "What happened to Apple services?",
                    "expected_doc_id": "apple",
                    "should_refuse": False,
                },
                {
                    "question": "How do I bake bread?",
                    "expected_doc_id": None,
                    "should_refuse": True,
                },
            ]
        ),
        encoding="utf-8",
    )

    result = run_offline_eval(
        corpus_dir=corpus_dir,
        cases_path=cases_path,
        collection=HarnessCollection(),
    )

    assert result["mode"] == "offline_contract"
    assert result["corpus_documents"] == 1
    assert result["cases"] == {
        "retrieval": 1,
        "agent": 2,
        "refusal": 2,
        "leakage_guard": 2,
        "orchestration": 5,
    }
    assert result["metrics"] == {
        "hit_at_1": 1.0,
        "mrr": 1.0,
        "citation_grounding_rate": 1.0,
        "tool_call_success_rate": 1.0,
        "refusal_accuracy": 1.0,
        "leakage_guard_rate": 1.0,
        "trajectory_contract_rate": 1.0,
        "trace_completeness_rate": 1.0,
        "recovery_contract_rate": 1.0,
        "mean_tool_steps": 1.4,
    }
    assert str(tmp_path) not in json.dumps(result)
