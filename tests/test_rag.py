from __future__ import annotations

import pytest

from app import rag


def test_chunk_preserves_overlap_between_adjacent_chunks():
    text = "a" * rag.CHUNK_CHARS + "b" * rag.CHUNK_CHARS

    chunks = rag._chunk(text)

    assert len(chunks) >= 2
    assert len(chunks[0]) == rag.CHUNK_CHARS
    assert chunks[1].startswith("a" * rag.CHUNK_OVERLAP)


def test_search_refuses_when_collection_is_empty(monkeypatch):
    class EmptyCollection:
        def query(self, query_texts: list[str], n_results: int) -> dict:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    monkeypatch.setattr(rag, "_collection", EmptyCollection())

    passages, refused, reason = rag.search("what did revenue do?", k=3)

    assert passages == []
    assert refused is True
    assert reason is not None


def test_search_refuses_when_best_similarity_is_weak(monkeypatch):
    class WeakCollection:
        def query(self, query_texts: list[str], n_results: int) -> dict:
            return {
                "documents": [["unrelated passage"]],
                "metadatas": [[{"doc_id": "doc", "chunk_id": 7}]],
                "distances": [[0.95]],
            }

    monkeypatch.setattr(rag, "_collection", WeakCollection())

    passages, refused, reason = rag.search("capital allocation?", k=1)

    assert passages[0].doc_id == "doc"
    assert passages[0].chunk_id == 7
    assert passages[0].citation == "doc::7"
    assert passages[0].distance == pytest.approx(0.95)
    assert passages[0].score == pytest.approx(0.05)
    assert passages[0].score_kind == "cosine_similarity"
    assert refused is True
    assert reason is not None


def test_search_returns_citations_when_similarity_is_strong(monkeypatch):
    class StrongCollection:
        def query(self, query_texts: list[str], n_results: int) -> dict:
            return {
                "documents": [["capital allocation details"]],
                "metadatas": [[{"doc_id": "10k", "chunk_id": 3}]],
                "distances": [[0.10]],
            }

    monkeypatch.setattr(rag, "_collection", StrongCollection())

    passages, refused, reason = rag.search("capital allocation?", k=1)

    assert passages[0].doc_id == "10k"
    assert passages[0].chunk_id == 3
    assert passages[0].citation == "10k::3"
    assert passages[0].distance == pytest.approx(0.10)
    assert passages[0].score == pytest.approx(0.90)
    assert passages[0].score_kind == "cosine_similarity"
    assert refused is False
    assert reason is None
