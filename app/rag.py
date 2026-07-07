"""
RAG core — Day 1-2 deliverable (working).

Chunk documents, embed + store in Chroma, retrieve top-k with citations, and REFUSE when the
best match is too weak. The refusal behavior is a deliberate rigor signal, not an afterthought.

Uses Chroma's default embedding function (ONNXMiniLM_L6_V2 in the tested local install).
Swap `embedding_function` for an API embedder later if you want; the interface is the same.
"""
from __future__ import annotations
import os
from typing import Optional

import chromadb
from .models import Citation

# --- config knobs ---
CHUNK_CHARS = 1000          # ~200-250 tokens per chunk
CHUNK_OVERLAP = 150
DISTANCE_METRIC = "cosine"
SCORE_KIND = "cosine_similarity"
REFUSE_SCORE_THRESHOLD = 0.25  # if the best passage's score < this, refuse (tune on real data)
REFUSE_SIMILARITY = REFUSE_SCORE_THRESHOLD  # backwards-compatible alias for early docs/tests

_client = chromadb.PersistentClient(path=os.getenv("CHROMA_DIR", ".chroma"))
_collection = _client.get_or_create_collection(
    "docs",
    metadata={"hnsw:space": DISTANCE_METRIC},
)


def _read(path: str) -> str:
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _chunk(text: str) -> list[str]:
    text = " ".join(text.split())
    step = CHUNK_CHARS - CHUNK_OVERLAP
    return [text[i:i + CHUNK_CHARS] for i in range(0, max(1, len(text)), step) if text[i:i + CHUNK_CHARS].strip()]


def ingest(path: str, doc_id: Optional[str] = None) -> tuple[str, int]:
    """Chunk a file and add it to Chroma. Returns (doc_id, n_chunks)."""
    doc_id = doc_id or os.path.basename(path)
    chunks = _chunk(_read(path))
    if not chunks:
        return doc_id, 0
    _collection.add(
        ids=[f"{doc_id}::{i}" for i in range(len(chunks))],
        documents=chunks,
        metadatas=[{"doc_id": doc_id, "chunk_id": i} for i in range(len(chunks))],
    )
    return doc_id, len(chunks)


def search(query: str, k: int = 4) -> tuple[list[Citation], bool, Optional[str]]:
    """Retrieve top-k passages with citations. Returns (passages, refused, reason)."""
    res = _collection.query(query_texts=[query], n_results=k)
    docs = res["documents"][0] if res["documents"] else []
    metas = res["metadatas"][0] if res["metadatas"] else []
    dists = res["distances"][0] if res["distances"] else []
    passages = [
        Citation(
            doc_id=m["doc_id"],
            chunk_id=m["chunk_id"],
            citation=f"{m['doc_id']}::{m['chunk_id']}",
            text=d,
            distance=dist,
            score=1.0 - dist,
            score_kind=SCORE_KIND,
        )
        for d, m, dist in zip(docs, metas, dists)
    ]
    if not passages or passages[0].score < REFUSE_SCORE_THRESHOLD:
        return passages, True, "No sufficiently relevant passage found — refusing rather than guessing."
    return passages, False, None


def list_documents() -> list[dict]:
    """Distinct doc_ids and their chunk counts."""
    got = _collection.get(include=["metadatas"])
    counts: dict[str, int] = {}
    for m in got["metadatas"]:
        counts[m["doc_id"]] = counts.get(m["doc_id"], 0) + 1
    return [{"doc_id": d, "n_chunks": n} for d, n in sorted(counts.items())]
