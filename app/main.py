"""
FastAPI surface. /ingest, /search, /documents, and /event-study are live.
/research (agent) lands on Day 4 — it returns 501 for now.
"""
from __future__ import annotations
from fastapi import FastAPI, HTTPException
from . import rag, tools
from .models import (
    IngestRequest, IngestResponse, SearchResponse, Citation,
    ResearchRequest, ResearchResponse, EventStudyRequest, DocumentInfo,
)

app = FastAPI(title="Agent Quant Research", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    try:
        doc_id, n = rag.ingest(req.path, req.doc_id)
    except FileNotFoundError:
        raise HTTPException(404, f"file not found: {req.path}")
    if n == 0:
        raise HTTPException(422, "no extractable text in file")
    return IngestResponse(doc_id=doc_id, n_chunks=n)


@app.get("/search", response_model=SearchResponse)
def search(q: str, k: int = 4):
    out = tools.search_docs(q, k=k)
    return SearchResponse(
        query=q,
        passages=[Citation(**p) for p in out["passages"]],
        refused=out["refused"],
        reason=out["reason"],
    )


@app.get("/documents", response_model=list[DocumentInfo])
def documents():
    return [DocumentInfo(**d) for d in rag.list_documents()]


@app.post("/research", response_model=ResearchResponse)
def research(req: ResearchRequest):
    raise HTTPException(501, "agent loop not implemented yet (Day 4)")


@app.post("/event-study")
def event_study(req: EventStudyRequest):
    return tools.run_event_study(req.ticker, req.event_date, req.window)
