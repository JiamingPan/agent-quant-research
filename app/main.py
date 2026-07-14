"""FastAPI surface for ingestion, retrieval, agent research, and event studies."""
from __future__ import annotations
from fastapi import FastAPI, HTTPException
from . import agent, rag, tools
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
    try:
        out = agent.run_agent(req.question)
    except agent.AgentConfigurationError as exc:
        raise HTTPException(503, str(exc)) from exc
    return ResearchResponse(
        question=req.question,
        answer=out["answer"],
        citations=[Citation(**item) for item in out["citations"]],
        confidence=out["confidence"],
        refused=out["refused"],
    )


@app.post("/event-study")
def event_study(req: EventStudyRequest):
    try:
        return tools.run_event_study(req.ticker, req.event_date, req.window)
    except tools.EventInputError as exc:
        raise HTTPException(422, str(exc)) from exc
