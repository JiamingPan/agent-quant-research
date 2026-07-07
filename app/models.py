"""Pydantic request/response models — the API surface contract."""
from __future__ import annotations
from typing import Optional

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    path: str = Field(..., description="Path to a PDF/MD/TXT file to add to the knowledge base.")
    doc_id: Optional[str] = Field(None, description="Optional explicit id; defaults to the filename.")


class IngestResponse(BaseModel):
    doc_id: str
    n_chunks: int


class Citation(BaseModel):
    doc_id: str
    chunk_id: int
    text: str
    score: float  # similarity (higher = closer); we convert Chroma distance -> similarity


class SearchResponse(BaseModel):
    query: str
    passages: list[Citation]
    refused: bool = False
    reason: Optional[str] = None


class ResearchRequest(BaseModel):
    question: str


class ResearchResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    confidence: float
    refused: bool = False


class EventStudyRequest(BaseModel):
    ticker: str
    event_date: str  # YYYY-MM-DD
    window: int = 5   # +/- trading days around the event


class DocumentInfo(BaseModel):
    doc_id: str
    n_chunks: int
