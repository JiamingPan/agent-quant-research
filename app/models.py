"""Pydantic request/response models — the API surface contract."""
from __future__ import annotations
from typing import Optional

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    path: str = Field(..., description="Server-side path to a PDF/MD/TXT file to add to Chroma.")
    doc_id: Optional[str] = Field(None, description="Optional explicit id; defaults to the filename.")


class IngestResponse(BaseModel):
    doc_id: str
    n_chunks: int


class Citation(BaseModel):
    doc_id: str = Field(..., description="Document identifier assigned at ingest time.")
    chunk_id: int = Field(..., description="Zero-based chunk index inside the document.")
    citation: str = Field(..., description="Stable source pointer in the form doc_id::chunk_id.")
    text: str = Field(..., description="Retrieved source passage text.")
    distance: float = Field(..., description="Raw Chroma distance; lower means closer.")
    score: float = Field(..., description="Converted score: 1.0 - distance; higher means closer.")
    score_kind: str = Field("cosine_similarity", description="Meaning of score for this collection.")


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
    event_date: str = Field(
        ...,
        description=(
            "YYYY-MM-DD or a timezone-aware ISO timestamp. Timestamps at/after "
            "16:00 America/New_York align to the next observed trading day."
        ),
    )
    window: int = Field(5, ge=0, description="Trading observations on each side.")


class DocumentInfo(BaseModel):
    doc_id: str
    n_chunks: int
