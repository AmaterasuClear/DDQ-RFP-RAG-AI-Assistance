from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


ConfidenceLevel = Literal["HIGH", "MEDIUM", "LOW"]
ReviewStatus = Literal["PENDING", "APPROVED", "REJECTED"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_name: str
    page: int
    quote: str
    similarity: float = 0.0


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    doc_name: str
    page: int
    text: str
    embedding: list[float]
    doc_type: str = "general"
    version: str = "v1"
    upload_date: str = Field(default_factory=utc_now_iso)


class DocumentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_name: str
    source_path: str = ""
    file_type: str = ""
    doc_type: str = "general"
    version: str = "v1"
    chunk_count: int = 0
    content_hash: str = ""
    source_modified_at: str = ""
    indexed_at: str = Field(default_factory=utc_now_iso)


class RetrievedChunk(Chunk):
    similarity: float = Field(ge=0.0, le=1.0)


class AnswerResponse(BaseModel):
    """PRD-required machine-readable answer schema."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    has_source: bool
    confidence_level: ConfidenceLevel
    uncertainty_note: str = ""
    source_citations: list[Citation] = Field(default_factory=list)


class RetrievalSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_chunks_searched: int = 0
    chunks_passed_threshold: int = 0
    threshold: float = 0.0
    top_score: float = 0.0


class PipelineResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response: AnswerResponse
    review_required: bool
    review_reason: str = ""
    review_item_id: str | None = None
    top_similarity: float = 0.0
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    retrieval_summary: RetrievalSummary | None = None


class ReviewItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(default_factory=lambda: str(uuid4()))
    question: str
    answer_json: dict
    reason: str
    status: ReviewStatus = "PENDING"
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)


def to_json(model: BaseModel) -> str:
    return model.model_dump_json(indent=2)
