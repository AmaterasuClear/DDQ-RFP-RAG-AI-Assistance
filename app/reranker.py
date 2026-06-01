from __future__ import annotations

from app.embeddings import query_coverage
from app.schemas import RetrievedChunk


def rerank_chunks(question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    return sorted(
        chunks,
        key=lambda chunk: (chunk.similarity, query_coverage(question, chunk.text)),
        reverse=True,
    )

