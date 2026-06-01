from __future__ import annotations

from pathlib import Path

from app.config import SIMILARITY_THRESHOLD, TOP_K
from app.embeddings import calibrated_similarity, generate_embedding, query_coverage
from app.metadata_filters import apply_metadata_filters
from app.schemas import RetrievedChunk
from app.storage import audit_event, list_chunks


def retrieve_chunks(
    question: str,
    top_k: int = TOP_K,
    db_path: str | Path | None = None,
    doc_type: str | None = None,
    version: str | None = None,
    doc_name: str | None = None,
) -> tuple[list[RetrievedChunk], dict]:
    query_embedding = generate_embedding(question)
    all_chunks = list_chunks(db_path)
    filtered = apply_metadata_filters(
        all_chunks,
        doc_type=doc_type,
        version=version,
        doc_name=doc_name,
    )

    scored: list[RetrievedChunk] = []
    for chunk in filtered:
        text_similarity = calibrated_similarity(
            question,
            chunk.text,
            query_embedding,
            chunk.embedding,
        )
        title_similarity = query_coverage(question, chunk.doc_name)
        if title_similarity >= 0.5 and text_similarity >= 0.5:
            text_similarity = max(text_similarity, 0.72)
        elif title_similarity > 0:
            text_similarity = max(text_similarity, min(0.69, text_similarity + 0.08))
        scored.append(
            RetrievedChunk(
                **chunk.model_dump(),
                similarity=round(text_similarity, 4),
            )
        )
    ranked = sorted(scored, key=lambda chunk: chunk.similarity, reverse=True)[:top_k]
    audit_event(
        "chunks_retrieved",
        {
            "question": question,
            "top_k": top_k,
            "results": [
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_name": chunk.doc_name,
                    "page": chunk.page,
                    "similarity": chunk.similarity,
                }
                for chunk in ranked
            ],
        },
        db_path,
    )

    summary = {
        "total_indexed_chunks": len(all_chunks),
        "filtered_chunks": len(filtered),
        "scored_chunks": len(scored),
        "returned_chunks": len(ranked),
        "top_score": ranked[0].similarity if ranked else 0.0,
    }
    return ranked, summary


def filter_low_similarity(
    chunks: list[RetrievedChunk],
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[RetrievedChunk]:
    return [chunk for chunk in chunks if chunk.similarity >= threshold]
