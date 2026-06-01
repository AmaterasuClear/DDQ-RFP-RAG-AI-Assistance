from __future__ import annotations

from statistics import mean

from app.schemas import RetrievedChunk


def avg_top3_similarity(chunks: list[RetrievedChunk]) -> float:
    if not chunks:
        return 0.0
    top_scores = [chunk.similarity for chunk in chunks[:3]]
    return round(mean(top_scores), 4)


def weighted_confidence_score(chunks: list[RetrievedChunk]) -> float:
    if not chunks:
        return 0.0
    weights = [1.0, 0.7, 0.5, 0.3, 0.2]
    total_weight = sum(weights[: len(chunks)])
    score = sum(chunk.similarity * weights[index] for index, chunk in enumerate(chunks))
    return round(score / total_weight, 4)

