from __future__ import annotations

from app.config import HIGH_CONFIDENCE_THRESHOLD, SIMILARITY_THRESHOLD
from app.schemas import ConfidenceLevel


def calculate_confidence(
    top_similarity: float,
    has_source: bool,
    threshold: float = SIMILARITY_THRESHOLD,
) -> ConfidenceLevel:
    if top_similarity < threshold:
        return "LOW"
    if not has_source:
        return "LOW"
    if top_similarity >= HIGH_CONFIDENCE_THRESHOLD:
        return "HIGH"
    return "MEDIUM"


def should_escalate(
    confidence: ConfidenceLevel,
    has_source: bool,
    no_retrieval_results: bool = False,
    malformed_json: bool = False,
) -> bool:
    return (
        confidence == "LOW"
        or not has_source
        or no_retrieval_results
        or malformed_json
    )


def review_reason(
    confidence: ConfidenceLevel,
    has_source: bool,
    no_retrieval_results: bool = False,
    malformed_json: bool = False,
) -> str:
    reasons: list[str] = []
    if no_retrieval_results:
        reasons.append("no retrieval results")
    if malformed_json:
        reasons.append("malformed JSON")
    if confidence == "LOW":
        reasons.append("confidence is LOW")
    if not has_source:
        reasons.append("has_source is false")
    return ", ".join(dict.fromkeys(reasons)) or "manual review required"

