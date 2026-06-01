from __future__ import annotations

from dataclasses import dataclass

from app.retrieve import retrieve_chunks


@dataclass(frozen=True)
class RetrievalCase:
    question: str
    expected_doc_name: str


def evaluate_retrieval_precision(
    cases: list[RetrievalCase],
    db_path: str | None = None,
) -> dict[str, float]:
    if not cases:
        return {"precision_at_1": 0.0, "precision_at_5": 0.0}

    at_1 = 0
    at_5 = 0
    for case in cases:
        retrieved, _ = retrieve_chunks(case.question, db_path=db_path)
        doc_names = [chunk.doc_name for chunk in retrieved]
        if doc_names[:1] == [case.expected_doc_name]:
            at_1 += 1
        if case.expected_doc_name in doc_names[:5]:
            at_5 += 1

    return {
        "precision_at_1": round(at_1 / len(cases), 4),
        "precision_at_5": round(at_5 / len(cases), 4),
    }

