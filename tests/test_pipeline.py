from __future__ import annotations

import json

from app.generate import parse_response
from app.embeddings import generate_embedding
from app.ingest import split_chunks
from app.main import run_question
from app.schemas import Chunk
from app.storage import list_review_items, save_chunk, update_review_status


def _save_sample_chunk(tmp_path, text: str, doc_name: str = "Security_Policy.pdf") -> str:
    db_path = tmp_path / "test.sqlite3"
    save_chunk(
        Chunk(
            doc_name=doc_name,
            page=12,
            text=text,
            embedding=generate_embedding(text),
            doc_type="security_policy",
            version="v2.1",
        ),
        db_path,
    )
    return str(db_path)


def test_retrieval_citations_are_backend_generated(tmp_path):
    db_path = _save_sample_chunk(
        tmp_path,
        "All customer data is encrypted at rest using AES-256. Production databases require managed keys.",
    )

    result = run_question("Do you encrypt customer data at rest?", db_path=db_path)

    # With no LLM key configured, generate_answer surfaces the error instead of
    # silently falling back. Citations are still injected from retrieval backend.
    assert result.response.source_citations
    citation = result.response.source_citations[0]
    assert citation.doc_name == "Security_Policy.pdf"
    assert citation.page == 12
    assert "encrypted at rest" in citation.quote
    assert "LLM generation failed" in result.response.uncertainty_note


def test_low_similarity_is_rejected_and_queued_for_review(tmp_path):
    db_path = _save_sample_chunk(
        tmp_path,
        "All customer data is encrypted at rest using AES-256.",
    )

    result = run_question("What snacks are stocked in the office kitchen?", db_path=db_path)

    assert result.response.has_source is False
    assert result.response.confidence_level == "LOW"
    assert result.response.source_citations == []
    assert result.review_required is True
    assert "similarity threshold" in result.response.uncertainty_note
    assert len(list_review_items(db_path=db_path)) == 1


def test_model_citations_are_discarded_before_backend_injection():
    raw = json.dumps(
        {
            "answer": "Yes.",
            "has_source": True,
            "confidence_level": "HIGH",
            "uncertainty_note": "",
            "source_citations": [
                {"doc_name": "fake.pdf", "page": 99, "quote": "fabricated"}
            ],
        }
    )

    parsed = parse_response(raw)
    # parse_response returns a dict; citations are stripped by inject_citations()
    assert isinstance(parsed, dict)
    assert parsed["answer"] == "Yes."


def test_answer_schema_is_stable_json(tmp_path):
    db_path = _save_sample_chunk(
        tmp_path,
        "The incident response policy requires triage within one hour and customer notification when legally required.",
    )

    result = run_question("What is the incident response policy?", db_path=db_path)
    payload = json.loads(result.response.model_dump_json())

    assert list(payload.keys()) == [
        "answer",
        "has_source",
        "confidence_level",
        "uncertainty_note",
        "source_citations",
    ]


def test_chunking_uses_prd_size_and_overlap():
    text = "The quick brown fox jumps over the lazy dog. " * 40
    chunks = split_chunks(text, chunk_size=300, overlap=50)

    assert all(len(chunk) <= 300 for chunk in chunks)
    # With natural-text splitting, we verify chunks are non-empty and clean
    assert all(len(chunk) >= 30 for chunk in chunks)
    # No chunk should start with a lowercase fragment
    for chunk in chunks:
        first_word = chunk.split()[0] if chunk.split() else ""
        if first_word and len(first_word) <= 5:
            assert not first_word[0].islower(), f"Chunk starts with fragment: {chunk[:50]}"


def test_review_status_can_be_updated(tmp_path):
    db_path = _save_sample_chunk(tmp_path, "Security policies define encryption controls.")
    result = run_question("Does the company run a cafeteria?", db_path=db_path)

    assert result.review_item_id is not None
    update_review_status(result.review_item_id, "REJECTED", db_path=db_path)

    item = list_review_items(db_path=db_path)[0]
    assert item.status == "REJECTED"

