from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

# ── Enforce offline mode before any HF-adjacent imports ──────────────────────────
import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ── Load .env early so BGE_MODEL_PATH is available ───────────────────────────────
import sys as _sys
from pathlib import Path as _Path
_meipass = getattr(_sys, "_MEIPASS", None)
_env_candidates: list[_Path] = []
if _meipass:
    _env_candidates.append(_Path(_meipass) / ".env")
else:
    _env_candidates.append(_Path(__file__).resolve().parents[1] / ".env")
_env_candidates.append(_Path.cwd() / ".env")
for _env_path in _env_candidates:
    if _env_path.exists():
        with open(_env_path, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line or _line.startswith("#") or "=" not in _line:
                    continue
                _key, _, _value = _line.partition("=")
                _key = _key.strip()
                _value = _value.strip().strip('"').strip("'")
                if _key and _key not in _os.environ:
                    _os.environ[_key] = _value
        break

from app.confidence import calculate_confidence, review_reason, should_escalate
from app.config import SIMILARITY_THRESHOLD, TOP_K, ensure_runtime_dirs
from app.generate import generate_answer
from app.ingest import ingest_document
from app.library import index_folder
from app.retrieve import filter_low_similarity, retrieve_chunks
from app.schemas import AnswerResponse, PipelineResult, to_json
from app.storage import (
    audit_event,
    enqueue_review,
    init_db,
    list_documents,
    list_review_items,
)


def refusal_response(note: str) -> AnswerResponse:
    return AnswerResponse(
        answer=(
            "We are unable to provide a substantiated response to this question "
            "based on the documents currently indexed in our knowledge base. "
            "We recommend submitting additional relevant documentation for review."
        ),
        has_source=False,
        confidence_level="LOW",
        uncertainty_note=note,
        source_citations=[],
    )


def run_question(
    question: str,
    db_path: str | Path | None = None,
    top_k: int = TOP_K,
    threshold: float = SIMILARITY_THRESHOLD,
    doc_type: str | None = None,
    version: str | None = None,
    doc_name: str | None = None,
) -> PipelineResult:
    init_db(db_path)
    audit_event("question_received", {"question": question}, db_path)

    retrieved, retrieval_meta = retrieve_chunks(
        question,
        top_k=top_k,
        db_path=db_path,
        doc_type=doc_type,
        version=version,
        doc_name=doc_name,
    )
    top_similarity = retrieved[0].similarity if retrieved else 0.0
    gated_chunks = filter_low_similarity(retrieved, threshold=threshold)

    from app.schemas import RetrievalSummary
    retrieval_summary = RetrievalSummary(
        total_chunks_searched=retrieval_meta["total_indexed_chunks"],
        chunks_passed_threshold=len(gated_chunks),
        threshold=threshold,
        top_score=top_similarity,
    )

    if not retrieved or not gated_chunks:
        note = f"No retrieved chunks met the {threshold:.2f} similarity threshold."
        response = refusal_response(note)
        reason = review_reason(
            response.confidence_level,
            response.has_source,
            no_retrieval_results=not retrieved,
        )
        item = enqueue_review(question, response.model_dump(), reason, db_path)
        audit_event(
            "answer_rejected",
            {"question": question, "reason": reason, "top_similarity": top_similarity},
            db_path,
        )
        return PipelineResult(
            response=response,
            review_required=True,
            review_reason=reason,
            review_item_id=item.item_id,
            top_similarity=top_similarity,
            retrieved_chunks=retrieved,
            retrieval_summary=retrieval_summary,
        )

    malformed_json = False
    try:
        response = generate_answer(question, gated_chunks)
    except (ValidationError, ValueError, json.JSONDecodeError):
        malformed_json = True
        response = refusal_response("The model output could not be parsed as valid JSON.")

    confidence = calculate_confidence(top_similarity, response.has_source, threshold)
    response = AnswerResponse(
        answer=response.answer,
        has_source=response.has_source,
        confidence_level=confidence,
        uncertainty_note=response.uncertainty_note,
        source_citations=response.source_citations,
    )

    review_required = should_escalate(
        confidence,
        response.has_source,
        no_retrieval_results=False,
        malformed_json=malformed_json,
    )
    reason = review_reason(
        confidence,
        response.has_source,
        no_retrieval_results=False,
        malformed_json=malformed_json,
    )
    review_item_id = None
    if review_required:
        item = enqueue_review(question, response.model_dump(), reason, db_path)
        review_item_id = item.item_id

    audit_event(
        "answer_generated",
        {
            "question": question,
            "confidence_level": response.confidence_level,
            "has_source": response.has_source,
            "review_required": review_required,
            "top_similarity": top_similarity,
        },
        db_path,
    )
    return PipelineResult(
        response=response,
        review_required=review_required,
        review_reason=reason if review_required else "",
        review_item_id=review_item_id,
        top_similarity=top_similarity,
        retrieved_chunks=gated_chunks,
        retrieval_summary=retrieval_summary,
    )


def _cmd_ingest(args: argparse.Namespace) -> None:
    chunks = ingest_document(
        args.document,
        doc_type=args.doc_type,
        version=args.version,
        db_path=args.db,
        copy_to_uploads=not args.no_copy,
    )
    print(json.dumps({"ingested_chunks": len(chunks)}, indent=2))


def _cmd_ask(args: argparse.Namespace) -> None:
    result = run_question(
        args.question,
        db_path=args.db,
        doc_type=args.doc_type,
        version=args.version,
        doc_name=args.doc_name,
    )
    print(to_json(result.response))


def _cmd_pipeline(args: argparse.Namespace) -> None:
    ingest_document(
        args.document,
        doc_type=args.doc_type,
        version=args.version,
        db_path=args.db,
        copy_to_uploads=not args.no_copy,
    )
    result = run_question(args.question, db_path=args.db)
    print(to_json(result.response))


def _cmd_index_folder(args: argparse.Namespace) -> None:
    results = index_folder(
        args.folder,
        doc_type=args.doc_type,
        version=args.version,
        db_path=args.db,
    )
    print(json.dumps(results, indent=2))


def _cmd_documents(args: argparse.Namespace) -> None:
    print(json.dumps(list_documents(args.db), indent=2))


def _cmd_review(args: argparse.Namespace) -> None:
    items = [item.model_dump() for item in list_review_items(args.status, args.db)]
    print(json.dumps(items, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trustworthy Due Diligence RAG Assistant")
    parser.add_argument("--db", default=None, help="SQLite database path")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a document")
    ingest_parser.add_argument("document")
    ingest_parser.add_argument("--doc-type", default="general")
    ingest_parser.add_argument("--version", default="v1")
    ingest_parser.add_argument("--no-copy", action="store_true")
    ingest_parser.set_defaults(func=_cmd_ingest)

    ask_parser = subparsers.add_parser("ask", help="Ask a due diligence question")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--doc-type", default=None)
    ask_parser.add_argument("--version", default=None)
    ask_parser.add_argument("--doc-name", default=None)
    ask_parser.set_defaults(func=_cmd_ask)

    pipeline_parser = subparsers.add_parser("pipeline", help="Ingest a document and ask a question")
    pipeline_parser.add_argument("document")
    pipeline_parser.add_argument("question")
    pipeline_parser.add_argument("--doc-type", default="general")
    pipeline_parser.add_argument("--version", default="v1")
    pipeline_parser.add_argument("--no-copy", action="store_true")
    pipeline_parser.set_defaults(func=_cmd_pipeline)

    index_parser = subparsers.add_parser("index-folder", help="Index a folder into the RAG library")
    index_parser.add_argument("folder")
    index_parser.add_argument("--doc-type", default="internal_policy")
    index_parser.add_argument("--version", default="current")
    index_parser.set_defaults(func=_cmd_index_folder)

    docs_parser = subparsers.add_parser("documents", help="List ingested documents")
    docs_parser.set_defaults(func=_cmd_documents)

    review_parser = subparsers.add_parser("review", help="List review queue items")
    review_parser.add_argument("--status", default=None)
    review_parser.set_defaults(func=_cmd_review)
    return parser


def main() -> None:
    ensure_runtime_dirs()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
