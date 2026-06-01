# DDQ-RAG: Due Diligence RAG Assistant

Local-first, source-grounded RAG system for institutional due diligence document analysis.

## Architecture

RAG pipeline: **Ingest → Embed (BGE-M3) → Store (SQLite) → Retrieve → Filter → Generate → Cite**

- **Ingest**: PDF/DOCX/TXT/MD → clean text → smart chunk (400 chars, 60 overlap) → BGE-M3 embedding → SQLite
- **Retrieve**: Cosine + lexical calibrated similarity → top-7 → threshold gate (0.40) → LLM/extractive generation
- **Generate**: Gemini API, OpenRouter, or local extractive (no LLM needed)
- **Output**: Stable JSON schema with backend-injected citations, confidence levels, review escalation

## Key Files

| Path | Purpose |
|------|---------|
| `app/main.py` | CLI entry point & pipeline orchestrator |
| `app/config.py` | Config (chunk size, thresholds, providers, paths) |
| `app/ingest.py` | Document parsing & smart chunking |
| `app/embeddings.py` | BGE-M3 embedding (offline) & similarity scoring |
| `app/retrieve.py` | Retrieval with calibrated similarity |
| `app/generate.py` | Answer generation (LLM / extractive) |
| `app/storage.py` | SQLite operations (chunks, docs, review queue, audit) |
| `app/confidence.py` | Confidence scoring & escalation logic |
| `app/library.py` | Document library management (index folder, stale detection) |
| `app/excel_processor.py` | Batch Excel questionnaire processing |
| `app/api.py` | FastAPI server (ingest, ask, review, process-excel) |
| `app/prompts.py` | LLM system prompts |
| `app/schemas.py` | Pydantic models (Chunk, AnswerResponse, PipelineResult, etc.) |
| `app/text_cleaner.py` | PDF/DOCX extraction artifact cleanup |
| `app/metadata_filters.py` | Filter chunks by doc_type/version/doc_name |
| `app/ranking.py` | Retrieval scoring helpers (avg top-3, weighted confidence) |
| `app/reranker.py` | Rerank by similarity + query coverage |
| `app/evaluation.py` | Retrieval precision evaluation |
| `desktop/client.py` | PySide6 desktop GUI (ask, excel, library, review tabs) |
| `tests/test_pipeline.py` | Integration tests for full pipeline |

## Commands

```powershell
# Ingest documents
python -m app.main ingest path/to/doc.pdf --doc-type security_policy --version v2

# Ask a question
python -m app.main ask "Do you encrypt data at rest?"

# Ingest + ask (pipeline)
python -m app.main pipeline doc.pdf "What is the incident response policy?"

# Index a folder of documents
python -m app.main index-folder ./RAGmaterial

# List documents
python -m app.main documents

# Review queue
python -m app.main review

# Start API server
uvicorn app.api:create_app --factory --reload

# Start desktop app
python -m desktop.client

# Run tests
python -m pytest
```

## Configuration (.env)

- `BGE_MODEL_PATH` — absolute path to local BGE-M3 model directory (required, offline)
- `GEMINI_API_KEY` / `GEMINI_MODEL` — for Gemini generation
- `OPENROUTER_API_KEY` / `OPENROUTER_LLM_MODEL` — alternative LLM provider
- `DDQ_GENERATION_PROVIDER` — "gemini", "openrouter", or "local" (default: local extractive)
- `DDQ_EMBEDDING_PROVIDER` — "local" only (BGE-M3)
- `DDQ_DB_PATH` — SQLite database path

## Key Config Values (app/config.py)

- Chunk size: 400, overlap: 60
- Top-K: 7
- Similarity threshold: 0.40
- High confidence threshold: 0.65

## Architecture Rules

- **Offline-first**: All HF Hub access blocked (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`)
- **BGE-M3 loaded once** at module import in `app/main.py` and `app/embeddings.py`
- **Citations always backend-generated**: LLM output citations are stripped, replaced with retrieval metadata in `generate.inject_citations()`
- **Gate before generate**: If no chunks pass 0.40 threshold, returns refusal + enqueues for human review
- **Review escalation**: LOW confidence, no source, no retrieval, or malformed JSON → human review queue

## Desktop Build

```powershell
./build_desktop.ps1
# Produces dist/DDQ-RAG-Workbench/DDQ-RAG-Workbench.exe
```
