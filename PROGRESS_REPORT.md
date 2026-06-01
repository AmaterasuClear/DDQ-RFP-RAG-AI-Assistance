# Project Progress Report: DDQ-RAG

**Due Diligence Retrieval-Augmented Generation Assistant**

**AI Agent Utilized: Claude Code**

---

## a. Technical Architecture & Tools

### System Architecture

The DDQ-RAG system implements a disciplined RAG pipeline with strict trust guarantees:

```
Document Source
      │
      ▼
┌──────────────┐
│    Ingest    │  PDF, DOCX, TXT, MD → clean text
│              │  Smart chunking: 400 chars, 60 overlap
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   BGE-M3     │  Offline embedding via sentence-transformers
│   Embed      │  Generates 1024-dim dense vectors
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   SQLite     │  Stores chunks + embeddings + metadata
│   Store      │  Documents, chunks, review queue, audit log
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Retrieve    │  Cosine similarity + lexical calibration
│              │  Top-7 candidates → similarity gate (≥ 0.40)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Gate       │  No chunks pass threshold?
│              │  → Refuse answer → enqueue human review
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Generate    │  Gemini API / OpenRouter / Local extractive
│              │  LLM citations STRIPPED, replaced with
└──────┬───────┘  backend retrieval metadata
       │
       ▼
┌──────────────┐
│ Confidence   │  HIGH (≥ 0.65) / MEDIUM / LOW
│  & Escalate  │  LOW → human review queue
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Output     │  JSON: answer + citations + confidence +
│              │  uncertainty_note + review_item_id
└──────────────┘
```

### Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.12 | Core implementation |
| **Embeddings** | BGE-M3 (sentence-transformers) | Offline, 1024-dim dense vectors |
| **Storage** | SQLite (via sqlite3) | Chunks, documents, review queue |
| **LLM Generation** | Gemini API / OpenRouter | Optional generative answers |
| **Local Fallback** | Extractive (TF-IDF + similarity) | No LLM key required |
| **REST API** | FastAPI + uvicorn | HTTP server interface |
| **Desktop GUI** | PySide6 (Qt) | Native desktop workbench |
| **Excel Processing** | openpyxl | Batch questionnaire I/O |
| **Packaging** | PyInstaller | Standalone EXE build |
| **Data Validation** | Pydantic | Schema enforcement |
| **Testing** | pytest | Integration tests |

### Interface Options

1. **CLI** — `python -m app.main {ingest|ask|pipeline|...}`
2. **REST API** — FastAPI server with 6 endpoints
3. **Desktop Workbench** — PySide6 GUI with 4 functional tabs
4. **Batch Processor** — `process_qa.py` for bulk Excel questionnaires

---

## b. Key Challenges Encountered

### 1. Offline Model Deployment

**Challenge:** The requirement for offline-first operation meant we couldn't rely on any embedding API. BGE-M3 needed to run fully locally, which introduced:
- Large model size (~2.2 GB) requiring explicit download instructions
- Significant RAM usage (8 GB+ for model loading)
- Slow initial load time (~30 seconds for model initialization)

**Resolution:** Implemented singleton model loading (load once at module import), enforced `HF_HUB_OFFLINE=1` to prevent accidental network calls, and documented the manual download process clearly.

### 2. Citation Trustworthiness

**Challenge:** Early tests showed LLMs would hallucinate citations — generating plausible-looking document references that didn't exist in the source material. This is unacceptable for institutional due diligence.

**Resolution:** Implemented a strict **backend-generated citation** system. The LLM's output citations are stripped before the response is assembled. All citations come exclusively from retrieval metadata (doc_name, page number, similarity score) injected at the application layer after generation.

### 3. The Similarity Threshold Trade-off

**Challenge:** Choosing the right similarity threshold was difficult:
- Too high (0.70): frequent false rejections, many valid queries sent to review
- Too low (0.30): low-quality answers with irrelevant source material

**Resolution:** Calibrated at **0.40** through empirical testing, combined with a confidence scoring system (HIGH ≥ 0.65, MEDIUM, LOW) that triages responses rather than a single pass/fail.

### 4. Cross-Platform Path Handling

**Challenge:** The application targets Windows primarily but needed to work across platforms. Windows path separators, temp directories, and encoding issues caused bugs during development.

**Resolution:** Used `pathlib.Path` throughout for cross-platform path handling, with clear separation between project root detection (PyInstaller `_MEIPASS` vs. development mode).

### 5. LLM Provider Integration Variance

**Challenge:** The Google Gemini API and OpenRouter have different response schemas, error formats, and rate limits. A unified abstraction was needed.

**Resolution:** Built a provider abstraction layer in `generate.py` that normalizes all LLM responses into a uniform schema, with per-provider error handling and fallback to local extractive mode.

---

## c. Iteration Progress & Improvements

### Phase 1: Proof of Concept (MVP)
- **Initial state:** Basic PDF ingestion with hardcoded chunking (500 chars, 50 overlap)
- **Retrieval:** Simple cosine similarity, top-5 results, 0.70 threshold
- **Generation:** OpenAI API dependency, no offline fallback
- **UI:** Streamlit web app only
- **Citations:** LLM-generated (unreliable)

### Phase 2: Trust Architecture
- Replaced OpenAI with Gemini API + local extractive fallback
- Implemented **backend citation injection** — stripped LLM citations, replaced with retrieval metadata
- Added **gate-before-generate** — refusal when no chunks pass threshold
- Added confidence scoring (HIGH/MEDIUM/LOW) with **human review escalation**
- Created SQLite review queue with status tracking

### Phase 3: BGE-M3 Migration
- Migrated from OpenAI embeddings to **fully offline BGE-M3**
- Added singleton model loading pattern to prevent reloads
- Implemented `HF_HUB_OFFLINE=1` enforcement
- Tuned chunk size to 400 chars with 60 overlap for BGE-M3's 512-token limit

### Phase 4: Enhanced Interfaces
- Replaced Streamlit with **PySide6 desktop GUI** — native Windows feel
- Added batch Excel questionnaire processing (`excel_processor.py`)
- Built FastAPI server with file upload endpoints
- Created PyInstaller build pipeline for standalone distribution
- Added document library management with stale detection

### Phase 5: Refinement & Cleanup
- Improved smart chunking with sentence boundary detection
- Added retrieval precision evaluation module
- Implemented similarity + query coverage reranking
- Cleaned project structure (removed debug artifacts, deduplicated files)
- Finalized documentation and progress reporting

---

## d. Lessons Learned

### 1. Offline-First Constrains Everything — and That's Good

Designing for offline operation forced better engineering decisions: no API dependency risks, no data leaving the machine, consistent performance regardless of network conditions. The BGE-M3 model, while large, provides embeddings that are predictable and auditable.

### 2. LLMs Cannot Be Trusted for Citations

This was the most important technical lesson. Even when instructed to "only cite from the provided context," LLMs frequently fabricated document names, page numbers, and quotes. The only reliable approach is to **decouple retrieval evidence from generative output** — use the LLM for answer composition, but source all citations from the deterministic retrieval layer.

### 3. Thresholds Are Context-Dependent

A similarity threshold that works for technical documents fails for nuanced policy questions. Rather than a single threshold, a **confidence band** (HIGH/MEDIUM/LOW) with graduated responses and human escalation handles the gray area far better.

### 4. Chunk Size Tuning Matters

The initial 500-char chunks caused BGE-M3 embedding degradation (the model's optimal input is ~512 tokens, not characters). Reducing to 400 characters with 60-character overlap improved retrieval quality significantly while maintaining document coverage.

### 5. Desktop GUI Over Web App

For institutional due diligence, the desktop app (PySide6) was preferred over the web interface (Streamlit) because:
- No server setup or port configuration
- Works fully offline
- Feels more secure (no browser exposure)
- Better file-system integration for document selection

---

## e. Future Opportunities

### 1. Hybrid Search Enhancement
**Current:** BGE-M3 dense vectors only.

**Opportunity:** Add BM25 sparse retrieval alongside dense vectors for hybrid search, improving recall for exact-match queries (policy numbers, clause references).

### 2. Multi-Document Cross-Referencing
**Current:** Answers are generated per-document or per-folder.

**Opportunity:** Implement cross-document synthesis — when no single document fully answers a question, the system could identify complementary sources and synthesize a composite answer with provenance from each.

### 3. Incremental Ingestion & Change Tracking
**Current:** Full re-ingestion on document update.

**Opportunity:** Implement delta ingestion — detect which chunks changed in a revised document and only re-embed those, preserving existing embeddings for unchanged content.

### 4. Interactive Review Dashboard
**Current:** CLI-based review queue with status flags.

**Opportunity:** A dedicated review UI within the desktop app showing: pending items, answer drafts, source document preview, and one-click approve/reject/edit workflow.

### 5. Export & Audit Trails
**Current:** Basic audit logging to SQLite.

**Opportunity:** Generate formal audit reports (PDF) for each question-answer session, including: the exact chunks retrieved, similarity scores, LLM prompt, raw and post-processed output, confidence calculation, and review disposition.

### 6. Model Fine-Tuning for Domain Language
**Opportunity:** Fine-tune BGE-M3 on institutional due diligence documents to improve embedding quality for domain-specific terminology (e.g., "custody," "sub-custodian," "liquidity coverage ratio").

### 7. Continuous Benchmarking
**Opportunity:** Build an evaluation harness with a curated set of due diligence questions and golden answers. Track retrieval precision, answer accuracy, and confidence calibration across every pipeline change.

### 8. Human-in-the-Loop Knowledge Reinforcement
**Current:** When a Compliance Officer rejects or edits an AI-generated answer during review, the correction is recorded as a status flag only — the improved answer is not fed back into the knowledge base.

**Opportunity:** Implement a closed-loop learning mechanism within the review workflow. When a reviewer submits a final approved answer that differs from the AI draft, the system automatically indexes the question-answer pair as a high-authority knowledge chunk, tagged with reviewer identity, approval timestamp, and source question context. Subsequent similar queries will preferentially retrieve these human-validated answers over raw policy document chunks, progressively improving response accuracy for recurring due diligence topics without requiring model retraining. Over time, this transforms the review workflow from a quality gate into an active knowledge accumulation engine.

---

## Appendix: Tools & Versions

| Tool | Version | Role |
|------|---------|------|
| Python | 3.12.1 | Runtime |
| BGE-M3 | via sentence-transformers | Embeddings |
| FastAPI | latest | REST framework |
| PySide6 | latest | Desktop GUI |
| SQLite | built-in | Storage |
| openpyxl | latest | Excel processing |
| Pydantic | v2 | Data validation |
| PyInstaller | latest | EXE packaging |
| pytest | latest | Test runner |
