# DDQ-RAG: Trustworthy Due Diligence RAG Assistant

A **local-first, source-grounded** Retrieval-Augmented Generation system for institutional due diligence document analysis. Built for accuracy, verifiability, and offline operation.

**Presentation:** [`GroupF_DDQ-RAG_Presentation.pptx`](GroupF_DDQ-RAG_Presentation.pptx)

**Important Instructions:**

Please run `DDQ-AI-Assistant.py` in the `desktop` folder to launch the application interface.

The desktop application is not included in this repository due to size limitations.

---

## Architecture Overview

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Ingest  │ →  │  Embed   │ →  │  Store   │ →  │Retrieve  │ →  │  Filter  │ →  │ Generate │ →  │   Cite   │
│ PDF/DOCX │    │ BGE-M3   │    │ SQLite   │    │ cosine+  │    │ threshold│    │ LLM  /   │    │backend-  │
│ TXT/MD   │    │ offline  │    │ chunks   │    │ lexical  │    │ ≥ 0.40   │    │extractive│    │injected  │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
```

**Key design principles:**
- **Gate before generate** — answers are only produced when retrieval confidence exceeds threshold
- **Citations always backend-generated** — LLM output citations are stripped and replaced with retrieval metadata for accuracy
- **Offline-first** — embedding runs entirely locally via BGE-M3; LLM generation is optional
- **Review escalation** — low-confidence or unanswerable queries are queued for human review

---

## Getting Started

### Prerequisites

- Python 3.10+
- 8GB+ RAM (for BGE-M3 model)
- Windows, macOS, or Linux

### 1. Clone & Install

```powershell
git clone <repo-url>
cd DDQ-RAG
python -m pip install -r requirements.txt
```

### 2. Download BGE-M3 Model (Required)

The system uses BGE-M3 for offline embeddings. You need to download the model once:

**Option A — Hugging Face (recommended):**

```powershell
# Install huggingface-hub if missing
pip install huggingface-hub

# Download BGE-M3 model to a local directory
huggingface-cli download BAAI/bge-m3 --local-dir C:\models\bge-m3
```

**Option B — Manual download:**

1. Go to [BAAI/bge-m3 on Hugging Face](https://huggingface.co/BAAI/bge-m3)
2. Download all files into a local folder (e.g., `C:\models\bge-m3`)
3. You need at minimum: `config.json`, `tokenizer.json`, `tokenizer_config.json`, and model weights files

### 3. Configure Environment

Copy `.env.example` to `.env` and edit:

```ini
# Required: path to your local BGE-M3 model directory
BGE_MODEL_PATH=C:\models\bge-m3

# Optional: LLM provider for generative answers (comment out for local-only)
DDQ_GENERATION_PROVIDER=gemini
# GEMINI_API_KEY=your_key_here
# GEMINI_MODEL=gemini-2.0-flash

# Optional: alternative LLM provider
# OPENROUTER_API_KEY=your_key_here
# OPENROUTER_LLM_MODEL=anthropic/claude-sonnet-4-20250514

# Optional: custom database path
# DDQ_DB_PATH=data/ddqrag.sqlite3
```

### 4. Ingest Documents

```powershell
# Index a folder of documents
python -m app.main index-folder ./RAGmaterial

# Or ingest individual files
python -m app.main ingest "path/to/policy.pdf" --doc-type security_policy --version v1
```

### 5. Ask Questions

```powershell
# Single question
python -m app.main ask "Do you encrypt data at rest?"

# Pipeline: ingest then ask
python -m app.main pipeline "policy.pdf" "What is the incident response policy?"
```

---

## Usage

### CLI

| Command | Description |
|---------|-------------|
| `python -m app.main ingest <file>` | Ingest a single document |
| `python -m app.main ask <question>` | Ask a question |
| `python -m app.main pipeline <file> <question>` | Ingest + ask in one step |
| `python -m app.main index-folder <dir>` | Index all documents in a folder |
| `python -m app.main documents` | List ingested documents |
| `python -m app.main review` | View human review queue |

### API Server

```powershell
uvicorn app.api:create_app --factory --reload
```

Endpoints:
- `GET /health` — Health check
- `POST /ingest` — Upload and ingest a document
- `POST /ask` — Ask a question
- `POST /process-excel` — Batch process an Excel questionnaire
- `GET /review-queue` — View review queue
- `POST /review-queue/{id}/{status}` — Update review status

### Desktop Workbench

```powershell
python -m desktop.client
```

A PySide6 GUI with tabs for:
- **Ask** — Question-answering interface with source citations
- **Excel** — Batch questionnaire processing
- **Library** — Document library management
- **Review** — Human review queue

### Build Standalone Executable

```powershell
./build_desktop.ps1
# Produces: dist/DDQ-RAG-Workbench/DDQ-RAG-Workbench.exe
```

---

## Configuration Reference

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `BGE_MODEL_PATH` | — | **Required.** Absolute path to BGE-M3 model directory |
| `DDQ_GENERATION_PROVIDER` | `local` | `gemini`, `openrouter`, or `local` |
| `DDQ_EMBEDDING_PROVIDER` | `local` | Only `local` (BGE-M3) is supported |
| `DDQ_DB_PATH` | `data/ddqrag.sqlite3` | SQLite database path |
| `GEMINI_API_KEY` | — | Gemini API key |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model name |
| `OPENROUTER_API_KEY` | — | OpenRouter API key |
| `OPENROUTER_LLM_MODEL` | — | OpenRouter model identifier |

### Pipeline Configuration (app/config.py)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Chunk size | 400 chars | Document chunk size for embedding |
| Chunk overlap | 60 chars | Overlap between consecutive chunks |
| Top-K | 7 | Number of chunks retrieved per query |
| Similarity threshold | 0.40 | Minimum similarity gate for generation |
| High confidence | ≥ 0.65 | Threshold for HIGH confidence level |

---

## Project Structure

```
DDQ-RAG/
├── app/                    # Core RAG pipeline
│   ├── main.py             # CLI entry point & orchestrator
│   ├── config.py           # Configuration & env loading
│   ├── ingest.py           # Document parsing & chunking
│   ├── embeddings.py       # BGE-M3 embedding & similarity
│   ├── storage.py          # SQLite operations
│   ├── retrieve.py         # Retrieval with calibration
│   ├── generate.py         # Answer generation (LLM/extractive)
│   ├── confidence.py       # Confidence scoring & escalation
│   ├── library.py          # Document library management
│   ├── excel_processor.py  # Batch Excel processing
│   ├── api.py              # FastAPI server
│   ├── prompts.py          # LLM system prompts
│   ├── schemas.py          # Pydantic data models
│   ├── text_cleaner.py     # Text extraction cleanup
│   ├── metadata_filters.py # Document metadata filtering
│   ├── ranking.py          # Retrieval scoring helpers
│   └── reranker.py         # Similarity + coverage reranking
├── desktop/
│   └── client.py           # PySide6 desktop GUI
├── data/                   # Runtime data (SQLite DB)
├── RAGmaterial/            # Sample due diligence documents
├── tests/
│   └── test_pipeline.py    # Integration tests
├── .env.example            # Environment template
├── requirements.txt        # Python dependencies
├── build_desktop.ps1       # PyInstaller build script
├── CLAUDE.md               # AI assistant instructions
└── PROGRESS_REPORT.md      # Project progress report
```

---

## Testing

```powershell
python -m pytest
```

---

## License

This project is developed for educational / institutional due diligence purposes.
