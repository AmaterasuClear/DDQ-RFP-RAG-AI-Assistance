from __future__ import annotations

import os
import sys
from pathlib import Path


# ── Offline enforcement ──────────────────────────────────────────────────────────
# Must be set BEFORE any HF/sentence-transformers imports to block all network access.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


_MEIPASS = getattr(sys, "_MEIPASS", None)

if getattr(sys, "frozen", False) and _MEIPASS:
    PROJECT_ROOT = Path(_MEIPASS)
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"
UPLOADS_DIR = DOCS_DIR / "uploads"
RAG_MATERIAL_DIR = PROJECT_ROOT / "RAGmaterial"
DEFAULT_DB_PATH = Path(os.getenv("DDQ_DB_PATH", DATA_DIR / "ddqrag.sqlite3"))


def _load_dotenv() -> None:
    """Load .env file into os.environ. Does NOT override existing vars."""
    candidates = [
        PROJECT_ROOT / ".env",
        Path.cwd() / ".env",
    ]
    for env_path in candidates:
        if not env_path.exists():
            continue
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        break


_load_dotenv()

CHUNK_SIZE = 400
CHUNK_OVERLAP = 60
TOP_K = 7
SIMILARITY_THRESHOLD = 0.40
HIGH_CONFIDENCE_THRESHOLD = 0.65

GENERATION_PROVIDER = os.getenv("DDQ_GENERATION_PROVIDER", "gemini").lower()
EMBEDDING_PROVIDER = os.getenv("DDQ_EMBEDDING_PROVIDER", "local").lower()
SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def ensure_runtime_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
