from __future__ import annotations

import math
import os
import re
import threading
from typing import Any

# Enforce offline mode — must be set before any HF-adjacent library touches the network
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Local BGE-M3 model — loaded once at module import
_BGE_MODEL: Any = None
_BGE_LOADED = False
_LOAD_LOCK = threading.Lock()
_MODEL_LOCK = threading.Lock()


def _load_bge_model():
    """Load BGE-M3 model from a local directory (fully offline, no HF Hub access)."""
    global _BGE_MODEL, _BGE_LOADED
    if _BGE_LOADED and _BGE_MODEL is not None:
        return

    with _LOAD_LOCK:
        if _BGE_LOADED and _BGE_MODEL is not None:
            return

        import sys
        st_module = sys.modules.get("sentence_transformers")
        if st_module is not None:
            existing = [obj for obj in vars(st_module).values()
                         if hasattr(obj, "encode") and type(obj).__name__ == "SentenceTransformer"]
            if existing:
                _BGE_MODEL = existing[0]
                _BGE_LOADED = True
                return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError(
                "sentence-transformers is required for local BGE-M3 embeddings. "
                "Install with: pip install sentence-transformers"
            )

        local_path = os.getenv("BGE_MODEL_PATH", "").strip()
        if not local_path:
            raise RuntimeError(
                "BGE_MODEL_PATH is not set. It must be an absolute path to a local "
                "BGE-M3 model directory (e.g. C:/models/bge-m3). HF Hub access is disabled."
            )
        if not os.path.isdir(local_path):
            raise RuntimeError(
                f"BGE_MODEL_PATH does not exist or is not a directory: {local_path}"
            )
        _BGE_MODEL = SentenceTransformer(local_path)
        _BGE_LOADED = True


def bge_m3_embedding(text: str) -> list[float]:
    """Generate semantic embedding using local BGE-M3 model."""
    _load_bge_model()
    with _MODEL_LOCK:
        embedding = _BGE_MODEL.encode(text, normalize_embeddings=True)
    return embedding.tolist()


EMBEDDING_DIMENSIONS = 1024
TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]*")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "company",
    "do", "does", "for", "from", "has", "have", "in", "is", "it", "of",
    "on", "or", "our", "policy", "the", "there", "to", "we", "what",
    "when", "where", "whether", "which", "who", "with", "you", "your",
}


def normalize_token(token: str) -> str:
    token = token.lower().strip("_-")
    if len(token) > 5 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    tokens = [normalize_token(match.group(0)) for match in TOKEN_RE.finditer(text)]
    return [token for token in tokens if token and token not in STOPWORDS]


def generate_embedding(text: str) -> list[float]:
    """Generate embedding using the configured EMBEDDING_PROVIDER."""
    from app.config import EMBEDDING_PROVIDER

    provider = EMBEDDING_PROVIDER.lower()
    if provider in ("bge", "local"):
        return bge_m3_embedding(text)
    raise RuntimeError(
        f"Unsupported embedding provider: {provider!r}. Supported: bge, local"
    )


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    length = min(len(left), len(right))
    dot = sum(left[i] * right[i] for i in range(length))
    left_norm = math.sqrt(sum(v * v for v in left[:length]))
    right_norm = math.sqrt(sum(v * v for v in right[:length]))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


def query_coverage(query: str, text: str) -> float:
    query_tokens = set(tokenize(query))
    text_tokens = set(tokenize(text))
    if not query_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


def calibrated_similarity(
    query: str,
    text: str,
    query_embedding: list[float],
    text_embedding: list[float],
) -> float:
    vector_score = cosine_similarity(query_embedding, text_embedding)
    coverage_score = query_coverage(query, text)
    query_tokens = tokenize(query)
    text_token_string = " ".join(tokenize(text))

    phrase_bonus = 0.0
    if len(query_tokens) >= 2:
        for first, second in zip(query_tokens, query_tokens[1:]):
            if f"{first} {second}" in text_token_string:
                phrase_bonus = 0.05
                break

    lexical_score = min(0.99, coverage_score + phrase_bonus)

    if vector_score >= 0.45:
        blended = vector_score + (lexical_score * 0.08)
    else:
        blended = max(vector_score, lexical_score * 0.5)
    return round(min(blended, 1.0), 4)