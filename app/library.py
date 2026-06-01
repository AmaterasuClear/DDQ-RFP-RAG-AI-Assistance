from __future__ import annotations

from pathlib import Path

from app.config import RAG_MATERIAL_DIR, SUPPORTED_DOCUMENT_EXTENSIONS
from app.ingest import file_sha256, ingest_document
from app.storage import delete_document, list_documents


def discover_documents(folder: str | Path = RAG_MATERIAL_DIR) -> list[Path]:
    root = Path(folder)
    if not root.exists():
        return []
    return sorted(
        [
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS
        ],
        key=lambda path: path.name.lower(),
    )


def index_document(
    path: str | Path,
    doc_type: str = "internal_policy",
    version: str = "current",
    db_path: str | Path | None = None,
    copy_to_uploads: bool = False,
) -> int:
    chunks = ingest_document(
        path,
        doc_type=doc_type,
        version=version,
        db_path=db_path,
        copy_to_uploads=copy_to_uploads,
        replace_existing=True,
    )
    return len(chunks)


def index_folder(
    folder: str | Path = RAG_MATERIAL_DIR,
    doc_type: str = "internal_policy",
    version: str = "current",
    db_path: str | Path | None = None,
) -> dict[str, int]:
    results: dict[str, int] = {}
    for path in discover_documents(folder):
        results[path.name] = index_document(
            path,
            doc_type=doc_type,
            version=version,
            db_path=db_path,
            copy_to_uploads=False,
        )
    return results


def delete_library_document(
    doc_name: str,
    db_path: str | Path | None = None,
) -> None:
    delete_document(doc_name, db_path)


def stale_documents(db_path: str | Path | None = None) -> list[dict[str, str]]:
    stale: list[dict[str, str]] = []
    for row in list_documents(db_path):
        source_path = Path(row.get("source_path", ""))
        if not source_path.exists():
            stale.append({**row, "stale_reason": "source missing"})
            continue
        current_hash = file_sha256(source_path)
        if row.get("content_hash") and current_hash != row["content_hash"]:
            stale.append({**row, "stale_reason": "source changed"})
    return stale
