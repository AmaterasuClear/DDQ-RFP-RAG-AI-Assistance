from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from app.config import DEFAULT_DB_PATH, ensure_runtime_dirs
from app.schemas import Chunk, DocumentRecord, ReviewItem, utc_now_iso


def _resolve_db_path(db_path: str | Path | None = None) -> Path:
    path = Path(db_path) if db_path else Path(DEFAULT_DB_PATH)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    ensure_runtime_dirs()
    path = _resolve_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path | None = None) -> None:
    with connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_name TEXT NOT NULL,
                page INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding TEXT NOT NULL,
                doc_type TEXT NOT NULL,
                version TEXT NOT NULL,
                upload_date TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_name TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                file_type TEXT NOT NULL,
                doc_type TEXT NOT NULL,
                version TEXT NOT NULL,
                chunk_count INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                source_modified_at TEXT NOT NULL,
                indexed_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS review_queue (
                item_id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                answer_json TEXT NOT NULL,
                reason TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save_document_record(
    record: DocumentRecord,
    db_path: str | Path | None = None,
) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO documents (
                doc_name, source_path, file_type, doc_type, version, chunk_count,
                content_hash, source_modified_at, indexed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.doc_name,
                record.source_path,
                record.file_type,
                record.doc_type,
                record.version,
                record.chunk_count,
                record.content_hash,
                record.source_modified_at,
                record.indexed_at,
            ),
        )
        conn.commit()


def save_chunk(chunk: Chunk, db_path: str | Path | None = None) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO chunks (
                chunk_id, doc_name, page, text, embedding, doc_type, version, upload_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk.chunk_id,
                chunk.doc_name,
                chunk.page,
                chunk.text,
                json.dumps(chunk.embedding),
                chunk.doc_type,
                chunk.version,
                chunk.upload_date,
            ),
        )
        conn.commit()


def save_chunks(chunks: list[Chunk], db_path: str | Path | None = None) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO chunks (
                chunk_id, doc_name, page, text, embedding, doc_type, version, upload_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    chunk.chunk_id,
                    chunk.doc_name,
                    chunk.page,
                    chunk.text,
                    json.dumps(chunk.embedding),
                    chunk.doc_type,
                    chunk.version,
                    chunk.upload_date,
                )
                for chunk in chunks
            ],
        )
        conn.commit()


def delete_document(doc_name: str, db_path: str | Path | None = None) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        conn.execute("DELETE FROM chunks WHERE doc_name = ?", (doc_name,))
        conn.execute("DELETE FROM documents WHERE doc_name = ?", (doc_name,))
        conn.commit()
    audit_event("document_deleted", {"doc_name": doc_name}, db_path)


def list_chunks(
    db_path: str | Path | None = None,
    doc_name: str | None = None,
) -> list[Chunk]:
    init_db(db_path)
    params: tuple[Any, ...] = ()
    where = ""
    if doc_name:
        where = "WHERE doc_name = ?"
        params = (doc_name,)
    with connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT chunk_id, doc_name, page, text, embedding, doc_type, version, upload_date
            FROM chunks
            {where}
            ORDER BY upload_date DESC, doc_name ASC, page ASC
            """,
            params,
        ).fetchall()

    return [
        Chunk(
            chunk_id=row["chunk_id"],
            doc_name=row["doc_name"],
            page=row["page"],
            text=row["text"],
            embedding=json.loads(row["embedding"]),
            doc_type=row["doc_type"],
            version=row["version"],
            upload_date=row["upload_date"],
        )
        for row in rows
    ]


def list_documents(db_path: str | Path | None = None) -> list[dict[str, Any]]:
    init_db(db_path)
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT doc_name, source_path, file_type, doc_type, version, chunk_count,
                   content_hash, source_modified_at, indexed_at
            FROM documents
            ORDER BY indexed_at DESC, doc_name ASC
            """
        ).fetchall()
        if rows:
            return [dict(row) for row in rows]

        fallback_rows = conn.execute(
            """
            SELECT doc_name, '' AS source_path, '' AS file_type, doc_type, version,
                   COUNT(*) AS chunk_count, '' AS content_hash,
                   '' AS source_modified_at, MAX(upload_date) AS indexed_at
            FROM chunks
            GROUP BY doc_name, doc_type, version
            ORDER BY indexed_at DESC, doc_name ASC
            """
        ).fetchall()
    return [dict(row) for row in fallback_rows]


def audit_event(
    event_type: str,
    payload: dict[str, Any],
    db_path: str | Path | None = None,
) -> None:
    from uuid import uuid4

    init_db(db_path)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO audit_events (event_id, event_type, payload, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (str(uuid4()), event_type, json.dumps(payload, default=str), utc_now_iso()),
        )
        conn.commit()


def enqueue_review(
    question: str,
    answer_json: dict[str, Any],
    reason: str,
    db_path: str | Path | None = None,
) -> ReviewItem:
    init_db(db_path)
    item = ReviewItem(question=question, answer_json=answer_json, reason=reason)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO review_queue (
                item_id, question, answer_json, reason, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item.item_id,
                item.question,
                json.dumps(item.answer_json),
                item.reason,
                item.status,
                item.created_at,
                item.updated_at,
            ),
        )
        conn.commit()
    audit_event("review_enqueued", {"item_id": item.item_id, "reason": reason}, db_path)
    return item


def list_review_items(
    status: str | None = None,
    db_path: str | Path | None = None,
) -> list[ReviewItem]:
    init_db(db_path)
    query = """
        SELECT item_id, question, answer_json, reason, status, created_at, updated_at
        FROM review_queue
    """
    params: tuple[Any, ...] = ()
    if status:
        query += " WHERE status = ?"
        params = (status,)
    query += " ORDER BY created_at DESC"

    with connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()

    return [
        ReviewItem(
            item_id=row["item_id"],
            question=row["question"],
            answer_json=json.loads(row["answer_json"]),
            reason=row["reason"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
        for row in rows
    ]


def clear_all_reviews(db_path: str | Path | None = None) -> int:
    """Delete all review queue items. Returns count of deleted rows."""
    init_db(db_path)
    with connect(db_path) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM review_queue")
        count = cursor.fetchone()[0]
        conn.execute("DELETE FROM review_queue")
        conn.commit()
    audit_event("review_cleared_all", {"deleted_count": count}, db_path)
    return count


def update_review_status(
    item_id: str,
    status: str,
    db_path: str | Path | None = None,
) -> None:
    if status not in {"APPROVED", "REJECTED", "PENDING"}:
        raise ValueError("status must be APPROVED, REJECTED, or PENDING")
    init_db(db_path)
    with connect(db_path) as conn:
        conn.execute(
            """
            UPDATE review_queue
            SET status = ?, updated_at = ?
            WHERE item_id = ?
            """,
            (status, utc_now_iso(), item_id),
        )
        conn.commit()
    audit_event("review_status_updated", {"item_id": item_id, "status": status}, db_path)
