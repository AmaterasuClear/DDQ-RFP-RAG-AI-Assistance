from __future__ import annotations

from app.schemas import Chunk


def apply_metadata_filters(
    chunks: list[Chunk],
    doc_type: str | None = None,
    version: str | None = None,
    doc_name: str | None = None,
) -> list[Chunk]:
    filtered = chunks
    if doc_type:
        filtered = [chunk for chunk in filtered if chunk.doc_type == doc_type]
    if version:
        filtered = [chunk for chunk in filtered if chunk.version == version]
    if doc_name:
        filtered = [chunk for chunk in filtered if chunk.doc_name == doc_name]
    return filtered

