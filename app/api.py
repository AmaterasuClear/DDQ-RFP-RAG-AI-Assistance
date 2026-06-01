from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

from pydantic import BaseModel

from app.excel_processor import process_questionnaire
from app.ingest import ingest_pdf
from app.main import run_question
from app.storage import list_review_items, update_review_status


class AskRequest(BaseModel):
    question: str
    doc_type: str | None = None
    version: str | None = None
    doc_name: str | None = None


def create_app():
    from fastapi import FastAPI, File, UploadFile
    from fastapi.responses import Response

    api = FastAPI(title="Due Diligence RAG — Institutional")

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.post("/ingest")
    async def ingest(
        file: UploadFile = File(...),
        doc_type: str = "general",
        version: str = "v1",
    ) -> dict[str, int | str]:
        suffix = Path(file.filename or "document.pdf").suffix or ".pdf"
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)
        try:
            chunks = ingest_pdf(tmp_path, doc_type=doc_type, version=version)
        finally:
            tmp_path.unlink(missing_ok=True)
        return {"doc_name": file.filename or "document.pdf", "chunks": len(chunks)}

    @api.post("/ask")
    def ask(request: AskRequest):
        return run_question(
            request.question,
            doc_type=request.doc_type,
            version=request.version,
            doc_name=request.doc_name,
        ).model_dump()

    @api.post("/process-excel")
    async def process_excel(file: UploadFile = File(...)):
        """Upload an Excel questionnaire and receive answered file."""
        data = await file.read()
        output_bytes, output_filename, summary = process_questionnaire(
            data, file.filename or "questionnaire.xlsx"
        )
        return Response(
            content=output_bytes,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f'attachment; filename="{output_filename}"',
                "X-Processing-Summary": str(summary),
            },
        )

    @api.get("/review-queue")
    def review_queue(status: str | None = None):
        return [item.model_dump() for item in list_review_items(status=status)]

    @api.post("/review-queue/{item_id}/{status}")
    def update_review(item_id: str, status: str) -> dict[str, str]:
        update_review_status(item_id, status)
        return {"item_id": item_id, "status": status}

    return api