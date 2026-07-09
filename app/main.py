"""FastAPI application: document Q&A over uploaded files (RAG).

Serves a JSON API and the single-page frontend from the same origin, so the
whole app runs from one command with no CORS setup.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import config
from app.rag import llm, parsing, pipeline, vectorstore

app = FastAPI(title="Document Q&A (RAG)")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


class QueryIn(BaseModel):
    question: str
    doc_ids: list[str] | None = None
    history: list[dict] | None = None


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "llm_configured": bool(config.OPENROUTER_API_KEY),
        "model": config.LLM_MODEL,
        "embed_model": config.EMBED_MODEL,
    }


@app.get("/api/documents")
def documents() -> dict:
    return {"documents": vectorstore.list_documents()}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    name = file.filename or "untitled"
    ext = Path(name).suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported file type '{ext}'. Allowed: "
            + ", ".join(sorted(config.ALLOWED_EXTENSIONS)),
        )

    contents = await file.read()
    if len(contents) > config.MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(400, f"File exceeds {config.MAX_FILE_MB} MB limit.")

    dest = config.UPLOAD_DIR / name
    dest.write_bytes(contents)

    try:
        result = pipeline.ingest_file(dest, name)
    except (parsing.UnsupportedFileType, ValueError) as e:
        raise HTTPException(400, str(e))
    return result


@app.post("/api/query")
def query(body: QueryIn) -> dict:
    if not body.question.strip():
        raise HTTPException(400, "Question cannot be empty.")
    try:
        return pipeline.answer_question(body.question, body.doc_ids, body.history)
    except llm.LLMConfigError as e:
        raise HTTPException(503, str(e))


@app.delete("/api/documents/{doc_id}")
def delete(doc_id: str) -> dict:
    vectorstore.delete_document(doc_id)
    return {"deleted": doc_id}


# ---- Frontend (mounted last so it doesn't shadow /api routes) ----
@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/", StaticFiles(directory=str(STATIC_DIR)), name="static")