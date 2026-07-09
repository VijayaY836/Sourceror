"""Orchestrates the RAG flow: ingest documents, answer questions."""
from __future__ import annotations

from pathlib import Path

from app import config
from app.rag import chunking, llm, parsing, vectorstore


def ingest_file(path: str | Path, filename: str) -> dict:
    """Parse -> chunk -> embed -> store a single document."""
    text = parsing.parse_document(path)
    if not text.strip():
        raise ValueError("No extractable text found in this file.")

    chunks = chunking.chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    doc_id = vectorstore.new_doc_id()
    stored = vectorstore.add_chunks(doc_id, filename, chunks)
    return {"doc_id": doc_id, "filename": filename, "chunks": stored}


def answer_question(
    question: str,
    doc_ids: list[str] | None = None,
    history: list[dict] | None = None,
) -> dict:
    """Retrieve relevant chunks and generate a grounded answer."""
    hits = vectorstore.query(question, config.TOP_K, doc_ids)
    text = llm.answer(question, hits, history)
    return {"answer": text, "sources": hits}