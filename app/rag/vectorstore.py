"""Persistent vector store backed by ChromaDB.

We supply our own embeddings (from embeddings.py) rather than letting Chroma
compute them, which keeps the embedding model in one place and swappable.
"""
from __future__ import annotations

import uuid

import chromadb

from app import config
from app.rag import embeddings

_COLLECTION = "documents"


def _client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def _collection():
    return _client().get_or_create_collection(
        name=_COLLECTION, metadata={"hnsw:space": "cosine"}
    )


def add_chunks(doc_id: str, filename: str, chunks: list[str]) -> int:
    """Embed and store chunks for a document. Returns number stored."""
    if not chunks:
        return 0
    vectors = embeddings.embed_texts(chunks)
    col = _collection()
    col.add(
        ids=[f"{doc_id}:{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=vectors,
        metadatas=[
            {"doc_id": doc_id, "filename": filename, "chunk_index": i}
            for i in range(len(chunks))
        ],
    )
    return len(chunks)


def query(question: str, top_k: int, doc_ids: list[str] | None = None) -> list[dict]:
    """Return the most relevant chunks, optionally filtered to specific docs."""
    col = _collection()
    if col.count() == 0:
        return []
    where = {"doc_id": {"$in": doc_ids}} if doc_ids else None
    res = col.query(
        query_embeddings=[embeddings.embed_query(question)],
        n_results=top_k,
        where=where,
    )
    hits: list[dict] = []
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for text, meta, dist in zip(docs, metas, dists):
        hits.append(
            {
                "text": text,
                "filename": meta.get("filename", "unknown"),
                "chunk_index": meta.get("chunk_index"),
                # cosine distance -> similarity in [0, 1]
                "score": round(1 - float(dist), 3),
            }
        )
    return hits


def list_documents() -> list[dict]:
    """Summarise stored documents (one row per file)."""
    col = _collection()
    if col.count() == 0:
        return []
    data = col.get(include=["metadatas"])
    counts: dict[str, dict] = {}
    for meta in data.get("metadatas", []):
        did = meta["doc_id"]
        row = counts.setdefault(
            did, {"doc_id": did, "filename": meta["filename"], "chunks": 0}
        )
        row["chunks"] += 1
    return sorted(counts.values(), key=lambda r: r["filename"])


def delete_document(doc_id: str) -> None:
    _collection().delete(where={"doc_id": doc_id})


def new_doc_id() -> str:
    return uuid.uuid4().hex[:12]