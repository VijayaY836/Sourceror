"""Embedding model wrapper.

Uses fastembed (ONNX runtime, no torch) so the repo stays light and starts fast.
The model is downloaded once on first use and cached locally.
"""
from __future__ import annotations

from functools import lru_cache

from fastembed import TextEmbedding

from app import config


@lru_cache(maxsize=1)
def _model() -> TextEmbedding:
    return TextEmbedding(model_name=config.EMBED_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of documents/passages."""
    return [vec.tolist() for vec in _model().embed(texts)]


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    return next(_model().query_embed(text)).tolist()