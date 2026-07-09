"""Split long text into overlapping chunks.

Strategy: recursively split on the largest natural boundary that keeps a chunk
under the target size (paragraphs -> lines -> sentences -> words), then pack
pieces into chunks with a sliding overlap so context isn't cut mid-thought.
"""
from __future__ import annotations

import re

_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split_recursive(text: str, size: int, seps: list[str]) -> list[str]:
    if len(text) <= size or not seps:
        return [text]
    sep = seps[0]
    pieces = text.split(sep)
    out: list[str] = []
    for piece in pieces:
        piece = piece if sep == "" else piece + sep
        if len(piece) <= size:
            out.append(piece)
        else:
            out.extend(_split_recursive(piece, size, seps[1:]))
    return out


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Return a list of text chunks with character overlap between neighbours."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []

    pieces = _split_recursive(text, chunk_size, _SEPARATORS)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if len(current) + len(piece) <= chunk_size:
            current += piece
        else:
            if current.strip():
                chunks.append(current.strip())
            # start next chunk with a tail of the previous one (overlap)
            tail = current[-overlap:] if overlap and current else ""
            current = tail + piece
    if current.strip():
        chunks.append(current.strip())

    return chunks