"""Extract plain text from uploaded documents.

Supported: .pdf (via pypdf), .txt, .md
Returns text with page markers for PDFs so chunks can reference page numbers.
"""
from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


class UnsupportedFileType(Exception):
    pass


def parse_document(path: str | Path) -> str:
    """Return extracted text for a supported document."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _parse_pdf(path)
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")

    raise UnsupportedFileType(f"Cannot parse '{suffix}' files.")


def _parse_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            # Page markers let us attribute retrieved chunks back to a page.
            parts.append(f"[page {i}]\n{text}")
    return "\n\n".join(parts)