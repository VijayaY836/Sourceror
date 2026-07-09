"""Central configuration, loaded from environment variables (.env supported)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"

for _d in (DATA_DIR, UPLOAD_DIR, CHROMA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- Retrieval / chunking ----
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))          # characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))     # overlap between chunks
TOP_K = int(os.getenv("TOP_K", "5"))                       # chunks retrieved per query

# ---- Embeddings ----
# fastembed model (ONNX, no torch). bge-small is a strong, tiny default.
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# ---- LLM (OpenRouter, OpenAI-compatible) ----
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "google/gemini-2.0-flash-001")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "4"))

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "25"))