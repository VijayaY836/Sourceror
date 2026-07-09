# Sourcerer — AI Document Q&A (RAG)

Upload documents, ask questions in plain language, and get answers grounded in
the source text — with the exact retrieved passages shown for every answer.

Built as a single service: a FastAPI backend that runs a full Retrieval-Augmented
Generation pipeline and serves a polished web UI from the same origin. One command,
one URL, no separate frontend build.

> Tip: record a short screen capture or drop a screenshot here once it's running —
> a GIF of an upload + a couple of questions makes the repo land instantly.

---

## What it does

- **Upload** one or more PDF, TXT, or Markdown files.
- **Parse** them to text (page-aware for PDFs).
- **Chunk** the text into overlapping, boundary-aware passages.
- **Embed** each chunk locally and store it in a vector database.
- **Retrieve** the most relevant passages for a question (semantic search).
- **Generate** an answer with a modern LLM, grounded only in retrieved passages,
  with inline `[n]` citations and expandable source snippets.
- **Multi-turn**: follow-up questions use recent conversation context.
- **Scope**: tick documents in the sidebar to search only a subset.

---

## Architecture

```
                ┌──────────── Browser (single-page UI) ────────────┐
                │  upload · chat · click a citation to see source  │
                └───────────────────────┬──────────────────────────┘
                                         │ JSON over HTTP (same origin)
                ┌────────────────────────▼─────────────────────────┐
                │                  FastAPI (app/main.py)            │
                └───┬───────────────┬────────────────┬─────────────┘
       ingest       │               │  query         │
   ┌────────────────▼──┐   ┌────────▼────────┐   ┌───▼──────────────┐
   │ parsing.py        │   │ vectorstore.py  │   │ llm.py           │
   │ pypdf / txt / md  │   │ ChromaDB        │   │ OpenRouter chat  │
   ├───────────────────┤   │ (cosine, HNSW)  │   │ grounded prompt  │
   │ chunking.py       │   └────────▲────────┘   │ + [n] citations  │
   │ overlap splitter  │            │            └──────────────────┘
   ├───────────────────┤   ┌────────┴────────┐
   │ embeddings.py     │──▶│ fastembed (ONNX)│
   │ bge-small-en      │   │ local, no torch │
   └───────────────────┘   └─────────────────┘
```

**Pipeline stages** (`app/rag/pipeline.py`):

1. `parsing.py` — extract text; PDFs get `[page n]` markers.
2. `chunking.py` — recursive boundary splitter (paragraph → line → sentence →
   word) with sliding character overlap so context isn't cut mid-thought.
3. `embeddings.py` — [fastembed](https://github.com/qdrant/fastembed) with
   `BAAI/bge-small-en-v1.5` (ONNX runtime, no PyTorch → light install, fast start).
4. `vectorstore.py` — [ChromaDB](https://www.trychroma.com/) persistent client,
   cosine similarity. Chunks carry `doc_id`, `filename`, `chunk_index` metadata.
5. `llm.py` — retrieved passages are formatted as numbered context; the model is
   instructed to answer **only** from them, cite with `[n]`, and say so when the
   documents don't cover the question. Uses OpenRouter (OpenAI-compatible), so any
   hosted model works.

---

## Tech stack

| Layer        | Choice                                   |
|--------------|------------------------------------------|
| Backend      | FastAPI + Uvicorn                        |
| PDF parsing  | pypdf                                     |
| Embeddings   | fastembed (`bge-small-en-v1.5`, local)   |
| Vector DB    | ChromaDB (persistent, cosine)            |
| LLM          | OpenRouter (default `gemini-2.0-flash`)  |
| Frontend     | Vanilla HTML/CSS/JS, served by FastAPI   |

---

## Setup

Requires **Python 3.10+**.

```bash
# 1. clone
git clone https://github.com/<you>/sourcerer-docqa.git
cd sourcerer-docqa

# 2. install
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. configure
cp .env.example .env
# open .env and set OPENROUTER_API_KEY  (get one at https://openrouter.ai/keys)

# 4. run
uvicorn app.main:app --reload --port 8000
```

Or just `bash run.sh` (creates the venv, installs, copies `.env`, starts).

Then open **http://localhost:8000**.

> **First run** downloads the embedding model (~90 MB) once and caches it. Give
> the very first upload a few extra seconds. A `sample_docs/photosynthesis.txt`
> file is included to test with immediately.

### Configuration (`.env`)

| Variable             | Default                        | Notes                          |
|----------------------|--------------------------------|--------------------------------|
| `OPENROUTER_API_KEY` | —                              | required for answer generation |
| `LLM_MODEL`          | `google/gemini-2.0-flash-001`  | any OpenRouter chat model      |
| `EMBED_MODEL`        | `BAAI/bge-small-en-v1.5`       | any fastembed model            |
| `CHUNK_SIZE`         | `1000`                         | characters per chunk           |
| `CHUNK_OVERLAP`      | `150`                          | overlap between chunks         |
| `TOP_K`              | `5`                            | passages retrieved per query   |

---

## API

| Method   | Route                    | Purpose                              |
|----------|--------------------------|--------------------------------------|
| `GET`    | `/api/health`            | status + whether an LLM key is set   |
| `POST`   | `/api/upload`            | upload + ingest a document           |
| `GET`    | `/api/documents`         | list ingested documents              |
| `DELETE` | `/api/documents/{id}`    | remove a document                    |
| `POST`   | `/api/query`             | `{question, doc_ids?, history?}` → `{answer, sources[]}` |

Interactive API docs at **http://localhost:8000/docs**.

---

## Demo script (for the video)

1. Open `http://localhost:8000`.
2. Drag in `sample_docs/photosynthesis.txt` (and a PDF of your own).
3. Ask, one after another:
   - "What are the two stages of photosynthesis?"
   - "Which one needs light, and where does it happen?"   ← tests follow-up context
   - "At what temperature do the enzymes start to denature?"
   - "Does this document mention the discovery of DNA?"    ← shows honest "not covered"
4. Expand **source passages** under an answer and click a `[n]` citation to jump
   to the passage it came from — this visibly proves retrieval is working.

---

## Deployment (optional)

Single service, so it deploys as one web process. On Render / Railway / Fly:

- Build: `pip install -r requirements.txt`
- Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Set `OPENROUTER_API_KEY` in the environment.

Note: free tiers with ephemeral disks reset `data/` on redeploy (re-upload after
a restart). Mount a persistent volume at `data/` to keep the index.

---

## Project layout

```
app/
  main.py            FastAPI app + routes + serves the frontend
  config.py          settings from environment
  rag/
    parsing.py       PDF/TXT/MD → text
    chunking.py      overlapping boundary-aware splitter
    embeddings.py    fastembed wrapper
    vectorstore.py   ChromaDB wrapper
    llm.py           OpenRouter client + grounded RAG prompt
    pipeline.py      ingest + query orchestration
static/index.html    single-file UI
sample_docs/         a text file to test with
```