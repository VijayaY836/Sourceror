"""LLM answer generation via OpenRouter (OpenAI-compatible API)."""
from __future__ import annotations

from openai import OpenAI

from app import config

_SYSTEM = (
    "You are a careful document analyst. Answer the user's question using ONLY the "
    "numbered context passages provided. Ground every claim in the passages and cite "
    "the passages you used with bracketed numbers like [1] or [2]. If the answer is "
    "not contained in the passages, say plainly that the documents do not cover it — "
    "do not use outside knowledge or guess. Be concise and specific."
)


class LLMConfigError(Exception):
    pass


def _client() -> OpenAI:
    if not config.OPENROUTER_API_KEY:
        raise LLMConfigError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file."
        )
    return OpenAI(
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
    )


def _format_context(chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"[{i}] (source: {c['filename']})\n{c['text']}")
    return "\n\n".join(blocks)


def answer(question: str, chunks: list[dict], history: list[dict] | None = None) -> str:
    """Generate a grounded answer from retrieved chunks."""
    if not chunks:
        return "I couldn't find anything relevant in the uploaded documents for that question."

    messages: list[dict] = [{"role": "system", "content": _SYSTEM}]

    # brief prior turns for follow-up questions
    for turn in (history or [])[-config.MAX_HISTORY_TURNS * 2 :]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    user = (
        f"Context passages:\n\n{_format_context(chunks)}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the passages above and cite them with [n]."
    )
    messages.append({"role": "user", "content": user})

    resp = _client().chat.completions.create(
        model=config.LLM_MODEL,
        messages=messages,
        temperature=config.LLM_TEMPERATURE,
    )
    return resp.choices[0].message.content.strip()