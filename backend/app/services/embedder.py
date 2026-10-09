import hashlib
import logging
import math
import re
import time

import httpx

from app.config import settings
from app.services.chunker import ChunkData

logger = logging.getLogger(__name__)

_BASE = settings.OPENROUTER_BASE_URL or "https://openrouter.ai/api/v1"
_EMBED_MODEL = settings.EMBED_MODEL
EMBEDDING_DIM = settings.EMBED_DIM
_BATCH_SIZE = 16
_MAX_RETRIES = 3

# Warn about embedding-dimension truncation only once per process.
_DIM_WARNED = False


def _fallback_embed(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    vec = [0.0] * dim
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return vec
    for token in tokens:
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if ((h >> 10) & 1) else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def _embed_batch(texts: list[str]) -> list[list[float]]:
    if not settings.OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")

    url = f"{_BASE}/embeddings"
    headers = {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    input_data = texts[0] if len(texts) == 1 else texts
    payload = {"model": _EMBED_MODEL, "input": input_data}

    response = httpx.post(url, headers=headers, json=payload, timeout=60.0)
    response.raise_for_status()
    data = response.json()

    results = []
    for item in data["data"]:
        emb = item.get("embedding", [])
        # OpenRouter's nemotron-3-embed-1b returns 2048 dims; we store 1024 to
        # match the local Nemotron-3-Embed-1B checkpoint and the vector(1024)
        # column. Truncate + renormalize. Warn once, not per chunk (uploads
        # embed hundreds of chunks and the per-item log swamped the output).
        if len(emb) != EMBEDDING_DIM:
            global _DIM_WARNED
            if not _DIM_WARNED:
                logger.warning(
                    "Embedding dimension %d differs from configured %d; truncating. "
                    "Set EMBED_DIM=%d to use the full vector.",
                    len(emb),
                    EMBEDDING_DIM,
                    len(emb),
                )
                _DIM_WARNED = True
        vec = emb[:EMBEDDING_DIM]
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        results.append(vec)

    return results


def _embed_with_retry(texts: list[str]) -> list[list[float]]:
    last_error: Exception | None = None
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return _embed_batch(texts)
        except Exception as exc:
            last_error = exc
            logger.warning(
                "OpenRouter embedding attempt %d/%d failed: %s",
                attempt + 1,
                max_retries,
                exc,
            )
            # If 429 Too Many Requests, break early to fallback rather than looping
            if "429" in str(exc):
                break
            time.sleep(2**attempt)

    if last_error and "429" in str(last_error):
        logger.warning(
            "OpenRouter daily free-model limit reached (429). Using deterministic local fallback embeddings."
        )
        return [_fallback_embed(t) for t in texts]

    raise RuntimeError(
        f"OpenRouter embedding failed after {max_retries} attempts: {last_error}"
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        embeddings.extend(_embed_with_retry(texts[i : i + _BATCH_SIZE]))
    return embeddings


def embed_chunks(chunks: list[ChunkData]) -> list[list[float]]:
    return embed_texts([c.content for c in chunks])


def embed_query(query: str) -> list[float]:
    if not query:
        return []
    return _embed_with_retry([query])[0]
