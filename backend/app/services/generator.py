import logging
import re
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import settings
from app.services.retriever import ChunkResult
from app.services.tracing import Tracer

logger = logging.getLogger(__name__)

_BASE = settings.OPENROUTER_BASE_URL or "https://openrouter.ai/api/v1"

_SYSTEM_PROMPT = (
    "You are Paper Pilot, a research paper assistant. "
    "Answer the user's question based ONLY on the provided paper excerpts. "
    "Cite sources using [1], [2] markers corresponding to the provided chunks. "
    "If the context doesn't contain enough information to answer, "
    "say 'The provided paper excerpts do not contain information about this.' "
    "If the user quotes or pastes a passage and asks to complete, continue, or "
    "reproduce it, quote the continuation verbatim from the excerpts rather than paraphrasing."
)

_GENERAL_SYSTEM_PROMPT = (
    "You are Paper Pilot, an AI assistant for research paper Q&A. "
    "Respond conversationally and introduce yourself. Explain that users can upload PDF research papers and ask questions to get grounded answers with section citations. "
    "If they ask a specific question without uploading a paper, answer politely and suggest uploading a PDF paper for section citations."
)

_MARKER_PATTERN = re.compile(r"[\[【]([\d\s,–-]+)[\]】]")


@dataclass
class Citation:
    chunk_id: str
    paper_id: str
    section_heading: str
    marker: str
    snippet: str = ""


@dataclass
class GenerationResult:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    finish_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


def _build_context(chunks: list[ChunkResult]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        heading = c.section_heading or "Untitled"
        text = c.metadata.get("parent_content") or c.content
        lines.append(f"[{i}] (Section: {heading}) {text}")
    return "\n\n".join(lines)


def _parse_marker_numbers(content: str) -> list[int]:
    range_match = re.search(r"^(\d+)\s*[-–]\s*(\d+)$", content.strip())
    if range_match:
        start, end = int(range_match.group(1)), int(range_match.group(2))
        if start <= end and (end - start) < 20:
            return list(range(start, end + 1))
    return [int(p) for p in re.findall(r"\d+", content)]


def _extract_citations(
    answer: str, chunks: list[ChunkResult]
) -> list[Citation]:
    seen = set()
    citations = []
    for match in _MARKER_PATTERN.finditer(answer):
        numbers = _parse_marker_numbers(match.group(1))
        for n in numbers:
            if n < 1 or n > len(chunks):
                continue
            chunk = chunks[n - 1]
            key = (chunk.chunk_id, n)
            if key in seen:
                continue
            seen.add(key)
            # Create a concise, clean excerpt snippet of the cited chunk
            raw_snippet = chunk.content.strip()
            # If the chunk starts with the heading, strip it for cleaner display
            if chunk.section_heading and raw_snippet.startswith(chunk.section_heading):
                raw_snippet = raw_snippet[len(chunk.section_heading):].strip()
            snippet = raw_snippet[:240].strip()
            if len(raw_snippet) > 240:
                snippet += "..."
            citations.append(
                Citation(
                    chunk_id=chunk.chunk_id,
                    paper_id=chunk.paper_id,
                    section_heading=chunk.section_heading,
                    marker=f"[{n}]",
                    snippet=snippet,
                )
            )
    return citations


def _extractive_fallback(query: str, chunks: list[ChunkResult]) -> GenerationResult:
    citations: list[Citation] = []
    lines = [
        "> **Notice**: OpenRouter daily free-model limit reached (50 requests/day). Providing an extractive answer synthesized directly from retrieved document sections.\n",
        f"Based on the retrieved sections for **\"{query}\"**:\n",
    ]
    for i, chunk in enumerate(chunks[:3]):
        marker = f"[{i + 1}]"
        raw_text = (chunk.content or "").strip()
        snippet = raw_text[:240].strip()
        if len(raw_text) > 240:
            snippet += "..."
        citations.append(
            Citation(
                chunk_id=chunk.chunk_id,
                paper_id=chunk.paper_id,
                section_heading=chunk.section_heading,
                marker=marker,
                snippet=snippet,
            )
        )
        heading = chunk.section_heading or "Relevant Section"
        lines.append(f"### {heading} {marker}\n{raw_text}\n")

    answer = "\n".join(lines)
    return GenerationResult(
        answer=answer,
        citations=citations,
        finish_reason="extractive_fallback",
        input_tokens=0,
        output_tokens=max(1, len(answer) // 4),
    )


async def generate(
    query: str,
    chunks: list[ChunkResult],
    model: str | None = None,
    temperature: float = 0.1,
    tracer: Tracer | None = None,
) -> GenerationResult:
    if model is None:
        model = settings.GENERATOR_MODEL or "z-ai/glm-5.2:free"

    if not query:
        return GenerationResult(answer="Please enter a question.", finish_reason="no_input")

    if not chunks:
        if tracer:
            async with tracer.span(
                "generate",
                attributes={"model": model, "input_tokens": 0, "output_tokens": 0, "finish_reason": "general_chat", "context_truncated": False, "context_token_count": 0},
            ):
                pass
        try:
            llm = ChatOpenAI(
                model=model,
                openai_api_key=settings.OPENROUTER_API_KEY,
                openai_api_base=_BASE,
                temperature=temperature,
                max_tokens=512,
            )
            response = llm.invoke([SystemMessage(content=_GENERAL_SYSTEM_PROMPT), HumanMessage(content=query)])
            return GenerationResult(answer=str(response.content), finish_reason="general_chat")
        except Exception:
            return GenerationResult(
                answer="Hello! I am Paper Pilot, your AI research paper assistant. Upload a PDF paper to ask questions with section citations.",
                finish_reason="fallback",
            )

    context = _build_context(chunks)
    context_token_count = max(1, len(context) // 4)

    llm = ChatOpenAI(
        model=model,
        openai_api_key=settings.OPENROUTER_API_KEY,
        openai_api_base=_BASE,
        temperature=temperature,
        max_tokens=1024,
        streaming=False,
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Paper excerpts:\n\n{context}\n\n---\n\nQuestion: {query}"
        ),
    ]

    def _call_llm():
        try:
            response = llm.invoke(messages)
            ans = response.content or ""
            cits = _extract_citations(ans, chunks)
            usage = response.usage_metadata if hasattr(response, "usage_metadata") else None
            in_toks = usage.get("input_tokens", 0) if usage else 0
            out_toks = usage.get("output_tokens", 0) if usage else 0
            freason = response.response_metadata.get("finish_reason") if response.response_metadata else None
            return ans, cits, freason, in_toks, out_toks
        except Exception as exc:
            if "429" in str(exc) or "rate limit" in str(exc).lower():
                logger.warning("OpenRouter rate limit (429) hit during generation. Using extractive fallback.")
                fb = _extractive_fallback(query, chunks)
                return fb.answer, fb.citations, fb.finish_reason, fb.input_tokens, fb.output_tokens
            raise

    if tracer:
        async with tracer.span(
            "generate",
            attributes={
                "model": model,
                "context_token_count": context_token_count,
                "context_truncated": context_token_count > 4000,
            },
        ) as span:
            answer, citations, finish_reason, input_tokens, output_tokens = _call_llm()
            span.attributes["input_tokens"] = input_tokens
            span.attributes["output_tokens"] = output_tokens
            span.attributes["finish_reason"] = finish_reason

            return GenerationResult(
                answer=answer,
                citations=citations,
                finish_reason=finish_reason,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
    else:
        answer, citations, finish_reason, input_tokens, output_tokens = _call_llm()
        return GenerationResult(
            answer=answer,
            citations=citations,
            finish_reason=finish_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


async def stream_generate(
    query: str,
    chunks: list[ChunkResult],
    model: str | None = None,
    temperature: float = 0.1,
    session_id: str | None = None,
    tracer: Tracer | None = None,
) -> AsyncGenerator[str, None]:
    if model is None:
        model = settings.GENERATOR_MODEL or "z-ai/glm-5.2:free"
    from app.services.streamer import stream_generate as _sg

    async for event in _sg(query, chunks, model, temperature, session_id, tracer):
        yield event
