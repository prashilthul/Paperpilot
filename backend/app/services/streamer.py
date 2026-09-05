import json
import logging
import uuid
from collections.abc import AsyncGenerator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import settings
from app.services.generator import _SYSTEM_PROMPT, _build_context, _extract_citations, _extractive_fallback
from app.services.retriever import ChunkResult
from app.services.tracing import Tracer

from app.services.chunker import _num_tokens

logger = logging.getLogger(__name__)

_BASE = settings.OPENROUTER_BASE_URL or "https://openrouter.ai/api/v1"


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
    trace_id = tracer.trace_id if tracer else str(uuid.uuid4())

    if not query:
        yield f"event: token\ndata: {json.dumps({'text': 'Please enter a question.'})}\n\n"
        yield f"event: done\ndata: {json.dumps({'citations': [], 'trace_id': trace_id, 'session_id': session_id})}\n\n"
        return

    if not chunks:
        if tracer:
            async with tracer.span(
                "generate",
                attributes={"model": model, "input_tokens": 0, "output_tokens": 0, "finish_reason": "general_chat", "context_truncated": False, "context_token_count": 0},
            ):
                pass

        _GENERAL_SYSTEM_PROMPT = (
            "You are Paper Pilot, an AI assistant for research paper Q&A. "
            "Respond conversationally and introduce yourself. Explain that users can upload PDF research papers and ask questions to get grounded answers with section citations. "
            "If they ask a specific question without uploading a paper, answer politely and suggest uploading a PDF paper for section citations."
        )

        try:
            llm = ChatOpenAI(
                model=model,
                openai_api_key=settings.OPENROUTER_API_KEY,
                openai_api_base=_BASE,
                temperature=temperature,
                max_tokens=512,
                streaming=True,
            )
            messages = [
                SystemMessage(content=_GENERAL_SYSTEM_PROMPT),
                HumanMessage(content=query),
            ]
            async for chunk in llm.astream(messages):
                token = chunk.content or ""
                if token:
                    yield f"event: token\ndata: {json.dumps({'text': token})}\n\n"
            spans_summary = tracer.get_trace_spans_summary() if tracer else []
            yield f"event: done\ndata: {json.dumps({'citations': [], 'trace_id': trace_id, 'trace_spans': spans_summary, 'session_id': session_id})}\n\n"
            return
        except Exception:
            no_info_text = "Hello! I am Paper Pilot, your AI research paper assistant. Upload a PDF paper to ask questions with section citations."
            spans_summary = tracer.get_trace_spans_summary() if tracer else []
            yield f"event: token\ndata: {json.dumps({'text': no_info_text})}\n\n"
            yield f"event: done\ndata: {json.dumps({'citations': [], 'trace_id': trace_id, 'trace_spans': spans_summary, 'session_id': session_id})}\n\n"
            return

    context = _build_context(chunks)
    context_token_count = max(1, len(context) // 4)

    llm = ChatOpenAI(
        model=model,
        openai_api_key=settings.OPENROUTER_API_KEY,
        openai_api_base=_BASE,
        temperature=temperature,
        max_tokens=1024,
        streaming=True,
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Paper excerpts:\n\n{context}\n\n---\n\nQuestion: {query}"
        ),
    ]

    if tracer:
        async with tracer.span(
            "generate",
            attributes={
                "model": model,
                "context_token_count": context_token_count,
                "context_truncated": context_token_count > 4000,
            },
        ) as span:
            full_text = ""
            output_token_count = 0
            stream_error: Exception | None = None

            try:
                async for chunk in llm.astream(messages):
                    token = chunk.content or ""
                    if token:
                        full_text += token
                        output_token_count += _num_tokens(token)
                        yield f"event: token\ndata: {json.dumps({'text': token})}\n\n"
            except Exception as e:
                logger.warning("stream generation failed: %s", e)
                span.attributes["finish_reason"] = "error"
                stream_error = e

            citations = _extract_citations(full_text, chunks)
            if not stream_error:
                span.attributes["output_tokens"] = output_token_count
                span.attributes["finish_reason"] = "stop"

        spans_summary = tracer.get_trace_spans_summary()
        if stream_error:
            if "429" in str(stream_error) or "rate limit" in str(stream_error).lower():
                logger.warning("Stream generation hit 429 rate limit. Yielding extractive fallback.")
                fb = _extractive_fallback(query, chunks)
                yield f"event: token\ndata: {json.dumps({'text': fb.answer})}\n\n"
                yield f"event: done\ndata: {json.dumps({'citations': [c.__dict__ for c in fb.citations], 'trace_id': trace_id, 'trace_spans': spans_summary, 'session_id': session_id})}\n\n"
            else:
                yield f"event: error\ndata: {json.dumps({'message': str(stream_error)})}\n\n"
        else:
            yield f"event: done\ndata: {json.dumps({'citations': [c.__dict__ for c in citations], 'trace_id': trace_id, 'trace_spans': spans_summary, 'session_id': session_id})}\n\n"
    else:
        full_text = ""
        stream_error: Exception | None = None

        try:
            async for chunk in llm.astream(messages):
                token = chunk.content or ""
                if token:
                    full_text += token
                    yield f"event: token\ndata: {json.dumps({'text': token})}\n\n"
        except Exception as e:
            logger.warning("stream generation failed: %s", e)
            stream_error = e

        citations = _extract_citations(full_text, chunks)
        spans_summary = tracer.get_trace_spans_summary() if tracer else []
        if stream_error:
            if "429" in str(stream_error) or "rate limit" in str(stream_error).lower():
                logger.warning("Stream generation hit 429 rate limit. Yielding extractive fallback.")
                fb = _extractive_fallback(query, chunks)
                yield f"event: token\ndata: {json.dumps({'text': fb.answer})}\n\n"
                yield f"event: done\ndata: {json.dumps({'citations': [c.__dict__ for c in fb.citations], 'trace_id': trace_id, 'trace_spans': spans_summary, 'session_id': session_id})}\n\n"
            else:
                yield f"event: error\ndata: {json.dumps({'message': str(stream_error)})}\n\n"
        else:
            yield f"event: done\ndata: {json.dumps({'citations': [c.__dict__ for c in citations], 'trace_id': trace_id, 'trace_spans': spans_summary, 'session_id': session_id})}\n\n"
