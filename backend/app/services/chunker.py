import re
from dataclasses import dataclass

import tiktoken

from app.schemas import SectionSchema

_ENCODING = tiktoken.get_encoding("cl100k_base")
_MAX_TOKENS = 512
_OVERLAP_TOKENS = 64
_SMALL_MAX_TOKENS = 128
_SMALL_OVERLAP_TOKENS = 32

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|[\n\r]+|(?<=[.!?])(?=[–—\-])|(?<=[;:])\s+")


@dataclass
class ChunkData:
    paper_id: str
    section_heading: str
    section_level: int
    content: str
    chunk_index: int
    total_sections: int
    parent_content: str | None = None


def _num_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def _group_sentences(
    sentences: list[str],
    max_tokens: int,
    overlap_tokens: int,
    base_tokens: int,
) -> list[list[str]]:
    groups: list[list[str]] = []
    overlap: list[str] = []
    idx = 0

    while idx < len(sentences):
        current: list[str] = list(overlap)
        tokens = base_tokens + sum(_num_tokens(s) for s in current)
        added_any = False

        while idx < len(sentences):
            s_text = sentences[idx]
            s_tokens = _num_tokens(s_text)
            if tokens + s_tokens > max_tokens:
                if not current:
                    current.append(s_text[:1500])
                    idx += 1
                    added_any = True
                break
            current.append(s_text)
            tokens += s_tokens
            idx += 1
            added_any = True

        if not added_any and idx < len(sentences):
            current.append(sentences[idx][:1000])
            idx += 1

        groups.append(current)

        overlap = []
        used = 0
        for s in reversed(current):
            s_t = _num_tokens(s)
            if used + s_t > overlap_tokens:
                break
            overlap.insert(0, s)
            used += s_t

    return groups


def _chunk_section_content(
    content: str,
    heading: str,
    section_level: int,
    paper_id: str,
    total_sections: int,
    chunk_start_index: int,
) -> list[ChunkData]:
    heading_tokens = _num_tokens(heading)

    # Single chunk if under limit
    text_for_counting = f"{heading}\n{content}" if content else heading
    if _num_tokens(text_for_counting) <= _MAX_TOKENS:
        return [
            ChunkData(
                paper_id=paper_id,
                section_heading=heading,
                section_level=section_level,
                content=text_for_counting,
                chunk_index=chunk_start_index,
                total_sections=total_sections,
            )
        ]

    sentences = _SENTENCE_SPLIT.split(content)
    if not sentences:
        return [
            ChunkData(
                paper_id=paper_id,
                section_heading=heading,
                section_level=section_level,
                content=heading,
                chunk_index=chunk_start_index,
                total_sections=total_sections,
            )
        ]

    groups = _group_sentences(sentences, _MAX_TOKENS, _OVERLAP_TOKENS, heading_tokens)

    return [
        ChunkData(
            paper_id=paper_id,
            section_heading=heading,
            section_level=section_level,
            content=f"{heading}\n{' '.join(group)}",
            chunk_index=chunk_start_index + i,
            total_sections=total_sections,
        )
        for i, group in enumerate(groups)
    ]


def _small_subchunks(medium: ChunkData) -> list[ChunkData]:
    if _num_tokens(medium.content) <= _SMALL_MAX_TOKENS:
        medium.parent_content = medium.content
        return [medium]

    parts = medium.content.split("\n", 1)
    heading = parts[0]
    body = parts[1] if len(parts) > 1 else ""
    sentences = _SENTENCE_SPLIT.split(body) if body else []
    if not sentences:
        medium.parent_content = medium.content
        return [medium]

    groups = _group_sentences(
        sentences, _SMALL_MAX_TOKENS, _SMALL_OVERLAP_TOKENS, _num_tokens(heading)
    )

    return [
        ChunkData(
            paper_id=medium.paper_id,
            section_heading=medium.section_heading,
            section_level=medium.section_level,
            content=f"{heading}\n{' '.join(group)}",
            chunk_index=medium.chunk_index,
            total_sections=medium.total_sections,
            parent_content=medium.content,
        )
        for group in groups
    ]


def chunk_sections(sections: list[SectionSchema], paper_id: str) -> list[ChunkData]:
    total = len(sections)
    all_chunks: list[ChunkData] = []

    for section in sections:
        medium_chunks = _chunk_section_content(
            content=section.content,
            heading=section.heading,
            section_level=section.level,
            paper_id=paper_id,
            total_sections=total,
            chunk_start_index=len(all_chunks),
        )
        for medium in medium_chunks:
            all_chunks.extend(_small_subchunks(medium))

    for i, chunk in enumerate(all_chunks):
        chunk.chunk_index = i

    return all_chunks
