import re
from collections import Counter
from pathlib import Path
from typing import Optional

import fitz

from app.schemas import CitationSchema, PaperSchema, ReferenceSchema, SectionSchema

# ---------------------------------------------------------------------------
# Citation regex patterns
# ---------------------------------------------------------------------------
_CITATION_RE = re.compile(
    r"\[\d+(?:[,\s]+\d+)*(?:[,-]\d+)*\]"  # [1], [1,2], [1-3]
    r"|"
    r"\([A-Z][a-z]+(?:\s+et\s+al\.?)?,\s*\d{4}[a-z]?\s*\)"  # (Author et al., 2020)
    r"|"
    r"\([A-Z][a-z]+(?:\s+(?:and|&)\s+[A-Z][a-z]+)?,\s*\d{4}[a-z]?\s*\)"  # (Author and Author, 2020)
)

# ---------------------------------------------------------------------------
# Heading detection helpers
# ---------------------------------------------------------------------------
_HEADING_NUMBERED = re.compile(r"^(?:[1-9]\d*|[IVXLCDM]+)\s*\.[\s]")
_HEADING_SUBNUMBERED = re.compile(r"^[1-9]\d*(?:\.[1-9]\d*)+\s+")
_HEADING_ALL_CAPS = re.compile(r"^[A-Z][A-Z\s/]{3,}$")

_COMMON_HEADINGS = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "methodology",
    "method",
    "methods",
    "experimental setup",
    "experiments",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "references",
    "bibliography",
    "acknowledgments",
    "acknowledgements",
    "appendix",
    "supplementary material",
    "future work",
}

# ---------------------------------------------------------------------------
# Block extraction
# ---------------------------------------------------------------------------


class _Block:
    __slots__ = ("text", "font_size", "is_bold", "page_num")

    def __init__(self, text: str, font_size: float, is_bold: bool, page_num: int) -> None:
        self.text = text
        self.font_size = font_size
        self.is_bold = is_bold
        self.page_num = page_num


def _extract_blocks(doc: fitz.Document) -> list[_Block]:
    blocks: list[_Block] = []
    for page in doc:
        for raw in page.get_text("dict")["blocks"]:
            if raw["type"] != 0:
                continue
            text_parts: list[str] = []
            max_size = 0.0
            has_bold = False
            for line in raw.get("lines", []):
                for span in line.get("spans", []):
                    text_parts.append(span["text"])
                    sz = span.get("size", 0)
                    if sz > max_size:
                        max_size = sz
                    font = (span.get("font") or "").lower()
                    if "bold" in font or "heavy" in font or "black" in font:
                        has_bold = True
            text = "".join(text_parts).strip()
            if not text:
                continue
            blocks.append(_Block(text, max_size, has_bold, page.number))
    return blocks


def _body_font_size(blocks: list[_Block]) -> float:
    sizes = Counter(round(b.font_size, 1) for b in blocks if b.font_size > 0)
    return sizes.most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Heading classification
# ---------------------------------------------------------------------------


def _is_heading(block: _Block, body_size: float, threshold: float) -> bool:
    text = block.text.strip()
    if not text or len(text) < 2:
        return False

    if _HEADING_NUMBERED.match(text):
        return True
    if _HEADING_SUBNUMBERED.match(text):
        return True
    if _HEADING_ALL_CAPS.match(text):
        return True

    if text.lower().rstrip(".:") in _COMMON_HEADINGS:
        return True

    if block.font_size >= threshold and len(text) < 120:
        return True

    if block.is_bold and block.font_size >= body_size * 1.05 and len(text) < 100:
        return True

    return False


# ---------------------------------------------------------------------------
# Metadata extraction (title, authors, abstract)
# ---------------------------------------------------------------------------


def _extract_title(blocks: list[_Block]) -> str:
    first_page = [b for b in blocks if b.page_num == 0]
    if not first_page:
        return "Untitled"

    candidate = max(first_page, key=lambda b: b.font_size)
    title = candidate.text.strip()
    return title if len(title) > 3 else "Untitled"


def _is_abstract_heading(text: str) -> bool:
    clean = re.sub(r"^(?:\d+|[ivxlcdm]+)\s*\.\s*", "", text.strip().lower().rstrip(".:"))
    return clean == "abstract"


def _extract_authors(blocks: list[_Block]) -> tuple[list[str], int]:
    first_page_blocks = [b for b in blocks if b.page_num == 0]
    if not first_page_blocks:
        return [], 0

    title_block = max(first_page_blocks, key=lambda b: b.font_size)
    title_idx = blocks.index(title_block)

    abstract_idx = -1
    for i in range(title_idx + 1, len(blocks)):
        if blocks[i].page_num > 0:
            break
        if _is_abstract_heading(blocks[i].text):
            abstract_idx = i
            break

    if abstract_idx == -1:
        return [], title_idx

    author_texts = []
    for i in range(title_idx + 1, abstract_idx):
        if blocks[i].page_num == 0:
            author_texts.append(blocks[i].text.strip())

    author_str = ", ".join(author_texts).strip()
    if not author_str:
        return [], title_idx

    author_str = re.sub(r"[\d*†‡§¶‖*]+", "", author_str).strip()
    parts = re.split(r"\s+and\s+|,\s*", author_str)
    authors = [p.strip().rstrip(",") for p in parts if re.match(r"^[A-Z]", p.strip())]
    authors = [a for a in authors if len(a.split()) >= 1]

    return authors, abstract_idx


def _extract_abstract(blocks: list[_Block], start_idx: int, body_size: float = 10.0, threshold: float = 12.0) -> Optional[str]:
    parts: list[str] = []
    recording = False
    for i in range(start_idx, min(start_idx + 20, len(blocks))):
        b = blocks[i]
        if _is_abstract_heading(b.text):
            recording = True
            continue
        if recording:
            if _is_heading(b, body_size, threshold) and not _is_abstract_heading(b.text):
                break
            parts.append(b.text.strip())

    return " ".join(parts).strip() if parts else None


# ---------------------------------------------------------------------------
# Section extraction
# ---------------------------------------------------------------------------


def _section_level(text: str, font_size: float, body_size: float) -> int:
    numbered = re.match(r"^(\d+(?:\.\d+)*)", text)
    if numbered:
        dots = numbered.group(1).count(".")
        return min(dots + 1, 6)
    if font_size >= body_size * 1.3:
        return 1
    if font_size >= body_size * 1.1:
        return 2
    return 1


def _extract_sections(blocks: list[_Block], body_size: float, threshold: float, abstract_end: int) -> list[SectionSchema]:
    sections: list[SectionSchema] = []
    current_heading = "Document Content"
    current_level = 1
    current_content: list[str] = []
    order = 0

    def flush() -> None:
        nonlocal order
        if current_heading and current_content:
            content = " ".join(current_content).strip()
            if content:
                sections.append(
                    SectionSchema(
                        heading=current_heading,
                        level=current_level,
                        content=content,
                        order_index=order,
                    )
                )
                order += 1

    for i, b in enumerate(blocks):
        if i <= abstract_end:
            continue
        text = b.text.strip()
        low = text.lower().strip(".:")

        if low in ("references", "bibliography"):
            if current_heading:
                flush()
            current_heading = text
            current_level = 1
            current_content = []
            break

        if _is_heading(b, body_size, threshold):
            if current_heading:
                flush()
            current_heading = text
            current_level = _section_level(text, b.font_size, body_size)
            current_content = []
        else:
            current_content.append(text)

    flush()
    return sections


# ---------------------------------------------------------------------------
# Citation extraction
# ---------------------------------------------------------------------------


def _extract_citations(text: str) -> list[CitationSchema]:
    seen = set()
    citations: list[CitationSchema] = []
    for m in _CITATION_RE.finditer(text):
        marker = m.group()
        if marker in seen:
            continue
        seen.add(marker)
        start = max(0, m.start() - 50)
        end = min(len(text), m.end() + 50)
        context = text[start:end].strip()
        citations.append(CitationSchema(marker=marker, context=context))
    return citations


# ---------------------------------------------------------------------------
# Reference extraction
# ---------------------------------------------------------------------------


def _extract_references(blocks: list[_Block]) -> list[ReferenceSchema]:
    ref_start = -1
    for i, b in enumerate(blocks):
        low = b.text.strip().lower().rstrip(".:")
        if low in ("references", "bibliography"):
            ref_start = i
            break

    if ref_start == -1:
        return []

    ref_blocks = blocks[ref_start + 1 :]
    ref_text = " ".join(b.text.strip() for b in ref_blocks if b.text.strip())

    ref_splitter = re.compile(r"\[(\d+)\]\s*")
    parts = ref_splitter.split(ref_text)

    references: list[ReferenceSchema] = []
    for j in range(1, len(parts) - 1, 2):
        if j + 1 >= len(parts):
            continue
        marker = f"[{parts[j]}]"
        raw = parts[j + 1].strip()
        if not raw:
            continue
        references.append(_parse_reference(marker, raw))

    return references


_REF_YEAR = re.compile(r"\((\d{4})\)")
_REF_AUTHORS_END = re.compile(r"[A-Z][a-z]+(?:\s+(?:and|&)\s+[A-Z][a-z]+)?\.")


def _parse_reference(marker: str, raw: str) -> ReferenceSchema:
    title: Optional[str] = None
    authors: Optional[str] = None
    year: Optional[int] = None
    source: Optional[str] = None

    year_match = _REF_YEAR.search(raw)
    if year_match:
        year = int(year_match.group(1))

    quote_match = re.search(r'"([^"]+)"', raw)
    if quote_match:
        title = quote_match.group(1)
    else:
        # fallback: title is between the first period and the year
        dot_idx = raw.find(".")
        if dot_idx > 0 and dot_idx < 200:
            after_dot = raw[dot_idx + 1 :].strip()
            year_pos = raw.find("(")
            if year_pos > dot_idx:
                title = after_dot[: year_pos - dot_idx - 1].strip()
            elif after_dot:
                title = after_dot[:150].strip()

    # authors are typically before the first period or before the year
    first_dot = raw.find(".")
    if first_dot > 0 and first_dot < 300:
        authors = raw[:first_dot].strip()
        # remove leading marker
        authors = re.sub(r"^\[\d+\]\s*", "", authors).strip()

    return ReferenceSchema(marker=marker, title=title, authors=authors, year=year, source=source, raw_text=raw)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_pdf(file_path: str | Path | None = None, *, stream: bytes | None = None) -> PaperSchema:
    if stream is not None:
        doc = fitz.open(stream=stream, filetype="pdf")
    elif file_path is not None:
        doc = fitz.open(str(file_path))
    else:
        raise ValueError("Either file_path or stream must be provided")
    try:
        blocks = _extract_blocks(doc)
        if not blocks:
            return PaperSchema(title="Untitled", authors=[])

        body_size = _body_font_size(blocks)
        threshold = body_size * 1.2

        title = _extract_title(blocks)
        authors, author_end = _extract_authors(blocks)
        abstract = _extract_abstract(blocks, author_end, body_size, threshold)

        abstract_end = author_end
        if abstract:
            for i in range(author_end, len(blocks)):
                low = blocks[i].text.strip().lower().rstrip(".:")
                if _is_abstract_heading(blocks[i].text):
                    abstract_end = i
                    break

        sections = _extract_sections(blocks, body_size, threshold, abstract_end)
        if sections and title and title != "Untitled":
            if title.lower() not in sections[0].content.lower():
                sections[0].content = f"{title}\n{sections[0].content}"

        full_text = " ".join(b.text for b in blocks)
        citations = _extract_citations(full_text)
        references = _extract_references(blocks)

        return PaperSchema(
            title=title,
            authors=authors,
            abstract=abstract,
            sections=sections,
            citations=citations,
            references=references,
        )
    finally:
        doc.close()
