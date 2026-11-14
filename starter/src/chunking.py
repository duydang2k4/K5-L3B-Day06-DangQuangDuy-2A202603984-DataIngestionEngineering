"""Fixed-size and paragraph-aware chunking with stable source locations."""
import hashlib
import re

from .models import Chunk, DocumentPage


def _chunk(page: DocumentPage, text: str, index: int, start: int,
           end: int) -> Chunk:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return Chunk(
        chunk_id=f"{page.document_id}:{page.page}:{start}:{digest}",
        document_id=page.document_id,
        source=page.source,
        page=page.page,
        text=text,
        metadata={
            **page.metadata,
            "chunk_index": index,
            "start_char": start,
            "end_char": end,
        },
    )


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Split a page into overlapping windows and retain source/page metadata."""
    if size <= 0:
        raise ValueError("size must be greater than zero")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be non-negative and smaller than size")
    if not page.text:
        return []

    chunks = []
    step = size - overlap
    for index, start in enumerate(range(0, len(page.text), step)):
        end = min(start + size, len(page.text))
        chunks.append(_chunk(page, page.text[start:end], index, start, end))
        if end == len(page.text):
            break
    return chunks


def _split_long_paragraph(text: str, offset: int,
                          max_size: int) -> list[tuple[str, int, int]]:
    pieces = []
    start = 0
    while start < len(text):
        end = min(start + max_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + max_size // 2, end)
            if boundary > start:
                end = boundary
        if end == start:
            end = min(start + max_size, len(text))
        pieces.append((text[start:end], offset + start, offset + end))
        start = end
    return pieces


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Group source paragraphs up to max_size and split oversized paragraphs."""
    if max_size <= 0:
        raise ValueError("max_size must be greater than zero")
    if not page.text:
        return []

    paragraphs = []
    offset = 0
    for line in page.text.splitlines(keepends=True):
        paragraph = line.rstrip("\r\n").strip()
        if paragraph:
            leading = len(line) - len(line.lstrip())
            paragraph_offset = offset + leading
            paragraphs.extend(
                _split_long_paragraph(paragraph, paragraph_offset, max_size)
            )
        offset += len(line)

    grouped = []
    current: list[tuple[str, int, int]] = []
    current_size = 0
    for paragraph in paragraphs:
        text, start, end = paragraph
        added_size = len(text) + (1 if current else 0)
        if current and current_size + added_size > max_size:
            grouped.append(current)
            current = []
            current_size = 0
            added_size = len(text)
        current.append(paragraph)
        current_size += added_size
    if current:
        grouped.append(current)

    chunks = []
    for index, group in enumerate(grouped):
        text = "\n".join(part[0] for part in group)
        chunks.append(_chunk(page, text, index, group[0][1], group[-1][2]))
    return chunks
