"""Read PDF and DOCX sources while preserving page-level provenance."""
from pathlib import Path
import re
import unicodedata
import warnings
import zipfile
from xml.etree import ElementTree

from pypdf import PdfReader

from .models import DocumentPage

_WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
_WORD_PAGE_BREAK = f"{{{_WORD_NS['w']}}}lastRenderedPageBreak"
_WORD_BREAK_TYPE = f"{{{_WORD_NS['w']}}}type"


def _page(document_id: str, source: str, number: int, text: str,
          boundary_source: str) -> DocumentPage:
    normalized = normalize_text(text)
    empty = not bool(normalized)
    if empty:
        warnings.warn(
            f"No extractable text on page {number} of {source}",
            RuntimeWarning,
            stacklevel=2,
        )
    return DocumentPage(
        document_id=document_id,
        source=source,
        page=number,
        text=normalized,
        metadata={"empty": empty, "page_boundary_source": boundary_source},
    )


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Extract each physical PDF page and mark pages without extractable text."""
    reader = PdfReader(str(path))
    return [
        _page(
            document_id,
            source,
            number,
            page.extract_text() or "",
            "pdf_page",
        )
        for number, page in enumerate(reader.pages, start=1)
    ]


def load_docx(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Extract DOCX text, honoring Word's saved rendered-page boundaries."""
    with zipfile.ZipFile(path) as archive:
        try:
            root = ElementTree.fromstring(archive.read("word/document.xml"))
        except KeyError as exc:
            raise ValueError(f"{path} is missing word/document.xml") from exc

    pages = [""]

    def walk(node: ElementTree.Element) -> None:
        tag = node.tag.rsplit("}", 1)[-1]
        if tag == "t":
            pages[-1] += node.text or ""
        elif tag in {"tab", "ptab"}:
            pages[-1] += "\t"
        elif tag == "lastRenderedPageBreak":
            pages.append("")
        elif tag == "br" and node.attrib.get(_WORD_BREAK_TYPE) == "page":
            pages.append("")
        else:
            for child in node:
                walk(child)
            if tag == "p":
                pages[-1] += "\n"

    body = root.find("w:body", _WORD_NS)
    if body is None:
        raise ValueError(f"{path} does not contain a Word document body")
    for element in body:
        if element.tag != f"{{{_WORD_NS['w']}}}sectPr":
            walk(element)

    has_rendered_breaks = len(pages) > 1
    boundary_source = "docx_rendered_page_break" if has_rendered_breaks else "docx_single_unit"
    return [
        _page(document_id, source, number, text, boundary_source)
        for number, text in enumerate(pages, start=1)
    ]


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace while preserving paragraph boundaries."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00a0", " ").replace("\u00ad", "")
    normalized_lines = []
    previous_blank = False
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = re.sub(r"[^\S\n]+", " ", line).strip()
        if not line:
            if normalized_lines and not previous_blank:
                normalized_lines.append("")
            previous_blank = True
            continue
        normalized_lines.append(line)
        previous_blank = False
    while normalized_lines and not normalized_lines[-1]:
        normalized_lines.pop()
    return "\n".join(normalized_lines)
