"""Ingest the configured corpus and evaluate all three retrieval strategies."""
import json
import os
import re
import time
import unicodedata
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .embeddings import MyEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import load_docx, load_pdf
from .models import Chunk, DocumentPage
from .store import PgVectorStore

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_STARTER_ROOT = Path(__file__).resolve().parents[1]
_CHAPTER_PATTERN = re.compile(r"^\s*Chương\s+(I|II|III|IV|V)\b", re.IGNORECASE)


def _read_benchmark(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        benchmark = json.load(file)
    if not isinstance(benchmark, dict) or not isinstance(benchmark.get("questions"), list):
        raise ValueError(f"{path} must contain a questions list")
    questions = benchmark["questions"]
    split_counts = {
        split: sum(question.get("split") == split for question in questions)
        for split in ("dev", "test")
    }
    if len(questions) < 20 or split_counts["dev"] < 15 or split_counts["test"] < 5:
        raise ValueError("Benchmark requires at least 20 questions, 15 dev and 5 test")
    if sum(bool(question.get("filter")) for question in questions) < 3:
        raise ValueError("Benchmark requires at least three metadata-filter questions")
    if not any(
        question.get("split") == "test" and question.get("filter")
        for question in questions
    ):
        raise ValueError("Benchmark requires at least one filtered test question")
    return benchmark


def _load_corpus(benchmark: dict[str, Any]) -> tuple[list[DocumentPage], dict[str, Any]]:
    corpus = benchmark.get("corpus")
    if not isinstance(corpus, list) or not corpus:
        raise ValueError("Benchmark corpus metadata is missing")
    documents = []
    for item in corpus:
        document_id = item.get("document_id")
        source = item.get("source")
        original_file = item.get("original_file")
        if not document_id or not source or not original_file:
            raise ValueError("Corpus entries require document_id, source and original_file")
        data_dir = _STARTER_ROOT / "data"
        path = data_dir / original_file
        if not path.is_file():
            candidates = [
                candidate for candidate in data_dir.iterdir()
                if candidate.is_file()
                and candidate.suffix.casefold() == Path(original_file).suffix.casefold()
            ]
            if len(candidates) != 1:
                raise FileNotFoundError(
                    f"Corpus file does not exist: {path}; found "
                    f"{len(candidates)} unambiguous candidates"
                )
            path = candidates[0]
        if path.suffix.casefold() == ".docx":
            pages = load_docx(path, document_id, source)
        elif path.suffix.casefold() == ".pdf":
            pages = load_pdf(path, document_id, source)
        else:
            raise ValueError(f"Unsupported corpus format: {path.suffix}")
        for page in pages:
            page.metadata.update({
                "document_id": document_id,
                "source": source,
                "year": item.get("document_year"),
                "version": item.get("version"),
                "title": item.get("document_title"),
            })
        documents.extend(pages)
    nonempty_page_count = sum(bool(page.text) for page in documents)
    if nonempty_page_count < 30:
        raise ValueError(
            f"Corpus has {nonempty_page_count} non-empty content pages; "
            "at least 30 are required"
        )
    expected_page_count = sum(item.get("page_count", 0) for item in corpus)
    if expected_page_count and len(documents) != expected_page_count:
        raise ValueError(
            f"Extracted {len(documents)} pages but benchmark expects "
            f"{expected_page_count}; verify DOCX/PDF page boundaries"
        )
    _assign_chapters(documents)
    evidence_warnings = _validate_benchmark_evidence(
        benchmark["questions"], documents
    )
    return documents, {
        "document_count": len(corpus),
        "page_count": len(documents),
        "nonempty_page_count": nonempty_page_count,
        "character_count": sum(len(page.text) for page in documents),
        "evidence_validation_warnings": evidence_warnings,
    }


def _assign_chapters(pages: list[DocumentPage]) -> None:
    chapter = None
    for page in pages:
        for line in page.text.splitlines():
            match = _CHAPTER_PATTERN.match(line)
            if match:
                chapter = f"chapter_{match.group(1).upper()}"
        if chapter:
            page.metadata["chapter"] = chapter


def _evidence_key(text: str) -> str:
    return "".join(unicodedata.normalize("NFC", text).casefold().split())


def _validate_benchmark_evidence(questions: list[dict[str, Any]],
                                 pages: list[DocumentPage]) -> list[dict[str, Any]]:
    text_by_document = {}
    for page in pages:
        text_by_document.setdefault(page.document_id, "")
        text_by_document[page.document_id] += page.text
    missing = []
    for question in questions:
        references = question.get("evidence") or []
        if not references and question.get("evidence_quote"):
            references = [{
                "source_id": question.get("gold_source_id"),
                "quote": question["evidence_quote"],
            }]
        for reference in references:
            document_id = reference.get("source_id", question.get("gold_source_id"))
            quote = _evidence_key(reference.get("quote", ""))
            source_text = _evidence_key(text_by_document.get(document_id, ""))
            if not quote or quote not in source_text:
                missing.append({
                    "question_id": question.get("id"),
                    "source_id": document_id,
                    "page": reference.get("page"),
                })
    if missing:
        warnings.warn(
            f"{len(missing)} benchmark evidence quote(s) do not exactly match "
            "the supplied corpus text; review corpus/benchmark version alignment",
            RuntimeWarning,
            stacklevel=2,
        )
    return missing


def _build_chunks(pages: list[DocumentPage]) -> dict[str, list[Chunk]]:
    fixed = [
        chunk
        for page in pages
        for chunk in fixed_size(page, size=800, overlap=100)
    ]
    paragraphs = [
        chunk
        for page in pages
        for chunk in by_paragraph(page, max_size=1000)
    ]
    return {"A_fixed": fixed, "B_paragraph": paragraphs, "C_filtered": paragraphs}


def _chunk_statistics(chunks: list[Chunk]) -> dict[str, Any]:
    lengths = [len(chunk.text) for chunk in chunks]
    return {
        "chunks": len(chunks),
        "total_characters": sum(lengths),
        "min_characters": min(lengths) if lengths else 0,
        "max_characters": max(lengths) if lengths else 0,
        "average_characters": sum(lengths) / len(lengths) if lengths else 0,
    }


def _prepare_manual_assessment(
    pages: list[DocumentPage],
    strategy_chunks: dict[str, list[Chunk]],
    traces: list[dict[str, Any]],
    questions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Prepare fixed coherence, grounding and metadata-review samples."""
    coherence_pages = [page for page in pages if page.text][:10]
    if len(coherence_pages) < 10:
        raise ValueError("At least 10 non-empty pages are required for coherence samples")

    chunks_by_page: dict[str, dict[tuple[str, int], list[Chunk]]] = {}
    for strategy, chunks in strategy_chunks.items():
        by_page: dict[tuple[str, int], list[Chunk]] = {}
        for chunk in chunks:
            by_page.setdefault((chunk.document_id, chunk.page), []).append(chunk)
        chunks_by_page[strategy] = by_page

    coherence_samples = []
    for sample_index, page in enumerate(coherence_pages, start=1):
        offset = 0
        location = ""
        position = 0
        for line in page.text.splitlines(keepends=True):
            candidate = line.strip()
            if candidate:
                leading = len(line) - len(line.lstrip())
                location = candidate[:500]
                position = offset + leading
                break
            offset += len(line)
        strategy_samples = {}
        for strategy in ("A_fixed", "B_paragraph", "C_filtered"):
            page_chunks = chunks_by_page[strategy].get(
                (page.document_id, page.page), []
            )
            chosen = next(
                (
                    chunk for chunk in page_chunks
                    if chunk.metadata["start_char"] <= position
                    < chunk.metadata["end_char"]
                ),
                page_chunks[0] if page_chunks else None,
            )
            strategy_samples[strategy] = {
                "chunk_id": chosen.chunk_id if chosen else None,
                "chunk_text": chosen.text if chosen else None,
                "score_0_to_2": None,
                "rationale": "",
            }
        coherence_samples.append({
            "sample_id": f"coherence_{sample_index:02d}",
            "document_id": page.document_id,
            "source": page.source,
            "page": page.page,
            "source_position_char": position,
            "source_excerpt": location,
            "strategies": strategy_samples,
        })

    trace_by_key = {
        (trace["strategy"], trace["question_id"]): trace for trace in traces
    }
    test_questions = [
        question for question in questions if question.get("split") == "test"
    ][:5]
    grounding_samples = []
    metadata_samples = []
    for strategy in ("A_fixed", "B_paragraph", "C_filtered"):
        for question in test_questions:
            trace = trace_by_key[(strategy, question["id"])]
            grounding_samples.append({
                "strategy": strategy,
                "question_id": question["id"],
                "split": "test",
                "top_3": trace["hits"],
                "score_0_to_2": None,
                "rationale": "",
            })
            for hit in trace["hits"]:
                metadata = hit.get("metadata") or {}
                filters = trace.get("filter")
                filter_compliant = (
                    all(metadata.get(key) == value for key, value in filters.items())
                    if filters else None
                )
                metadata_samples.append({
                    "strategy": strategy,
                    "question_id": question["id"],
                    "chunk_id": hit["chunk_id"],
                    "source": hit["source"],
                    "page": hit["page"],
                    "metadata": metadata,
                    "filter": filters,
                    "filter_compliant": filter_compliant,
                    "metadata_correct_manual": None,
                    "notes": "",
                })

    return {
        "coherence_scale": "0=not understandable, 1=partly understandable, 2=self-contained",
        "grounding_scale": "0=no evidence, 1=partial evidence, 2=sufficient evidence and source",
        "coherence_samples": coherence_samples,
        "grounding_samples": grounding_samples,
        "metadata_usefulness_samples": metadata_samples,
    }


def main() -> None:
    """Run fixed-size, paragraph and metadata-filtered retrieval benchmarks."""
    load_dotenv(_STARTER_ROOT / ".env")
    benchmark = _read_benchmark(_PROJECT_ROOT / "benchmark" / "benchmark.json")
    pages, corpus_statistics = _load_corpus(benchmark)
    for question in benchmark["questions"]:
        filters = question.get("filter") or {}
        for key, value in filters.items():
            if not any(page.metadata.get(key) == value for page in pages):
                raise ValueError(
                    f"Question {question.get('id')} filters on unavailable "
                    f"metadata value {key}={value!r}"
                )

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is missing from starter/.env")
    model_name = (
        os.getenv("EMBEDDING_MODEL")
        or "Horizon-Labs/multilingual-embedding-small"
    )
    embedder = MyEmbedder(model_name)
    store = PgVectorStore(database_url, embedder.dimensions)
    store.create_schema()

    strategy_chunks = _build_chunks(pages)
    vectors_by_strategy = {}
    timings = {"embedding_seconds": {}, "upsert_seconds": {}}
    for strategy in ("A_fixed", "B_paragraph"):
        start = time.perf_counter()
        vectors = embedder.embed_documents(
            [chunk.text for chunk in strategy_chunks[strategy]]
        )
        timings["embedding_seconds"][strategy] = time.perf_counter() - start
        vectors_by_strategy[strategy] = vectors

    vectors_by_strategy["C_filtered"] = vectors_by_strategy["B_paragraph"]
    for strategy, chunks in strategy_chunks.items():
        start = time.perf_counter()
        store.upsert(strategy, chunks, vectors_by_strategy[strategy])
        timings["upsert_seconds"][strategy] = time.perf_counter() - start

    queries = [question["question"] for question in benchmark["questions"]]
    query_vectors = embedder.embed_queries(queries)
    traces = []
    for strategy in ("A_fixed", "B_paragraph", "C_filtered"):
        for question, query_vector in zip(benchmark["questions"], query_vectors):
            filters = question.get("filter") if strategy == "C_filtered" else None
            hits = store.search(strategy, query_vector, top_k=3, filters=filters)
            evaluation = evaluate_query(question, hits)
            traces.append({
                "strategy": strategy,
                "question_id": question["id"],
                "split": question["split"],
                "question": question["question"],
                "filter": filters,
                "expected_answer": question.get("expected_answer"),
                "evidence": question.get("evidence", []),
                "hits": hits,
                **evaluation,
            })

    results = {
        "benchmark_id": benchmark.get("benchmark_id"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": model_name,
        "embedding_dimensions": embedder.dimensions,
        "vector_store": "PostgreSQL + pgvector",
        "search_metric": "cosine similarity, exact search",
        "corpus": corpus_statistics,
        "strategies": {
            "A_fixed": {
                "chunking": "fixed-size",
                "size": 800,
                "overlap": 100,
                **_chunk_statistics(strategy_chunks["A_fixed"]),
            },
            "B_paragraph": {
                "chunking": "paragraph grouping",
                "max_size": 1000,
                "metadata_filter": False,
                **_chunk_statistics(strategy_chunks["B_paragraph"]),
            },
            "C_filtered": {
                "chunking": "same paragraph chunks as B",
                "max_size": 1000,
                "metadata_filter": "chapter, when supplied by question",
                **_chunk_statistics(strategy_chunks["C_filtered"]),
            },
        },
        "timings": timings,
        "metrics": summarize(traces),
        "manual_assessment": _prepare_manual_assessment(
            pages, strategy_chunks, traces, benchmark["questions"]
        ),
        "traces": traces,
    }
    results_path = _STARTER_ROOT / "results" / "benchmark_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    print(json.dumps(results["metrics"], ensure_ascii=False, indent=2))
    print(f"Full top-3 traces saved to {results_path}")


if __name__ == "__main__":
    main()
