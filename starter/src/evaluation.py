"""Evaluate source/page evidence and aggregate retrieval metrics."""
import unicodedata
from typing import Any


def _match_text(text: str) -> str:
    return "".join(unicodedata.normalize("NFC", text).casefold().split())


def _matches_source_page(hit: dict[str, Any],
                         reference: dict[str, Any]) -> bool:
    metadata = hit.get("metadata") or {}
    source_id = reference.get("source_id", reference.get("document_id"))
    source = reference.get("source")
    page = reference.get("page")

    actual_document_id = hit.get("document_id", metadata.get("document_id"))
    actual_source = hit.get("source", metadata.get("source"))
    actual_page = hit.get("page", metadata.get("page"))
    if source_id is not None and actual_document_id != source_id:
        return False
    if source is not None and actual_source != source:
        return False
    return page is None or actual_page == page


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Check that a single retrieved chunk contains the recorded gold evidence.

    All evidence references in the question must be present in one chunk for the
    primary evidence hit. A partial multi-chunk match is reported separately.
    This exact-match check does not assign human grounding or coherence labels.
    """
    evidence = question.get("evidence") or []
    if evidence:
        references = [
            {
                "source_id": item.get("source_id"),
                "source": item.get("source"),
                "page": item.get("page"),
                "quote": item.get("quote", ""),
            }
            for item in evidence
        ]
    elif question.get("evidence_quote"):
        references = [{
            "source_id": question.get("gold_source_id"),
            "source": question.get("gold_source"),
            "page": question.get("gold_page"),
            "quote": question["evidence_quote"],
        }]
    else:
        references = []

    gold_sources = question.get("gold_sources") or []
    if not gold_sources and (
        question.get("gold_source_id") is not None
        or question.get("gold_page") is not None
    ):
        gold_sources = [{
            "source_id": question.get("gold_source_id"),
            "source": question.get("gold_source"),
            "page": question.get("gold_page"),
        }]

    normalized_quotes = [_match_text(item["quote"]) for item in references]
    source_page_matches = []
    evidence_matches = []
    for hit in hits:
        source_page_matches.append(
            any(_matches_source_page(hit, reference) for reference in gold_sources)
        )
        hit_text = _match_text(hit.get("text", ""))
        matched = [
            index
            for index, (reference, quote) in enumerate(zip(references, normalized_quotes))
            if quote and _matches_source_page(hit, reference) and quote in hit_text
        ]
        evidence_matches.append(matched)

    complete_matches = [
        bool(references) and len(matched) == len(references)
        for matched in evidence_matches
    ]
    partial_evidence_hit = any(evidence_matches)
    return {
        "question_id": question.get("id"),
        "split": question.get("split"),
        "evidence_required": len(references),
        "source_page_match_by_hit": source_page_matches,
        "matched_evidence_by_hit": evidence_matches,
        "hit_at_1": bool(complete_matches and complete_matches[0]),
        "hit_at_3": any(complete_matches[:3]),
        "source_page_hit_at_1": bool(source_page_matches and source_page_matches[0]),
        "source_page_hit_at_3": any(source_page_matches[:3]),
        "partial_evidence_hit_at_3": partial_evidence_hit,
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate retrieval counts and rates by strategy and dev/test split."""
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for trace in traces:
        strategy = trace["strategy"]
        split = trace.get("split") or "all"
        grouped.setdefault(strategy, {}).setdefault(split, []).append(trace)

    def metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(items)
        hit_at_1 = sum(bool(item.get("hit_at_1")) for item in items)
        hit_at_3 = sum(bool(item.get("hit_at_3")) for item in items)
        source_page_hit_at_3 = sum(
            bool(item.get("source_page_hit_at_3")) for item in items
        )
        return {
            "questions": total,
            "hit_at_1_count": hit_at_1,
            "hit_at_1": hit_at_1 / total if total else None,
            "evidence_hit_at_3_count": hit_at_3,
            "evidence_hit_at_3": hit_at_3 / total if total else None,
            "source_page_hit_at_3_count": source_page_hit_at_3,
            "source_page_hit_at_3": source_page_hit_at_3 / total if total else None,
        }

    result = {}
    for strategy, splits in grouped.items():
        result[strategy] = {
            split: metrics(items) for split, items in sorted(splits.items())
        }
        result[strategy]["overall"] = metrics(
            [trace for items in splits.values() for trace in items]
        )
    return result
