import unittest
from pathlib import Path

from src.chunking import by_paragraph, fixed_size
from src.evaluation import evaluate_query, summarize
from src.ingestion import load_docx, normalize_text
from src.models import DocumentPage
from src.run import _prepare_manual_assessment


class IngestionTests(unittest.TestCase):
    def test_normalize_text_preserves_paragraphs_and_unicode(self):
        self.assertEqual(
            normalize_text("Cafe\u0301\u00a0  X\t\r\n \r\nY"),
            "Café X\n\nY",
        )

    def test_docx_loader_uses_rendered_page_boundaries(self):
        path = Path(__file__).resolve().parents[1] / "data" / "nghi_dinh_94_2026.docx"
        pages = load_docx(path, "94/2026/NĐ-CP", "source.pdf")
        self.assertEqual(len(pages), 32)
        self.assertEqual(pages[0].page, 1)
        self.assertEqual(pages[-1].page, 32)
        self.assertTrue(all(page.text for page in pages))
        self.assertEqual(
            pages[0].metadata["page_boundary_source"],
            "docx_rendered_page_break",
        )


class ChunkingTests(unittest.TestCase):
    def setUp(self):
        self.page = DocumentPage(
            document_id="doc",
            source="source.pdf",
            page=4,
            text="first paragraph\nsecond paragraph is a little longer",
            metadata={"chapter": "chapter_II"},
        )

    def test_fixed_size_chunks_keep_page_and_overlap(self):
        chunks = fixed_size(self.page, size=12, overlap=3)
        self.assertEqual(chunks[0].text[-3:], chunks[1].text[:3])
        self.assertTrue(all(chunk.page == 4 for chunk in chunks))
        self.assertTrue(all(chunk.metadata["chapter"] == "chapter_II" for chunk in chunks))

    def test_paragraph_chunking_respects_maximum_and_keeps_all_content(self):
        chunks = by_paragraph(self.page, max_size=18)
        self.assertTrue(all(len(chunk.text) <= 18 for chunk in chunks))
        self.assertEqual(
            "".join("".join(chunk.text.split()) for chunk in chunks),
            "".join(self.page.text.split()),
        )

    def test_invalid_chunk_sizes_are_rejected(self):
        with self.assertRaises(ValueError):
            fixed_size(self.page, size=4, overlap=4)
        with self.assertRaises(ValueError):
            by_paragraph(self.page, max_size=0)


class EvaluationTests(unittest.TestCase):
    def test_hit_requires_full_evidence_on_a_gold_source_and_page(self):
        question = {
            "id": "Q1",
            "split": "test",
            "gold_sources": [
                {"source_id": "doc", "source": "source.pdf", "page": 2}
            ],
            "evidence": [
                {
                    "source_id": "doc",
                    "source": "source.pdf",
                    "page": 2,
                    "quote": "first required passage",
                },
                {
                    "source_id": "doc",
                    "source": "source.pdf",
                    "page": 2,
                    "quote": "second required passage",
                },
            ],
        }
        hits = [
            {
                "document_id": "doc",
                "source": "source.pdf",
                "page": 2,
                "text": "first required passage",
            },
            {
                "document_id": "doc",
                "source": "source.pdf",
                "page": 2,
                "text": "second required passage",
            },
        ]
        result = evaluate_query(question, hits)
        self.assertFalse(result["hit_at_3"])
        self.assertTrue(result["partial_evidence_hit_at_3"])
        self.assertEqual(result["matched_evidence_by_hit"], [[0], [1]])

        combined = evaluate_query(question, [{**hits[0], "text":
                                              "first required passage; second required passage"}])
        self.assertTrue(combined["hit_at_1"])
        self.assertTrue(combined["hit_at_3"])

    def test_wrong_page_is_not_a_source_page_hit(self):
        question = {
            "id": "Q2",
            "gold_source_id": "doc",
            "gold_page": 2,
            "evidence_quote": "gold quote",
        }
        result = evaluate_query(question, [{
            "document_id": "doc",
            "source": "source.pdf",
            "page": 3,
            "text": "gold quote",
        }])
        self.assertFalse(result["hit_at_1"])
        self.assertFalse(result["source_page_hit_at_1"])

    def test_summary_reports_counts_and_split_denominators(self):
        metrics = summarize([
            {"strategy": "A", "split": "dev", "hit_at_1": True, "hit_at_3": True},
            {"strategy": "A", "split": "test", "hit_at_1": False, "hit_at_3": True},
        ])
        self.assertEqual(metrics["A"]["dev"]["hit_at_1_count"], 1)
        self.assertEqual(metrics["A"]["test"]["questions"], 1)
        self.assertEqual(metrics["A"]["overall"]["evidence_hit_at_3"], 1.0)


class ManualAssessmentTests(unittest.TestCase):
    def test_prepares_fixed_coherence_and_test_review_samples(self):
        pages = [
            DocumentPage("doc", "source.pdf", page, f"Page {page} has a complete paragraph.",
                         {"chapter": "chapter_II"})
            for page in range(1, 11)
        ]
        strategies = {}
        for strategy in ("A_fixed", "B_paragraph", "C_filtered"):
            strategies[strategy] = [
                chunk
                for page in pages
                for chunk in by_paragraph(page, 100)
            ]

        questions = [
            {
                "id": f"T{index}",
                "split": "test",
                "filter": {"chapter": "chapter_II"} if index == 0 else None,
            }
            for index in range(5)
        ]
        traces = []
        for strategy in strategies:
            for question in questions:
                traces.append({
                    "strategy": strategy,
                    "question_id": question["id"],
                    "hits": [
                        {
                            "chunk_id": f"{strategy}-{question['id']}-{rank}",
                            "source": "source.pdf",
                            "page": 1,
                            "text": "Retrieved passage",
                            "score": 1.0,
                            "metadata": {"chapter": "chapter_II"},
                        }
                        for rank in range(3)
                    ],
                    "filter": question["filter"] if strategy == "C_filtered" else None,
                })

        samples = _prepare_manual_assessment(pages, strategies, traces, questions)
        self.assertEqual(len(samples["coherence_samples"]), 10)
        self.assertEqual(len(samples["grounding_samples"]), 15)
        self.assertEqual(len(samples["metadata_usefulness_samples"]), 45)
        self.assertTrue(all(
            sample["score_0_to_2"] is None
            for sample in samples["grounding_samples"]
        ))
        filtered = [
            sample for sample in samples["metadata_usefulness_samples"]
            if sample["strategy"] == "C_filtered" and sample["question_id"] == "T0"
        ]
        self.assertTrue(all(sample["filter_compliant"] for sample in filtered))


if __name__ == "__main__":
    unittest.main()
