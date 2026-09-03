from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import fitz

TEST_DIR = Path(__file__).resolve().parent
CODE_DIR = TEST_DIR.parent
MLR_DIR = CODE_DIR.parent / "MLR"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import extract_mlr as mlr


class ProgressTests(unittest.TestCase):
    def test_noninteractive_progress_reports_start_and_completion(self):
        output = io.StringIO()
        progress = mlr.ConsoleProgress(stream=output, width=10)

        progress.start("Reading pages", 4)
        progress.update(2)
        progress.finish()

        rendered = output.getvalue()
        self.assertIn("Reading pages", rendered)
        self.assertIn("0.0% (0/4)", rendered)
        self.assertIn("50.0% (2/4)", rendered)
        self.assertIn("100.0% (4/4)", rendered)

    def test_progress_can_be_disabled(self):
        output = io.StringIO()
        progress = mlr.ConsoleProgress(enabled=False, stream=output)
        progress.start("Hidden", 1)
        progress.finish()
        self.assertEqual(output.getvalue(), "")


class RefillTests(unittest.TestCase):
    def test_refill_preserves_grounded_concepts_and_reaches_minimum(self):
        chapter = mlr.CHAPTERS[0]
        initial = [asdict(candidate(index, chapter)) for index in range(20)]
        retained = initial[:16]
        refill = [asdict(candidate(index, chapter)) for index in range(20, 24)]

        class FakeBook:
            def chapter_chunks(self, *_args, **_kwargs):
                return [f"[PDF_PAGE {chapter.pdf_start} | BOOK_PAGE {chapter.book_start}]\ntext"]

            def page_text(self, *_args, **_kwargs):
                return " ".join(item["evidence"] for item in [*initial, *refill])

        class FakeClient:
            model = "fake-qwen"

            def __init__(self):
                self.responses = [initial, retained, refill, refill]

            def complete_json(self, *_args, **_kwargs):
                concepts = self.responses.pop(0)
                payload = {"concepts": concepts}
                return payload, json.dumps(payload), False

        with patch.object(mlr, "extract_assets", return_value=([], [])):
            result = mlr.extract_chapter(FakeBook(), FakeClient(), chapter, [], [])

        names = {item["canonical"] for item in result.concepts}
        self.assertEqual(len(result.concepts), 20)
        self.assertTrue({item["canonical"] for item in retained}.issubset(names))
        self.assertIn("refill", {item["stage"] for item in result.raw_responses})


def candidate(index: int, chapter: mlr.Chapter | None = None, **overrides):
    chapter = chapter or mlr.CHAPTERS[0]
    data = {
        "canonical": f"useful machine learning concept {index}",
        "aliases": [],
        "evidence_page": chapter.pdf_start,
        "evidence": f"Evidence for useful concept {index}.",
        "relevance": 0.9,
        "anchor_page": chapter.pdf_start,
    }
    data.update(overrides)
    return mlr.ConceptCandidate.from_mapping(data, chapter)


def chapter_result(chapter_number: int = 1, concept_count: int = 20) -> mlr.ChapterResult:
    chapter = mlr.CHAPTERS[chapter_number - 1]
    concepts = [asdict(candidate(index, chapter)) for index in range(concept_count)]
    return mlr.ChapterResult(
        version=mlr.PROGRAM_VERSION,
        chapter=chapter.number,
        chapter_title=chapter.title,
        pdf_start=chapter.pdf_start,
        pdf_end=chapter.pdf_end,
        model="fake-qwen",
        completed_at="2026-01-01T00:00:00+00:00",
        concepts=concepts,
        formulas=[],
        figures=[],
    )


class ChapterMapTests(unittest.TestCase):
    def test_exact_boundaries_and_appendix_exclusion(self):
        self.assertEqual(len(mlr.CHAPTERS), 14)
        self.assertEqual((mlr.CHAPTERS[0].pdf_start, mlr.CHAPTERS[0].pdf_end), (20, 37))
        self.assertEqual((mlr.CHAPTERS[-1].pdf_start, mlr.CHAPTERS[-1].pdf_end), (483, 510))
        included = {page for chapter in mlr.CHAPTERS for page in range(chapter.pdf_start, chapter.pdf_end + 1)}
        self.assertNotIn(19, included)
        self.assertNotIn(39, included)  # Part I divider
        self.assertNotIn(125, included)  # Part II divider
        self.assertNotIn(307, included)  # Part III divider
        self.assertNotIn(511, included)  # Part IV / appendices

    def test_context_window_is_clamped_to_chapter(self):
        chapter = mlr.CHAPTERS[0]
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "small.pdf"
            document = fitz.open()
            document.new_page()
            document.save(pdf_path)
            document.close()
            with mlr.PdfBook(pdf_path, validate=False) as book:
                self.assertEqual(book.context_window(chapter, 20), [20, 21])
                self.assertEqual(book.context_window(chapter, 37), [36, 37])


class NormalizationTests(unittest.TestCase):
    def test_case_hyphen_plural_normalization(self):
        self.assertEqual(mlr.clean_concept_name("  K–Means  "), "k-means")
        self.assertEqual(mlr.normalize_key("neural networks"), mlr.normalize_key("neural network"))

    def test_local_duplicate_merge_preserves_alias(self):
        chapter = mlr.CHAPTERS[0]
        first = candidate(1, chapter, canonical="neural network", aliases=["neural networks"])
        second = candidate(2, chapter, canonical="neural networks", relevance=0.8)
        merged, decisions = mlr.merge_candidates_locally([first, second])
        self.assertEqual(len(merged), 1)
        self.assertEqual(len(decisions), 1)

    def test_concept_count_limits(self):
        with self.assertRaises(mlr.ExtractionError):
            mlr.select_chapter_concepts([candidate(index) for index in range(19)])
        selected = mlr.select_chapter_concepts([candidate(index) for index in range(105)])
        self.assertEqual(len(selected), 100)


class JsonAndApiTests(unittest.TestCase):
    def test_json_parser_accepts_markdown_fence(self):
        parsed = mlr.parse_json_object('```json\n{"concepts": []}\n```')
        self.assertEqual(parsed, {"concepts": []})

    def test_api_retries_429(self):
        calls = []

        def transport(endpoint, payload, headers, timeout):
            calls.append(payload)
            if len(calls) == 1:
                raise mlr.ApiError("busy", status=429)
            return {"choices": [{"message": {"content": '{"concepts": []}'}}]}

        client = mlr.QwenClient(
            "fake-key", "https://example.invalid/v1", "fake-model", transport=transport, sleeper=lambda _: None
        )
        payload, _, used_vision = client.complete_json("system", "user")
        self.assertEqual(payload, {"concepts": []})
        self.assertFalse(used_vision)
        self.assertEqual(len(calls), 2)

    def test_invalid_json_gets_one_repair_call(self):
        responses = iter(["not-json", '{"concepts": []}'])

        def transport(endpoint, payload, headers, timeout):
            return {"choices": [{"message": {"content": next(responses)}}]}

        client = mlr.QwenClient("fake-key", "https://example.invalid/v1", "fake-model", transport=transport)
        payload, _, _ = client.complete_json("system", "user")
        self.assertEqual(payload, {"concepts": []})

    def test_vision_error_falls_back_to_text(self):
        content_types = []

        def transport(endpoint, payload, headers, timeout):
            content = payload["messages"][1]["content"]
            content_types.append(type(content))
            if isinstance(content, list):
                raise mlr.ApiError("image input unsupported", status=400)
            return {"choices": [{"message": {"content": '{"formulas": [], "figures": []}'}}]}

        client = mlr.QwenClient("fake-key", "https://example.invalid/v1", "fake-model", transport=transport)
        payload, _, used_vision = client.complete_json("system", "user", images=[b"jpeg"], use_vision_model=True)
        self.assertEqual(payload["figures"], [])
        self.assertFalse(used_vision)
        self.assertEqual(content_types, [list, str])


class ParsingTests(unittest.TestCase):
    def test_concept_parser_filters_bad_pages_and_low_relevance(self):
        payload = {
            "concepts": [
                asdict(candidate(1)),
                {**asdict(candidate(2)), "evidence_page": 511},
                {**asdict(candidate(3)), "relevance": 0.2},
            ]
        }
        parsed = mlr.parse_concept_response(payload, mlr.CHAPTERS[0])
        self.assertEqual([item.canonical for item in parsed], ["useful machine learning concept 1"])

    def test_concept_parser_rejects_a_page_outside_its_chunk(self):
        payload = {"concepts": [asdict(candidate(1))]}
        parsed = mlr.parse_concept_response(payload, mlr.CHAPTERS[0], allowed_pages=[21, 22])
        self.assertEqual(parsed, [])

    def test_asset_parser_enforces_exact_context_window(self):
        chapter = mlr.CHAPTERS[0]
        payload = {
            "formulas": [
                {
                    "concept": "supervised learning",
                    "latex": "y=f(x)",
                    "pdf_page": 21,
                    "confidence": 0.95,
                },
                {
                    "concept": "supervised learning",
                    "latex": "z=g(x)",
                    "pdf_page": 30,
                    "confidence": 0.95,
                },
            ],
            "figures": [],
        }
        formulas, _ = mlr.parse_asset_response(
            payload, chapter, ["supervised learning"], allowed_pages=[20, 21]
        )
        self.assertEqual(len(formulas), 1)
        self.assertEqual(formulas[0].pdf_page, 21)


class OutputTests(unittest.TestCase):
    def test_rebuild_is_idempotent_and_formula_columns_are_first(self):
        result = chapter_result()
        result.formulas = [
            asdict(
                mlr.FormulaRecord(
                    concept=result.concepts[0]["canonical"],
                    latex=r"g(\mathbf{w})=\sum_p e_p^2",
                    pdf_page=20,
                    equation_label="(1.1)",
                    confidence=0.95,
                )
            )
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            state_dir = root / mlr.STATE_DIR_NAME
            missing_pdf = root / "not-needed.pdf"
            mlr.rebuild_outputs(root, missing_pdf, state_dir, [result])
            first = (root / "MLR_concepts.csv").read_bytes()
            mlr.rebuild_outputs(root, missing_pdf, state_dir, [result])
            second = (root / "MLR_concepts.csv").read_bytes()
            self.assertEqual(first, second)
            with (root / "MLR_formula.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0][:3], ["formula_id", "concept", "latex"])
            self.assertEqual(rows[1][0], "F0001")
            self.assertEqual(len(rows), 2)

    def test_global_duplicate_is_one_main_row_but_keeps_two_metadata_rows(self):
        result1 = chapter_result(1)
        result2 = chapter_result(2)
        result1.concepts = [
            asdict(candidate(1, mlr.CHAPTERS[0], canonical="neural network", aliases=["neural networks"]))
        ]
        result2.concepts = [
            asdict(candidate(1, mlr.CHAPTERS[1], canonical="neural networks", aliases=[]))
        ]
        concepts, metadata, _, _ = mlr.aggregate_results([result1, result2])
        self.assertEqual(len(concepts), 1)
        self.assertEqual(len(metadata), 2)

    def test_invalid_checkpoint_does_not_pass_validation(self):
        result = chapter_result(concept_count=19)
        with self.assertRaises(mlr.ExtractionError):
            mlr.validate_result(result)


class FigureCropTests(unittest.TestCase):
    @staticmethod
    def make_figure_pdf(path: Path) -> None:
        document = fitz.open()
        page = document.new_page(width=600, height=800)
        page.draw_rect(fitz.Rect(130, 160, 470, 380), color=(0, 0, 0), fill=(0.9, 0.9, 0.9))
        page.insert_text((140, 410), "Figure 1.1 A useful machine learning diagram.", fontsize=11)
        document.save(path)
        document.close()

    def test_vector_figure_crop_is_png_and_not_whole_page(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "figure.pdf"
            self.make_figure_pdf(pdf_path)
            with mlr.PdfBook(pdf_path, validate=False) as book:
                png = book.crop_figure_png(1, "Figure 1.1", "A useful machine learning diagram")
            self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertLess(len(png), 2_000_000)

    def test_same_crop_is_saved_once_for_multiple_concepts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf_path = root / "figure.pdf"
            self.make_figure_pdf(pdf_path)
            figures = [
                {
                    "concept": concept,
                    "chapter": 1,
                    "pdf_page": 1,
                    "book_page": -7,
                    "figure_label": "Figure 1.1",
                    "caption": "A useful machine learning diagram.",
                }
                for concept in ["concept alpha", "concept beta"]
            ]
            with mlr.PdfBook(pdf_path, validate=False) as book:
                rows = mlr.materialize_figures(
                    root, pdf_path, root / mlr.STATE_DIR_NAME, figures, book=book
                )
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0][1], rows[1][1])
            self.assertEqual(len(list((root / "MLR_graph").glob("*.png"))), 1)


class RealPdfTests(unittest.TestCase):
    @unittest.skipUnless((MLR_DIR / mlr.DEFAULT_PDF_NAME).exists(), "repository PDF is unavailable")
    def test_real_pdf_edition_and_boundaries_validate(self):
        with mlr.PdfBook(MLR_DIR / mlr.DEFAULT_PDF_NAME) as book:
            self.assertEqual(len(book.document), mlr.EXPECTED_PDF_PAGES)
            self.assertIn("Introduction", book.page_text(20, strip_margins=False))

    @unittest.skipUnless((MLR_DIR / mlr.DEFAULT_PDF_NAME).exists(), "repository PDF is unavailable")
    def test_real_pdf_evidence_grounding(self):
        grounded = candidate(
            1,
            canonical="machine learning",
            evidence="Machine learning is a unified algorithmic framework",
        )
        invented = candidate(
            2,
            canonical="invented concept",
            evidence="This sentence is absolutely absent from the textbook page",
        )
        with mlr.PdfBook(MLR_DIR / mlr.DEFAULT_PDF_NAME) as book:
            self.assertTrue(mlr.evidence_is_supported(book, grounded))
            self.assertFalse(mlr.evidence_is_supported(book, invented))


if __name__ == "__main__":
    unittest.main()
