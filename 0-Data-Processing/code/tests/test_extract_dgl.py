from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import fitz

TEST_DIR = Path(__file__).resolve().parent
CODE_DIR = TEST_DIR.parent
DGL_DIR = CODE_DIR.parent / "DGL"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import extract_dgl as dgl


def make_lecture_pdf(root: Path, lecture: dgl.Lecture, *, draw_visual: bool = False) -> Path:
    path = root / lecture.relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    for page_index in range(lecture.expected_pages):
        page = document.new_page(width=dgl.EXPECTED_PAGE_WIDTH, height=dgl.EXPECTED_PAGE_HEIGHT)
        page.insert_text((45, 30), "special thanks and https://example.invalid", fontsize=9)
        page.insert_text((60, 75), f"Graph learning body page {page_index + 1}", fontsize=14)
        page.insert_text((60, 110), "Message passing aggregates neighbor features.", fontsize=11)
        if draw_visual and page_index == 0:
            page.draw_rect(fitz.Rect(110, 150, 360, 340), color=(0, 0, 0), fill=(0.9, 0.9, 0.9))
        page.insert_text((60, 610), "BASIRA YouTube GitHub", fontsize=9)
    document.save(path)
    document.close()
    return path


def concept(name: str = "message passing", *, source: str = "printed", confidence: float = 0.95):
    return dgl.ConceptCandidate(
        canonical=name,
        aliases=[],
        evidence="Message passing aggregates neighbor features.",
        source_type=source,
        confidence=confidence,
        bbox=[20, 20, 300, 120],
    )


def page_result(
    *,
    lecture: int = 1,
    page: int = 1,
    concepts=None,
    formulas=None,
    visuals=None,
) -> dgl.PageResult:
    lecture_info = dgl.LECTURES[lecture - 1]
    return dgl.PageResult(
        version=dgl.PROGRAM_VERSION,
        lecture=lecture,
        pdf_file=lecture_info.pdf_name,
        pdf_page=page,
        page_title="Graph message passing",
        summary="This page presents graph message passing and neighborhood aggregation.",
        key_topics=["message passing", "neighborhood aggregation"],
        model="fake-vision-model",
        completed_at="2026-01-01T00:00:00+00:00",
        body_bbox=[38, 55, 774, 580],
        concepts=[asdict(item) for item in (concepts or [])],
        formulas=[asdict(item) for item in (formulas or [])],
        visuals=[asdict(item) for item in (visuals or [])],
    )


class ManifestAndGeometryTests(unittest.TestCase):
    def test_manifest_has_only_six_annotated_pdfs_and_23_pages(self):
        self.assertEqual(len(dgl.LECTURES), 6)
        self.assertEqual(sum(item.expected_pages for item in dgl.LECTURES), 23)
        self.assertTrue(all(item.pdf_name.endswith("_Annotated.pdf") for item in dgl.LECTURES))
        self.assertTrue(all("_Clean" not in item.pdf_name for item in dgl.LECTURES))

    def test_tile_coordinate_transform(self):
        full = dgl.NormalizedBox(0, 0, 1000, 1000)
        self.assertEqual(dgl.TILES[0].local_to_body(full).as_list(), [0.0, 0.0, 550.0, 550.0])
        self.assertEqual(dgl.TILES[3].local_to_body(full).as_list(), [450.0, 450.0, 1000.0, 1000.0])

    def test_body_crop_excludes_header_and_footer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            lecture = dgl.LECTURES[0]
            make_lecture_pdf(root, lecture)
            with dgl.AnnotatedLecturePdf(root, lecture) as pdf:
                rect = pdf.body_rect(1)
                self.assertAlmostEqual(rect.x0, 38.0, places=2)
                self.assertAlmostEqual(rect.y0, 55.0, places=2)
                self.assertAlmostEqual(rect.x1, 774.0, places=2)
                self.assertAlmostEqual(rect.y1, 580.0, places=2)
                text = pdf.body_text(1).casefold()
                self.assertIn("message passing", text)
                self.assertNotIn("special thanks", text)
                self.assertNotIn("youtube", text)

    def test_non_annotated_pdf_is_rejected(self):
        lecture = dgl.Lecture(1, "DGL_Lecture_1", "DGL_Lecture_1_Clean.pdf", 3)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            make_lecture_pdf(root, lecture)
            with self.assertRaises(dgl.ExtractionError):
                dgl.AnnotatedLecturePdf(root, lecture)


class ApiTests(unittest.TestCase):
    def test_images_are_mandatory(self):
        client = dgl.QwenVisionClient(
            "fake", "https://example.invalid/v1", "fake", "fake-vision", transport=lambda *args: {}
        )
        with self.assertRaises(dgl.ExtractionError):
            client.complete_json("system", "user", [])

    def test_vision_rejection_never_falls_back_to_text(self):
        calls = []

        def transport(endpoint, payload, headers, timeout):
            calls.append(payload)
            raise dgl.ApiError("image input is unsupported", status=400)

        client = dgl.QwenVisionClient(
            "fake", "https://example.invalid/v1", "fake", "fake-vision", transport=transport
        )
        with self.assertRaisesRegex(dgl.ExtractionError, "will not fall back"):
            client.complete_json("system", "user", [b"jpeg"])
        self.assertEqual(len(calls), 1)
        self.assertIsInstance(calls[0]["messages"][1]["content"], list)

    def test_429_is_retried(self):
        calls = []

        def transport(endpoint, payload, headers, timeout):
            calls.append(payload)
            if len(calls) == 1:
                raise dgl.ApiError("busy", status=429)
            return {"choices": [{"message": {"content": '{"ok": true}'}}]}

        client = dgl.QwenVisionClient(
            "fake",
            "https://example.invalid/v1",
            "fake",
            "fake-vision",
            transport=transport,
            sleeper=lambda _: None,
        )
        payload, _ = client.complete_json("system", "user", [b"jpeg"])
        self.assertTrue(payload["ok"])
        self.assertEqual(len(calls), 2)

    def test_invalid_json_gets_repair_request(self):
        responses = iter(["not-json", '{"ok": true}'])

        def transport(endpoint, payload, headers, timeout):
            return {"choices": [{"message": {"content": next(responses)}}]}

        client = dgl.QwenVisionClient(
            "fake", "https://example.invalid/v1", "fake", "fake-vision", transport=transport
        )
        payload, _ = client.complete_json("system", "user", [b"jpeg"])
        self.assertTrue(payload["ok"])


class ParsingTests(unittest.TestCase):
    def test_unknown_source_type_is_rejected_not_downgraded(self):
        raw = {
            "canonical": "message passing",
            "aliases": [],
            "evidence": "clear evidence",
            "source_type": "unknown",
            "confidence": 0.99,
            "bbox": [0, 0, 100, 100],
        }
        with self.assertRaises(ValueError):
            dgl.ConceptCandidate.from_mapping(raw)

    def test_low_confidence_handwriting_is_excluded(self):
        payload = {
            "page_title": "Graph learning",
            "summary": "This page introduces a graph learning method.",
            "key_topics": ["graph learning"],
            "concepts": [
                asdict(concept("message passing", source="handwritten", confidence=0.89)),
                asdict(concept("graph convolution", source="printed", confidence=0.80)),
            ],
            "formulas": [],
            "visuals": [],
            "uncertain_handwriting": [],
        }
        _, _, _, concepts, _, _, _, warnings = dgl.parse_page_response(payload)
        self.assertEqual([item.canonical for item in concepts], ["graph convolution"])
        self.assertTrue(any("low-confidence concept" in warning for warning in warnings))

    def test_zero_concept_page_is_valid(self):
        payload = {
            "page_title": "Lecture recap",
            "summary": "This page is a recap with no additional retained concepts.",
            "key_topics": [],
            "concepts": [],
            "formulas": [],
            "visuals": [],
            "uncertain_handwriting": [],
        }
        _, _, _, concepts, formulas, visuals, _, _ = dgl.parse_page_response(payload)
        self.assertEqual((concepts, formulas, visuals), ([], [], []))
        dgl.validate_page_result(page_result())

    def test_invalid_or_unlinked_formula_is_excluded(self):
        payload = {
            "page_title": "Message passing",
            "summary": "This page defines message passing.",
            "key_topics": ["message passing"],
            "concepts": [asdict(concept())],
            "formulas": [
                {
                    "latex": "H_{k+1} = ?",
                    "related_concepts": ["message passing"],
                    "source_type": "handwritten",
                    "confidence": 0.99,
                    "bbox": [100, 100, 300, 200],
                    "label": "",
                },
                {
                    "latex": "H_{k+1}=A H_k W_k",
                    "related_concepts": ["not retained"],
                    "source_type": "printed",
                    "confidence": 0.95,
                    "bbox": [100, 100, 300, 200],
                    "label": "",
                },
            ],
            "visuals": [],
            "uncertain_handwriting": [],
        }
        _, _, _, _, formulas, _, _, _ = dgl.parse_page_response(payload)
        self.assertEqual(formulas, [])

    def test_unsafe_visual_bbox_is_rejected(self):
        raw = {
            "kind": "diagram",
            "description": "Nearly the whole page",
            "related_concepts": ["message passing"],
            "source_type": "printed",
            "confidence": 0.99,
            "bbox": [0, 0, 1000, 1000],
        }
        with self.assertRaises(ValueError):
            dgl.VisualCandidate.from_mapping(raw)


class AggregationAndCheckpointTests(unittest.TestCase):
    def test_cross_page_concept_dedup(self):
        first = page_result(concepts=[concept("graph neural network")])
        second = page_result(page=2, concepts=[concept("graph neural networks")])
        aggregate = dgl.aggregate_results([first, second])
        self.assertEqual(len(aggregate["concept_rows"]), 1)
        self.assertEqual(len(aggregate["metadata_rows"]), 2)

    def test_formula_and_visual_many_to_many(self):
        concepts = [concept("message passing"), concept("graph convolution")]
        formula = dgl.FormulaCandidate(
            latex=r"H_{k+1}=A H_k W_k",
            related_concepts=["message passing", "graph convolution"],
            source_type="printed",
            confidence=0.95,
            bbox=[100, 100, 350, 200],
        )
        visual = dgl.VisualCandidate(
            kind="diagram",
            description="Message-passing architecture",
            related_concepts=["message passing", "graph convolution"],
            source_type="mixed",
            confidence=0.95,
            bbox=[90, 220, 450, 600],
        )
        result = page_result(concepts=concepts, formulas=[formula], visuals=[visual])
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            make_lecture_pdf(root, dgl.LECTURES[0], draw_visual=True)
            state_dir = root / dgl.STATE_DIR_NAME
            dgl.rebuild_outputs(root, state_dir, [result])
            with (root / "DGL_formula.csv").open(encoding="utf-8", newline="") as handle:
                formula_rows = list(csv.reader(handle))
            with (root / "DGL_graph" / "index.csv").open(encoding="utf-8", newline="") as handle:
                graph_rows = list(csv.reader(handle))
            self.assertEqual(formula_rows[0][:3], ["formula_id", "concept", "latex"])
            self.assertEqual(len(formula_rows), 3)
            self.assertEqual(formula_rows[1][0], formula_rows[2][0])
            self.assertEqual(len(graph_rows), 3)
            self.assertEqual(graph_rows[1][0], graph_rows[2][0])
            self.assertEqual(graph_rows[1][2], graph_rows[2][2])
            self.assertEqual(len(list((root / "DGL_graph").glob("*.png"))), 1)

    def test_commit_is_idempotent_and_next_resumes(self):
        result = page_result()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            state_dir = root / dgl.STATE_DIR_NAME
            dgl.commit_page(root, state_dir, result)
            first = (root / "DGL_page_summaries.csv").read_bytes()
            dgl.commit_page(root, state_dir, result)
            second = (root / "DGL_page_summaries.csv").read_bytes()
            self.assertEqual(first, second)
            self.assertEqual(dgl.next_incomplete_lecture(state_dir).number, 1)
            self.assertTrue(dgl.checkpoint_path(state_dir, 1, 1).exists())

    def test_invalid_replacement_does_not_overwrite_checkpoint(self):
        good = page_result()
        bad = page_result()
        bad.summary = ""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            state_dir = root / dgl.STATE_DIR_NAME
            dgl.commit_page(root, state_dir, good)
            checkpoint = dgl.checkpoint_path(state_dir, 1, 1)
            before = checkpoint.read_bytes()
            with self.assertRaises(dgl.ExtractionError):
                dgl.commit_page(root, state_dir, bad)
            self.assertEqual(before, checkpoint.read_bytes())


class EndToEndMockTests(unittest.TestCase):
    def test_one_page_uses_four_tiles_and_one_overview(self):
        class FakeClient:
            vision_model = "fake-vision"

            def __init__(self):
                self.calls = 0

            def complete_json(self, system_prompt, user_prompt, images):
                self.calls += 1
                self.assert_images(images)
                if self.calls <= 4:
                    payload = {
                        "concepts": [],
                        "formulas": [],
                        "visuals": [],
                        "uncertain_handwriting": [],
                    }
                else:
                    payload = {
                        "page_title": "Graph message passing",
                        "summary": "This page introduces graph message passing.",
                        "key_topics": ["message passing"],
                        "concepts": [asdict(concept())],
                        "formulas": [],
                        "visuals": [],
                        "uncertain_handwriting": [],
                    }
                return payload, json.dumps(payload)

            @staticmethod
            def assert_images(images):
                if len(images) != 1 or not images[0]:
                    raise AssertionError("expected one rendered image")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            lecture = dgl.LECTURES[0]
            make_lecture_pdf(root, lecture)
            client = FakeClient()
            with dgl.AnnotatedLecturePdf(root, lecture) as pdf:
                result = dgl.extract_page(pdf, client, 1, [], [])
            self.assertEqual(client.calls, 5)
            self.assertEqual(len(result.concepts), 1)
            self.assertEqual(len(result.raw_responses), 5)


class RealPdfTests(unittest.TestCase):
    @unittest.skipUnless(
        all((DGL_DIR / lecture.relative_path).exists() for lecture in dgl.LECTURES),
        "repository Annotated PDFs are unavailable",
    )
    def test_all_real_annotated_pdfs_validate(self):
        total_pages = 0
        for lecture in dgl.LECTURES:
            with dgl.AnnotatedLecturePdf(DGL_DIR, lecture) as pdf:
                total_pages += len(pdf.document)
                body = pdf.render_body_jpeg(1)
                self.assertTrue(body.startswith(b"\xff\xd8"))
                text = pdf.body_text(1).casefold()
                self.assertNotIn("special thanks", text)
        self.assertEqual(total_pages, 23)


if __name__ == "__main__":
    unittest.main()
