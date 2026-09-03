from __future__ import annotations

import csv
import io
import tempfile
import unittest
from pathlib import Path

import audit_and_revise_evidence as audit


class FakeMLRSource:
    def grounded(self, quote: str, page: int) -> bool:
        return quote == "A grounded explanation." and page in {10, 11}

    def page_meta(self, page: int) -> dict[str, str]:
        return {
            "chapter": "2",
            "chapter_title": "Test chapter",
            "pdf_page": str(page),
            "book_page": str(page - 8),
        }


class FakeBox:
    @classmethod
    def from_value(cls, value):
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ValueError("bad box")
        if not (0 <= value[0] < value[2] <= 1000 and 0 <= value[1] < value[3] <= 1000):
            raise ValueError("bad box")
        return cls()


class FakeDGLSource:
    class Mod:
        NormalizedBox = FakeBox

    mod = Mod()

    def grounded(self, quote: str, lecture: int, page: int, source_type: str) -> bool:
        return quote == "A grounded explanation." and (lecture, page) == (1, 1)

    def meta(self, lecture: int, page: int) -> dict[str, str]:
        return {
            "lecture": str(lecture),
            "pdf_file": "lecture.pdf",
            "pdf_page": str(page),
            "page_title": "Test page",
        }


class EvidenceHelpersTests(unittest.TestCase):
    def test_console_progress_reports_completion(self):
        stream = io.StringIO()
        progress = audit.ConsoleProgress("Test", 2, stream=stream)
        progress.start()
        progress.update(1, "working")
        progress.finish()
        self.assertIn("[Test] 2/2 (100%)", stream.getvalue())

    def test_append_uses_one_period_and_is_idempotent(self):
        value = audit.append_sentences("GCN.", "Graph convolutional networks aggregate neighboring features.")
        self.assertEqual(value, "GCN. Graph convolutional networks aggregate neighboring features")
        self.assertEqual(audit.append_sentences(value, "Graph convolutional networks aggregate neighboring features."), value)

    def test_union_bbox(self):
        self.assertEqual(audit.union_bbox("[10,20,30,40]", "[5,25,50,60]"), "[5.0,20.0,50.0,60.0]")

    def test_mlr_append_rejects_cross_page(self):
        row = {"evidence": "Mention", "pdf_page": "10"}
        decision = {"evidence": "A grounded explanation.", "confidence": 0.9, "target": {"page": 11}}
        self.assertFalse(audit.apply_candidate(row, decision, "MLR", FakeMLRSource(), "APPEND"))
        self.assertEqual(row["evidence"], "Mention")

    def test_mlr_migrate_updates_all_locator_fields(self):
        row = {"evidence": "Wrong", "chapter": "1", "chapter_title": "Old", "pdf_page": "10", "book_page": "2"}
        decision = {"evidence": "A grounded explanation.", "confidence": 0.9, "target": {"page": 11}}
        self.assertTrue(audit.apply_candidate(row, decision, "MLR", FakeMLRSource(), "MIGRATE"))
        self.assertEqual(row["pdf_page"], "11")
        self.assertEqual(row["book_page"], "3")
        self.assertEqual(row["chapter_title"], "Test chapter")

    def test_dgl_append_unions_bbox_and_marks_mixed(self):
        row = {
            "evidence": "GCN",
            "lecture": "1",
            "pdf_page": "1",
            "bbox": "[10,20,30,40]",
            "source_type": "printed",
            "confidence": "0.950",
        }
        decision = {
            "evidence": "A grounded explanation.",
            "confidence": 0.9,
            "target": {"page": 1, "lecture": 1, "bbox": [5, 25, 50, 60], "source_type": "handwritten"},
        }
        self.assertTrue(audit.apply_candidate(row, decision, "DGL", FakeDGLSource(), "APPEND"))
        self.assertEqual(row["bbox"], "[5.0,20.0,50.0,60.0]")
        self.assertEqual(row["source_type"], "mixed")
        self.assertEqual(row["confidence"], "0.900")

    def test_dgl_rejects_missing_new_bbox(self):
        row = {"evidence": "GCN", "lecture": "1", "pdf_page": "1", "bbox": "[10,20,30,40]", "source_type": "printed"}
        decision = {"evidence": "A grounded explanation.", "confidence": 0.9, "target": {"page": 1, "lecture": 1}}
        self.assertFalse(audit.apply_candidate(row, decision, "DGL", FakeDGLSource(), "REPLACE"))

    def test_dry_run_does_not_rewrite_v2(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "MLR"
            folder.mkdir()
            (folder / "MLR_concepts.csv").write_text("concept\n", encoding="utf-8")
            path = folder / "MLR_concepts_metadata_v2.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["concept", "chapter", "pdf_page", "evidence"])
                writer.writerow(["concept", "1", "10", "evidence"])
            before = path.read_bytes()
            audit.run_dataset(root, "MLR", apply=False, dry_run=True, client=None)
            self.assertEqual(path.read_bytes(), before)

    def test_reconcile_restores_baseline_and_deduplicates_v2(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "MLR"
            folder.mkdir()
            (folder / "MLR_concepts.csv").write_text("concept\nminimum\n", encoding="utf-8")
            headers = ["concept", "chapter", "chapter_title", "pdf_page", "book_page", "evidence"]
            original = [{"concept": "minimum", "chapter": "1", "chapter_title": "T", "pdf_page": "10", "book_page": "2", "evidence": "A complete definition of a minimum point is given here."}]
            revised = [dict(original[0], evidence="x") , dict(original[0], evidence="x")]
            for name, rows in [("MLR_concepts_metadata.csv", original), ("MLR_concepts_metadata_v2.csv", revised)]:
                with (folder / name).open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=headers)
                    writer.writeheader()
                    writer.writerows(rows)
            result = audit.reconcile_v2(root, "MLR")
            self.assertEqual(result["after"], 1)
            rows = list(csv.DictReader((folder / "MLR_concepts_metadata_v2.csv").open(encoding="utf-8-sig", newline="")))
            self.assertEqual(rows[0]["evidence"], original[0]["evidence"])


if __name__ == "__main__":
    unittest.main()
