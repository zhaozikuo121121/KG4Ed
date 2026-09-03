from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import generate_profiles_v2 as profiles


def write_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerows(rows)


def make_fixture(root: Path, name: str = "MLR") -> profiles.DatasetSpec:
    dataset_root = root / name
    spec = profiles.DatasetSpec(
        name=name,
        root=dataset_root,
        concepts_name=f"{name}_concepts.csv",
        metadata_name=f"{name}_concepts_metadata_v2.csv",
        output_name=f"{name}_profiles_v2.jsonl",
    )
    write_csv(spec.concepts_path, [["message passing", "neighborhood aggregation"], ["graph learning"]])
    if name == "MLR":
        write_csv(
            spec.metadata_path,
            [
                ["concept", "chapter", "chapter_title", "pdf_page", "book_page", "evidence"],
                ["message passing", "1", "Graphs", "20", "12", "Neighbors exchange features."],
                ["message passing", "1", "Graphs", "20", "12", "The operation aggregates local information."],
                ["graph learning", "2", "Learning", "30", "22", "Models learn from graph-structured data."],
            ],
        )
    else:
        write_csv(
            spec.metadata_path,
            [
                ["concept", "lecture", "pdf_file", "pdf_page", "page_title", "evidence", "source_type", "confidence", "bbox"],
                ["message passing", "1", "lecture.pdf", "1", "GNN", "Neighbors exchange features.", "printed", "0.95", "[1,2,3,4]"],
                ["graph learning", "2", "lecture2.pdf", "2", "Graphs", "Models learn from graph data.", "printed", "0.90", "[5,6,7,8]"],
            ],
        )
    return spec


def profile_input_for(data: profiles.DatasetData, concept_id: int = 0) -> profiles.ProfileInput:
    return profiles.build_profile_input(data, data.concepts[concept_id])


class LoaderTests(unittest.TestCase):
    def test_v2_loader_keeps_all_rows_and_source_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            spec = make_fixture(Path(temp), "DGL")
            data = profiles.load_dataset(spec)
            self.assertEqual(len(data.concepts), 2)
            self.assertEqual(len(data.evidence_by_id[0]), 1)
            self.assertEqual(data.evidence_by_id[0][0]["source_type"], "printed")
            self.assertEqual(data.evidence_by_id[0][0]["bbox"], "[1,2,3,4]")
            self.assertEqual(data.evidence_by_id[0][0]["evidence_id"], "DGL-E0001")

    def test_real_datasets_have_complete_v2_joins(self):
        mlr = profiles.load_dataset(profiles.DATASET_SPECS["MLR"])
        dgl = profiles.load_dataset(profiles.DATASET_SPECS["DGL"])
        self.assertEqual(len(mlr.concepts), 285)
        self.assertEqual(len(dgl.concepts), 392)
        self.assertEqual(sum(len(v) for v in mlr.evidence_by_id.values()), 371)
        self.assertEqual(sum(len(v) for v in dgl.evidence_by_id.values()), 601)


class ValidationTests(unittest.TestCase):
    def test_draft_rejects_unknown_ids_and_multiple_background_claims(self):
        with tempfile.TemporaryDirectory() as temp:
            data = profiles.load_dataset(make_fixture(Path(temp)))
            item = profile_input_for(data)
            base = {
                "profile": "message passing: Neighbors exchange features.",
                "claims": [
                    {"claim": "Neighbors exchange features.", "evidence_ids": ["MLR-E0001"], "classification": "source_supported"}
                ],
                "warnings": [],
            }
            self.assertIsNotNone(profiles.validate_draft(base, item))
            bad = json.loads(json.dumps(base))
            bad["claims"][0]["evidence_ids"] = ["MLR-E9999"]
            with self.assertRaises(profiles.ResponseValidationError):
                profiles.validate_draft(bad, item)
            bad = json.loads(json.dumps(base))
            bad["claims"].extend(
                [
                    {"claim": "It is useful.", "evidence_ids": [], "classification": "background_context"},
                    {"claim": "It is common.", "evidence_ids": [], "classification": "background_context"},
                ]
            )
            with self.assertRaises(profiles.ResponseValidationError):
                profiles.validate_draft(bad, item)

    def test_review_requires_exact_sentence_reviews_and_one_background(self):
        with tempfile.TemporaryDirectory() as temp:
            data = profiles.load_dataset(make_fixture(Path(temp)))
            item = profile_input_for(data)
            profile = "message passing: Neighbors exchange features. It is useful."
            value = {
                "final_profile": profile,
                "sentence_reviews": [
                    {"sentence": "message passing: Neighbors exchange features.", "verdict": "source_supported", "evidence_ids": ["MLR-E0001"], "reason": "definition"},
                    {"sentence": "It is useful.", "verdict": "allowed_background", "evidence_ids": [], "reason": "brief bridge"},
                ],
                "warnings": [],
            }
            self.assertEqual(profiles.validate_review(value, item).final_profile, profile)
            value["sentence_reviews"][1]["sentence"] = "wrong sentence."
            with self.assertRaises(profiles.ResponseValidationError):
                profiles.validate_review(value, item)


class ClientAndCheckpointTests(unittest.TestCase):
    def test_progress_reports_skip_count_and_completion(self):
        stream = StringIO()
        progress = profiles.ProgressReporter(3, completed=1, skipped=1, stream=stream)
        progress.render("ready")
        progress.checkpointed("generated")
        self.assertIn("1/3", stream.getvalue())
        self.assertIn("skipped 1", stream.getvalue())
        self.assertIn("2/3", stream.getvalue())

    def test_two_stage_generation_returns_reviewed_profile(self):
        with tempfile.TemporaryDirectory() as temp:
            data = profiles.load_dataset(make_fixture(Path(temp)))
            item = profile_input_for(data)
            draft = {
                "profile": "message passing: Neighbors exchange features.",
                "claims": [
                    {"claim": "Neighbors exchange features.", "evidence_ids": ["MLR-E0001"], "classification": "source_supported"}
                ],
                "warnings": [],
            }
            review = {
                "final_profile": draft["profile"],
                "sentence_reviews": [
                    {"sentence": draft["profile"], "verdict": "source_supported", "evidence_ids": ["MLR-E0001"], "reason": "direct metadata definition"}
                ],
                "warnings": [],
            }
            responses = [
                {"choices": [{"message": {"content": json.dumps(draft)}}]},
                {"choices": [{"message": {"content": json.dumps(review)}}]},
            ]

            def transport(*_args):
                return responses.pop(0)

            config = profiles.ApiConfig("key", "https://example.test/v1", "deepseek-v4-flash", 3, 1)
            client = profiles.DeepSeekClient(config, transport=transport, sleeper=lambda _x: None)
            checkpoint = profiles.generate_checkpoint(client, item)
            self.assertEqual(checkpoint["profile"], review["final_profile"])
            self.assertEqual(len(checkpoint["raw_responses"]), 2)

    def test_client_repairs_invalid_json_once(self):
        responses = [
            {"choices": [{"message": {"content": "not json"}}]},
            {"choices": [{"message": {"content": '{"ok": true}'}}]},
        ]

        def transport(*_args):
            return responses.pop(0)

        config = profiles.ApiConfig("key", "https://example.test/v1", "deepseek-v4-flash", 3, 1)
        client = profiles.DeepSeekClient(config, transport=transport, sleeper=lambda _x: None)
        self.assertEqual(client.complete_json("system", "user").value, {"ok": True})
        self.assertEqual(responses, [])

    def test_publish_is_atomic_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temp:
            spec = make_fixture(Path(temp))
            data = profiles.load_dataset(spec)
            for concept in data.concepts:
                item = profiles.build_profile_input(data, concept)
                profile = f"{concept.name}: {item.evidence[0]['evidence']}"
                review = {
                    "final_profile": profile,
                    "sentence_reviews": [
                        {"sentence": profile, "verdict": "source_supported", "evidence_ids": [item.evidence[0]["evidence_id"]], "reason": "source"}
                    ],
                    "warnings": [],
                }
                checkpoint = {
                    "version": profiles.PROGRAM_VERSION,
                    "prompt_version": profiles.PROMPT_VERSION,
                    "dataset": "MLR",
                    "input_hash": item.input_hash,
                    "concept_id": concept.concept_id,
                    "concept_name": concept.name,
                    "model": "deepseek-v4-flash",
                    "profile": profile,
                    "review": review,
                }
                profiles.atomic_write_json(profiles.checkpoint_path(spec, concept.concept_id), checkpoint)
            self.assertTrue(profiles.publish_if_complete(data))
            first = spec.output_path.read_bytes()
            self.assertTrue(profiles.publish_if_complete(data))
            self.assertEqual(first, spec.output_path.read_bytes())
            rows = [json.loads(line) for line in spec.output_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["concept_id"] for row in rows], [0, 1])
            self.assertEqual(list(rows[0]), ["concept_id", "concept_name", "aliases", "profile", "model", "prompt_version"])

    def test_root_env_overrides_file_values(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / ".env").write_text("DEEPSEEK_MODEL=file-model\nDEEPSEEK_TIMEOUT_SECONDS=12\n", encoding="utf-8")
            with patch.dict(os.environ, {"DEEPSEEK_MODEL": "environment-model"}, clear=True):
                config = profiles.load_api_config(root)
            self.assertEqual(config.model, "environment-model")
            self.assertEqual(config.timeout, 12)


if __name__ == "__main__":
    unittest.main()
