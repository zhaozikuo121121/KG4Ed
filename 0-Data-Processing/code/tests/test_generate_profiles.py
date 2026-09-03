from __future__ import annotations

import base64
import csv
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


TEST_DIR = Path(__file__).resolve().parent
ROOT_DIR = TEST_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import generate_profiles as profiles


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUB"
    "AScY42YAAAAASUVORK5CYII="
)


def write_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerows(rows)


def make_spec(root: Path, name: str = "MLR") -> profiles.DatasetSpec:
    return profiles.DatasetSpec(
        name=name,
        root=root,
        concepts_name=f"{name}_concepts.csv",
        metadata_name=f"{name}_concepts_metadata.csv",
        formula_name=f"{name}_formula.csv",
        graph_dir_name=f"{name}_graph",
        output_name=f"{name}_profiles.jsonl",
    )


def make_dataset(root: Path, name: str = "MLR", concepts: list[list[str]] | None = None):
    spec = make_spec(root, name)
    concepts = concepts or [["message passing", "neighborhood aggregation"]]
    write_csv(spec.concepts_path, concepts)
    if name == "MLR":
        write_csv(
            spec.metadata_path,
            [
                ["concept", "chapter", "chapter_title", "pdf_page", "book_page", "evidence"],
                ["message passing", "1", "Graphs", "20", "12", "Neighbors exchange features."],
            ],
        )
        write_csv(
            spec.formula_path,
            [
                ["formula_id", "concept", "latex", "chapter", "pdf_page", "book_page", "equation_label"],
                ["F0001", "message passing", "h_i'=sum_j h_j", "1", "20", "12", "1.1"],
            ],
        )
        write_csv(
            spec.graph_index_path,
            [
                ["concept", "file_name", "chapter", "pdf_page", "book_page", "figure_label", "caption"],
                ["message passing", "message.png", "1", "20", "12", "Figure 1", "Neighbors send messages."],
            ],
        )
    else:
        write_csv(
            spec.metadata_path,
            [
                ["concept", "lecture", "pdf_file", "pdf_page", "page_title", "evidence", "source_type", "confidence", "bbox"],
                ["message passing", "1", "lecture.pdf", "1", "GNN", "Neighbors exchange features.", "printed", "0.95", "[]"],
            ],
        )
        write_csv(
            spec.formula_path,
            [
                ["formula_id", "concept", "latex", "lecture", "pdf_file", "pdf_page", "source_type", "confidence", "bbox", "label"],
                ["DGL-F0001", "message passing", "h_i'=sum_j h_j", "1", "lecture.pdf", "1", "printed", "0.95", "[]", "aggregation"],
            ],
        )
        write_csv(
            spec.graph_index_path,
            [
                ["visual_id", "concept", "file_name", "lecture", "pdf_file", "pdf_page", "kind", "description", "source_type", "confidence", "bbox"],
                ["DGL-G0001", "message passing", "message.png", "1", "lecture.pdf", "1", "diagram", "Neighbors send messages.", "printed", "0.95", "[]"],
            ],
        )
    (spec.graph_dir / "message.png").write_bytes(PNG_1X1)
    return spec


def profile_input(name: str = "message passing", *, with_image: bool = False):
    concept = profiles.Concept(0, name, (name,))
    visual = {
        "visual_id": "G1",
        "file_name": "g.png",
        "description": "A neighborhood aggregation diagram.",
        "image_available": True,
        "image_slot": 1,
    }
    return profiles.ProfileInput(
        dataset="DGL",
        concept=concept,
        evidence=[{"evidence": "Neighbors exchange and aggregate messages."}],
        formulas=[{"formula_id": "F1", "latex": "h_i'=sum_j h_j"}],
        visuals=[visual] if with_image else [],
        images=[profiles.ImagePayload("image/png", PNG_1X1)] if with_image else [],
        warnings=[],
        input_hash="abc123",
    )


def valid_response(name: str = "message passing") -> dict[str, object]:
    return {
        "profile": (
            f"{name}: A graph learning operation that exchanges information between connected "
            "nodes and aggregates neighboring features to update each node representation. It lets "
            "graph neural networks combine local structure with feature information for prediction."
        ),
        "used_formula_ids": [],
        "used_visual_ids": [],
        "warnings": [],
    }


class LoaderTests(unittest.TestCase):
    def test_aliases_include_canonical_first_and_unique_alias_resolves_resources(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            spec = make_dataset(root, concepts=[["message passing", "neighborhood aggregation"]])
            # Use the alias in all resource tables; the unique alias must map to the canonical row.
            for path in (spec.metadata_path, spec.formula_path, spec.graph_index_path):
                text = path.read_text(encoding="utf-8").replace("message passing", "neighborhood aggregation")
                path.write_text(text, encoding="utf-8")
            data = profiles.load_dataset(spec)

            self.assertEqual(data.concepts[0].aliases, ("message passing", "neighborhood aggregation"))
            self.assertEqual(len(data.evidence_by_id[0]), 1)
            self.assertEqual(len(data.formulas_by_id[0]), 1)
            self.assertEqual(len(data.visuals_by_id[0]), 1)
            prepared = profiles.build_profile_input(data, data.concepts[0], include_image_bytes=True)
            self.assertEqual(prepared.formulas[0]["formula_id"], "F0001")
            self.assertTrue(prepared.visuals[0]["visual_id"].startswith("MLR-G-"))
            self.assertEqual(len(prepared.images), 1)

    def test_duplicate_canonical_names_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_name:
            path = Path(temp_name) / "concepts.csv"
            write_csv(path, [["Graph Convolution"], ["graph-convolution"]])
            with self.assertRaises(profiles.DataValidationError):
                profiles.load_concepts(path)

    def test_dgl_ids_and_multiconcept_resources_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            spec = make_dataset(
                root,
                "DGL",
                concepts=[["message passing"], ["graph neural network", "gnn"]],
            )
            with spec.formula_path.open("a", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerow(
                    ["DGL-F0001", "gnn", "h_i'=sum_j h_j", "1", "lecture.pdf", "1", "printed", "0.9", "[]", "aggregation"]
                )
            with spec.graph_index_path.open("a", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerow(
                    ["DGL-G0001", "gnn", "message.png", "1", "lecture.pdf", "1", "diagram", "Neighbors send messages.", "printed", "0.9", "[]"]
                )
            data = profiles.load_dataset(spec)

            self.assertEqual(data.formulas_by_id[0][0]["formula_id"], "DGL-F0001")
            self.assertEqual(data.formulas_by_id[1][0]["formula_id"], "DGL-F0001")
            self.assertEqual(data.visuals_by_id[0][0]["visual_id"], "DGL-G0001")
            self.assertEqual(data.visuals_by_id[1][0]["visual_id"], "DGL-G0001")

    def test_resource_limits_and_missing_image_warning(self):
        with tempfile.TemporaryDirectory() as temp_name:
            spec = make_dataset(Path(temp_name))
            data = profiles.load_dataset(spec)
            data.evidence_by_id[0] *= 5
            data.formulas_by_id[0] = [
                {"formula_id": f"F{i}", "latex": f"x_{i}", "confidence": 1, "order": i}
                for i in range(5)
            ]
            data.visuals_by_id[0] = [
                {
                    "visual_id": f"G{i}",
                    "file_name": f"missing{i}.png",
                    "path": spec.graph_dir / f"missing{i}.png",
                    "description": "diagram",
                    "confidence": 1,
                    "order": i,
                }
                for i in range(4)
            ]
            prepared = profiles.build_profile_input(data, data.concepts[0], include_image_bytes=True)
            self.assertLessEqual(len(prepared.evidence), 3)
            self.assertEqual(len(prepared.formulas), 3)
            self.assertEqual(len(prepared.visuals), 2)
            self.assertEqual(len(prepared.images), 0)
            self.assertEqual(len(prepared.warnings), 2)


class JsonAndClientTests(unittest.TestCase):
    def test_fenced_json_is_parsed(self):
        self.assertEqual(profiles.parse_json_object("```json\n{\"ok\": true}\n```"), {"ok": True})

    def test_invalid_json_gets_one_repair_request(self):
        responses = [
            {"choices": [{"message": {"content": "not json"}}]},
            {"choices": [{"message": {"content": json.dumps(valid_response())}}]},
        ]

        def transport(*_args):
            return responses.pop(0)

        client = profiles.QwenClient(
            profiles.ApiConfig("key", "https://example.invalid/v1", "text", "vision", 5, 2),
            transport=transport,
            sleeper=lambda _seconds: None,
        )
        completion = client.complete_json("system", "user")
        self.assertEqual(completion.value["used_formula_ids"], [])
        self.assertEqual(len(responses), 0)

    def test_retry_and_vision_rejection_fall_back_to_text(self):
        payload_models: list[str] = []
        sleeps: list[float] = []
        calls = 0

        def transport(_endpoint, payload, _headers, _timeout):
            nonlocal calls
            calls += 1
            payload_models.append(payload["model"])
            if calls == 1:
                raise profiles.ApiError("busy", status=429)
            if calls == 2:
                raise profiles.ApiError("no image support", status=415)
            return {"choices": [{"message": {"content": json.dumps(valid_response())}}]}

        client = profiles.QwenClient(
            profiles.ApiConfig("key", "https://example.invalid/v1", "text", "vision", 5, 3),
            transport=transport,
            sleeper=sleeps.append,
        )
        completion = client.complete_json(
            "system", "user", images=[profiles.ImagePayload("image/png", PNG_1X1)]
        )
        self.assertEqual(payload_models, ["vision", "vision", "text"])
        self.assertEqual(sleeps, [1.0])
        self.assertFalse(completion.used_images)
        self.assertEqual(completion.model, "text")
        self.assertTrue(completion.warnings)


class ValidationAndGenerationTests(unittest.TestCase):
    def test_word_limit_invalid_ids_and_context_phrases_are_rejected(self):
        item = profile_input()
        within_tolerance = valid_response()
        within_tolerance["profile"] = "message passing: " + "word " * 151
        validated = profiles.validate_profile_response(within_tolerance, item)
        self.assertEqual(profiles.english_word_count(validated.profile), 153)

        too_long = valid_response()
        too_long["profile"] = "message passing: " + "word " * 201
        with self.assertRaises(profiles.ResponseValidationError):
            profiles.validate_profile_response(too_long, item)

        unknown = valid_response()
        unknown["used_formula_ids"] = ["DOES-NOT-EXIST"]
        with self.assertRaises(profiles.ResponseValidationError):
            profiles.validate_profile_response(unknown, item)

        contextual = valid_response()
        contextual["profile"] = "message passing: The attached image shows information flow."
        with self.assertRaises(profiles.ResponseValidationError):
            profiles.validate_profile_response(contextual, item)

    def test_overlong_profile_is_repaired_without_local_truncation(self):
        item = profile_input()
        responses = [
            profiles.Completion(
                {**valid_response(), "profile": "message passing: " + "word " * 201},
                "too long",
                "vision",
                False,
            ),
            profiles.Completion(valid_response(), "valid", "text", False),
        ]

        class FakeClient:
            def complete_json(self, *_args, **_kwargs):
                return responses.pop(0)

        checkpoint = profiles.generate_checkpoint(FakeClient(), item)  # type: ignore[arg-type]
        self.assertLessEqual(checkpoint["word_count"], 200)
        self.assertEqual(checkpoint["model"], "text")
        self.assertEqual(len(checkpoint["raw_responses"]), 2)

    def test_failed_generation_does_not_touch_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_name:
            checkpoint_path = Path(temp_name) / "checkpoint.json"
            checkpoint_path.write_text("old checkpoint", encoding="utf-8")

            class FailingClient:
                def complete_json(self, *_args, **_kwargs):
                    raise profiles.ApiError("network failed")

            with self.assertRaises(profiles.ApiError):
                profiles.generate_checkpoint(FailingClient(), profile_input())  # type: ignore[arg-type]
            self.assertEqual(checkpoint_path.read_text(encoding="utf-8"), "old checkpoint")


class CheckpointAndPublishingTests(unittest.TestCase):
    def make_checkpoint(self, prepared: profiles.ProfileInput) -> dict[str, object]:
        response = valid_response(prepared.concept.name)
        return {
            "version": profiles.PROGRAM_VERSION,
            "prompt_version": profiles.PROMPT_VERSION,
            "dataset": prepared.dataset,
            "input_hash": prepared.input_hash,
            "concept_id": prepared.concept.concept_id,
            "concept_name": prepared.concept.name,
            "aliases": list(prepared.concept.aliases),
            "profile": response["profile"],
            "model": "fake-qwen",
            "used_formula_ids": [],
            "used_visual_ids": [],
            "warnings": [],
        }

    def test_publish_requires_all_current_checkpoints_and_preserves_order(self):
        with tempfile.TemporaryDirectory() as temp_name:
            spec = make_dataset(
                Path(temp_name),
                concepts=[["message passing", "neighborhood aggregation"], ["graph convolution"]],
            )
            data = profiles.load_dataset(spec)
            spec.output_path.write_text("old output\n", encoding="utf-8")

            first = profiles.build_profile_input(data, data.concepts[0], include_image_bytes=False)
            profiles.atomic_write_json(
                profiles.checkpoint_path(spec, 0), self.make_checkpoint(first)
            )
            self.assertFalse(profiles.publish_if_complete(data))
            self.assertEqual(spec.output_path.read_text(encoding="utf-8"), "old output\n")

            second = profiles.build_profile_input(data, data.concepts[1], include_image_bytes=False)
            profiles.atomic_write_json(
                profiles.checkpoint_path(spec, 1), self.make_checkpoint(second)
            )
            self.assertTrue(profiles.publish_if_complete(data))
            rows = [json.loads(line) for line in spec.output_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["concept_id"] for row in rows], [0, 1])
            self.assertEqual(rows[0]["aliases"][0], "message passing")
            self.assertEqual(
                list(rows[0]),
                ["concept_id", "concept_name", "aliases", "profile", "model", "prompt_version"],
            )

    def test_resource_change_invalidates_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_name:
            spec = make_dataset(Path(temp_name))
            data = profiles.load_dataset(spec)
            before = profiles.build_profile_input(data, data.concepts[0], include_image_bytes=False)
            checkpoint = self.make_checkpoint(before)
            self.assertTrue(profiles.checkpoint_is_valid(checkpoint, before))

            spec.formula_path.write_text(
                spec.formula_path.read_text(encoding="utf-8").replace("sum_j h_j", "mean_j h_j"),
                encoding="utf-8",
            )
            changed_data = profiles.load_dataset(spec)
            after = profiles.build_profile_input(
                changed_data, changed_data.concepts[0], include_image_bytes=False
            )
            self.assertNotEqual(before.input_hash, after.input_hash)
            self.assertFalse(profiles.checkpoint_is_valid(checkpoint, after))

    def test_next_resumes_after_valid_checkpoint_and_force_selects_valid_rows(self):
        with tempfile.TemporaryDirectory() as temp_name:
            spec = make_dataset(
                Path(temp_name), concepts=[["message passing"], ["graph convolution"]]
            )
            data = profiles.load_dataset(spec)
            first = profiles.build_profile_input(data, data.concepts[0], include_image_bytes=False)
            profiles.atomic_write_json(
                profiles.checkpoint_path(spec, 0), self.make_checkpoint(first)
            )

            targets = profiles.determine_targets(
                [data], concept_id=None, force=False, next_only=True
            )
            self.assertEqual([(item.spec.name, concept.concept_id) for item, concept in targets], [("MLR", 1)])

            forced = profiles.determine_targets(
                [data], concept_id=0, force=True, next_only=False
            )
            self.assertEqual([concept.concept_id for _item, concept in forced], [0])

    def test_complete_publish_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temp_name:
            spec = make_dataset(Path(temp_name))
            data = profiles.load_dataset(spec)
            prepared = profiles.build_profile_input(data, data.concepts[0], include_image_bytes=False)
            profiles.atomic_write_json(
                profiles.checkpoint_path(spec, 0), self.make_checkpoint(prepared)
            )
            self.assertTrue(profiles.publish_if_complete(data))
            first_bytes = spec.output_path.read_bytes()
            self.assertTrue(profiles.publish_if_complete(data))
            self.assertEqual(spec.output_path.read_bytes(), first_bytes)


class ConfigurationAndRealDataTests(unittest.TestCase):
    def test_process_environment_overrides_project_env(self):
        with tempfile.TemporaryDirectory() as temp_name:
            spec = make_spec(Path(temp_name))
            with patch.object(
                profiles,
                "read_env_file",
                return_value={
                    "QWEN_API_KEY": "file-key",
                    "QWEN_MODEL": "file-model",
                    "QWEN_VISION_MODEL": "file-vision",
                },
            ), patch.dict(os.environ, {"QWEN_MODEL": "environment-model"}, clear=True):
                config = profiles.load_api_config(spec)
            self.assertEqual(config.api_key, "file-key")
            self.assertEqual(config.model, "environment-model")
            self.assertEqual(config.vision_model, "file-vision")

    def test_real_datasets_have_expected_concept_counts_and_complete_joins(self):
        mlr = profiles.load_dataset(profiles.DATASET_SPECS["MLR"])
        dgl = profiles.load_dataset(profiles.DATASET_SPECS["DGL"])
        self.assertEqual(len(mlr.concepts), 285)
        self.assertEqual(len(dgl.concepts), 392)
        self.assertEqual(mlr.unmatched_metadata + mlr.unmatched_formulas + mlr.unmatched_visuals, 0)
        self.assertEqual(dgl.unmatched_metadata + dgl.unmatched_formulas + dgl.unmatched_visuals, 0)


if __name__ == "__main__":
    unittest.main()
