from __future__ import annotations

import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import generate_prerequisites as pairwise
import generate_prerequisites_by_concept as conceptwise
import generate_profiles as profiles


def write_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerows(rows)


def make_data(
    count: int = 4,
    dataset: str = "MLR",
    root: Path | None = None,
) -> conceptwise.ConceptCentricData:
    concepts = [
        profiles.Concept(index, f"concept {index}", (f"concept {index}", f"alias {index}"))
        for index in range(count)
    ]
    pairs = pairwise.enumerate_pairs(concepts)
    dataset_root = root or Path(tempfile.gettempdir()) / "concept-centric-tests" / dataset
    spec = pairwise.DatasetSpec(
        dataset,
        dataset_root,
        f"{dataset}_concepts.csv",
        f"{dataset}_profiles.jsonl",
        f"{dataset}_prerequisite.csv",
    )
    return conceptwise.ConceptCentricData(
        spec=spec,
        concepts=concepts,
        pairs=pairs,
        tasks=conceptwise.group_anchor_tasks(pairs),
        input_hash=conceptwise._input_hash(spec, concepts, pairs),
    )


class GroupingAndPromptTests(unittest.TestCase):
    def test_every_unordered_pair_is_assigned_once_to_lower_id_anchor(self):
        data = make_data(4)
        self.assertEqual(
            [(task.anchor_id, task.candidate_ids, task.pair_ids) for task in data.tasks],
            [
                (0, (1, 2, 3), (0, 1, 2)),
                (1, (2, 3), (3, 4)),
                (2, (3,), (5,)),
            ],
        )
        all_pair_ids = [pair_id for task in data.tasks for pair_id in task.pair_ids]
        self.assertEqual(all_pair_ids, [pair.pair_id for pair in data.pairs])
        with self.assertRaises(conceptwise.PrerequisiteValidationError):
            conceptwise.group_anchor_tasks([pairwise.ConceptPair(0, 2, 1)])

    def test_prompt_contains_only_anchor_and_candidate_canonical_names(self):
        data = make_data(4, "MLR")
        prompt = conceptwise.build_user_prompt(data, data.tasks[0])
        combined = conceptwise.PREREQUISITE_BY_CONCEPT_SYSTEM_PROMPT + prompt
        self.assertIn("Dataset: MLR", prompt)
        self.assertIn('"concept_name": "concept 0"', prompt)
        self.assertIn('"concept_name": "concept 3"', prompt)
        self.assertNotIn("alias 0", prompt)
        self.assertNotIn('"aliases"', prompt)
        self.assertNotIn("profile", combined.lower())
        self.assertNotIn("DGL", prompt)

    def test_real_dataset_request_counts_match_full_pair_universes(self):
        mlr = conceptwise.load_dataset(pairwise.DATASET_SPECS["MLR"])
        dgl = conceptwise.load_dataset(pairwise.DATASET_SPECS["DGL"])
        self.assertEqual(
            (
                len(mlr.concepts),
                len(mlr.pairs),
                len(mlr.tasks),
            ),
            (285, 285 * 284 // 2, 284),
        )
        self.assertEqual(
            (
                len(dgl.concepts),
                len(dgl.pairs),
                len(dgl.tasks),
            ),
            (392, 392 * 391 // 2, 391),
        )


class ResponseValidationTests(unittest.TestCase):
    def setUp(self):
        self.data = make_data(3)
        self.task = self.data.tasks[0]

    def test_empty_and_bidirectional_anchor_relations_are_valid(self):
        self.assertEqual(conceptwise.validate_response({"relations": []}, self.task), [])
        value = {
            "relations": [
                {"source_concept_id": 2, "target_concept_id": 0},
                {"source_concept_id": 0, "target_concept_id": 1},
            ]
        }
        self.assertEqual(
            conceptwise.validate_response(value, self.task),
            [
                conceptwise.DirectedRelation(0, 1),
                conceptwise.DirectedRelation(2, 0),
            ],
        )

    def test_invalid_ids_loops_duplicates_conflicts_and_extra_fields_are_rejected(self):
        invalid_values = [
            {"relations": [], "explanation": "none"},
            {
                "relations": [
                    {"source_concept_id": 0, "target_concept_id": 1, "label": 1}
                ]
            },
            {"relations": [{"source_concept_id": 0, "target_concept_id": 99}]},
            {"relations": [{"source_concept_id": 0, "target_concept_id": 0}]},
            {"relations": [{"source_concept_id": 1, "target_concept_id": 2}]},
            {
                "relations": [
                    {"source_concept_id": 0, "target_concept_id": 1},
                    {"source_concept_id": 0, "target_concept_id": 1},
                ]
            },
            {
                "relations": [
                    {"source_concept_id": 0, "target_concept_id": 1},
                    {"source_concept_id": 1, "target_concept_id": 0},
                ]
            },
        ]
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(conceptwise.PrerequisiteValidationError):
                    conceptwise.validate_response(value, self.task)


class FakeClient:
    def __init__(self, values):
        self.values = list(values)
        self.calls = 0

    def complete_json(self, _system_prompt, _user_prompt):
        self.calls += 1
        value = self.values.pop(0)
        return profiles.Completion(
            value,
            json.dumps(value),
            conceptwise.PREREQUISITE_MODEL,
            False,
        )


class CheckpointAndPublishingTests(unittest.TestCase):
    def test_invalid_response_is_repaired_and_returned_in_candidate_order(self):
        data = make_data(3)
        task = data.tasks[0]
        client = FakeClient(
            [
                {"relations": [{"source_concept_id": 0, "target_concept_id": 99}]},
                {
                    "relations": [
                        {"source_concept_id": 2, "target_concept_id": 0},
                        {"source_concept_id": 0, "target_concept_id": 1},
                    ]
                },
            ]
        )
        relations, history, model = conceptwise.request_anchor_relations(client, data, task)
        self.assertEqual(
            relations,
            [conceptwise.DirectedRelation(0, 1), conceptwise.DirectedRelation(2, 0)],
        )
        self.assertEqual(len(history), 2)
        self.assertEqual(model, conceptwise.PREREQUISITE_MODEL)

    def test_checkpoint_is_strategy_local_and_input_changes_make_it_stale(self):
        data = make_data(3)
        task = data.tasks[0]
        relation = conceptwise.DirectedRelation(0, 1)
        value = conceptwise._checkpoint_value(
            data, task, [relation], conceptwise.PREREQUISITE_MODEL, []
        )
        self.assertTrue(conceptwise.valid_checkpoint(value, data, task))
        self.assertFalse(
            conceptwise.valid_checkpoint(value, replace(data, input_hash="changed"), task)
        )
        self.assertNotEqual(
            conceptwise.checkpoint_path(data.spec, task.anchor_id),
            pairwise.checkpoint_path(data.spec, task.anchor_id),
        )

    def test_publication_requires_every_anchor_and_appends_positive_edges_only(self):
        with tempfile.TemporaryDirectory() as temp_name:
            data = make_data(3, root=Path(temp_name) / "MLR")
            write_csv(data.spec.output_path, [["existing source", "existing target", "1"]])
            first_task, second_task = data.tasks
            first = conceptwise._checkpoint_value(
                data,
                first_task,
                [conceptwise.DirectedRelation(0, 1)],
                conceptwise.PREREQUISITE_MODEL,
                [],
            )
            profiles.atomic_write_json(
                conceptwise.checkpoint_path(data.spec, first_task.anchor_id), first
            )
            before = data.spec.output_path.read_bytes()
            self.assertFalse(conceptwise.publish_if_complete(data))
            self.assertEqual(data.spec.output_path.read_bytes(), before)

            second = conceptwise._checkpoint_value(
                data,
                second_task,
                [conceptwise.DirectedRelation(2, 1)],
                conceptwise.PREREQUISITE_MODEL,
                [],
            )
            profiles.atomic_write_json(
                conceptwise.checkpoint_path(data.spec, second_task.anchor_id), second
            )
            self.assertTrue(conceptwise.publish_if_complete(data))
            with data.spec.output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(
                rows,
                [
                    ["existing source", "existing target", "1"],
                    ["concept 0", "concept 1", "1"],
                    ["concept 2", "concept 1", "1"],
                ],
            )
            self.assertTrue(all(row[2] == "1" for row in rows))
            published = data.spec.output_path.read_bytes()
            self.assertTrue(conceptwise.publish_if_complete(data))
            self.assertEqual(data.spec.output_path.read_bytes(), published)

    def test_next_concept_resumes_and_force_regenerates_all_anchors(self):
        with tempfile.TemporaryDirectory() as temp_name:
            data = make_data(4, root=Path(temp_name) / "MLR")
            config = profiles.ApiConfig(
                "key", "url", conceptwise.PREREQUISITE_MODEL,
                conceptwise.PREREQUISITE_MODEL, 1, 1
            )
            requested: list[int] = []

            def fake_request(_client, _data, task):
                requested.append(task.anchor_id)
                return [], [], conceptwise.PREREQUISITE_MODEL

            with patch.object(profiles, "QwenClient", return_value=object()), patch.object(
                conceptwise, "request_anchor_relations", side_effect=fake_request
            ):
                self.assertFalse(
                    conceptwise.process_dataset(data, config, next_only=True)
                )
                self.assertEqual(requested, [0])
                self.assertTrue(conceptwise.process_dataset(data, config))
                self.assertEqual(requested, [0, 1, 2])
                self.assertTrue(conceptwise.process_dataset(data, config, force=True))
                self.assertEqual(requested, [0, 1, 2, 0, 1, 2])

    def test_dry_run_calls_no_qwen_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temp_name:
            data = make_data(3, root=Path(temp_name) / "MLR")
            config = profiles.ApiConfig(
                "", "url", conceptwise.PREREQUISITE_MODEL,
                conceptwise.PREREQUISITE_MODEL, 1, 1
            )
            with patch.object(
                profiles, "QwenClient", side_effect=AssertionError("Qwen called")
            ):
                self.assertFalse(conceptwise.process_dataset(data, config, dry_run=True))
            self.assertFalse(data.spec.output_path.exists())
            self.assertFalse(conceptwise.checkpoint_dir(data.spec).exists())


class CliTests(unittest.TestCase):
    def test_cli_is_dataset_specific_and_has_no_batch_or_worker_options(self):
        args = conceptwise.build_parser().parse_args(["--dataset", "MLR", "--dry-run"])
        self.assertEqual(args.dataset, "MLR")
        with self.assertRaises(SystemExit):
            conceptwise.build_parser().parse_args(["--dataset", "all"])
        with self.assertRaises(SystemExit):
            conceptwise.build_parser().parse_args(
                ["--dataset", "MLR", "--workers", "2"]
            )
        with self.assertRaises(SystemExit):
            conceptwise.build_parser().parse_args(
                ["--dataset", "MLR", "--batch-size", "20"]
            )
        with self.assertRaises(SystemExit):
            conceptwise.main(["--dataset", "MLR", "--next-concept", "--force"])


if __name__ == "__main__":
    unittest.main()
