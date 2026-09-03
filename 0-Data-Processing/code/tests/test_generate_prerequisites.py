from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


import generate_prerequisites as prereq
import generate_profiles as profiles


def write_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerows(rows)


def make_data(count: int = 4, dataset: str = "MLR") -> prereq.DatasetData:
    concepts = [
        profiles.Concept(index, f"concept {index}", (f"concept {index}", f"alias {index}"))
        for index in range(count)
    ]
    profiles_by_id = {
        index: f"concept {index}: A standalone technical concept profile."
        for index in range(count)
    }
    root = Path(tempfile.gettempdir()) / "prerequisite-tests"
    spec = prereq.DatasetSpec(dataset, root, f"{dataset}_concepts.csv", f"{dataset}_profiles.jsonl", f"{dataset}_prerequisite.csv")
    pairs = prereq.enumerate_pairs(concepts)
    return prereq.DatasetData(
        spec,
        concepts,
        profiles_by_id,
        pairs,
        prereq._input_hash(spec, concepts, profiles_by_id, pairs),
    )


class PairAndPromptTests(unittest.TestCase):
    def test_enumerates_all_unordered_pairs_in_stable_order(self):
        data = make_data(4)
        self.assertEqual(
            data.pairs,
            [
                prereq.ConceptPair(0, 0, 1),
                prereq.ConceptPair(1, 0, 2),
                prereq.ConceptPair(2, 0, 3),
                prereq.ConceptPair(3, 1, 2),
                prereq.ConceptPair(4, 1, 3),
                prereq.ConceptPair(5, 2, 3),
            ],
        )
        self.assertEqual(len(prereq.batched(data.pairs, 2)), 3)

    def test_prompt_deduplicates_concept_records_and_keeps_dataset_local_context(self):
        data = make_data(3, "MLR")
        prompt = prereq.build_user_prompt(data, [data.pairs[0], data.pairs[1]])
        payload_start = prompt.index("Concept records:")
        payload = prompt[payload_start:]
        self.assertIn('"concept_id": 0', payload)
        self.assertEqual(payload.count('"concept_id": 0'), 1)
        self.assertIn('"concept_name": "concept 0"', payload)
        self.assertNotIn('"aliases"', payload)
        self.assertNotIn('"profile"', payload)
        self.assertIn("Dataset: MLR", prompt)
        self.assertNotIn("DGL", prompt)
        self.assertNotIn("profile", (prereq.PREREQUISITE_SYSTEM_PROMPT + prompt).lower())

    def test_real_datasets_enumerate_all_concept_pairs(self):
        with patch.object(prereq, "load_profiles", side_effect=AssertionError("profiles loaded")):
            mlr = prereq.load_dataset(prereq.DATASET_SPECS["MLR"])
            dgl = prereq.load_dataset(prereq.DATASET_SPECS["DGL"])
        self.assertEqual(len(mlr.pairs), 285 * 284 // 2)
        self.assertEqual(len(dgl.pairs), 392 * 391 // 2)


class ValidationAndOutputTests(unittest.TestCase):
    def test_validation_requires_exact_pair_ids_and_labels(self):
        data = make_data(3)
        batch = data.pairs[:2]
        valid = {"decisions": [{"pair_id": 1, "label": -1}, {"pair_id": 0, "label": 0}]}
        self.assertEqual(
            prereq._validate_decisions(valid, batch),
            [prereq.PairDecision(0, 0), prereq.PairDecision(1, -1)],
        )
        with self.assertRaises(prereq.PrerequisiteValidationError):
            prereq._validate_decisions({"decisions": [{"pair_id": 0, "label": 2}]}, batch)
        with self.assertRaises(prereq.PrerequisiteValidationError):
            prereq._validate_decisions({"decisions": [{"pair_id": 0, "label": 0}]}, batch)
        with self.assertRaises(prereq.PrerequisiteValidationError):
            prereq._validate_decisions(
                {"decisions": [{"pair_id": 0, "label": 0}, {"pair_id": 0, "label": 1}]},
                batch,
            )

    def test_direction_conversion_and_deterministic_negative_sampling(self):
        data = make_data(4)
        decisions = {
            0: prereq.PairDecision(0, 1),
            1: prereq.PairDecision(1, -1),
            2: prereq.PairDecision(2, 0),
            3: prereq.PairDecision(3, 0),
            4: prereq.PairDecision(4, 0),
            5: prereq.PairDecision(5, 0),
        }
        rows = prereq._public_rows(data, decisions, negative_ratio=3, seed=42)
        self.assertEqual(rows[:2], [("concept 0", "concept 1", 1), ("concept 2", "concept 0", 1)])
        self.assertEqual(len(rows), 6)  # two positives plus all four 0 candidates are below the 3x limit
        self.assertEqual(rows, prereq._public_rows(data, decisions, negative_ratio=3, seed=42))

    def test_no_positive_case_is_capped(self):
        data = make_data(50)
        decisions = {pair.pair_id: prereq.PairDecision(pair.pair_id, 0) for pair in data.pairs}
        rows = prereq._public_rows(data, decisions, seed=42)
        self.assertEqual(len(rows), 1000)
        self.assertTrue(all(row[2] == 0 for row in rows))

    def test_append_preserves_existing_order_and_existing_semantics_win(self):
        data = make_data(3)
        with tempfile.TemporaryDirectory() as temp_name:
            path = Path(temp_name) / "prerequisite.csv"
            write_csv(path, [["alias 0", "concept 1", "1"]])
            incoming = [
                ("concept 0", "concept 1", 1),
                ("concept 1", "concept 0", 1),
                ("concept 0", "concept 2", 1),
            ]
            stats = prereq.append_public_rows(path, data.concepts, incoming)
            self.assertEqual(stats, prereq.AppendStats(1, 1, 1, 1))
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(
                rows,
                [
                    ["alias 0", "concept 1", "1"],
                    ["concept 0", "concept 2", "1"],
                ],
            )

            rerun = prereq.append_public_rows(path, data.concepts, incoming)
            self.assertEqual(rerun, prereq.AppendStats(2, 0, 2, 1))
            with path.open("r", encoding="utf-8", newline="") as handle:
                self.assertEqual(len(list(csv.reader(handle))), 2)


class FakeClient:
    def __init__(self, values):
        self.values = list(values)
        self.calls = 0

    def complete_json(self, _system_prompt, _user_prompt):
        self.calls += 1
        value = self.values.pop(0)
        return profiles.Completion(value, json.dumps(value), prereq.PREREQUISITE_MODEL, False)


class CheckpointTests(unittest.TestCase):
    def test_request_batch_decisions_repairs_schema_and_returns_batch_order(self):
        data = make_data(3)
        batch = data.pairs[:2]
        client = FakeClient(
            [
                {"decisions": [{"pair_id": batch[0].pair_id, "label": 3}]},
                {"decisions": [{"pair_id": batch[1].pair_id, "label": -1}, {"pair_id": batch[0].pair_id, "label": 0}]},
            ]
        )
        decisions, history, model = prereq.request_batch_decisions(client, data, batch)
        self.assertEqual(decisions, [prereq.PairDecision(0, 0), prereq.PairDecision(1, -1)])
        self.assertEqual(len(history), 2)
        self.assertEqual(model, prereq.PREREQUISITE_MODEL)

    def test_publish_is_atomic_in_practice_and_requires_all_batches(self):
        with tempfile.TemporaryDirectory() as temp_name:
            data = make_data(3)
            data.spec = prereq.DatasetSpec(
                "MLR", Path(temp_name), "MLR_concepts.csv", "MLR_profiles.jsonl", "MLR_prerequisite.csv"
            )
            batches = prereq.batched(data.pairs, 2)
            write_csv(data.spec.output_path, [["existing source", "existing target", "1"]])
            first = [prereq.PairDecision(item.pair_id, 1) for item in batches[0]]
            value = prereq._checkpoint_value(data, 0, batches[0], 2, first, prereq.PREREQUISITE_MODEL, [])
            profiles.atomic_write_json(prereq.checkpoint_path(data.spec, 0), value)
            self.assertFalse(prereq.publish_if_complete(data, batches, 2))
            self.assertEqual(
                data.spec.output_path.read_text(encoding="utf-8"),
                "existing source,existing target,1\n",
            )

            second = [prereq.PairDecision(item.pair_id, 0) for item in batches[1]]
            value = prereq._checkpoint_value(data, 1, batches[1], 2, second, prereq.PREREQUISITE_MODEL, [])
            profiles.atomic_write_json(prereq.checkpoint_path(data.spec, 1), value)
            self.assertTrue(prereq.publish_if_complete(data, batches, 2, negative_ratio=3))
            with data.spec.output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(
                rows,
                [
                    ["existing source", "existing target", "1"],
                    ["concept 0", "concept 1", "1"],
                    ["concept 0", "concept 2", "1"],
                    ["concept 1", "concept 2", "0"],
                ],
            )
            before = data.spec.output_path.read_bytes()
            self.assertTrue(prereq.publish_if_complete(data, batches, 2, negative_ratio=3))
            self.assertEqual(data.spec.output_path.read_bytes(), before)

    def test_load_profiles_rejects_missing_profile(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            concepts_path = root / "concepts.csv"
            profiles_path = root / "profiles.jsonl"
            concepts_path.write_text("a\nb\n", encoding="utf-8")
            profiles_path.write_text(
                json.dumps({"concept_id": 0, "concept_name": "a", "profile": "a: definition"}) + "\n",
                encoding="utf-8",
            )
            concepts = profiles.load_concepts(concepts_path)
            with self.assertRaises(prereq.PrerequisiteValidationError):
                prereq.load_profiles(profiles_path, concepts)


class CliTests(unittest.TestCase):
    def test_parser_rejects_workers(self):
        with self.assertRaises(SystemExit):
            prereq.build_parser().parse_args(["--dataset", "MLR", "--workers", "4"])

    def test_dataset_all_is_ordered_mlr_then_dgl(self):
        self.assertEqual([item.name for item in prereq.selected_specs("all")], ["MLR", "DGL"])

    def test_dataset_all_stops_before_dgl_when_mlr_is_incomplete(self):
        mlr = make_data(3, "MLR")
        dgl = make_data(3, "DGL")
        specs = [mlr.spec, dgl.spec]
        process_calls = []
        with patch.object(prereq, "selected_specs", return_value=specs), patch.object(
            prereq, "load_dataset", side_effect=[mlr, dgl]
        ), patch.object(
            prereq,
            "_model_config",
            return_value=profiles.ApiConfig("key", "url", prereq.PREREQUISITE_MODEL, prereq.PREREQUISITE_MODEL, 1, 1),
        ), patch.object(
            prereq,
            "process_dataset",
            side_effect=lambda data, *_args, **_kwargs: process_calls.append(data.spec.name) or False,
        ):
            self.assertEqual(prereq.main(["--dataset", "all"]), 1)
        self.assertEqual(process_calls, ["MLR"])

    def test_model_config_forces_plus_model(self):
        spec = prereq.DATASET_SPECS["MLR"]
        with patch.object(
            profiles,
            "load_api_config",
            return_value=profiles.ApiConfig("key", "url", "other", "other-vision", 1, 1),
        ):
            config = prereq._model_config(spec)
        self.assertEqual(config.model, prereq.PREREQUISITE_MODEL)
        self.assertEqual(config.vision_model, prereq.PREREQUISITE_MODEL)

    def test_dry_run_does_not_call_qwen_or_modify_output(self):
        with tempfile.TemporaryDirectory() as temp_name:
            data = make_data(3)
            data.spec = prereq.DatasetSpec(
                "MLR",
                Path(temp_name),
                "MLR_concepts.csv",
                "MLR_profiles.jsonl",
                "MLR_prerequisite.csv",
            )
            write_csv(data.spec.output_path, [["existing", "row", "1"]])
            before = data.spec.output_path.read_bytes()
            config = profiles.ApiConfig("", "url", prereq.PREREQUISITE_MODEL, prereq.PREREQUISITE_MODEL, 1, 1)
            with patch.object(profiles, "QwenClient", side_effect=AssertionError("Qwen called")):
                self.assertFalse(
                    prereq.process_dataset(data, config, batch_size=2, dry_run=True)
                )
            self.assertEqual(data.spec.output_path.read_bytes(), before)

if __name__ == "__main__":
    unittest.main()
