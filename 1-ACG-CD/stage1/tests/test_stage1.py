from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stage1"))

from src.config import load_config
from src.negatives import make_hard_negatives, make_random_non_prerequisite_negatives, make_reverse_negatives
from src.data import concept_maps, load_stage0_concepts, parse_gold_positive_edges, parse_label_file_negative_edges
from src.pipeline import run_stage1


class Stage1DataTests(unittest.TestCase):
    def test_parse_gold_positive_edges_ignores_file_negatives(self):
        stage0_outputs = ROOT / "stage0" / "outputs"
        data_dir = ROOT / "data" / "MOOCML"
        concepts_df = load_stage0_concepts(stage0_outputs)
        name_to_id, _ = concept_maps(concepts_df)
        positives, stats = parse_gold_positive_edges(data_dir / "MOOCML_prerequisites.csv", name_to_id)
        self.assertEqual(len(positives), 1735)
        self.assertEqual(len(set(positives)), 1735)
        self.assertEqual(stats["label_0"], 4977)
        self.assertEqual(stats["ignored_file_negative_rows"], 4977)
        self.assertEqual(stats["invalid_rows"], 0)

    def test_parse_label_file_negative_edges_reads_moocml_dash_rows(self):
        stage0_outputs = ROOT / "stage0" / "outputs"
        data_dir = ROOT / "data" / "MOOCML"
        concepts_df = load_stage0_concepts(stage0_outputs)
        name_to_id, _ = concept_maps(concepts_df)
        negatives, stats = parse_label_file_negative_edges(data_dir / "MOOCML_prerequisites.csv", name_to_id)
        self.assertEqual(stats["label_file_negative_rows"], 4977)
        self.assertGreaterEqual(len(negatives), 2 * 300)
        self.assertEqual(len(negatives), len(set(negatives)))

    def test_zero_negative_counts_return_empty_lists(self):
        concepts_df = load_stage0_concepts(ROOT / "stage0" / "outputs")
        name_to_id, _ = concept_maps(concepts_df)
        positives, _ = parse_gold_positive_edges(ROOT / "data" / "MOOCML" / "MOOCML_prerequisites.csv", name_to_id)
        train_pos = positives[:5]
        gold_pos = set(positives)
        similarity = np.eye(len(concepts_df), dtype=float)
        self.assertEqual(make_reverse_negatives(train_pos, gold_pos, max_count=0), [])
        self.assertEqual(make_hard_negatives(train_pos, gold_pos, set(), similarity, max_count=0), [])
        self.assertEqual(make_random_non_prerequisite_negatives(len(concepts_df), 0, gold_pos, set(), seed=42), [])

    def test_default_pipeline_outputs_and_leakage_guard(self):
        cfg = load_config(ROOT / "stage1" / "config" / "config.yaml")
        cfg["stage0_outputs_dir"] = str(ROOT / "stage0" / "outputs")
        tmpdir = Path(tempfile.mkdtemp(prefix="stage1_test_"))
        try:
            cfg["output_dir"] = str(tmpdir)
            summary = run_stage1(cfg)
            counts = summary["split_counts"]
            train_count = int(round(counts["positive_edges_all"] * float(cfg["train_ratio"])))
            val_count = int(round(train_count * float(cfg["val_ratio_to_train"])))
            test_count = counts["positive_edges_all"] - train_count - val_count
            self.assertEqual(counts["positive_edges_all"], 1735)
            self.assertEqual(counts["train_pos"], train_count)
            self.assertEqual(counts["val_pos"], val_count)
            self.assertEqual(counts["test_pos"], test_count)
            self.assertEqual(counts["train_neg_reverse"], int(cfg["reverse_negative_count"]))
            self.assertEqual(counts["train_neg_hard"], int(cfg["hard_negative_count"]))
            expected_file_neg = int(round(float(cfg.get("label_file_negative_ratio_to_train", 1.5)) * train_count))
            self.assertEqual(counts["train_neg_file_random"], expected_file_neg)
            self.assertEqual(counts["val_neg"], val_count)
            self.assertEqual(counts["test_neg"], test_count)
            self.assertEqual(
                counts["train_labels_initial"],
                train_count + int(cfg["reverse_negative_count"]) + int(cfg["hard_negative_count"]) + expected_file_neg,
            )
            self.assertEqual(counts["val_labels"], 2 * val_count)
            self.assertEqual(counts["test_labels"], 2 * test_count)
            self.assertEqual(counts["heldout_pairs"], 2 * (val_count + test_count))
            self.assertTrue((tmpdir / "train_neg_file_random.csv").exists())
            self.assertEqual(summary["label_file_random_negative_stats"]["sampled"], expected_file_neg)

            train = pd.read_csv(tmpdir / "train_labels_initial.csv")
            file_random = pd.read_csv(tmpdir / "train_neg_file_random.csv")
            heldout = pd.read_csv(tmpdir / "heldout_pairs.csv")
            for df in [train, heldout]:
                self.assertIn("source_index", df.columns)
                self.assertIn("target_index", df.columns)
                self.assertTrue((df.source_id == df.source_index).all())
                self.assertTrue((df.target_id == df.target_index).all())
            self.assertEqual(set(file_random.negative_type.astype(str)), {"label_file_random"})
            train_pairs = set(zip(train.source_id, train.target_id))
            heldout_pairs = set(zip(heldout.source_id, heldout.target_id))
            self.assertFalse(train_pairs & heldout_pairs)
            reverse_pairs = set(
                zip(
                    train[train.negative_type == "reverse"].source_id,
                    train[train.negative_type == "reverse"].target_id,
                )
            )
            hard_pairs = set(
                zip(
                    train[train.negative_type == "hard"].source_id,
                    train[train.negative_type == "hard"].target_id,
                )
            )
            file_pairs = set(zip(file_random.source_id, file_random.target_id))
            self.assertFalse(file_pairs & reverse_pairs)
            self.assertFalse(file_pairs & hard_pairs)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_dataset_registry_points_to_new_csv_datasets(self):
        cfg = load_config(ROOT / "stage1" / "config" / "config.yaml")
        self.assertEqual(Path(cfg["data_dir"]).name, "MOOCML")
        self.assertEqual(cfg["label_file"], "MOOCML_prerequisites.csv")
        self.assertTrue(str(cfg["stage0_outputs_dir"]).endswith("outputs_moocml"))
        self.assertTrue(str(cfg["stage1_outputs_dir"]).endswith("outputs_moocml"))

        class Args:
            dataset_name = "lecturebank"
            train_ratio = None
            seed = None
            output_dir = None
            reverse_negative_count = None
            hard_negative_count = None

        from src.config import apply_cli_overrides

        cfg = apply_cli_overrides(cfg, Args())
        self.assertEqual(Path(cfg["data_dir"]).name, "LectureBank")
        self.assertEqual(cfg["label_file"], "LectureBank_prerequisites.csv")
        self.assertTrue(str(cfg["stage0_outputs_dir"]).endswith("outputs_lecturebank"))

    def test_moocml_label_file_only_config_disables_reverse_and_hard(self):
        cfg = load_config(ROOT / "stage1" / "config" / "config_moocml_label_file_only.yaml")
        cfg["stage0_outputs_dir"] = str(ROOT / "stage0" / "outputs")
        tmpdir = Path(tempfile.mkdtemp(prefix="stage1_file_only_test_"))
        try:
            cfg["output_dir"] = str(tmpdir)
            summary = run_stage1(cfg)
            counts = summary["split_counts"]
            train_count = int(round(counts["positive_edges_all"] * float(cfg["train_ratio"])))
            self.assertEqual(counts["train_pos"], train_count)
            self.assertEqual(counts["train_neg_reverse"], 0)
            self.assertEqual(counts["train_neg_hard"], 0)
            expected_file_neg = int(round(float(cfg.get("label_file_negative_ratio_to_train", 1.5)) * train_count))
            self.assertEqual(counts["train_neg_file_random"], expected_file_neg)
            self.assertEqual(counts["train_labels_initial"], train_count + expected_file_neg)
            self.assertEqual(summary["label_file_random_negative_stats"]["sampled"], expected_file_neg)

            train = pd.read_csv(tmpdir / "train_labels_initial.csv")
            heldout = pd.read_csv(tmpdir / "heldout_pairs.csv")
            self.assertEqual(set(train.negative_type.astype(str)), {"none", "label_file_random"})
            self.assertEqual(len(pd.read_csv(tmpdir / "train_neg_reverse.csv")), 0)
            self.assertEqual(len(pd.read_csv(tmpdir / "train_neg_hard.csv")), 0)
            train_pairs = set(zip(train.source_id, train.target_id))
            heldout_pairs = set(zip(heldout.source_id, heldout.target_id))
            self.assertFalse(train_pairs & heldout_pairs)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_moocml_label_file_negative_count_overrides_ratio(self):
        cfg = load_config(ROOT / "stage1" / "config" / "config_moocml_label_file_only.yaml")
        cfg["stage0_outputs_dir"] = str(ROOT / "stage0" / "outputs")
        cfg["label_file_negative_count"] = 123
        tmpdir = Path(tempfile.mkdtemp(prefix="stage1_file_count_test_"))
        try:
            cfg["output_dir"] = str(tmpdir)
            summary = run_stage1(cfg)
            counts = summary["split_counts"]
            self.assertEqual(counts["train_neg_file_random"], 123)
            self.assertEqual(summary["label_file_random_negative_stats"]["requested"], 123)
            self.assertEqual(summary["label_file_random_negative_stats"]["sampled"], 123)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

