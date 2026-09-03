from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stage2"))

from src.data_loader import load_stage2_data
from src.utils import pair_set


class Stage2DataLoadingTests(unittest.TestCase):
    def test_loads_stage0_stage1_and_checks_no_heldout_train_overlap(self):
        data = load_stage2_data(ROOT / "stage0" / "outputs", ROOT / "stage1" / "outputs")
        self.assertEqual(data.concept_embeddings.shape[0], 244)
        self.assertGreater(data.concept_embeddings.shape[1], 0)
        self.assertEqual(data.adjacency.shape, (244, 244))
        self.assertGreater(len(data.train_labels), 0)
        self.assertEqual(pair_set(data.heldout_pairs), pair_set(data.val_labels) | pair_set(data.test_labels))
        self.assertFalse(pair_set(data.train_labels) & pair_set(data.heldout_pairs))
        self.assertTrue({"source_index", "target_index"}.issubset(data.train_labels.columns))


if __name__ == "__main__":
    unittest.main()
