from __future__ import annotations

import sys
import unittest
import os
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stage2"))

from src.data_loader import load_stage2_data
import src.pseudo_labeling as pseudo_labeling_module
from src.pseudo_labeling import QwenLLMJudge, add_label_confidence, generate_pseudo_labels, resolve_pseudo_conflicts
from src.utils import PSEUDO_COLUMNS, pair_set


class PseudoLabelingTests(unittest.TestCase):
    def _pseudo_row(self, src, dst, label, combined, pseudo_type):
        return {
            "source_id": src,
            "target_id": dst,
            "source_index": src,
            "target_index": dst,
            "source": f"c{src}",
            "target": f"c{dst}",
            "label": label,
            "teacher_model": "test",
            "model_score": combined,
            "llm_score": "",
            "combined_score": combined,
            "label_confidence": combined if label == 1 else 1.0 - combined,
            "iteration": 0,
            "pseudo_type": pseudo_type,
        }

    def _label_row(self, src, dst, label, negative_type="none"):
        return {
            "source_id": src,
            "target_id": dst,
            "source_index": src,
            "target_index": dst,
            "source": f"c{src}",
            "target": f"c{dst}",
            "label": label,
            "split": "train",
            "negative_type": negative_type,
            "similarity": np.nan,
        }

    def test_conflict_resolution_uses_label_confidence_not_raw_combined_score(self):
        r_train = pd.DataFrame([self._label_row(0, 1, 1)])
        r_neg_dynamic = pd.DataFrame([self._label_row(3, 4, 0, "hard")])
        r_syn_pos = pd.DataFrame([self._pseudo_row(7, 8, 1, 0.91, "positive")])
        r_syn_neg = pd.DataFrame([self._pseudo_row(1, 2, 0, 0.05, "negative")])
        new_pseudo = pd.DataFrame(
            [
                self._pseudo_row(1, 2, 1, 0.70, "positive"),
                self._pseudo_row(7, 8, 0, 0.20, "negative"),
            ]
        )
        out_pos, out_neg, out_r_neg, metrics = resolve_pseudo_conflicts(
            new_pseudo, r_syn_pos, r_syn_neg, r_neg_dynamic, r_train
        )
        self.assertFalse(((out_pos.source_index == 1) & (out_pos.target_index == 2)).any())
        self.assertTrue(((out_neg.source_index == 1) & (out_neg.target_index == 2)).any())
        self.assertTrue(((out_pos.source_index == 7) & (out_pos.target_index == 8)).any())
        self.assertFalse(((out_neg.source_index == 7) & (out_neg.target_index == 8)).any())
        self.assertEqual(metrics["pos_vs_r_syn_neg_keep_existing"], 1)
        self.assertEqual(metrics["neg_vs_r_syn_pos_keep_existing"], 1)
        self.assertEqual(len(out_r_neg), 1)

    def test_label_confidence_is_recomputed_from_label_and_combined_score(self):
        stale = pd.DataFrame(
            [
                {**self._pseudo_row(1, 2, 1, 0.80, "positive"), "label_confidence": 0.01},
                {**self._pseudo_row(3, 4, 0, 0.10, "negative"), "label_confidence": 0.02},
            ]
        )
        out = add_label_confidence(stale)
        self.assertAlmostEqual(float(out.iloc[0].label_confidence), 0.80)
        self.assertAlmostEqual(float(out.iloc[1].label_confidence), 0.90)

    def test_same_label_duplicate_keeps_higher_label_confidence(self):
        new_pseudo = pd.DataFrame(
            [
                self._pseudo_row(4, 5, 1, 0.60, "positive"),
                self._pseudo_row(4, 5, 1, 0.92, "positive"),
                self._pseudo_row(6, 7, 0, 0.40, "negative"),
                self._pseudo_row(6, 7, 0, 0.05, "negative"),
            ]
        )
        out_pos, out_neg, _, _ = resolve_pseudo_conflicts(
            new_pseudo,
            pd.DataFrame(columns=PSEUDO_COLUMNS),
            pd.DataFrame(columns=PSEUDO_COLUMNS),
            pd.DataFrame(columns=["source_index", "target_index"]),
            pd.DataFrame(columns=["source_index", "target_index"]),
        )
        self.assertEqual(len(out_pos), 1)
        self.assertEqual(len(out_neg), 1)
        self.assertAlmostEqual(float(out_pos.iloc[0].combined_score), 0.92)
        self.assertAlmostEqual(float(out_neg.iloc[0].combined_score), 0.05)

    def test_new_positive_keeps_constructed_negative_but_never_gold_positive(self):
        r_train = pd.DataFrame([self._label_row(0, 1, 1)])
        r_neg_dynamic = pd.DataFrame([self._label_row(2, 3, 0, "hard")])
        new_pseudo = pd.DataFrame(
            [
                self._pseudo_row(2, 3, 1, 0.85, "positive"),
                self._pseudo_row(0, 1, 0, 0.01, "negative"),
            ]
        )
        out_pos, out_neg, out_r_neg, metrics = resolve_pseudo_conflicts(
            new_pseudo, pd.DataFrame(columns=PSEUDO_COLUMNS), pd.DataFrame(columns=PSEUDO_COLUMNS), r_neg_dynamic, r_train
        )
        self.assertTrue(((out_pos.source_index == 2) & (out_pos.target_index == 3)).any())
        self.assertFalse(((out_neg.source_index == 0) & (out_neg.target_index == 1)).any())
        self.assertTrue(((out_r_neg.source_index == 2) & (out_r_neg.target_index == 3)).any())
        self.assertEqual(len(out_r_neg), 1)
        self.assertEqual(metrics["stage1_neg_removed_by_new_pos"], 0)
        self.assertEqual(metrics["new_neg_dropped_against_gold_pos"], 1)

    def test_pseudo_generation_excludes_heldout_train_and_self_loops(self):
        data = load_stage2_data(ROOT / "stage0" / "outputs", ROOT / "stage1" / "outputs")
        cfg = {
            "seed": 42,
            "pseudo_pos_threshold": 0.8,
            "pseudo_neg_threshold": 0.2,
            "use_llm_judge": False,
        }
        def scorer(df):
            return np.linspace(0.0, 1.0, len(df), dtype=np.float32)
        pseudo = generate_pseudo_labels(data, scorer, "mock", 0, cfg)
        self.assertTrue(set(PSEUDO_COLUMNS).issubset(pseudo.columns))
        pairs = pair_set(pseudo)
        self.assertFalse(pairs & pair_set(data.heldout_pairs))
        self.assertFalse(pairs & pair_set(data.train_labels))
        self.assertFalse((pseudo.source_index == pseudo.target_index).any())
        self.assertTrue((pseudo.combined_score.astype(float) == pseudo.model_score.astype(float)).all())
        self.assertTrue((pseudo.llm_score.fillna("").astype(str) == "").all())
    def test_llm_judge_receives_only_top_bottom_candidates(self):
        data = load_stage2_data(ROOT / "stage0" / "outputs", ROOT / "stage1" / "outputs")
        cfg = {
            "seed": 42,
            "pseudo_pos_threshold": 0.8,
            "pseudo_neg_threshold": 0.2,
            "use_llm_judge": True,
            "llm": {"judge_weight": 0.5},
        }
        calls = []

        class RecordingJudge:
            def judge(self, rows):
                calls.append(len(rows))
                return rows["model_score"].astype(float).to_numpy()

            def combine(self, model_scores, llm_scores=None):
                return model_scores, [str(float(x)) for x in llm_scores]

        original = pseudo_labeling_module.build_llm_judge
        try:
            pseudo_labeling_module.build_llm_judge = lambda cfg: RecordingJudge()
            pseudo = generate_pseudo_labels(
                data,
                lambda df: np.linspace(0.0, 1.0, len(df), dtype=np.float32),
                "mock",
                0,
                cfg,
            )
        finally:
            pseudo_labeling_module.build_llm_judge = original
        self.assertEqual(calls, [5, 5])
        self.assertEqual(len(pseudo), 4)
        counts = pseudo["pseudo_type"].value_counts().to_dict()
        self.assertEqual(counts.get("positive", 0), 2)
        self.assertEqual(counts.get("negative", 0), 2)
        train_positive_count = int((data.train_labels["label"].astype(int) == 1).sum())
        self.assertLessEqual(counts.get("positive", 0), int(0.1 * train_positive_count))
        self.assertLessEqual(counts.get("negative", 0), int(0.1 * train_positive_count))

    def test_qwen_llm_judge_fails_fast_without_api_key(self):
        data = load_stage2_data(ROOT / "stage0" / "outputs", ROOT / "stage1" / "outputs")
        cfg = {
            "seed": 42,
            "pseudo_pos_threshold": 0.8,
            "pseudo_neg_threshold": 0.2,
            "use_llm_judge": True,
            "llm": {
                "provider": "qwen",
                "model": "qwen3.8-max",
                "api_key_env": "DEFINITELY_MISSING_DASHSCOPE_KEY_FOR_TEST",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "timeout_seconds": 30,
                "judge_weight": 0.5,
            },
        }
        with self.assertRaises(RuntimeError) as ctx:
            generate_pseudo_labels(data, lambda df: np.linspace(0, 1, len(df), dtype=np.float32), "mock", 0, cfg)
        self.assertIn("DEFINITELY_MISSING_DASHSCOPE_KEY_FOR_TEST", str(ctx.exception))

    def test_qwen_llm_judge_uses_short_local_ids_and_retries_bad_ids(self):
        tmp = Path(tempfile.mkdtemp(prefix="qwen_judge_ids_"))
        old_key = os.environ.get("TEST_QWEN_KEY")
        os.environ["TEST_QWEN_KEY"] = "dummy"
        try:
            class FlakyJudge(QwenLLMJudge):
                def __init__(self, cfg):
                    super().__init__(cfg)
                    self.prompts = []

                def _call_qwen_json(self, prompt: str) -> dict:
                    self.prompts.append(prompt)
                    if len(self.prompts) == 1:
                        return {"results": [{"id": "pair_98_95", "score": 0.1}]}
                    payload = json.loads(prompt.split("Input:\n", 1)[1].split("\n\nOutput format:", 1)[0])
                    return {"results": [{"id": item["id"], "score": 0.7} for item in payload["pairs"]]}

            judge = FlakyJudge(
                {
                    "provider": "qwen",
                    "model": "qwen3.8-max",
                    "api_key_env": "TEST_QWEN_KEY",
                    "cache_path": str(tmp / "cache.jsonl"),
                    "batch_size": 2,
                    "max_retries": 2,
                    "retry_backoff_seconds": 0,
                }
            )
            rows = pd.DataFrame(
                [
                    {"source_index": 98, "target_index": 95, "source": "A", "target": "B"},
                    {"source_index": 12, "target_index": 34, "source": "C", "target": "D"},
                ]
            )
            scores = judge.judge(rows)
            self.assertTrue(np.allclose(scores, [0.7, 0.7]))
            self.assertIn('"id": "p000"', judge.prompts[0])
            self.assertIn('"id": "p001"', judge.prompts[0])
            self.assertNotIn('"id": "pair_98_95"', judge.prompts[0])
        finally:
            if old_key is None:
                os.environ.pop("TEST_QWEN_KEY", None)
            else:
                os.environ["TEST_QWEN_KEY"] = old_key
            shutil.rmtree(tmp, ignore_errors=True)

    @unittest.skipUnless(
        os.environ.get("RUN_REAL_LLM_TESTS") == "1" and os.environ.get("DASHSCOPE_API_KEY"),
        "Set RUN_REAL_LLM_TESTS=1 and DASHSCOPE_API_KEY to run real LLM smoke tests.",
    )
    def test_real_qwen_llm_judge_optional_smoke(self):
        data = load_stage2_data(ROOT / "stage0" / "outputs", ROOT / "stage1" / "outputs")
        cfg = {
            "seed": 42,
            "pseudo_pos_threshold": 0.8,
            "pseudo_neg_threshold": 0.2,
            "use_llm_judge": True,
            "llm": {
                "provider": "qwen",
                "model": "qwen3.8-max",
                "api_key_env": "DASHSCOPE_API_KEY",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "timeout_seconds": 30,
                "batch_size": 2,
                "judge_weight": 0.5,
            },
        }
        pseudo = generate_pseudo_labels(data, lambda df: np.linspace(0, 1, len(df), dtype=np.float32), "real_llm", 0, cfg)
        self.assertTrue(set(PSEUDO_COLUMNS).issubset(pseudo.columns))


if __name__ == "__main__":
    unittest.main()

