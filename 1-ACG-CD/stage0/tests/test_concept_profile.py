from __future__ import annotations

import os
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stage0"))

from src.concept_profile import load_or_create_profiles


class ConceptProfileCacheTests(unittest.TestCase):
    def _concepts(self):
        return pd.DataFrame(
            [
                {"concept_id": 0, "concept": "gradient descent", "aliases": "gradient descent", "slide_snippets": []},
                {
                    "concept_id": 1,
                    "concept": "loss function",
                    "aliases": "loss function::;cost function",
                    "slide_snippets": [],
                },
                {"concept_id": 2, "concept": "neural network", "aliases": "neural network", "slide_snippets": []},
            ]
        )

    def test_fallback_profiles_write_and_reuse_jsonl_cache(self):
        tmp = Path(tempfile.mkdtemp(prefix="profiles_test_"))
        try:
            cfg = {
                "stage0_outputs_dir": str(tmp),
                "profile": {
                    "use_llm_profiles": False,
                    "cache_path": str(tmp / "concept_profiles.jsonl"),
                    "max_words": 120,
                    "prompt_version": "profile_v3",
                },
                "llm": {},
            }
            profiles = load_or_create_profiles(self._concepts(), cfg)
            self.assertEqual(set(profiles), {0, 1, 2})
            self.assertTrue(all(len(p.split()) <= 120 for p in profiles.values()))
            lines1 = (tmp / "concept_profiles.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines1), 3)
            profiles2 = load_or_create_profiles(self._concepts(), cfg)
            lines2 = (tmp / "concept_profiles.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(profiles, profiles2)
            self.assertEqual(lines1, lines2)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_partial_cache_only_appends_missing_concepts(self):
        tmp = Path(tempfile.mkdtemp(prefix="profiles_partial_"))
        try:
            cache = tmp / "concept_profiles.jsonl"
            cache.write_text(
                '{"concept_id": 0, "concept_name": "gradient descent", "profile": "Cached profile.", "model": "fallback", "prompt_version": "profile_v3"}\n',
                encoding="utf-8",
            )
            cfg = {
                "stage0_outputs_dir": str(tmp),
                "profile": {
                    "use_llm_profiles": False,
                    "cache_path": str(cache),
                    "max_words": 120,
                    "prompt_version": "profile_v3",
                },
                "llm": {},
            }
            profiles = load_or_create_profiles(self._concepts(), cfg)
            self.assertEqual(profiles[0], "Cached profile.")
            self.assertEqual(len(cache.read_text(encoding="utf-8").splitlines()), 3)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_dataset_profile_file_is_normalized_to_stage0_cache_prompt_version(self):
        tmp = Path(tempfile.mkdtemp(prefix="profiles_source_"))
        try:
            data_dir = tmp / "data"
            data_dir.mkdir()
            source_profile = data_dir / "Dataset_profiles.jsonl"
            with source_profile.open("w", encoding="utf-8") as f:
                for row in self._concepts().itertuples(index=False):
                    f.write(
                        json.dumps(
                            {
                                "concept_id": int(row.concept_id),
                                "concept_name": str(row.concept),
                                "profile": f"{row.concept} imported profile.",
                                "model": "qwen3.8-max",
                                "prompt_version": "concept_profile_qwen_v1",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            cache = tmp / "concept_profiles.jsonl"
            cfg = {
                "data_dir": str(data_dir),
                "profile_file": "Dataset_profiles.jsonl",
                "stage0_outputs_dir": str(tmp),
                "profile": {
                    "use_llm_profiles": True,
                    "cache_path": str(cache),
                    "prompt_version": "profile_v3",
                },
                "llm": {},
            }
            profiles = load_or_create_profiles(self._concepts(), cfg)
            self.assertEqual(set(profiles), {0, 1, 2})
            rows = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines()]
            self.assertEqual({row["prompt_version"] for row in rows}, {"profile_v3"})
            self.assertEqual({row["source_prompt_version"] for row in rows}, {"concept_profile_qwen_v1"})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @unittest.skipUnless(
        os.environ.get("RUN_REAL_LLM_TESTS") == "1" and os.environ.get("DASHSCOPE_API_KEY"),
        "Set RUN_REAL_LLM_TESTS=1 and DASHSCOPE_API_KEY to run real LLM smoke tests.",
    )
    def test_real_qwen_profile_generation_optional_smoke(self):
        tmp = Path(tempfile.mkdtemp(prefix="profiles_real_llm_"))
        try:
            concepts = pd.DataFrame(
                [
                    {
                        "concept_id": 999001,
                        "concept": "gradient descent",
                        "aliases": "gradient descent",
                        "slide_snippets": [
                            "Gradient descent updates parameters by moving opposite the gradient to reduce loss."
                        ],
                    }
                ]
            )
            cfg = {
                "stage0_outputs_dir": str(tmp),
                "profile": {
                    "use_llm_profiles": True,
                    "cache_path": str(tmp / "concept_profiles.jsonl"),
                    "batch_size": 1,
                    "min_words": 20,
                    "max_words": 60,
                    "prompt_version": "profile_v3",
                },
                "llm": {
                    "provider": "qwen",
                    "model": "qwen3.8-max",
                    "api_key_env": "DASHSCOPE_API_KEY",
                    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "timeout_seconds": 30,
                    "max_retries": 2,
                    "retry_backoff_seconds": 1,
                },
            }
            profiles = load_or_create_profiles(concepts, cfg)
            self.assertIn(999001, profiles)
            self.assertGreater(len(profiles[999001].split()), 5)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
