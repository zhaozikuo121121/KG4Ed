from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stage2"))

from src.concept_profile import load_stage0_profiles


class Stage0ConceptProfileLoaderTests(unittest.TestCase):
    def _concepts(self):
        return pd.DataFrame(
            [
                {"concept_id": 0, "concept": "gradient descent"},
                {"concept_id": 1, "concept": "loss function"},
            ]
        )

    def test_loads_complete_stage0_profile_cache(self):
        tmp = Path(tempfile.mkdtemp(prefix="stage2_profiles_"))
        try:
            cache = tmp / "concept_profiles.jsonl"
            cache.write_text(
                "\n".join(
                    [
                        '{"concept_id": 0, "concept_name": "gradient descent", "profile": "Profile A.", "model": "fallback", "prompt_version": "profile_v3"}',
                        '{"concept_id": 1, "concept_name": "loss function", "profile": "Profile B.", "model": "fallback", "prompt_version": "profile_v3"}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            profiles = load_stage0_profiles(
                self._concepts(),
                {"stage0_outputs_dir": str(tmp), "profile": {"cache_path": str(cache), "prompt_version": "profile_v3"}},
            )
            self.assertEqual(profiles, {0: "Profile A.", 1: "Profile B."})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_missing_stage0_profile_cache_fails_without_generating(self):
        tmp = Path(tempfile.mkdtemp(prefix="stage2_profiles_missing_"))
        try:
            with self.assertRaises(FileNotFoundError):
                load_stage0_profiles(
                    self._concepts(),
                    {
                        "stage0_outputs_dir": str(tmp),
                        "profile": {"cache_path": str(tmp / "missing.jsonl"), "prompt_version": "profile_v3"},
                    },
                )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_prompt_version_mismatch_fails_fast(self):
        tmp = Path(tempfile.mkdtemp(prefix="stage2_profiles_version_"))
        try:
            cache = tmp / "concept_profiles.jsonl"
            cache.write_text(
                '{"concept_id": 0, "concept_name": "gradient descent", "profile": "Old.", "model": "fallback", "prompt_version": "profile_v2"}\n',
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                load_stage0_profiles(
                    self._concepts(),
                    {"stage0_outputs_dir": str(tmp), "profile": {"cache_path": str(cache), "prompt_version": "profile_v3"}},
                )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
