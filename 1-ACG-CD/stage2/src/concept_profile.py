from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd


def load_stage0_profiles(concepts: pd.DataFrame, cfg: Dict) -> Dict[int, str]:
    profile_cfg = dict(cfg.get("profile", {}) or {})
    profile_path = Path(profile_cfg.get("cache_path", Path(cfg["stage0_outputs_dir"]) / "concept_profiles.jsonl"))
    prompt_version = str(profile_cfg.get("prompt_version", "profile_v3"))
    if not profile_path.exists():
        raise FileNotFoundError(
            f"Missing Stage0 concept profiles: {profile_path}. "
            "Run stage0/run_stage0.py first; Stage2 only reads profiles and never calls the profile LLM."
        )

    cached: Dict[int, dict] = {}
    with profile_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if str(obj.get("prompt_version")) != prompt_version:
                continue
            try:
                cid = int(obj["concept_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid concept profile cache row at {profile_path}:{line_no}") from exc
            profile = str(obj.get("profile", "")).strip()
            if profile:
                cached[cid] = obj

    concept_ids = {int(row.concept_id) for row in concepts.itertuples(index=False)}
    missing = concept_ids - set(cached)
    if missing:
        raise RuntimeError(
            f"Stage0 concept profiles are missing {len(missing)} concepts for prompt_version={prompt_version}: "
            f"{sorted(missing)[:10]}. Re-run Stage0 to generate the static profile cache."
        )
    return {cid: str(cached[cid]["profile"]) for cid in sorted(concept_ids)}
