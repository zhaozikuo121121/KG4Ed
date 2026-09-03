from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, Sequence

import pandas as pd

from .data_io import Concept, normalize_text

PROFILE_COLUMNS = ["concept_id", "concept_name", "profile", "model", "prompt_version"]


def _chunks(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _extract_json_object(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def _call_qwen_json(prompt: str, llm_cfg: Dict, max_retries: int, backoff: float) -> dict:
    api_key_env = str(llm_cfg.get("api_key_env", "DASHSCOPE_API_KEY"))
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Qwen profile generation requires environment variable {api_key_env}.")
    base_url = str(llm_cfg.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")).rstrip("/")
    url = base_url + "/chat/completions"
    payload = {
        "model": str(llm_cfg.get("model", "qwen3.8-max")),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    last_error: Exception | None = None
    for attempt in range(max_retries):
        req = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=float(llm_cfg.get("timeout_seconds", 30))) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                content = body["choices"][0]["message"]["content"] or ""
                return _extract_json_object(content)
        except Exception as exc:
            last_error = exc
            if attempt + 1 < max_retries:
                time.sleep(backoff * (2 ** attempt))
    raise RuntimeError(f"Qwen JSON call failed after {max_retries} attempts: {last_error}")


def build_related_slide_texts(
    concepts: Sequence[Concept],
    caption_texts: Sequence[str],
    max_snippets: int = 3,
    max_chars: int = 500,
) -> Dict[int, list[str]]:
    normalized_captions = [(text, normalize_text(text)) for text in caption_texts]
    related: Dict[int, list[str]] = {}
    for concept in concepts:
        aliases = [normalize_text(alias) for alias in concept.aliases if alias.strip()]
        snippets: list[str] = []
        for original, normalized in normalized_captions:
            if any(alias and alias in normalized for alias in aliases):
                compact = " ".join(original.split())
                snippets.append(compact[:max_chars])
                if len(snippets) >= max_snippets:
                    break
        related[concept.concept_id] = snippets
    return related


def build_fallback_profile(row, max_words: int = 120) -> str:
    concept = str(row.concept)
    aliases = str(getattr(row, "aliases", ""))
    alias_list = [a.strip() for a in aliases.split("::;") if a.strip()]
    alias_text = ", ".join(alias_list[:3]) if alias_list else concept
    profile = (
        f"{concept} is a machine learning course concept. "
        f"It appears with related terms such as {alias_text} and is used to describe model ideas, data, optimization, or evaluation."
    )
    words = profile.split()
    if len(words) > max_words:
        profile = " ".join(words[:max_words]).rstrip(".,;:") + "."
    return profile


def _read_profile_cache(path: Path, prompt_version: str) -> Dict[int, dict]:
    cached: Dict[int, dict] = {}
    if not path.exists():
        return cached
    with path.open("r", encoding="utf-8") as f:
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
                raise ValueError(f"Invalid concept profile cache row at {path}:{line_no}") from exc
            if obj.get("profile"):
                cached[cid] = obj
    return cached


def _read_source_profiles(path: Path) -> Dict[int, dict]:
    cached: Dict[int, dict] = {}
    if not path.exists():
        return cached
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            try:
                cid = int(obj["concept_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid source concept profile row at {path}:{line_no}") from exc
            profile = str(obj.get("profile", "")).strip()
            if profile:
                cached[cid] = obj
    return cached


def _append_profiles(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _profile_prompt(batch: list[dict], min_words: int, max_words: int) -> str:
    return (
        "You are given machine learning concepts and related slide snippets where each concept appears.\n"
        "For each concept, write a concise educational explanation in the context of a machine learning course.\n\n"
        "Requirements:\n"
        "1. Explain what the concept means.\n"
        "2. Explain its role in machine learning.\n"
        "3. Mention key ideas, formulas, or related terms if necessary.\n"
        "4. Do not state whether this concept is a prerequisite of another concept.\n"
        "5. Do not list learning order or prerequisite relations.\n"
        f"6. Keep each explanation within {min_words}-{max_words} words.\n"
        "7. Return valid JSON only.\n\n"
        "Input concepts:\n"
        + json.dumps(batch, ensure_ascii=False, indent=2)
        + "\n\nOutput format:\n{\n  \"profiles\": [\n    {\"id\": \"...\", \"name\": \"...\", \"profile\": \"...\"}\n  ]\n}"
    )


def _validate_profile_response(payload: dict, expected_ids: set[str]) -> dict[str, str]:
    profiles = payload.get("profiles")
    if not isinstance(profiles, list) or len(profiles) != len(expected_ids):
        raise ValueError("Profile response must contain one profile per input concept.")
    result: dict[str, str] = {}
    for item in profiles:
        cid = str(item.get("id"))
        profile = str(item.get("profile", "")).strip()
        if cid not in expected_ids or not profile:
            raise ValueError(f"Invalid profile item: {item!r}")
        result[cid] = profile
    if set(result) != expected_ids:
        raise ValueError("Profile response IDs do not match input IDs.")
    return result


def _generate_llm_profiles(missing_rows: list, profile_cfg: Dict, llm_cfg: Dict) -> list[dict]:
    batch_size = int(profile_cfg.get("batch_size", llm_cfg.get("batch_size", 20)))
    min_words = int(profile_cfg.get("min_words", 80))
    max_words = int(profile_cfg.get("max_words", 120))
    prompt_version = str(profile_cfg.get("prompt_version", "profile_v3"))
    max_retries = int(llm_cfg.get("max_retries", 3))
    backoff = float(llm_cfg.get("retry_backoff_seconds", 2))
    model = str(llm_cfg.get("model", "qwen3.8-max"))
    generated: list[dict] = []
    for batch_rows in _chunks(missing_rows, batch_size):
        batch = [
            {
                "id": str(int(row.concept_id)),
                "name": str(row.concept),
                "slide_snippets": list(getattr(row, "slide_snippets", []) or []),
            }
            for row in batch_rows
        ]
        payload = _call_qwen_json(_profile_prompt(batch, min_words, max_words), llm_cfg, max_retries, backoff)
        profiles = _validate_profile_response(payload, {item["id"] for item in batch})
        for row in batch_rows:
            cid = int(row.concept_id)
            generated.append(
                {
                    "concept_id": cid,
                    "concept_name": str(row.concept),
                    "profile": profiles[str(cid)],
                    "model": model,
                    "prompt_version": prompt_version,
                }
            )
    return generated


def load_or_create_profiles(concepts: pd.DataFrame, cfg: Dict, force_rebuild: bool = False) -> Dict[int, str]:
    profile_cfg = dict(cfg.get("profile", {}) or {})
    llm_cfg = dict(cfg.get("llm", {}) or {})
    cache_path = Path(profile_cfg.get("cache_path", Path(cfg["stage0_outputs_dir"]) / "concept_profiles.jsonl"))
    prompt_version = str(profile_cfg.get("prompt_version", "profile_v3"))
    max_words = int(profile_cfg.get("max_words", 120))
    use_llm = bool(profile_cfg.get("use_llm_profiles", False))

    cached = {} if force_rebuild else _read_profile_cache(cache_path, prompt_version)
    concept_ids = {int(row.concept_id) for row in concepts.itertuples(index=False)}
    missing = [row for row in concepts.itertuples(index=False) if int(row.concept_id) not in cached]

    source_profile_file = cfg.get("profile_file")
    if missing and source_profile_file:
        source_path = Path(str(source_profile_file))
        if not source_path.is_absolute():
            source_path = Path(cfg["data_dir"]) / source_path
        source_profiles = _read_source_profiles(source_path)
        imported_rows: list[dict] = []
        for row in missing:
            cid = int(row.concept_id)
            source_obj = source_profiles.get(cid)
            if not source_obj:
                continue
            imported_rows.append(
                {
                    "concept_id": cid,
                    "concept_name": str(row.concept),
                    "aliases": [
                        alias.strip()
                        for alias in str(getattr(row, "aliases", "") or "").split("::;")
                        if alias.strip()
                    ],
                    "profile": str(source_obj["profile"]).strip(),
                    "model": str(source_obj.get("model", "source_profile_file")),
                    "prompt_version": prompt_version,
                    "source_prompt_version": str(source_obj.get("prompt_version", "")),
                    "source_profile_file": str(source_path),
                }
            )
        if imported_rows:
            _append_profiles(cache_path, imported_rows)
            for row in imported_rows:
                cached[int(row["concept_id"])] = row
            missing = [row for row in concepts.itertuples(index=False) if int(row.concept_id) not in cached]

    if missing:
        if use_llm:
            new_rows = _generate_llm_profiles(missing, profile_cfg, llm_cfg)
        else:
            new_rows = [
                {
                    "concept_id": int(row.concept_id),
                    "concept_name": str(row.concept),
                    "profile": build_fallback_profile(row, max_words=max_words),
                    "model": "fallback",
                    "prompt_version": prompt_version,
                }
                for row in missing
            ]
        _append_profiles(cache_path, new_rows)
        for row in new_rows:
            cached[int(row["concept_id"])] = row

    missing_after = concept_ids - set(cached)
    if missing_after:
        raise RuntimeError(f"Missing concept profiles after generation: {sorted(missing_after)[:10]}")
    return {cid: str(cached[cid]["profile"]) for cid in sorted(concept_ids)}
