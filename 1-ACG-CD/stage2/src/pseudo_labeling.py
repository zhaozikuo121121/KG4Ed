from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Callable, Dict, Sequence, Set

import numpy as np
import pandas as pd

from .utils import PSEUDO_COLUMNS, pair_set


class NoLLMJudge:
    """Offline judge used only when use_llm_judge=false."""

    def combine(self, model_scores: np.ndarray, llm_scores: np.ndarray | None = None) -> tuple[np.ndarray, list[str]]:
        return model_scores.astype(np.float32), [""] * len(model_scores)

    def judge(self, rows: pd.DataFrame) -> np.ndarray | None:
        return None


class QwenLLMJudge:
    """Batch JSON Qwen/DashScope judge for prerequisite pseudo-label validation."""

    def __init__(self, llm_cfg: Dict):
        self.provider = str(llm_cfg.get("provider", "qwen"))
        if self.provider.lower() != "qwen":
            raise ValueError(f"Unsupported llm.provider={self.provider!r}; currently only 'qwen' is implemented.")
        self.model = str(llm_cfg.get("model", "qwen3.8-max"))
        self.api_key_env = str(llm_cfg.get("api_key_env", "DASHSCOPE_API_KEY"))
        self.base_url = str(llm_cfg.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")).rstrip("/")
        self.timeout_seconds = float(llm_cfg.get("timeout_seconds", 30))
        self.batch_size = int(llm_cfg.get("batch_size", 20))
        self.max_retries = int(llm_cfg.get("max_retries", 3))
        self.retry_backoff_seconds = float(llm_cfg.get("retry_backoff_seconds", 2))
        self.cache_path = Path(str(llm_cfg.get("cache_path", "stage2/outputs/llm_judge_cache.jsonl")))
        self.judge_weight = float(llm_cfg.get("judge_weight", 0.5))
        self.prompt_version = str(llm_cfg.get("prompt_version", "prereq_judge_v2"))
        self.api_key = os.environ.get(self.api_key_env)
        if not self.api_key:
            raise RuntimeError(
                f"use_llm_judge=true but environment variable {self.api_key_env} is not set. "
                "Set it to your DashScope/Qwen API key; the key is never stored in code."
            )
        self.cache = self._read_cache()

    @staticmethod
    def _chunks(items: list, size: int):
        for start in range(0, len(items), size):
            yield items[start : start + size]

    def _cache_key(self, source_index: int, target_index: int) -> str:
        return f"{self.model}:{self.prompt_version}:{source_index}->{target_index}"

    def _read_cache(self) -> dict[str, float]:
        cache: dict[str, float] = {}
        if not self.cache_path.exists():
            return cache
        with self.cache_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("model") != self.model or obj.get("prompt_version") != self.prompt_version:
                    continue
                cache[str(obj["cache_key"])] = float(obj["score"])
        return cache

    def _append_cache(self, rows: list[dict]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
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

    @staticmethod
    def _prompt(pairs: list[dict]) -> str:
        return (
            "You are an expert machine learning professor.\n\n"
            "Task:\nFor each candidate pair, evaluate whether Concept A is a strict prerequisite for Concept B.\n\n"
            "Definition:\nA strict prerequisite means that understanding Concept A is substantially necessary before a student can properly understand Concept B.\n"
            "Concepts may be related without being prerequisites.\n\n"
            "Scoring:\n"
            "- 1.0 = A is clearly required to understand B.\n"
            "- 0.7 = A is probably a prerequisite.\n"
            "- 0.5 = uncertain or weak prerequisite.\n"
            "- 0.3 = related but probably not a prerequisite.\n"
            "- 0.0 = A is not a prerequisite of B.\n\n"
            "Requirements:\n"
            "1. Judge directionally: A -> B is different from B -> A.\n"
            "2. Do not infer prerequisite status from topical similarity alone.\n"
            "3. Return valid JSON only.\n"
            "4. Do not include explanations.\n\n"
            "Input:\n"
            + json.dumps({"pairs": pairs}, ensure_ascii=False, indent=2)
            + "\n\nOutput format:\n{\n  \"results\": [\n    {\"id\": \"pair_001\", \"score\": 0.0}\n  ]\n}"
        )

    def _call_qwen_json(self, prompt: str) -> dict:
        url = self.base_url + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(
                url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    content = body["choices"][0]["message"]["content"] or ""
                    return self._extract_json_object(content)
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(self.retry_backoff_seconds * (2 ** attempt))
        raise RuntimeError(f"Qwen judge call failed after {self.max_retries} attempts: {last_error}")

    @staticmethod
    def _validate_scores(payload: dict, expected_ids: set[str]) -> dict[str, float]:
        results = payload.get("results")
        if not isinstance(results, list) or len(results) != len(expected_ids):
            raise ValueError("LLM judge response must contain one result per input pair.")
        out: dict[str, float] = {}
        for item in results:
            pair_id = str(item.get("id"))
            if pair_id not in expected_ids:
                raise ValueError(f"Unexpected LLM result id: {pair_id}")
            score = float(item.get("score"))
            if not (0.0 <= score <= 1.0):
                raise ValueError(f"LLM score out of range for {pair_id}: {score}")
            out[pair_id] = score
        if set(out) != expected_ids:
            raise ValueError("LLM result IDs do not match input IDs.")
        return out

    def _judge_uncached_batch(self, batch: list[tuple[int, object, str]]) -> dict[int, float]:
        """Judge one uncached batch with short local IDs and schema retries.

        The cache key remains the real directed pair ``source_index->target_index``.
        Only the LLM-facing JSON IDs are shortened to ``p000``, ``p001``...
        to reduce copy errors such as returning ``pair_98_95`` for a batch that
        did not contain that exact pair.  If a whole batch repeatedly fails
        schema validation, split to single-pair calls.  If a single pair still
        fails, drop that candidate rather than using an unreviewed model score.
        """

        if not batch:
            return {}

        pair_payload = []
        local_id_to_pos: dict[str, int] = {}
        local_id_to_row: dict[str, object] = {}
        for local_idx, (pos, row, _old_pair_id) in enumerate(batch):
            local_id = f"p{local_idx:03d}"
            local_id_to_pos[local_id] = int(pos)
            local_id_to_row[local_id] = row
            pair_payload.append(
                {
                    "id": local_id,
                    "concept_a": str(row.source),
                    "concept_b": str(row.target),
                }
            )

        expected_ids = set(local_id_to_pos)
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                payload = self._call_qwen_json(self._prompt(pair_payload))
                parsed = self._validate_scores(payload, expected_ids)
                return {local_id_to_pos[local_id]: float(score) for local_id, score in parsed.items()}
            except Exception as exc:  # noqa: BLE001 - schema/API retries share the same recovery path.
                last_error = exc
                if attempt + 1 < self.max_retries:
                    print(
                        f"[llm judge retry] batch_size={len(batch)} attempt={attempt + 1}/{self.max_retries} "
                        f"failed: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    time.sleep(self.retry_backoff_seconds * (2 ** attempt))

        if len(batch) > 1:
            print(
                f"[llm judge split] batch_size={len(batch)} failed schema validation after retries; "
                "retrying as single-pair requests.",
                flush=True,
            )
            recovered: dict[int, float] = {}
            for item in batch:
                recovered.update(self._judge_uncached_batch([item]))
            return recovered

        pos, row, _old_pair_id = batch[0]
        print(
            f"[llm judge skip] dropping candidate {int(row.source_index)}->{int(row.target_index)} "
            f"after repeated invalid LLM responses: {last_error}",
            flush=True,
        )
        return {}

    def judge(self, rows: pd.DataFrame) -> np.ndarray:
        if len(rows) == 0:
            return np.asarray([], dtype=np.float32)
        scores: list[float | None] = [None] * len(rows)
        missing: list[tuple[int, object, str]] = []
        for pos, row in enumerate(rows.itertuples(index=False)):
            key = self._cache_key(int(row.source_index), int(row.target_index))
            if key in self.cache:
                scores[pos] = self.cache[key]
            else:
                pair_id = f"pair_{int(row.source_index)}_{int(row.target_index)}"
                missing.append((pos, row, pair_id))
        new_cache_rows: list[dict] = []
        for batch in self._chunks(missing, self.batch_size):
            parsed_by_pos = self._judge_uncached_batch(batch)
            for pos, row, pair_id in batch:
                if pos not in parsed_by_pos:
                    continue
                score = float(parsed_by_pos[pos])
                scores[pos] = score
                key = self._cache_key(int(row.source_index), int(row.target_index))
                self.cache[key] = score
                new_cache_rows.append(
                    {
                        "cache_key": key,
                        "source_index": int(row.source_index),
                        "target_index": int(row.target_index),
                        "source": str(row.source),
                        "target": str(row.target),
                        "score": score,
                        "model": self.model,
                        "prompt_version": self.prompt_version,
                    }
                )
        if new_cache_rows:
            self._append_cache(new_cache_rows)
        return np.asarray([np.nan if x is None else float(x) for x in scores], dtype=np.float32)

    def combine(self, model_scores: np.ndarray, llm_scores: np.ndarray | None) -> tuple[np.ndarray, list[str]]:
        if llm_scores is None:
            raise ValueError("QwenLLMJudge.combine requires llm_scores")
        combined = self.judge_weight * llm_scores + (1.0 - self.judge_weight) * model_scores
        return combined.astype(np.float32), [str(float(x)) for x in llm_scores]


def build_llm_judge(cfg: Dict):
    if not bool(cfg.get("use_llm_judge", False)):
        return NoLLMJudge()
    return QwenLLMJudge(dict(cfg.get("llm", {}) or {}))


def add_label_confidence(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if len(out) == 0:
        out["label_confidence"] = pd.Series(dtype=float)
        return out
    combined = out["combined_score"].astype(float)
    labels = out["label"].astype(int)
    out["label_confidence"] = np.where(labels == 1, combined, 1.0 - combined)
    return out


def _empty_pseudo_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=PSEUDO_COLUMNS)


def _pair_keys(df: pd.DataFrame) -> list[tuple[int, int]]:
    if len(df) == 0:
        return []
    return list(zip(df["source_index"].astype(int), df["target_index"].astype(int)))


def _dedupe_by_confidence(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df.reindex(columns=PSEUDO_COLUMNS).copy()
    out = add_label_confidence(df)
    out = out.sort_values("label_confidence", ascending=False, kind="mergesort")
    out = out.drop_duplicates(subset=["source_index", "target_index"], keep="first")
    return out[PSEUDO_COLUMNS].reset_index(drop=True)


class StablePseudoTracker:
    """Track pseudo labels that have been confirmed across models or rounds.

    Online consistency pseudo labels are not stored here. This tracker is only fed pseudo labels that already passed the
    existing model-threshold + optional LLM-judge path.
    """

    def __init__(self, min_hits: int = 2):
        self.min_hits = max(1, int(min_hits))
        self._records: dict[tuple[int, int], dict] = {}

    def update(self, new_pseudo: pd.DataFrame, iteration: int) -> tuple[pd.DataFrame, Dict[str, int]]:
        metrics = {
            "input_rows": int(len(new_pseudo)),
            "newly_stable_rows": 0,
            "tracked_rows": int(len(self._records)),
            "stable_rows": 0,
            "resets_by_label_conflict": 0,
        }
        if len(new_pseudo) == 0:
            metrics["stable_rows"] = int(sum(1 for rec in self._records.values() if int(rec["hits"]) >= self.min_hits))
            return pd.DataFrame(columns=PSEUDO_COLUMNS), metrics

        newly_stable: list[dict] = []
        for row in add_label_confidence(new_pseudo).itertuples(index=False):
            key = (int(row.source_index), int(row.target_index))
            label = int(row.label)
            teacher = str(row.teacher_model)
            previous = self._records.get(key)
            was_stable = bool(previous and int(previous["hits"]) >= self.min_hits)

            if previous is None or int(previous["label"]) != label:
                if previous is not None and int(previous["label"]) != label:
                    metrics["resets_by_label_conflict"] += 1
                rec = {
                    "label": label,
                    "hits": 1,
                    "last_iteration": int(iteration),
                    "teachers": {teacher},
                    "row": row._asdict(),
                }
            else:
                teachers = set(previous.get("teachers", set()))
                hits = int(previous["hits"])
                should_increment = False
                if int(previous.get("last_iteration", -1)) != int(iteration):
                    should_increment = True
                elif teacher not in teachers:
                    # Content and graph agreeing in the same AKD round counts
                    # as an additional confirmation.
                    should_increment = True
                if should_increment:
                    hits += 1
                teachers.add(teacher)
                best_row = previous["row"]
                current_conf = float(getattr(row, "label_confidence"))
                previous_conf = float(best_row.get("label_confidence", 0.0))
                if current_conf >= previous_conf:
                    best_row = row._asdict()
                rec = {
                    "label": label,
                    "hits": hits,
                    "last_iteration": int(iteration),
                    "teachers": teachers,
                    "row": best_row,
                }

            self._records[key] = rec
            is_stable = int(rec["hits"]) >= self.min_hits
            if is_stable and not was_stable:
                newly_stable.append(rec["row"])

        metrics["tracked_rows"] = int(len(self._records))
        metrics["stable_rows"] = int(sum(1 for rec in self._records.values() if int(rec["hits"]) >= self.min_hits))
        metrics["newly_stable_rows"] = int(len(newly_stable))
        if not newly_stable:
            return pd.DataFrame(columns=PSEUDO_COLUMNS), metrics
        return _dedupe_by_confidence(pd.DataFrame(newly_stable)), metrics

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for rec in self._records.values():
            row = dict(rec["row"])
            row["stable_hits"] = int(rec["hits"])
            row["stable_teachers"] = ",".join(sorted(rec.get("teachers", set())))
            row["stable_last_iteration"] = int(rec.get("last_iteration", -1))
            rows.append(row)
        if not rows:
            return pd.DataFrame(columns=PSEUDO_COLUMNS + ["stable_hits", "stable_teachers", "stable_last_iteration"])
        return pd.DataFrame(rows)


def _confidence_by_pair(df: pd.DataFrame) -> dict[tuple[int, int], float]:
    out = add_label_confidence(df)
    return {
        (int(row.source_index), int(row.target_index)): float(row.label_confidence)
        for row in out.itertuples(index=False)
    }


def _concat_pseudo_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [df for df in frames if len(df)]
    if not non_empty:
        return _empty_pseudo_frame()
    return pd.concat(non_empty, ignore_index=True)


def resolve_pseudo_conflicts(
    new_pseudo: pd.DataFrame,
    r_syn_pos: pd.DataFrame,
    r_syn_neg: pd.DataFrame,
    r_neg_dynamic: pd.DataFrame,
    r_train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    new_pseudo = add_label_confidence(new_pseudo)
    r_syn_pos = _dedupe_by_confidence(r_syn_pos)
    r_syn_neg = _dedupe_by_confidence(r_syn_neg)
    r_neg_dynamic = r_neg_dynamic.copy()
    d_hat_pos = new_pseudo[new_pseudo["label"].astype(int) == 1].copy()
    d_hat_neg = new_pseudo[new_pseudo["label"].astype(int) == 0].copy()
    metrics = {
        "new_pos_before": int(len(d_hat_pos)),
        "new_neg_before": int(len(d_hat_neg)),
        "stage1_neg_removed_by_new_pos": 0,
        "new_neg_dropped_against_gold_pos": 0,
        "pos_vs_r_syn_neg_keep_new": 0,
        "pos_vs_r_syn_neg_keep_existing": 0,
        "neg_vs_r_syn_pos_keep_new": 0,
        "neg_vs_r_syn_pos_keep_existing": 0,
    }

    gold_pos_pairs = pair_set(r_train)
    if len(d_hat_neg):
        before = len(d_hat_neg)
        d_hat_neg = d_hat_neg[
            ~pd.Series(_pair_keys(d_hat_neg), index=d_hat_neg.index).isin(gold_pos_pairs)
        ].copy()
        metrics["new_neg_dropped_against_gold_pos"] = int(before - len(d_hat_neg))

    if len(d_hat_pos) and len(r_syn_neg):
        neg_conf = _confidence_by_pair(r_syn_neg)
        keep_new_idx = []
        drop_existing_pairs: set[tuple[int, int]] = set()
        for idx, row in d_hat_pos.iterrows():
            pair = (int(row.source_index), int(row.target_index))
            if pair not in neg_conf:
                keep_new_idx.append(idx)
            elif float(row.label_confidence) > neg_conf[pair]:
                keep_new_idx.append(idx)
                drop_existing_pairs.add(pair)
                metrics["pos_vs_r_syn_neg_keep_new"] += 1
            else:
                metrics["pos_vs_r_syn_neg_keep_existing"] += 1
        d_hat_pos = d_hat_pos.loc[keep_new_idx].copy()
        if drop_existing_pairs:
            r_syn_neg = r_syn_neg[
                ~pd.Series(_pair_keys(r_syn_neg), index=r_syn_neg.index).isin(drop_existing_pairs)
            ].copy()

    if len(d_hat_neg) and len(r_syn_pos):
        pos_conf = _confidence_by_pair(r_syn_pos)
        keep_new_idx = []
        drop_existing_pairs: set[tuple[int, int]] = set()
        for idx, row in d_hat_neg.iterrows():
            pair = (int(row.source_index), int(row.target_index))
            if pair not in pos_conf:
                keep_new_idx.append(idx)
            elif float(row.label_confidence) > pos_conf[pair]:
                keep_new_idx.append(idx)
                drop_existing_pairs.add(pair)
                metrics["neg_vs_r_syn_pos_keep_new"] += 1
            else:
                metrics["neg_vs_r_syn_pos_keep_existing"] += 1
        d_hat_neg = d_hat_neg.loc[keep_new_idx].copy()
        if drop_existing_pairs:
            r_syn_pos = r_syn_pos[
                ~pd.Series(_pair_keys(r_syn_pos), index=r_syn_pos.index).isin(drop_existing_pairs)
            ].copy()

    r_syn_pos = _dedupe_by_confidence(_concat_pseudo_frames([r_syn_pos, d_hat_pos]))
    r_syn_neg = _dedupe_by_confidence(_concat_pseudo_frames([r_syn_neg, d_hat_neg]))
    metrics["new_pos_after"] = int(len(d_hat_pos))
    metrics["new_neg_after"] = int(len(d_hat_neg))
    metrics["r_syn_pos_rows"] = int(len(r_syn_pos))
    metrics["r_syn_neg_rows"] = int(len(r_syn_neg))
    metrics["r_neg_dynamic_rows"] = int(len(r_neg_dynamic))
    return r_syn_pos, r_syn_neg, r_neg_dynamic, metrics


def build_candidate_pairs(data, exclude_dfs: Sequence[pd.DataFrame], max_candidates: int | None = None, seed: int = 42) -> pd.DataFrame:
    excluded: Set[tuple[int, int]] = set()
    for df in exclude_dfs:
        excluded |= pair_set(df)
    rows = []
    for src in range(data.num_concepts):
        for dst in range(data.num_concepts):
            if src == dst or (src, dst) in excluded:
                continue
            rows.append(
                {
                    "source_id": src,
                    "target_id": dst,
                    "source_index": src,
                    "target_index": dst,
                    "source": data.id_to_name[src],
                    "target": data.id_to_name[dst],
                }
            )
    df = pd.DataFrame(rows)
    if max_candidates is not None and len(df) > max_candidates:
        df = df.sample(n=max_candidates, random_state=seed).reset_index(drop=True)
    return df.reset_index(drop=True)


def _select_initial_pseudo(candidates: pd.DataFrame, candidate_k: int, pos_threshold: float, neg_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    pos = candidates[candidates["model_score"] >= pos_threshold].sort_values("model_score", ascending=False).head(candidate_k)
    neg = candidates[candidates["model_score"] <= neg_threshold].sort_values("model_score", ascending=True).head(candidate_k)
    return pos.copy(), neg.copy()


def _print_score_stats(prefix: str, teacher_model: str, iteration: int, scores: pd.Series) -> None:
    values = scores.astype(float)
    print(
        f"[{teacher_model} round {iteration}] {prefix} score stats: "
        f"min={values.min():.4f}, "
        f"p01={values.quantile(0.01):.4f}, "
        f"p05={values.quantile(0.05):.4f}, "
        f"p10={values.quantile(0.10):.4f}, "
        f"p50={values.quantile(0.50):.4f}, "
        f"p90={values.quantile(0.90):.4f}, "
        f"p95={values.quantile(0.95):.4f}, "
        f"p99={values.quantile(0.99):.4f}, "
        f"max={values.max():.4f}",
        flush=True,
    )


def _apply_llm_and_finalize(frame: pd.DataFrame, judge, label: int, pseudo_type: str, teacher_model: str, iteration: int, final_k: int) -> pd.DataFrame:
    if len(frame) == 0:
        return pd.DataFrame(columns=PSEUDO_COLUMNS)
    llm_scores = judge.judge(frame)
    combined, llm_text = judge.combine(frame["model_score"].to_numpy(dtype=np.float32), llm_scores)
    out = frame.copy()
    out["llm_score"] = llm_text
    out["combined_score"] = combined
    if len(out):
        valid_llm = np.isfinite(out["combined_score"].astype(float).to_numpy())
        if not bool(valid_llm.all()):
            dropped = int((~valid_llm).sum())
            print(
                f"[llm judge drop] dropped {dropped} {teacher_model} {pseudo_type} candidates "
                "because they did not receive a valid LLM score.",
                flush=True,
            )
            out = out.loc[valid_llm].copy()
    if len(out) == 0:
        return pd.DataFrame(columns=PSEUDO_COLUMNS)
    out["label"] = label
    out = add_label_confidence(out)
    out["teacher_model"] = teacher_model
    out["iteration"] = iteration
    out["pseudo_type"] = pseudo_type
    ascending = pseudo_type == "negative"
    return out.sort_values("combined_score", ascending=ascending).head(final_k)[PSEUDO_COLUMNS]


def generate_pseudo_labels(
    data,
    scorer: Callable[[pd.DataFrame], np.ndarray],
    teacher_model: str,
    iteration: int,
    cfg: Dict,
    base_exclude_dfs: Sequence[pd.DataFrame] | None = None,
    extra_exclude_dfs: Sequence[pd.DataFrame] | None = None,
    raw_scorer: Callable[[pd.DataFrame], np.ndarray] | None = None,
) -> pd.DataFrame:
    exclude = (
        list(base_exclude_dfs)
        if base_exclude_dfs is not None
        else [data.train_labels, data.val_labels, data.test_labels, data.heldout_pairs]
    )
    if extra_exclude_dfs:
        exclude.extend(extra_exclude_dfs)
    candidates = build_candidate_pairs(data, exclude, max_candidates=None, seed=int(cfg.get("seed", 42)) + iteration)
    if len(candidates) == 0:
        return pd.DataFrame(columns=PSEUDO_COLUMNS)

    candidates = candidates.copy()
    if raw_scorer is not None:
        candidates["raw_model_score"] = np.asarray(raw_scorer(candidates), dtype=np.float32)
    candidates["model_score"] = np.asarray(scorer(candidates), dtype=np.float32)

    if "raw_model_score" in candidates.columns:
        _print_score_stats("raw", teacher_model, iteration, candidates["raw_model_score"])
        _print_score_stats("calibrated", teacher_model, iteration, candidates["model_score"])
    else:
        _print_score_stats("model", teacher_model, iteration, candidates["model_score"])

    train_positive_count = int((data.train_labels["label"].astype(int) == 1).sum())
    total_ratio = float(cfg.get("max_new_pseudo_ratio_per_round", cfg.get("max_pseudo_ratio", 0.1)))
    final_total = max(2, int(total_ratio * train_positive_count))
    final_k = max(1, final_total // 2)
    candidate_k = max(1, int(final_k * float(cfg.get("pre_llm_candidate_multiplier", 2))))
    pos, neg = _select_initial_pseudo(
        candidates,
        candidate_k,
        float(cfg.get("pseudo_pos_threshold", 0.8)),
        float(cfg.get("pseudo_neg_threshold", 0.2)),
    )

    judge = build_llm_judge(cfg)
    frames = []
    for label, pseudo_type, frame in [(1, "positive", pos), (0, "negative", neg)]:
        frames.append(_apply_llm_and_finalize(frame, judge, label, pseudo_type, teacher_model, iteration, final_k))
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PSEUDO_COLUMNS)
    return _dedupe_by_confidence(result)


def pseudo_to_training_labels(pseudo_df: pd.DataFrame, split: str = "train") -> pd.DataFrame:
    if len(pseudo_df) == 0:
        from .utils import LABEL_COLUMNS
        return pd.DataFrame(columns=LABEL_COLUMNS)
    pseudo_df = add_label_confidence(pseudo_df)
    return pd.DataFrame(
        {
            "source_id": pseudo_df["source_id"].astype(int),
            "target_id": pseudo_df["target_id"].astype(int),
            "source_index": pseudo_df["source_index"].astype(int),
            "target_index": pseudo_df["target_index"].astype(int),
            "source": pseudo_df["source"],
            "target": pseudo_df["target"],
            "label": pseudo_df["label"].astype(int),
            "split": split,
            "negative_type": "pseudo_" + pseudo_df["pseudo_type"].astype(str),
            "similarity": np.nan,
            "combined_score": pseudo_df["combined_score"].astype(float),
            "label_confidence": pseudo_df["label_confidence"].astype(float),
        }
    )
