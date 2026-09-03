from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .concept_profile import load_stage0_profiles
from .content_model import build_content_model
from .data_loader import load_stage2_data
from .diagnostic_plots import generate_diagnostic_plots
from .fusion import search_alpha_and_threshold
from .metrics import find_best_threshold
from .pseudo_labeling import (
    StablePseudoTracker,
    add_label_confidence,
    build_candidate_pairs,
    build_llm_judge,
    generate_pseudo_labels,
    pseudo_to_training_labels,
    resolve_pseudo_conflicts,
)
from .train_content import fit_content_model, save_content_checkpoint
from .train_graph import GraphTrainer
from .utils import PSEUDO_COLUMNS, ensure_dir, set_seed, write_json


def _empty_pseudo_df() -> pd.DataFrame:
    return pd.DataFrame(columns=PSEUDO_COLUMNS)


def _pseudo_union(r_syn_pos: pd.DataFrame, r_syn_neg: pd.DataFrame) -> pd.DataFrame:
    frames = [df for df in [r_syn_pos, r_syn_neg] if len(df)]
    return pd.concat(frames, ignore_index=True) if frames else _empty_pseudo_df()


def _content_train_frame(r_train: pd.DataFrame, r_neg_dynamic: pd.DataFrame, r_syn_pos: pd.DataFrame, r_syn_neg: pd.DataFrame) -> pd.DataFrame:
    pseudo_labels = pseudo_to_training_labels(_pseudo_union(r_syn_pos, r_syn_neg))
    frames = [r_train, r_neg_dynamic]
    if len(pseudo_labels):
        frames.append(pseudo_labels)
    return pd.concat(frames, ignore_index=True)


def _log(message: str) -> None:
    print(message, flush=True)


def _threshold_values(cfg: Dict) -> list[float]:
    return [float(x) for x in cfg.get("threshold_search_values", [i / 10.0 for i in range(1, 10)])]


def _validation_graph_metrics(val_df: pd.DataFrame, val_graph_scores: np.ndarray, cfg: Dict) -> Dict[str, float]:
    threshold, metrics = find_best_threshold(
        val_df["label"].astype(int).to_numpy(),
        val_graph_scores,
        _threshold_values(cfg),
    )
    return {
        "validation_graph_auc": float(metrics["auc"]),
        "validation_graph_f1": float(metrics["f1"]),
        "validation_graph_precision": float(metrics["precision"]),
        "validation_graph_recall": float(metrics["recall"]),
        "validation_graph_accuracy": float(metrics["accuracy"]),
        "validation_graph_threshold": float(threshold),
    }


def _prefixed_validation_metrics(prefix: str, val_df: pd.DataFrame, scores: np.ndarray, cfg: Dict) -> Dict[str, float]:
    threshold, metrics = find_best_threshold(
        val_df["label"].astype(int).to_numpy(),
        scores,
        _threshold_values(cfg),
    )
    return {
        f"validation_{prefix}_auc": float(metrics["auc"]),
        f"validation_{prefix}_f1": float(metrics["f1"]),
        f"validation_{prefix}_precision": float(metrics["precision"]),
        f"validation_{prefix}_recall": float(metrics["recall"]),
        f"validation_{prefix}_accuracy": float(metrics["accuracy"]),
        f"validation_{prefix}_threshold": float(threshold),
    }


def _fusion_params_for_round(
    val_df: pd.DataFrame,
    val_content: np.ndarray,
    val_graph: np.ndarray,
    cfg: Dict,
) -> Dict:
    alpha_values = [float(x) for x in cfg.get("alpha_search_values", [i / 10.0 for i in range(11)])]
    threshold_values = [float(x) for x in cfg.get("threshold_search_values", [i / 10.0 for i in range(1, 10)])]
    best = search_alpha_and_threshold(
        val_df["label"].astype(int).to_numpy(),
        val_content,
        val_graph,
        alpha_values,
        threshold_values,
    )
    return {
        "selection_set": "validation",
        "selected_alpha": float(best["alpha"]),
        "selected_threshold": float(best["threshold"]),
        "alpha_search_values": alpha_values,
        "threshold_search_values": threshold_values,
        "validation_fusion_metrics": best,
        "validation_content_metrics": _prefixed_validation_metrics("content", val_df, val_content, cfg),
        "validation_graph_metrics": _prefixed_validation_metrics("graph", val_df, val_graph, cfg),
    }


def _fusion_metric_order_keys(cfg: Dict) -> list[str]:
    primary = str(cfg.get("fusion_best_metric", "f1")).strip().lower()
    raw_ties = cfg.get("fusion_best_tie_breakers", ["auc", "accuracy"])
    if isinstance(raw_ties, str):
        ties = [x.strip().lower() for x in raw_ties.split(",") if x.strip()]
    else:
        ties = [str(x).strip().lower() for x in raw_ties]
    return [primary] + [x for x in ties if x != primary]


def _fusion_metric_tuple(fusion_params: Dict, cfg: Dict) -> tuple[float, ...]:
    metrics = dict(fusion_params.get("validation_fusion_metrics", {}) or {})
    return tuple(float(metrics.get(k, float("-inf"))) for k in _fusion_metric_order_keys(cfg))


def _pair_set(df: pd.DataFrame) -> set[tuple[int, int]]:
    if len(df) == 0 or "source_index" not in df.columns or "target_index" not in df.columns:
        return set()
    return set(zip(df["source_index"].astype(int), df["target_index"].astype(int)))


def _safe_numeric_mean(series: pd.Series) -> float | None:
    if series.empty:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) == 0:
        return None
    return float(values.mean())


def _pseudo_quality_metrics(r_syn_pos: pd.DataFrame, r_syn_neg: pd.DataFrame) -> Dict[str, float | int | None]:
    pseudo = _pseudo_union(r_syn_pos, r_syn_neg)
    if len(pseudo) == 0:
        return {
            "r_syn_total": 0,
            "pseudo_label_confidence_mean": 0.0,
            "pseudo_llm_score_mean": None,
        }
    confidence_mean = _safe_numeric_mean(pseudo["label_confidence"]) if "label_confidence" in pseudo.columns else None
    llm_score_mean = _safe_numeric_mean(pseudo["llm_score"]) if "llm_score" in pseudo.columns else None
    return {
        "r_syn_total": int(len(pseudo)),
        "pseudo_label_confidence_mean": float(confidence_mean) if confidence_mean is not None else 0.0,
        "pseudo_llm_score_mean": llm_score_mean,
    }


def _mine_hard_negatives(
    data,
    profiles,
    content_model,
    graph_trainer: GraphTrainer,
    r_neg_dynamic: pd.DataFrame,
    r_syn_pos: pd.DataFrame,
    r_syn_neg: pd.DataFrame,
    cfg: Dict,
    iteration: int,
) -> tuple[pd.DataFrame, Dict[str, int | float | bool]]:
    if not bool(cfg.get("hard_negative_mining_enabled", True)):
        return pd.DataFrame(columns=PSEUDO_COLUMNS), {"hard_negative_mining_enabled": False, "hard_negative_rows": 0}
    if not bool(cfg.get("use_llm_judge", False)):
        return pd.DataFrame(columns=PSEUDO_COLUMNS), {
            "hard_negative_mining_enabled": True,
            "hard_negative_skipped_reason": "use_llm_judge=false",
            "hard_negative_rows": 0,
        }
    train_positive_count = int((data.train_labels["label"].astype(int) == 1).sum())
    max_rows = max(0, int(float(cfg.get("hard_negative_max_ratio_to_positive_train", 0.1)) * train_positive_count))
    if max_rows <= 0:
        return pd.DataFrame(columns=PSEUDO_COLUMNS), {"hard_negative_mining_enabled": True, "hard_negative_rows": 0}
    exclude = [data.train_labels, data.val_labels, data.test_labels, data.heldout_pairs, r_neg_dynamic, r_syn_pos, r_syn_neg]
    candidates = build_candidate_pairs(data, exclude, max_candidates=None, seed=int(cfg.get("seed", 42)) + 1000 + iteration)
    if len(candidates) == 0:
        return pd.DataFrame(columns=PSEUDO_COLUMNS), {"hard_negative_mining_enabled": True, "hard_negative_rows": 0}
    candidates = candidates.copy()
    candidates["content_score"] = content_model.predict_scores(
        candidates, profiles, batch_size=int(cfg.get("content_batch_size", 8))
    )
    candidates["graph_score"] = graph_trainer.predict_scores(candidates, calibrated=True)
    gh_cl = (
        (candidates["graph_score"] >= float(cfg.get("hard_negative_graph_high_threshold", 0.65)))
        & (candidates["content_score"] <= float(cfg.get("hard_negative_content_low_threshold", 0.45)))
    )
    ch_gl = (
        (candidates["content_score"] >= float(cfg.get("hard_negative_content_high_threshold", 0.60)))
        & (candidates["graph_score"] <= float(cfg.get("hard_negative_graph_low_threshold", 0.35)))
    )
    selected = candidates[gh_cl | ch_gl].copy()
    if len(selected) == 0:
        return pd.DataFrame(columns=PSEUDO_COLUMNS), {
            "hard_negative_mining_enabled": True,
            "hard_negative_candidate_rows": int(len(candidates)),
            "hard_negative_disagreement_rows": 0,
            "hard_negative_rows": 0,
        }
    selected["disagreement"] = (selected["content_score"] - selected["graph_score"]).abs()
    selected = selected.sort_values("disagreement", ascending=False).head(max_rows * 2).copy()
    use_llm = bool(cfg.get("use_llm_judge", False))
    llm_values: np.ndarray | None = None
    if use_llm:
        judge = build_llm_judge(cfg)
        llm_values = judge.judge(selected)
        selected["llm_numeric"] = llm_values
        selected = selected[selected["llm_numeric"].astype(float) <= float(cfg.get("hard_negative_llm_max_score", 0.30))].copy()
    selected = selected.head(max_rows).copy()
    if len(selected) == 0:
        return pd.DataFrame(columns=PSEUDO_COLUMNS), {
            "hard_negative_mining_enabled": True,
            "hard_negative_candidate_rows": int(len(candidates)),
            "hard_negative_disagreement_rows": int((gh_cl | ch_gl).sum()),
            "hard_negative_rows": 0,
        }
    soft_min = float(cfg.get("hard_negative_soft_target_min", 0.15))
    soft_max = float(cfg.get("hard_negative_soft_target_max", 0.30))
    base = np.minimum(selected["content_score"].to_numpy(dtype=float), selected["graph_score"].to_numpy(dtype=float))
    if use_llm and "llm_numeric" in selected.columns:
        base = np.minimum(base, selected["llm_numeric"].astype(float).to_numpy())
    combined = np.clip(base, soft_min, soft_max).astype(np.float32)
    out = selected.copy()
    out["label"] = 0
    out["teacher_model"] = "hard_negative_mining"
    out["model_score"] = out[["content_score", "graph_score"]].max(axis=1).astype(float)
    out["llm_score"] = [str(float(x)) for x in selected["llm_numeric"].astype(float)] if use_llm and "llm_numeric" in selected.columns else ""
    out["combined_score"] = combined
    out["iteration"] = int(iteration)
    out["pseudo_type"] = "hard_negative"
    out = add_label_confidence(out)
    out = out[PSEUDO_COLUMNS].reset_index(drop=True)
    return out, {
        "hard_negative_mining_enabled": True,
        "hard_negative_candidate_rows": int(len(candidates)),
        "hard_negative_disagreement_rows": int((gh_cl | ch_gl).sum()),
        "hard_negative_rows": int(len(out)),
        "hard_negative_max_rows": int(max_rows),
        "hard_negative_used_llm": bool(use_llm),
        "hard_negative_mean_combined_score": float(out["combined_score"].astype(float).mean()) if len(out) else 0.0,
    }


def _metric_order_keys(cfg: Dict) -> list[str]:
    primary = str(cfg.get("graph_best_metric", "auc")).strip().lower()
    raw_ties = cfg.get("graph_best_tie_breakers", ["f1", "accuracy"])
    if isinstance(raw_ties, str):
        tie_breakers = [x.strip().lower() for x in raw_ties.split(",") if x.strip()]
    else:
        tie_breakers = [str(x).strip().lower() for x in raw_ties]
    return [primary] + [x for x in tie_breakers if x != primary]


def _graph_metric_tuple(metrics: Dict[str, float], cfg: Dict) -> tuple[float, ...]:
    values = []
    for key in _metric_order_keys(cfg):
        metric_key = key if key.startswith("validation_graph_") else f"validation_graph_{key}"
        values.append(float(metrics.get(metric_key, float("-inf"))))
    return tuple(values)


def run_akd(cfg: Dict, device) -> Dict:
    set_seed(int(cfg.get("seed", 42)))
    output_dir = ensure_dir(cfg["stage2_outputs_dir"])
    checkpoint_dir = ensure_dir(cfg["checkpoint_dir"])
    _log("Stage2 AKD start")
    _log(f"Device: {device}")
    _log(f"Output dir: {output_dir}")
    _log(f"Checkpoint dir: {checkpoint_dir}")
    data = load_stage2_data(cfg["stage0_outputs_dir"], cfg["stage1_outputs_dir"])
    profiles = load_stage0_profiles(data.concepts, cfg)

    rounds = int(cfg.get("akd_rounds", 4))
    _log(f"AKD rounds: {rounds}")
    _log(
        "Samples: "
        f"train={len(data.train_labels)}, val={len(data.val_labels)}, test={len(data.test_labels)}, "
        f"heldout={len(data.heldout_pairs)}"
    )
    r_train = data.train_labels[data.train_labels["label"].astype(int) == 1].copy().reset_index(drop=True)
    r_neg_dynamic = data.train_labels[data.train_labels["label"].astype(int) == 0].copy().reset_index(drop=True)
    r_syn_pos = _empty_pseudo_df()
    r_syn_neg = _empty_pseudo_df()
    candidate_exclude = [data.train_labels, data.val_labels, data.test_labels, data.heldout_pairs]
    stable_tracker = StablePseudoTracker(min_hits=int(cfg.get("stable_pseudo_min_hits", 1)))
    round_summaries = []
    content_history_frames: list[pd.DataFrame] = []
    graph_history_frames: list[pd.DataFrame] = []
    round_diagnostic_rows: list[Dict] = []

    content_model = build_content_model(cfg, data, device)
    graph_trainer = GraphTrainer(cfg, data, device)
    best_graph_round: int | None = None
    best_graph_checkpoint: Path | None = None
    best_graph_metrics: Dict[str, float] | None = None
    best_graph_key: tuple[float, ...] | None = None
    best_graph_val_scores: np.ndarray | None = None
    best_fusion_round: int | None = None
    best_fusion_snapshot: Dict | None = None
    best_fusion_metrics: Dict | None = None
    best_fusion_key: tuple[float, ...] | None = None

    for iteration in range(rounds):
        _log(f"Round {iteration + 1}/{rounds}")
        round_dir = ensure_dir(output_dir / f"akd_round_{iteration}")
        content_train = _content_train_frame(r_train, r_neg_dynamic, r_syn_pos, r_syn_neg)
        _log(f"content model training start: rows={len(content_train)}")
        content_metrics = fit_content_model(
            content_model,
            content_train,
            profiles,
            cfg,
            val_df=data.val_labels,
            iteration=iteration,
        )
        content_epoch_history = pd.DataFrame(content_metrics.pop("epoch_history", []))
        if len(content_epoch_history):
            content_epoch_history.to_csv(round_dir / "content_epoch_history.csv", index=False, encoding="utf-8")
            content_history_frames.append(content_epoch_history)
        else:
            pd.DataFrame(columns=["round", "epoch", "global_epoch", "train_total_loss", "val_bce_loss"]).to_csv(
                round_dir / "content_epoch_history.csv", index=False, encoding="utf-8"
            )
        _log(f"content model training done: {content_metrics}")
        content_round_checkpoint = checkpoint_dir / f"content_round_{iteration}.pt"
        save_content_checkpoint(content_model, content_round_checkpoint)
        _log(f"checkpoint saved: {content_round_checkpoint}")
        val_content_scores = content_model.predict_scores(
            data.val_labels, profiles, batch_size=int(cfg.get("content_batch_size", 8))
        )
        validation_content_metrics = _prefixed_validation_metrics("content", data.val_labels, val_content_scores, cfg)
        _log(f"content validation done: {validation_content_metrics}")

        _log("content pseudo label generation start")
        pseudo_from_content = generate_pseudo_labels(
            data=data,
            scorer=lambda pairs: content_model.predict_scores(pairs, profiles, batch_size=int(cfg.get("content_batch_size", 8))),
            teacher_model="content",
            iteration=iteration,
            cfg=cfg,
            base_exclude_dfs=candidate_exclude,
        )
        pseudo_from_content.to_csv(round_dir / "pseudo_from_content.csv", index=False, encoding="utf-8")
        _log(f"content pseudo label generation done: rows={len(pseudo_from_content)}")
        stable_from_content, content_stability = stable_tracker.update(pseudo_from_content, iteration)
        stable_from_content.to_csv(round_dir / "stable_from_content.csv", index=False, encoding="utf-8")
        stable_tracker.to_frame().to_csv(round_dir / "stable_pseudo_memory_after_content.csv", index=False, encoding="utf-8")
        _log(f"content pseudo stability done: {content_stability}")
        r_syn_pos, r_syn_neg, r_neg_dynamic, content_cleaning = resolve_pseudo_conflicts(
            stable_from_content, r_syn_pos, r_syn_neg, r_neg_dynamic, r_train
        )
        _log(f"content conflict cleaning done: {content_cleaning}")
        _pseudo_union(r_syn_pos, r_syn_neg).to_csv(round_dir / "r_syn_after_content.csv", index=False, encoding="utf-8")

        graph_supervised = pd.concat([r_train, r_neg_dynamic], ignore_index=True)
        graph_pseudo = _pseudo_union(r_syn_pos, r_syn_neg)
        graph_epochs = int(cfg.get("graph_epochs", 200))
        graph_diag = getattr(graph_trainer, "graph_stats", {})
        _log(
            "graph model training start: "
            f"supervised_rows={len(graph_supervised)}, pseudo_rows={len(graph_pseudo)}, epochs={graph_epochs}, "
            f"relations={graph_diag.get('relation_counts', {})}, "
            f"semantic_k={graph_diag.get('semantic_selected_k')}, "
            f"lcc={float(graph_diag.get('semantic_lcc_coverage', 0.0)):.4f}, "
            f"fallback={graph_diag.get('semantic_fallback_used')}"
        )
        graph_metrics = graph_trainer.fit(
            graph_supervised,
            pseudo_df=graph_pseudo,
            epochs=graph_epochs,
            iteration=iteration,
            val_df=data.val_labels,
        )
        graph_epoch_history = pd.DataFrame(graph_metrics.pop("epoch_history", []))
        if len(graph_epoch_history):
            graph_epoch_history.to_csv(round_dir / "graph_epoch_history.csv", index=False, encoding="utf-8")
            graph_history_frames.append(graph_epoch_history)
        else:
            pd.DataFrame(
                columns=[
                    "round",
                    "epoch",
                    "global_epoch",
                    "train_total_loss",
                    "val_bce_loss",
                    "gold_bce",
                    "pseudo_soft_bce",
                    "kd_bce",
                    "rank_loss",
                    "logit_cap_penalty",
                ]
            ).to_csv(round_dir / "graph_epoch_history.csv", index=False, encoding="utf-8")
        _log(f"graph model training done: {graph_metrics}")
        graph_calibration_metrics = graph_trainer.fit_calibration(data.val_labels)
        _log(f"graph calibration done: {graph_calibration_metrics}")
        if graph_calibration_metrics.get("graph_calibration_saturation_warning"):
            _log(f"WARNING: {graph_calibration_metrics.get('graph_calibration_warning_message')}")
        val_graph_scores = graph_trainer.predict_scores(data.val_labels, calibrated=True)
        validation_graph_metrics = _validation_graph_metrics(data.val_labels, val_graph_scores, cfg)
        validation_graph_metrics = {**validation_graph_metrics, **graph_calibration_metrics}
        _log(f"graph validation done: {validation_graph_metrics}")
        graph_round_checkpoint = checkpoint_dir / f"graph_round_{iteration}.pt"
        graph_trainer.save_checkpoint(graph_round_checkpoint)
        _log(f"checkpoint saved: {graph_round_checkpoint}")
        graph_key = _graph_metric_tuple(validation_graph_metrics, cfg)
        graph_is_best = best_graph_key is None or graph_key > best_graph_key
        if graph_is_best:
            best_graph_key = graph_key
            best_graph_round = int(iteration)
            best_graph_checkpoint = graph_round_checkpoint
            best_graph_metrics = validation_graph_metrics
            best_graph_val_scores = val_graph_scores
            _log(
                "best graph checkpoint updated: "
                f"round={best_graph_round}, checkpoint={best_graph_checkpoint}, metrics={best_graph_metrics}"
            )
        if best_graph_checkpoint is None or best_graph_val_scores is None or best_graph_round is None:
            raise RuntimeError("No best graph checkpoint available for fusion snapshot selection.")
        graph_best_until_checkpoint = checkpoint_dir / f"graph_best_until_{iteration}.pt"
        shutil.copy2(best_graph_checkpoint, graph_best_until_checkpoint)
        fusion_params_round = _fusion_params_for_round(data.val_labels, val_content_scores, best_graph_val_scores, cfg)
        fusion_params_round["content_checkpoint"] = str(content_round_checkpoint)
        fusion_params_round["graph_checkpoint"] = str(graph_best_until_checkpoint)
        fusion_params_round["graph_source_round"] = int(best_graph_round)
        fusion_params_round["fusion_round"] = int(iteration)
        write_json(round_dir / "fusion_params_round.json", fusion_params_round)
        fusion_key = _fusion_metric_tuple(fusion_params_round, cfg)
        fusion_is_best = bool(cfg.get("save_best_fusion_snapshot", True)) and (
            best_fusion_key is None or fusion_key > best_fusion_key
        )
        if fusion_is_best:
            best_fusion_key = fusion_key
            best_fusion_round = int(iteration)
            best_fusion_metrics = dict(fusion_params_round.get("validation_fusion_metrics", {}) or {})
            best_fusion_snapshot = {
                "round": int(iteration),
                "content_checkpoint": str(content_round_checkpoint),
                "graph_checkpoint": str(graph_best_until_checkpoint),
                "graph_source_round": int(best_graph_round),
                "fusion_params_path": str(round_dir / "fusion_params_round.json"),
                "fusion_params": fusion_params_round,
            }
            _log(f"best fusion snapshot updated: {best_fusion_snapshot}")

        _log("graph pseudo label generation start")
        pseudo_from_graph = generate_pseudo_labels(
            data=data,
            scorer=lambda pairs: graph_trainer.predict_scores(pairs, calibrated=True),
            raw_scorer=lambda pairs: graph_trainer.predict_scores(pairs, calibrated=False),
            teacher_model="graph",
            iteration=iteration,
            cfg=cfg,
            base_exclude_dfs=candidate_exclude,
        )
        pseudo_from_graph.to_csv(round_dir / "pseudo_from_graph.csv", index=False, encoding="utf-8")
        _log(f"graph pseudo label generation done: rows={len(pseudo_from_graph)}")
        stable_from_graph, graph_stability = stable_tracker.update(pseudo_from_graph, iteration)
        stable_from_graph.to_csv(round_dir / "stable_from_graph.csv", index=False, encoding="utf-8")
        stable_tracker.to_frame().to_csv(round_dir / "stable_pseudo_memory_after_graph.csv", index=False, encoding="utf-8")
        _log(f"graph pseudo stability done: {graph_stability}")
        r_syn_pos, r_syn_neg, r_neg_dynamic, graph_cleaning = resolve_pseudo_conflicts(
            stable_from_graph, r_syn_pos, r_syn_neg, r_neg_dynamic, r_train
        )
        _log(f"graph conflict cleaning done: {graph_cleaning}")
        hard_negatives, hard_negative_metrics = _mine_hard_negatives(
            data=data,
            profiles=profiles,
            content_model=content_model,
            graph_trainer=graph_trainer,
            r_neg_dynamic=r_neg_dynamic,
            r_syn_pos=r_syn_pos,
            r_syn_neg=r_syn_neg,
            cfg=cfg,
            iteration=iteration,
        )
        hard_negatives.to_csv(round_dir / "hard_negatives_mined.csv", index=False, encoding="utf-8")
        if len(hard_negatives):
            r_syn_pos, r_syn_neg, r_neg_dynamic, hard_negative_cleaning = resolve_pseudo_conflicts(
                hard_negatives, r_syn_pos, r_syn_neg, r_neg_dynamic, r_train
            )
        else:
            hard_negative_cleaning = {}
        hard_negative_metrics = {**hard_negative_metrics, "hard_negative_cleaning": hard_negative_cleaning}
        _log(f"hard negative mining done: {hard_negative_metrics}")
        r_syn_pos.to_csv(round_dir / "r_syn_pos.csv", index=False, encoding="utf-8")
        r_syn_neg.to_csv(round_dir / "r_syn_neg.csv", index=False, encoding="utf-8")
        r_neg_dynamic.to_csv(round_dir / "r_neg_dynamic.csv", index=False, encoding="utf-8")
        pseudo_quality = _pseudo_quality_metrics(r_syn_pos, r_syn_neg)
        round_diagnostic_row = {
            "round": int(iteration + 1),
            "iteration": int(iteration),
            "content_val_auc": float(validation_content_metrics.get("validation_content_auc", 0.0)),
            "graph_val_auc": float(validation_graph_metrics.get("validation_graph_auc", 0.0)),
            "r_syn_pos_rows": int(len(r_syn_pos)),
            "r_syn_neg_rows": int(len(r_syn_neg)),
            **pseudo_quality,
        }
        round_diagnostic_rows.append(round_diagnostic_row)

        summary = {
            "iteration": iteration,
            "content_train_rows": int(len(content_train)),
            "graph_train_rows": int(len(graph_supervised) + len(graph_pseudo)),
            "graph_supervised_rows": int(len(graph_supervised)),
            "graph_pseudo_rows": int(len(graph_pseudo)),
            "r_train_rows": int(len(r_train)),
            "r_neg_dynamic_rows": int(len(r_neg_dynamic)),
            "r_syn_pos_rows": int(len(r_syn_pos)),
            "r_syn_neg_rows": int(len(r_syn_neg)),
            "pseudo_from_content_rows": int(len(pseudo_from_content)),
            "pseudo_from_graph_rows": int(len(pseudo_from_graph)),
            "content_cleaning": content_cleaning,
            "graph_cleaning": graph_cleaning,
            "content_stability": content_stability,
            "graph_stability": graph_stability,
            "content_metrics": content_metrics,
            "validation_content_metrics": validation_content_metrics,
            "graph_metrics": graph_metrics,
            "validation_graph_metrics": validation_graph_metrics,
            "is_best_graph_round": bool(graph_is_best),
            "graph_best_until_round": int(best_graph_round),
            "graph_best_until_checkpoint": str(graph_best_until_checkpoint),
            "fusion_params_round": fusion_params_round,
            "validation_fusion_metrics": fusion_params_round.get("validation_fusion_metrics", {}),
            "is_best_fusion_round": bool(fusion_is_best),
            "hard_negative_metrics": hard_negative_metrics,
            "graph_structure": graph_diag,
            "content_epoch_history_path": str(round_dir / "content_epoch_history.csv"),
            "graph_epoch_history_path": str(round_dir / "graph_epoch_history.csv"),
            "round_diagnostics": round_diagnostic_row,
        }
        write_json(round_dir / "cleaning_metrics.json", {
            "content": content_cleaning,
            "graph": graph_cleaning,
            "hard_negative": hard_negative_cleaning,
            "content_stability": content_stability,
            "graph_stability": graph_stability,
        })
        write_json(round_dir / "round_metrics.json", summary)
        round_summaries.append(summary)
        _log(
            f"Round {iteration + 1}/{rounds} done: "
            f"content_pseudo={len(pseudo_from_content)}, graph_pseudo={len(pseudo_from_graph)}, "
            f"R_syn_pos={len(r_syn_pos)}, R_syn_neg={len(r_syn_neg)}, r_neg_dynamic={len(r_neg_dynamic)}"
        )

    if best_graph_checkpoint is None or best_graph_metrics is None or best_graph_round is None:
        raise RuntimeError("No best graph checkpoint was selected; AKD must run at least one graph round.")
    if best_fusion_snapshot is None or best_fusion_metrics is None or best_fusion_round is None:
        raise RuntimeError("No best fusion snapshot was selected; AKD must run at least one round.")
    content_final_path = checkpoint_dir / "content_final.pt"
    graph_final_path = checkpoint_dir / "graph_final.pt"
    shutil.copy2(best_fusion_snapshot["content_checkpoint"], content_final_path)
    shutil.copy2(best_fusion_snapshot["graph_checkpoint"], graph_final_path)
    fusion_params = dict(best_fusion_snapshot["fusion_params"])
    fusion_params["best_fusion_round"] = int(best_fusion_round)
    fusion_params["content_final_source_checkpoint"] = str(best_fusion_snapshot["content_checkpoint"])
    fusion_params["graph_final_source_checkpoint"] = str(best_fusion_snapshot["graph_checkpoint"])
    write_json(output_dir / "fusion_params.json", fusion_params)
    content_history = (
        pd.concat(content_history_frames, ignore_index=True)
        if content_history_frames
        else pd.DataFrame(columns=["round", "epoch", "global_epoch", "train_total_loss", "val_bce_loss"])
    )
    graph_history = (
        pd.concat(graph_history_frames, ignore_index=True)
        if graph_history_frames
        else pd.DataFrame(
            columns=[
                "round",
                "epoch",
                "global_epoch",
                "train_total_loss",
                "val_bce_loss",
                "gold_bce",
                "pseudo_soft_bce",
                "kd_bce",
                "rank_loss",
                "logit_cap_penalty",
            ]
        )
    )
    round_diagnostics = pd.DataFrame(round_diagnostic_rows)
    content_history_path = output_dir / "training_history_content.csv"
    graph_history_path = output_dir / "training_history_graph.csv"
    round_diagnostics_path = output_dir / "akd_round_diagnostics.csv"
    content_history.to_csv(content_history_path, index=False, encoding="utf-8")
    graph_history.to_csv(graph_history_path, index=False, encoding="utf-8")
    round_diagnostics.to_csv(round_diagnostics_path, index=False, encoding="utf-8")
    diagnostic_plot_paths = generate_diagnostic_plots(output_dir, content_history, graph_history, round_diagnostics)
    _log(f"diagnostic histories saved: {content_history_path}, {graph_history_path}, {round_diagnostics_path}")
    _log(f"diagnostic plots saved: {diagnostic_plot_paths}")
    _log(
        "checkpoint saved: "
        f"{content_final_path} (copied from best fusion round {best_fusion_round}: "
        f"{best_fusion_snapshot['content_checkpoint']})"
    )
    _log(
        "checkpoint saved: "
        f"{graph_final_path} (copied from best fusion round {best_fusion_round}: "
        f"{best_fusion_snapshot['graph_checkpoint']})"
    )
    _log(f"fusion / validation done: {fusion_params}")
    run_summary = {
        "stage": "stage2",
        "rounds_run": rounds,
        "round_summaries": round_summaries,
        "best_graph_round": int(best_graph_round),
        "best_graph_checkpoint": str(best_graph_checkpoint),
        "best_graph_metrics": best_graph_metrics,
        "best_fusion_round": int(best_fusion_round),
        "best_fusion_snapshot": best_fusion_snapshot,
        "best_fusion_metrics": best_fusion_metrics,
        "content_final_source_checkpoint": str(best_fusion_snapshot["content_checkpoint"]),
        "graph_final_source_checkpoint": str(best_fusion_snapshot["graph_checkpoint"]),
        "fusion_final_source_round": int(best_fusion_round),
        "fusion_params": fusion_params,
        "diagnostic_history_files": {
            "content": str(content_history_path),
            "graph": str(graph_history_path),
            "akd_rounds": str(round_diagnostics_path),
        },
        "diagnostic_plot_files": diagnostic_plot_paths,
        "final_checkpoints": {
            "content": str(content_final_path),
            "graph": str(graph_final_path),
        },
        "leakage_guard": "heldout_pairs.csv was excluded from pseudo-label candidates and train/heldout overlap was checked at load time.",
    }
    write_json(output_dir / "stage2_summary.json", run_summary)
    _log("Stage2 done")
    return run_summary

