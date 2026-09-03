from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Set

import numpy as np
import pandas as pd

from .data import (
    Edge,
    concept_maps,
    empty_label_frame,
    load_similarity,
    load_stage0_concepts,
    make_rows,
    parse_gold_positive_edges,
    parse_label_file_negative_edges,
    split_positive_edges,
)
from .negatives import make_hard_negatives, make_random_non_prerequisite_negatives, make_reverse_negatives


def _jsonable_config(cfg: Dict) -> Dict:
    return {key: str(value) if isinstance(value, Path) else value for key, value in cfg.items()}


def _edge_set(df: pd.DataFrame) -> Set[tuple[int, int]]:
    if df.empty:
        return set()
    return set(zip(df["source_id"].astype(int), df["target_id"].astype(int)))


def _assert_no_overlap(name_a: str, df_a: pd.DataFrame, name_b: str, df_b: pd.DataFrame) -> None:
    overlap = _edge_set(df_a) & _edge_set(df_b)
    if overlap:
        sample = sorted(overlap)[:5]
        raise AssertionError(f"{name_a} overlaps {name_b}: {sample}")


def validate_outputs(
    all_pos_df: pd.DataFrame,
    train_pos_df: pd.DataFrame,
    val_pos_df: pd.DataFrame,
    test_pos_df: pd.DataFrame,
    train_reverse_df: pd.DataFrame,
    train_hard_df: pd.DataFrame,
    train_file_random_df: pd.DataFrame,
    val_neg_df: pd.DataFrame,
    test_neg_df: pd.DataFrame,
    train_labels_df: pd.DataFrame,
    val_labels_df: pd.DataFrame,
    test_labels_df: pd.DataFrame,
    heldout_df: pd.DataFrame,
) -> None:
    gold = _edge_set(all_pos_df)
    if len(all_pos_df) != len(gold):
        raise AssertionError("positive_edges_all.csv contains duplicate directed pairs")

    for name, df in [
        ("train_pos", train_pos_df),
        ("val_pos", val_pos_df),
        ("test_pos", test_pos_df),
        ("train_neg_reverse", train_reverse_df),
        ("train_neg_hard", train_hard_df),
        ("train_neg_file_random", train_file_random_df),
        ("val_neg", val_neg_df),
        ("test_neg", test_neg_df),
        ("train_labels_initial", train_labels_df),
        ("val_labels", val_labels_df),
        ("test_labels", test_labels_df),
        ("heldout_pairs", heldout_df),
    ]:
        required = {
            "source_id",
            "target_id",
            "source_index",
            "target_index",
            "source",
            "target",
            "label",
            "split",
            "negative_type",
            "similarity",
        }
        missing = required - set(df.columns)
        if missing:
            raise AssertionError(f"{name} missing columns: {sorted(missing)}")
        if not df.empty:
            if not (df["source_id"].astype(int) == df["source_index"].astype(int)).all():
                raise AssertionError(f"{name}: source_id must equal source_index")
            if not (df["target_id"].astype(int) == df["target_index"].astype(int)).all():
                raise AssertionError(f"{name}: target_id must equal target_index")
            if (df["source_id"].astype(int) == df["target_id"].astype(int)).any():
                raise AssertionError(f"{name}: self-loop found")

    # Positive split is disjoint and reconstructs gold.
    _assert_no_overlap("train_pos", train_pos_df, "val_pos", val_pos_df)
    _assert_no_overlap("train_pos", train_pos_df, "test_pos", test_pos_df)
    _assert_no_overlap("val_pos", val_pos_df, "test_pos", test_pos_df)
    if _edge_set(train_pos_df) | _edge_set(val_pos_df) | _edge_set(test_pos_df) != gold:
        raise AssertionError("train/val/test positives do not reconstruct R_gold")

    # Negatives never collide with gold and do not repeat across negative pools.
    negative_frames = [train_reverse_df, train_hard_df, train_file_random_df, val_neg_df, test_neg_df]
    negative_union: set[tuple[int, int]] = set()
    for df in negative_frames:
        cur = _edge_set(df)
        if cur & gold:
            raise AssertionError("negative sample collides with R_gold")
        if cur & negative_union:
            raise AssertionError("negative sample pools overlap")
        negative_union |= cur

    _assert_no_overlap("heldout_pairs", heldout_df, "train_labels_initial", train_labels_df)
    if len(heldout_df) != len(val_labels_df) + len(test_labels_df):
        raise AssertionError("heldout_pairs must equal val_labels + test_labels")


def run_stage1(cfg: Dict) -> Dict:
    data_dir = Path(cfg["data_dir"])
    stage0_outputs_dir = Path(cfg["stage0_outputs_dir"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    concepts_df = load_stage0_concepts(stage0_outputs_dir)
    name_to_id, id_to_name = concept_maps(concepts_df)
    similarity = load_similarity(stage0_outputs_dir)
    if similarity.shape[0] != len(concepts_df):
        raise ValueError("Stage0 similarity matrix size does not match concepts.csv")

    label_file = data_dir / str(cfg["label_file"])
    positives, label_stats = parse_gold_positive_edges(label_file, name_to_id)
    gold_pos = set(positives)
    train_pos, val_pos, test_pos = split_positive_edges(
        positives=positives,
        train_ratio=float(cfg["train_ratio"]),
        val_ratio_to_train=float(cfg["val_ratio_to_train"]),
        seed=int(cfg["seed"]),
    )

    reverse_neg = make_reverse_negatives(
        train_pos=train_pos,
        gold_pos=gold_pos,
        max_count=int(cfg["reverse_negative_count"]),
    )
    if len(reverse_neg) < int(cfg["reverse_negative_count"]):
        raise ValueError(
            f"Only {len(reverse_neg)} reverse negatives available, requested {cfg['reverse_negative_count']}"
        )

    hard_neg = make_hard_negatives(
        train_pos=train_pos,
        gold_pos=gold_pos,
        excluded_negatives=set(reverse_neg),
        similarity=similarity,
        max_count=int(cfg["hard_negative_count"]),
    )
    if len(hard_neg) < int(cfg["hard_negative_count"]):
        raise ValueError(f"Only {len(hard_neg)} hard negatives available, requested {cfg['hard_negative_count']}")

    label_file_negative_stats = {
        "enabled": False,
        "requested": 0,
        "available_after_exclusion": 0,
        "sampled": 0,
    }
    file_random_neg: list[Edge] = []
    label_file_negatives, parsed_negative_stats = parse_label_file_negative_edges(label_file, name_to_id)
    explicit_count = cfg.get("label_file_negative_count", None)
    if explicit_count is not None:
        requested_file_random = int(explicit_count)
    else:
        requested_file_random = int(round(float(cfg.get("label_file_negative_ratio_to_train", 1.5)) * len(train_pos)))
    if requested_file_random > 0 and label_file_negatives:
        excluded_for_file_random = gold_pos | set(reverse_neg) | set(hard_neg)
        available_file_random = [edge for edge in label_file_negatives if edge not in excluded_for_file_random]
        sampled_file_random_count = min(requested_file_random, len(available_file_random))
        rng = np.random.default_rng(int(cfg["seed"]) + 303)
        chosen_idx = rng.choice(len(available_file_random), size=sampled_file_random_count, replace=False)
        file_random_neg = [available_file_random[int(i)] for i in chosen_idx]
        label_file_negative_stats = {
            **parsed_negative_stats,
            "enabled": True,
            "requested": int(requested_file_random),
            "available_after_exclusion": int(len(available_file_random)),
            "sampled": int(len(file_random_neg)),
            "capped_by_available": bool(sampled_file_random_count < requested_file_random),
        }
    else:
        label_file_negative_stats = {
            **parsed_negative_stats,
            "enabled": False,
            "requested": int(requested_file_random),
            "available_after_exclusion": 0,
            "sampled": 0,
        }

    excluded_for_eval_neg = set(reverse_neg) | set(hard_neg) | set(file_random_neg)
    val_neg = make_random_non_prerequisite_negatives(
        node_count=len(concepts_df),
        count=len(val_pos),
        gold_pos=gold_pos,
        excluded=excluded_for_eval_neg,
        seed=int(cfg["seed"]) + 101,
    )
    excluded_for_test_neg = excluded_for_eval_neg | set(val_neg)
    test_neg = make_random_non_prerequisite_negatives(
        node_count=len(concepts_df),
        count=len(test_pos),
        gold_pos=gold_pos,
        excluded=excluded_for_test_neg,
        seed=int(cfg["seed"]) + 202,
    )

    all_pos_df = make_rows(positives, id_to_name, similarity, label=1, split="gold_all", negative_type="none")
    train_pos_df = make_rows(train_pos, id_to_name, similarity, label=1, split="train", negative_type="none")
    val_pos_df = make_rows(val_pos, id_to_name, similarity, label=1, split="val", negative_type="none")
    test_pos_df = make_rows(test_pos, id_to_name, similarity, label=1, split="test", negative_type="none")

    train_reverse_df = make_rows(reverse_neg, id_to_name, similarity, label=0, split="train", negative_type="reverse")
    train_hard_df = make_rows(hard_neg, id_to_name, similarity, label=0, split="train", negative_type="hard")
    train_file_random_df = make_rows(
        file_random_neg, id_to_name, similarity, label=0, split="train", negative_type="label_file_random"
    )
    val_neg_df = make_rows(
        val_neg, id_to_name, similarity, label=0, split="val", negative_type="random_non_prerequisite"
    )
    test_neg_df = make_rows(
        test_neg, id_to_name, similarity, label=0, split="test", negative_type="random_non_prerequisite"
    )

    train_label_frames = [train_pos_df]
    if len(train_reverse_df):
        train_label_frames.append(train_reverse_df)
    if len(train_hard_df):
        train_label_frames.append(train_hard_df)
    if len(train_file_random_df):
        train_label_frames.append(train_file_random_df)
    train_labels_df = pd.concat(train_label_frames, ignore_index=True)
    val_labels_df = pd.concat([val_pos_df, val_neg_df], ignore_index=True)
    test_labels_df = pd.concat([test_pos_df, test_neg_df], ignore_index=True)
    heldout_df = pd.concat([val_labels_df, test_labels_df], ignore_index=True)
    pseudo_pos_df = empty_label_frame()
    pseudo_neg_df = empty_label_frame()

    validate_outputs(
        all_pos_df,
        train_pos_df,
        val_pos_df,
        test_pos_df,
        train_reverse_df,
        train_hard_df,
        train_file_random_df,
        val_neg_df,
        test_neg_df,
        train_labels_df,
        val_labels_df,
        test_labels_df,
        heldout_df,
    )

    outputs = {
        "positive_edges_all.csv": all_pos_df,
        "train_pos.csv": train_pos_df,
        "val_pos.csv": val_pos_df,
        "test_pos.csv": test_pos_df,
        "train_neg_reverse.csv": train_reverse_df,
        "train_neg_hard.csv": train_hard_df,
        "train_neg_file_random.csv": train_file_random_df,
        "val_neg.csv": val_neg_df,
        "test_neg.csv": test_neg_df,
        "train_labels_initial.csv": train_labels_df,
        "val_labels.csv": val_labels_df,
        "test_labels.csv": test_labels_df,
        "heldout_pairs.csv": heldout_df,
        "pseudo_pos_empty.csv": pseudo_pos_df,
        "pseudo_neg_empty.csv": pseudo_neg_df,
    }
    for filename, df in outputs.items():
        df.to_csv(output_dir / filename, index=False, encoding="utf-8")

    summary = {
        "stage": "stage1",
        "config": _jsonable_config(cfg),
        "concept_count": int(len(concepts_df)),
        "label_stats": label_stats,
        "split_counts": {
            "positive_edges_all": int(len(all_pos_df)),
            "train_pos": int(len(train_pos_df)),
            "val_pos": int(len(val_pos_df)),
            "test_pos": int(len(test_pos_df)),
            "train_neg_reverse": int(len(train_reverse_df)),
            "train_neg_hard": int(len(train_hard_df)),
            "train_neg_file_random": int(len(train_file_random_df)),
            "val_neg": int(len(val_neg_df)),
            "test_neg": int(len(test_neg_df)),
            "train_labels_initial": int(len(train_labels_df)),
            "val_labels": int(len(val_labels_df)),
            "test_labels": int(len(test_labels_df)),
            "heldout_pairs": int(len(heldout_df)),
        },
        "label_file_random_negative_stats": label_file_negative_stats,
        "schema": list(train_labels_df.columns),
        "leakage_guard": "Stage2 must exclude heldout_pairs.csv from supervised training and pseudo-label generation candidates.",
        "notes": [
            (
                f"Sampled negative rows in {cfg['label_file']} are added as label_file_random training negatives; "
                "remaining file negatives stay unused."
                if label_file_negative_stats.get("enabled")
                else f"No label-file negatives from {cfg['label_file']} were copied into Stage1 outputs."
            ),
            "source_id/source_index and target_id/target_index are both Stage0 concept_id values for direct indexing into Stage0 numpy arrays.",
        ],
        "outputs": {filename: str(output_dir / filename) for filename in outputs},
    }
    (output_dir / "stage1_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
