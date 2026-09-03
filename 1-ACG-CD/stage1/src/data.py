from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

LABEL_COLUMNS = [
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
]


@dataclass(frozen=True, order=True)
class Edge:
    source_id: int
    target_id: int


def clean_field(value: object) -> str:
    return " ".join(str(value).strip().lower().split())


def load_stage0_concepts(stage0_outputs_dir: str | Path) -> pd.DataFrame:
    path = Path(stage0_outputs_dir) / "concepts.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage0 concepts file: {path}")
    df = pd.read_csv(path)
    required = {"concept_id", "concept"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    df = df.sort_values("concept_id").reset_index(drop=True)
    expected_ids = list(range(len(df)))
    actual_ids = df["concept_id"].astype(int).tolist()
    if actual_ids != expected_ids:
        raise ValueError("Stage0 concept_id values must be contiguous 0-based indices.")
    return df


def concept_maps(concepts_df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    name_to_id: Dict[str, int] = {}
    id_to_name = {int(row.concept_id): str(row.concept) for row in concepts_df.itertuples(index=False)}
    for row in concepts_df.itertuples(index=False):
        concept_id = int(row.concept_id)
        names = [str(row.concept)]
        aliases = getattr(row, "aliases", "")
        if aliases is not None and not pd.isna(aliases):
            names.extend(str(aliases).split("::;"))
        for name in names:
            key = clean_field(name)
            if key and key not in name_to_id:
                name_to_id[key] = concept_id
    return name_to_id, id_to_name


def parse_gold_positive_edges(label_file: str | Path, name_to_id: Dict[str, int]) -> Tuple[List[Edge], Dict[str, int]]:
    path = Path(label_file)
    counts: Counter[str] = Counter()
    positives: List[Edge] = []
    seen: set[Edge] = set()
    invalid_rows = 0
    total_rows = 0
    ignored_file_negatives = 0
    duplicate_rows = 0
    self_loop_rows = 0

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.reader(f):
                if not row or not any(str(cell).strip() for cell in row):
                    continue
                if len(row) != 3:
                    invalid_rows += 1
                    continue
                a, b, label = (clean_field(cell) for cell in row)
                if not a or not b or label not in {"0", "1"}:
                    invalid_rows += 1
                    continue
                total_rows += 1
                counts[label] += 1
                if label == "0":
                    ignored_file_negatives += 1
                    continue
                if a not in name_to_id or b not in name_to_id:
                    invalid_rows += 1
                    continue
                edge = Edge(name_to_id[a], name_to_id[b])
                if edge.source_id == edge.target_id:
                    self_loop_rows += 1
                    continue
                if edge in seen:
                    duplicate_rows += 1
                    continue
                positives.append(edge)
                seen.add(edge)
        stats = {
            "label_file_rows": total_rows,
            "label_1": counts.get("1", 0),
            "label_0": counts.get("0", 0),
            "ignored_file_negative_rows": ignored_file_negatives,
            "invalid_rows": invalid_rows,
            "self_loop_rows": self_loop_rows,
            "duplicate_rows": duplicate_rows,
            "unique_directed_positive_edges": len(positives),
        }
        return positives, stats

    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            parts = [clean_field(p) for p in re.split(r"\t+", line.strip()) if p.strip()]
            if len(parts) < 3:
                invalid_rows += 1
                continue
            a, b, label = parts[:3]
            total_rows += 1
            counts[label] += 1
            if label == "-":
                ignored_file_negatives += 1
                continue
            if a not in name_to_id or b not in name_to_id:
                invalid_rows += 1
                continue
            if label == "1-":
                edge = Edge(name_to_id[a], name_to_id[b])
            elif label == "-1":
                edge = Edge(name_to_id[b], name_to_id[a])
            else:
                invalid_rows += 1
                continue
            if edge.source_id == edge.target_id:
                self_loop_rows += 1
                continue
            if edge in seen:
                duplicate_rows += 1
                continue
            positives.append(edge)
            seen.add(edge)

    stats = {
        "label_file_rows": total_rows,
        "label_1-": counts.get("1-", 0),
        "label_-1": counts.get("-1", 0),
        "label_-": counts.get("-", 0),
        "ignored_file_negative_rows": ignored_file_negatives,
        "invalid_rows": invalid_rows,
        "self_loop_rows": self_loop_rows,
        "duplicate_rows": duplicate_rows,
        "unique_directed_positive_edges": len(positives),
    }
    return positives, stats


def parse_label_file_negative_edges(label_file: str | Path, name_to_id: Dict[str, int]) -> Tuple[List[Edge], Dict[str, int]]:
    """Parse directed non-prerequisite pairs from label-file negative rows."""
    path = Path(label_file)
    negatives: List[Edge] = []
    seen: set[Edge] = set()
    total_negative_rows = 0
    invalid_rows = 0
    self_loop_rows = 0
    duplicate_rows = 0

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.reader(f):
                if not row or not any(str(cell).strip() for cell in row):
                    continue
                if len(row) != 3:
                    invalid_rows += 1
                    continue
                a, b, label = (clean_field(cell) for cell in row)
                if label != "0":
                    continue
                total_negative_rows += 1
                if not a or not b or a not in name_to_id or b not in name_to_id:
                    invalid_rows += 1
                    continue
                edge = Edge(name_to_id[a], name_to_id[b])
                if edge.source_id == edge.target_id:
                    self_loop_rows += 1
                    continue
                if edge in seen:
                    duplicate_rows += 1
                    continue
                negatives.append(edge)
                seen.add(edge)
    else:
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                if not line.strip():
                    continue
                parts = [clean_field(p) for p in re.split(r"\t+", line.strip()) if p.strip()]
                if len(parts) < 3:
                    continue
                a, b, label = parts[:3]
                if label != "-":
                    continue
                total_negative_rows += 1
                if a not in name_to_id or b not in name_to_id:
                    invalid_rows += 1
                    continue
                edge = Edge(name_to_id[a], name_to_id[b])
                if edge.source_id == edge.target_id:
                    self_loop_rows += 1
                    continue
                if edge in seen:
                    duplicate_rows += 1
                    continue
                negatives.append(edge)
                seen.add(edge)

    stats = {
        "label_file_negative_rows": total_negative_rows,
        "invalid_label_file_negative_rows": invalid_rows,
        "self_loop_label_file_negative_rows": self_loop_rows,
        "duplicate_label_file_negative_rows": duplicate_rows,
        "unique_label_file_negative_edges": len(negatives),
    }
    return negatives, stats


def load_similarity(stage0_outputs_dir: str | Path) -> np.ndarray:
    path = Path(stage0_outputs_dir) / "similarity_matrix.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage0 similarity matrix: {path}")
    similarity = np.load(path)
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("similarity_matrix.npy must be square")
    if not np.isfinite(similarity).all():
        raise ValueError("similarity_matrix.npy contains NaN or Inf")
    return similarity.astype(np.float32)


def split_positive_edges(
    positives: Sequence[Edge],
    train_ratio: float,
    val_ratio_to_train: float,
    seed: int,
) -> Tuple[List[Edge], List[Edge], List[Edge]]:
    if train_ratio not in (0.15, 0.30, 0.60):
        raise ValueError("train_ratio must be one of 0.15, 0.30, or 0.60")
    train_count = int(round(len(positives) * train_ratio))
    if train_count <= 0 or train_count >= len(positives):
        raise ValueError(f"train_ratio={train_ratio} leaves no validation/test positives")
    val_count = int(round(train_count * val_ratio_to_train))
    if val_count <= 0:
        raise ValueError("val_ratio_to_train produced zero validation samples")
    if train_count + val_count >= len(positives):
        raise ValueError("train_count + val_count must be smaller than total positives so test can exist")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(positives))
    shuffled = [positives[int(i)] for i in indices]
    train = shuffled[:train_count]
    val = shuffled[train_count : train_count + val_count]
    test = shuffled[train_count + val_count :]
    return train, val, test


def make_rows(
    edges: Sequence[Edge],
    id_to_name: Dict[int, str],
    similarity: np.ndarray,
    label: int,
    split: str,
    negative_type: str,
) -> pd.DataFrame:
    rows = []
    for edge in edges:
        sim = float(similarity[edge.source_id, edge.target_id])
        rows.append(
            {
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "source_index": edge.source_id,
                "target_index": edge.target_id,
                "source": id_to_name[edge.source_id],
                "target": id_to_name[edge.target_id],
                "label": int(label),
                "split": split,
                "negative_type": negative_type,
                "similarity": sim,
            }
        )
    return pd.DataFrame(rows, columns=LABEL_COLUMNS)


def empty_label_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=LABEL_COLUMNS)
