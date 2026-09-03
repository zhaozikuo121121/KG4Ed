from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .utils import LABEL_COLUMNS, pair_set


@dataclass
class Stage2Data:
    concepts: pd.DataFrame
    concept_embeddings: np.ndarray
    adjacency: np.ndarray
    similarity: np.ndarray
    train_labels: pd.DataFrame
    val_labels: pd.DataFrame
    test_labels: pd.DataFrame
    heldout_pairs: pd.DataFrame
    pseudo_pos_empty: pd.DataFrame
    pseudo_neg_empty: pd.DataFrame
    id_to_name: Dict[int, str]
    name_to_id: Dict[str, int]

    @property
    def num_concepts(self) -> int:
        return int(len(self.concepts))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def _validate_label_df(df: pd.DataFrame, name: str, num_concepts: int) -> None:
    missing = set(LABEL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    for col in ["source_index", "target_index"]:
        if len(df) and not np.issubdtype(df[col].dtype, np.integer):
            # Pandas may parse empty-compatible columns as float; try strict cast only if exact.
            casted = df[col].astype(int)
            if not np.allclose(df[col].to_numpy(), casted.to_numpy()):
                raise ValueError(f"{name}.{col} must contain integer indices")
            df[col] = casted
        if len(df) and ((df[col] < 0).any() or (df[col] >= num_concepts).any()):
            raise ValueError(f"{name}.{col} index out of range [0, {num_concepts})")
    if len(df):
        if not (df["source_id"].astype(int) == df["source_index"].astype(int)).all():
            raise ValueError(f"{name}: source_id must equal source_index")
        if not (df["target_id"].astype(int) == df["target_index"].astype(int)).all():
            raise ValueError(f"{name}: target_id must equal target_index")
        if (df["source_index"].astype(int) == df["target_index"].astype(int)).any():
            raise ValueError(f"{name}: self-loop found")


def _assert_disjoint(name_a: str, df_a: pd.DataFrame, name_b: str, df_b: pd.DataFrame) -> None:
    overlap = pair_set(df_a) & pair_set(df_b)
    if overlap:
        raise ValueError(f"Leakage detected: {name_a} overlaps {name_b}; sample={sorted(overlap)[:5]}")


def load_stage2_data(stage0_outputs_dir: str | Path, stage1_outputs_dir: str | Path) -> Stage2Data:
    s0 = Path(stage0_outputs_dir)
    s1 = Path(stage1_outputs_dir)
    concepts = _read_csv(s0 / "concepts.csv").sort_values("concept_id").reset_index(drop=True)
    if concepts["concept_id"].astype(int).tolist() != list(range(len(concepts))):
        raise ValueError("Stage0 concepts.csv must use contiguous 0-based concept_id values")

    X = np.load(s0 / "concept_embeddings.npy").astype(np.float32)
    A = np.load(s0 / "adjacency_matrix.npy").astype(np.float32)
    S = np.load(s0 / "similarity_matrix.npy").astype(np.float32)
    n = len(concepts)
    if X.shape[0] != n or A.shape != (n, n) or S.shape != (n, n):
        raise ValueError(f"Stage0 array shape mismatch: concepts={n}, X={X.shape}, A={A.shape}, S={S.shape}")
    if not (np.isfinite(X).all() and np.isfinite(A).all() and np.isfinite(S).all()):
        raise ValueError("Stage0 arrays contain NaN or Inf")

    train = _read_csv(s1 / "train_labels_initial.csv")
    val = _read_csv(s1 / "val_labels.csv")
    test = _read_csv(s1 / "test_labels.csv")
    heldout = _read_csv(s1 / "heldout_pairs.csv")
    pseudo_pos = _read_csv(s1 / "pseudo_pos_empty.csv")
    pseudo_neg = _read_csv(s1 / "pseudo_neg_empty.csv")
    for name, df in [
        ("train_labels_initial", train),
        ("val_labels", val),
        ("test_labels", test),
        ("heldout_pairs", heldout),
        ("pseudo_pos_empty", pseudo_pos),
        ("pseudo_neg_empty", pseudo_neg),
    ]:
        _validate_label_df(df, name, n)

    _assert_disjoint("train_labels_initial", train, "heldout_pairs", heldout)
    _assert_disjoint("train_labels_initial", train, "val_labels", val)
    _assert_disjoint("train_labels_initial", train, "test_labels", test)
    heldout_union = pair_set(val) | pair_set(test)
    if pair_set(heldout) != heldout_union:
        raise ValueError("heldout_pairs.csv must equal val_labels.csv + test_labels.csv as a pair set")

    id_to_name = {int(r.concept_id): str(r.concept) for r in concepts.itertuples(index=False)}
    name_to_id = {v.lower(): k for k, v in id_to_name.items()}
    return Stage2Data(
        concepts=concepts,
        concept_embeddings=X,
        adjacency=A,
        similarity=S,
        train_labels=train,
        val_labels=val,
        test_labels=test,
        heldout_pairs=heldout,
        pseudo_pos_empty=pseudo_pos,
        pseudo_neg_empty=pseudo_neg,
        id_to_name=id_to_name,
        name_to_id=name_to_id,
    )
