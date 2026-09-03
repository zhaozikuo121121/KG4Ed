from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from .metrics import binary_metrics, find_best_threshold
from .utils import write_json


def search_alpha_and_threshold(
    y_true,
    content_scores: np.ndarray,
    graph_scores: np.ndarray,
    alpha_values: Iterable[float],
    threshold_values: Iterable[float],
) -> Dict:
    best = None
    for alpha in alpha_values:
        fused = float(alpha) * content_scores + (1.0 - float(alpha)) * graph_scores
        threshold, metrics = find_best_threshold(y_true, fused, threshold_values)
        row = {"alpha": float(alpha), "threshold": float(threshold), **metrics}
        if best is None or (row["f1"], row["auc"], row["accuracy"]) > (best["f1"], best["auc"], best["accuracy"]):
            best = row
    assert best is not None
    return best


def make_predictions_df(df: pd.DataFrame, content_scores: np.ndarray, graph_scores: np.ndarray, alpha: float, threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["content_score"] = content_scores
    out["graph_score"] = graph_scores
    out["final_score"] = alpha * content_scores + (1.0 - alpha) * graph_scores
    out["pred_label"] = (out["final_score"] >= threshold).astype(int)
    return out


def select_fusion_params_and_save(
    val_df: pd.DataFrame,
    val_content: np.ndarray,
    val_graph: np.ndarray,
    cfg: Dict,
    output_dir: str | Path,
) -> Dict:
    alpha_values = [float(x) for x in cfg.get("alpha_search_values", [i / 10.0 for i in range(11)])]
    threshold_values = [float(x) for x in cfg.get("threshold_search_values", [i / 10.0 for i in range(1, 10)])]
    best = search_alpha_and_threshold(
        val_df["label"].astype(int).to_numpy(), val_content, val_graph, alpha_values, threshold_values
    )
    content_val_threshold, content_val_metrics = find_best_threshold(
        val_df["label"].astype(int).to_numpy(), val_content, threshold_values
    )
    graph_val_threshold, graph_val_metrics = find_best_threshold(
        val_df["label"].astype(int).to_numpy(), val_graph, threshold_values
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    params = {
        "selection_set": "validation",
        "selected_alpha": best["alpha"],
        "selected_threshold": best["threshold"],
        "alpha_search_values": alpha_values,
        "threshold_search_values": threshold_values,
        "validation_fusion_metrics": best,
        "validation_content_metrics": content_val_metrics,
        "validation_graph_metrics": graph_val_metrics,
    }
    write_json(output_dir / "fusion_params.json", params)
    return params


def predict_and_score(
    df: pd.DataFrame,
    content_scores: np.ndarray,
    graph_scores: np.ndarray,
    alpha: float,
    threshold: float,
) -> tuple[pd.DataFrame, Dict]:
    pred = make_predictions_df(df, content_scores, graph_scores, alpha, threshold)
    metrics = binary_metrics(df["label"].astype(int).to_numpy(), pred["final_score"].to_numpy(), threshold)
    return pred, metrics
