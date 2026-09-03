from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np


def binary_metrics(y_true, y_score, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (tp + tn) / len(y_true) if len(y_true) else 0.0
    return {
        "auc": float(auc_score(y_true, y_score)),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(acc),
        "threshold": float(threshold),
    }


def auc_score(y_true, y_score) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # Tie-aware average ranks.
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            avg = (i + 1 + j) / 2.0
            ranks[order[i:j]] = avg
        i = j
    sum_pos_ranks = ranks[pos].sum()
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def find_best_threshold(y_true, y_score, thresholds: Iterable[float]) -> Tuple[float, Dict[str, float]]:
    best_t = 0.5
    best = None
    for t in thresholds:
        cur = binary_metrics(y_true, y_score, threshold=float(t))
        if best is None or (cur["f1"], cur["accuracy"]) > (best["f1"], best["accuracy"]):
            best_t = float(t)
            best = cur
    assert best is not None
    return best_t, best
