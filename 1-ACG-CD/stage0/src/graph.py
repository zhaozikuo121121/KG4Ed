from __future__ import annotations

from collections import deque
from typing import Dict, List, Tuple

import numpy as np


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    x = embeddings.astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    x_norm = x / norms
    sim = x_norm @ x_norm.T
    sim = np.clip(sim, -1.0, 1.0).astype(np.float32)
    np.fill_diagonal(sim, 1.0)
    return sim


def build_knn_adjacency(similarity: np.ndarray, k: int) -> np.ndarray:
    n = similarity.shape[0]
    if similarity.shape != (n, n):
        raise ValueError("similarity must be a square matrix")
    if k <= 0 or k >= n:
        raise ValueError(f"k must be in [1, {n - 1}], got {k}")

    directed = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        scores = similarity[i].copy()
        scores[i] = -np.inf
        # argpartition is efficient; final sorting makes output deterministic.
        top_idx = np.argpartition(-scores, kth=k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx], kind="mergesort")]
        directed[i, top_idx] = similarity[i, top_idx]

    adjacency = np.maximum(directed, directed.T).astype(np.float32)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def largest_connected_component_coverage(adjacency: np.ndarray) -> float:
    n = adjacency.shape[0]
    seen = np.zeros(n, dtype=bool)
    largest = 0
    neighbors = [np.flatnonzero(adjacency[i] != 0.0) for i in range(n)]
    for start in range(n):
        if seen[start]:
            continue
        q: deque[int] = deque([start])
        seen[start] = True
        size = 0
        while q:
            cur = q.popleft()
            size += 1
            for nxt in neighbors[cur]:
                if not seen[nxt]:
                    seen[nxt] = True
                    q.append(int(nxt))
        largest = max(largest, size)
    return largest / float(n) if n else 0.0


def build_initial_graph(
    similarity: np.ndarray,
    k: int,
    fallback_k: int,
    min_lcc_coverage: float,
) -> Tuple[np.ndarray, Dict[str, float | int | bool]]:
    adjacency = build_knn_adjacency(similarity, k=k)
    coverage = largest_connected_component_coverage(adjacency)
    selected_k = k
    used_fallback = False
    if coverage < min_lcc_coverage and fallback_k != k:
        fallback_adj = build_knn_adjacency(similarity, k=fallback_k)
        fallback_coverage = largest_connected_component_coverage(fallback_adj)
        adjacency = fallback_adj
        coverage = fallback_coverage
        selected_k = fallback_k
        used_fallback = True
    stats: Dict[str, float | int | bool] = {
        "selected_k": selected_k,
        "fallback_used": used_fallback,
        "largest_connected_component_coverage": float(coverage),
        "edge_count_undirected": int(np.count_nonzero(np.triu(adjacency, k=1))),
    }
    return adjacency, stats


def edge_rows_from_adjacency(adjacency: np.ndarray, concept_names: List[str]) -> List[Dict[str, int | str | float]]:
    rows: List[Dict[str, int | str | float]] = []
    srcs, dsts = np.nonzero(np.triu(adjacency, k=1))
    for src, dst in zip(srcs.tolist(), dsts.tolist()):
        rows.append(
            {
                "source_id": int(src),
                "target_id": int(dst),
                "source": concept_names[src],
                "target": concept_names[dst],
                "weight": float(adjacency[src, dst]),
            }
        )
    rows.sort(key=lambda r: (int(r["source_id"]), int(r["target_id"])))
    return rows
