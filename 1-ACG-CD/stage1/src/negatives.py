from __future__ import annotations

from typing import Iterable, List, Sequence, Set, Tuple

import numpy as np

from .data import Edge


def make_reverse_negatives(train_pos: Sequence[Edge], gold_pos: Set[Edge], max_count: int) -> List[Edge]:
    if max_count <= 0:
        return []
    negatives: List[Edge] = []
    seen: Set[Edge] = set()
    for edge in train_pos:
        candidate = Edge(edge.target_id, edge.source_id)
        if candidate.source_id == candidate.target_id:
            continue
        if candidate in gold_pos or candidate in seen:
            continue
        negatives.append(candidate)
        seen.add(candidate)
        if len(negatives) >= max_count:
            break
    return negatives


def make_hard_negatives(
    train_pos: Sequence[Edge],
    gold_pos: Set[Edge],
    excluded_negatives: Set[Edge],
    similarity: np.ndarray,
    max_count: int,
) -> List[Edge]:
    if max_count <= 0:
        return []
    train_nodes = sorted({e.source_id for e in train_pos} | {e.target_id for e in train_pos})
    candidates: List[Tuple[float, int, int]] = []
    for source_id in train_nodes:
        for target_id in train_nodes:
            if source_id == target_id:
                continue
            edge = Edge(source_id, target_id)
            if edge in gold_pos or edge in excluded_negatives:
                continue
            candidates.append((float(similarity[source_id, target_id]), source_id, target_id))
    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
    return [Edge(src, dst) for _, src, dst in candidates[:max_count]]


def make_random_non_prerequisite_negatives(
    node_count: int,
    count: int,
    gold_pos: Set[Edge],
    excluded: Set[Edge],
    seed: int,
) -> List[Edge]:
    if count <= 0:
        return []
    candidates: List[Edge] = []
    for source_id in range(node_count):
        for target_id in range(node_count):
            if source_id == target_id:
                continue
            edge = Edge(source_id, target_id)
            if edge in gold_pos or edge in excluded:
                continue
            candidates.append(edge)
    if len(candidates) < count:
        raise ValueError(f"Not enough non-prerequisite candidates: need {count}, have {len(candidates)}")
    rng = np.random.default_rng(seed)
    chosen_idx = rng.choice(len(candidates), size=count, replace=False)
    return [candidates[int(i)] for i in chosen_idx]
