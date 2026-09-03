from __future__ import annotations

from collections import deque
import math
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn


REL_SELF_LOOP = 0
REL_SEMANTIC_SIM = 1
REL_PREREQ_FORWARD = 2
REL_PREREQ_BACKWARD = 3
NUM_RELATIONS = 4
RELATION_NAMES = {
    REL_SELF_LOOP: "self_loop",
    REL_SEMANTIC_SIM: "semantic_sim",
    REL_PREREQ_FORWARD: "prereq_forward",
    REL_PREREQ_BACKWARD: "prereq_backward",
}


class RelGraphSAGEConv(nn.Module):
    """Small dependency-free Rel-GraphSAGE convolution for Stage2 concept graphs.

    ``edge_index`` follows the PyG convention: row 0 is source nodes and row 1
    is destination nodes. Messages are aggregated per relation with mean
    normalization, keeping relation-specific parameters separate.
    """

    def __init__(self, in_channels: int, out_channels: int, num_relations: int = NUM_RELATIONS, bias: bool = True):
        super().__init__()
        if num_relations <= 0:
            raise ValueError("num_relations must be positive")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.num_relations = int(num_relations)
        self.self_weight = nn.Parameter(torch.empty(self.in_channels, self.out_channels))
        self.weight = nn.Parameter(torch.empty(self.num_relations, self.in_channels, self.out_channels))
        self.bias = nn.Parameter(torch.empty(self.out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.self_weight)
        for rel in range(self.num_relations):
            nn.init.xavier_uniform_(self.weight[rel])
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if edge_type.ndim != 1 or edge_type.shape[0] != edge_index.shape[1]:
            raise ValueError("edge_type must have shape [num_edges]")
        if edge_index.numel() and (int(edge_type.min()) < 0 or int(edge_type.max()) >= self.num_relations):
            raise ValueError("edge_type contains relation ids outside [0, num_relations)")

        num_nodes = x.shape[0]
        out = x @ self.self_weight
        src_all = edge_index[0].long()
        dst_all = edge_index[1].long()
        for rel in range(self.num_relations):
            mask = edge_type == rel
            if not bool(mask.any()):
                continue
            src = src_all[mask]
            dst = dst_all[mask]
            msg = x.index_select(0, src) @ self.weight[rel]
            deg = torch.bincount(dst, minlength=num_nodes).to(dtype=msg.dtype, device=msg.device).clamp_min_(1.0)
            msg = msg / deg.index_select(0, dst).unsqueeze(-1)
            out.index_add_(0, dst, msg)
        if self.bias is not None:
            out = out + self.bias
        return out


class RelGraphSAGEPrereqModel(nn.Module):
    """Rel-GraphSAGE encoder with source/target asymmetric prerequisite decoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_relations: int = NUM_RELATIONS,
        graphsage_layers: int = 2,
        dropout: float = 0.2,
        decoder_dropout: float | None = None,
        use_residual: bool = True,
        use_layernorm: bool = True,
    ):
        super().__init__()
        if int(graphsage_layers) <= 0:
            raise ValueError("graphsage_layers must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_relations = int(num_relations)
        self.graphsage_layers = int(graphsage_layers)
        self.use_residual = bool(use_residual)
        self.use_layernorm = bool(use_layernorm)
        # Kept for config compatibility; the diagonal bilinear decoder below
        # intentionally has no dropout or MLP capacity.
        _ = decoder_dropout

        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.graphsage_layers_module = nn.ModuleList(
            [
                RelGraphSAGEConv(self.hidden_dim, self.hidden_dim, num_relations=self.num_relations)
                for _ in range(self.graphsage_layers)
            ]
        )
        self.dropout = nn.Dropout(float(dropout))
        self.layer_norms = nn.ModuleList(
            [
                nn.LayerNorm(self.hidden_dim) if self.use_layernorm else nn.Identity()
                for _ in range(self.graphsage_layers)
            ]
        )

        node_dim = 2 * self.hidden_dim
        self.node_dim = int(node_dim)
        self.source_proj = nn.Linear(node_dim, node_dim)
        self.target_proj = nn.Linear(node_dim, node_dim)
        self.bilinear_diag = nn.Parameter(torch.ones(node_dim))
        self.decoder_bias = nn.Parameter(torch.zeros(1))

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        x0 = self.input_proj(x)
        h = x0
        for conv, norm in zip(self.graphsage_layers_module, self.layer_norms):
            h_next = torch.relu(conv(h, edge_index, edge_type))
            if self.use_residual:
                h = norm(h + self.dropout(h_next))
            else:
                h = norm(self.dropout(h_next))
        return torch.cat([x0, h], dim=-1)

    def decode(self, h: torch.Tensor, source_idx: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
        s_i = self.source_proj(h[source_idx])
        t_j = self.target_proj(h[target_idx])
        logits = ((s_i * self.bilinear_diag) * t_j).sum(dim=-1) / math.sqrt(float(self.node_dim))
        return logits + self.decoder_bias.squeeze(0)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        source_idx: torch.Tensor,
        target_idx: torch.Tensor,
    ) -> torch.Tensor:
        h = self.encode(x, edge_index, edge_type)
        return self.decode(h, source_idx, target_idx)


def _largest_connected_component_coverage_from_pairs(num_nodes: int, undirected_pairs: Sequence[tuple[int, int]]) -> float:
    if num_nodes <= 0:
        return 0.0
    neighbors: list[list[int]] = [[] for _ in range(num_nodes)]
    for src, dst in undirected_pairs:
        if src == dst:
            continue
        neighbors[int(src)].append(int(dst))
        neighbors[int(dst)].append(int(src))
    seen = np.zeros(num_nodes, dtype=bool)
    largest = 0
    for start in range(num_nodes):
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
                    q.append(nxt)
        largest = max(largest, size)
    return largest / float(num_nodes)


def _semantic_undirected_topk_pairs(similarity: np.ndarray, k: int) -> tuple[list[tuple[int, int]], float]:
    sim = np.asarray(similarity, dtype=np.float32)
    n = sim.shape[0]
    if sim.shape != (n, n):
        raise ValueError("similarity must be a square matrix")
    if not (0 < int(k) < n):
        raise ValueError(f"k must be in [1, {n - 1}], got {k}")
    directed: set[tuple[int, int]] = set()
    for src in range(n):
        scores = sim[src].copy()
        scores[src] = -np.inf
        top_idx = np.argpartition(-scores, kth=int(k) - 1)[: int(k)]
        top_idx = top_idx[np.argsort(-scores[top_idx], kind="mergesort")]
        for dst in top_idx.tolist():
            if src != int(dst):
                directed.add((src, int(dst)))
    undirected = sorted({(min(src, dst), max(src, dst)) for src, dst in directed if src != dst})
    coverage = _largest_connected_component_coverage_from_pairs(n, undirected)
    return undirected, float(coverage)


def _pair_set(df: pd.DataFrame | None) -> set[tuple[int, int]]:
    if df is None or len(df) == 0 or "source_index" not in df.columns or "target_index" not in df.columns:
        return set()
    return set(zip(df["source_index"].astype(int), df["target_index"].astype(int)))


def _relation_pair_set(edge_index_np: np.ndarray, edge_type_np: np.ndarray, relation_id: int) -> set[tuple[int, int]]:
    if edge_index_np.size == 0:
        return set()
    mask = edge_type_np == int(relation_id)
    return set(zip(edge_index_np[0, mask].astype(int).tolist(), edge_index_np[1, mask].astype(int).tolist()))


def build_rel_graphsage_graph(
    similarity: np.ndarray,
    train_labels: pd.DataFrame,
    num_nodes: int,
    cfg: Dict | None = None,
    val_labels: pd.DataFrame | None = None,
    test_labels: pd.DataFrame | None = None,
    pseudo_labels: pd.DataFrame | None = None,
    excluded_prereq_pairs: set[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Build the fixed four-relation graph used by the Rel-GraphSAGE encoder.

    Only self loops, Stage0 semantic similarity edges, and gold positive
    ``R_train`` prerequisite edges from ``train_labels`` are used for message
    passing. Negatives, held-out positives, and pseudo labels are deliberately
    excluded from graph construction.
    """

    cfg = {} if cfg is None else cfg
    excluded_prereq_pairs = set() if excluded_prereq_pairs is None else {
        (int(src), int(dst)) for src, dst in excluded_prereq_pairs
    }
    top_k = int(cfg.get("semantic_top_k", 10))
    fallback_k = int(cfg.get("semantic_top_k_fallback", 15))
    threshold = float(cfg.get("semantic_connectivity_threshold", 0.9))

    semantic_pairs, coverage = _semantic_undirected_topk_pairs(similarity, top_k)
    selected_k = top_k
    fallback_used = False
    if coverage < threshold and fallback_k != top_k:
        fallback_pairs, fallback_coverage = _semantic_undirected_topk_pairs(similarity, fallback_k)
        semantic_pairs = fallback_pairs
        coverage = fallback_coverage
        selected_k = fallback_k
        fallback_used = True

    edges: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()

    def add_edge(src: int, dst: int, rel: int) -> None:
        key = (int(src), int(dst), int(rel))
        if key not in seen:
            seen.add(key)
            edges.append(key)

    for node in range(int(num_nodes)):
        add_edge(node, node, REL_SELF_LOOP)
    for src, dst in semantic_pairs:
        add_edge(src, dst, REL_SEMANTIC_SIM)
        add_edge(dst, src, REL_SEMANTIC_SIM)

    train_pos = train_labels[train_labels["label"].astype(int) == 1].copy() if len(train_labels) else train_labels.copy()
    excluded_count = 0
    for row in train_pos.itertuples(index=False):
        src = int(row.source_index)
        dst = int(row.target_index)
        if src == dst:
            continue
        if (src, dst) in excluded_prereq_pairs:
            excluded_count += 1
            continue
        add_edge(src, dst, REL_PREREQ_FORWARD)
        add_edge(dst, src, REL_PREREQ_BACKWARD)

    if edges:
        edge_index = np.asarray([[src for src, _, _ in edges], [dst for _, dst, _ in edges]], dtype=np.int64)
        edge_type = np.asarray([rel for _, _, rel in edges], dtype=np.int64)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_type = np.zeros((0,), dtype=np.int64)

    relation_counts = {
        RELATION_NAMES[rel]: int((edge_type == rel).sum())
        for rel in range(NUM_RELATIONS)
    }
    forward_pairs = _relation_pair_set(edge_index, edge_type, REL_PREREQ_FORWARD)
    neg_pairs = _pair_set(train_labels[train_labels["label"].astype(int) == 0]) if len(train_labels) else set()
    val_pairs = _pair_set(val_labels)
    test_pairs = _pair_set(test_labels)
    pseudo_pairs = _pair_set(pseudo_labels)
    leakage = {
        "val_forward_prereq_overlap": int(len(forward_pairs & val_pairs)),
        "test_forward_prereq_overlap": int(len(forward_pairs & test_pairs)),
        "negative_forward_prereq_overlap": int(len(forward_pairs & neg_pairs)),
        "pseudo_forward_prereq_overlap": int(len(forward_pairs & pseudo_pairs)),
    }

    stats: Dict[str, object] = {
        "model_type": "rel_graphsage",
        "num_relations": NUM_RELATIONS,
        "relation_counts": relation_counts,
        "semantic_selected_k": int(selected_k),
        "semantic_fallback_used": bool(fallback_used),
        "semantic_lcc_coverage": float(coverage),
        "semantic_connectivity_threshold": float(threshold),
        "r_train_gold_positive_edges": int(len(train_pos)),
        "excluded_prereq_positive_edges": int(excluded_count),
        "num_edges": int(edge_type.shape[0]),
        "leakage_guard": leakage,
    }
    return edge_index, edge_type, stats


def reverse_ranking_loss(forward_logits: torch.Tensor, reverse_logits: torch.Tensor, margin: float = 0.25) -> torch.Tensor:
    """Probability-margin direction ranking loss.

    The loss only asks ``p(i->j)`` to exceed ``p(j->i)`` by ``margin`` and
    stops pushing once that probability gap is satisfied.  This avoids the
    unbounded logit growth caused by a raw-logit margin.
    """
    if forward_logits.numel() == 0:
        return forward_logits.new_tensor(0.0)
    forward_prob = torch.sigmoid(forward_logits)
    reverse_prob = torch.sigmoid(reverse_logits)
    return torch.relu(float(margin) - forward_prob + reverse_prob).mean()


def make_pair_tensors(df: pd.DataFrame, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    src = torch.tensor(df["source_index"].astype(int).to_numpy(), dtype=torch.long, device=device)
    dst = torch.tensor(df["target_index"].astype(int).to_numpy(), dtype=torch.long, device=device)
    labels = torch.tensor(df["label"].astype(float).to_numpy(), dtype=torch.float32, device=device)
    return src, dst, labels


def predict_graph_scores(
    model: RelGraphSAGEPrereqModel,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    pairs_df: pd.DataFrame,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    if len(pairs_df) == 0:
        return np.asarray([], dtype=np.float32)
    model.eval()
    src_all = pairs_df["source_index"].astype(int).to_numpy()
    dst_all = pairs_df["target_index"].astype(int).to_numpy()
    outs = []
    with torch.no_grad():
        h = model.encode(x, edge_index, edge_type)
        for start in range(0, len(pairs_df), batch_size):
            src = torch.tensor(src_all[start:start + batch_size], dtype=torch.long, device=device)
            dst = torch.tensor(dst_all[start:start + batch_size], dtype=torch.long, device=device)
            logits = model.decode(h, src, dst)
            outs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outs).astype(np.float32)

