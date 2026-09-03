from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .graph_model import (
    NUM_RELATIONS,
    REL_PREREQ_FORWARD,
    REL_SELF_LOOP,
    RelGraphSAGEPrereqModel,
    build_rel_graphsage_graph,
    make_pair_tensors,
    reverse_ranking_loss,
)


_EPS = 1.0e-6


def _to_numpy_probs(x: torch.Tensor) -> np.ndarray:
    return torch.sigmoid(x).detach().cpu().numpy().astype(np.float32)


def _prob_logit_np(prob: np.ndarray, eps: float = _EPS) -> np.ndarray:
    p = np.clip(np.asarray(prob, dtype=np.float32), eps, 1.0 - eps)
    return np.log(p / (1.0 - p)).astype(np.float32)


def _score_stats(scores: np.ndarray, prefix: str) -> Dict[str, float]:
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return {f"{prefix}_{k}": 0.0 for k in ["min", "p01", "p05", "p10", "p50", "p90", "p95", "p99", "max"]}
    return {
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_p01": float(np.quantile(arr, 0.01)),
        f"{prefix}_p05": float(np.quantile(arr, 0.05)),
        f"{prefix}_p10": float(np.quantile(arr, 0.10)),
        f"{prefix}_p50": float(np.quantile(arr, 0.50)),
        f"{prefix}_p90": float(np.quantile(arr, 0.90)),
        f"{prefix}_p95": float(np.quantile(arr, 0.95)),
        f"{prefix}_p99": float(np.quantile(arr, 0.99)),
        f"{prefix}_max": float(np.max(arr)),
    }


def _brier_score(labels: np.ndarray, probs: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.float32)
    p = np.asarray(probs, dtype=np.float32)
    return float(np.mean((p - y) ** 2)) if y.size else 0.0


def _ece_score(labels: np.ndarray, probs: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(labels, dtype=np.float32)
    p = np.asarray(probs, dtype=np.float32)
    if y.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi >= 1.0:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not mask.any():
            continue
        conf = float(p[mask].mean())
        acc = float(y[mask].mean())
        ece += float(mask.mean()) * abs(conf - acc)
    return float(ece)


def _logit_cap_penalty(logits: torch.Tensor, cap: float) -> torch.Tensor:
    if logits.numel() == 0 or float(cap) <= 0:
        return logits.new_tensor(0.0)
    return torch.relu(torch.abs(logits) - float(cap)).pow(2).mean()


def _weighted_soft_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits.numel() == 0:
        return logits.new_tensor(0.0)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if sample_weights is not None:
        loss = loss * sample_weights
    return loss.mean()


def _split_frame(df: pd.DataFrame, folds: int) -> list[pd.DataFrame]:
    if int(folds) <= 1:
        return [df.reset_index(drop=True)]
    indices = np.array_split(np.arange(len(df)), int(folds))
    return [df.iloc[idx].reset_index(drop=True) for idx in indices]


class GraphTrainer:
    def __init__(self, cfg: Dict, data, device: torch.device):
        self.cfg = cfg
        self.data = data
        self.device = device
        self.x = torch.tensor(data.concept_embeddings, dtype=torch.float32, device=device)
        edge_index_np, edge_type_np, graph_stats = build_rel_graphsage_graph(
            similarity=data.similarity,
            train_labels=data.train_labels,
            num_nodes=data.num_concepts,
            cfg=cfg,
            val_labels=data.val_labels,
            test_labels=data.test_labels,
        )
        self.edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=device)
        self.edge_type = torch.tensor(edge_type_np, dtype=torch.long, device=device)
        self.graph_stats = graph_stats
        self._forward_prereq_pairs = self._pairs_for_relations({REL_PREREQ_FORWARD})
        self._gold_pos_pairs = self._pair_set(data.train_labels[data.train_labels["label"].astype(int) == 1])
        self.calibration_temperature = float(cfg.get("graph_temperature_init", 1.0))
        self.calibration_bias = 0.0
        self.calibration_fitted = False
        self.calibration_metrics: Dict[str, float | str | bool] = {}
        self.model = RelGraphSAGEPrereqModel(
            input_dim=data.concept_embeddings.shape[1],
            hidden_dim=int(cfg.get("graph_hidden_dim", cfg.get("hidden_dim", 128))),
            num_relations=int(cfg.get("num_relations", NUM_RELATIONS)),
            graphsage_layers=int(cfg.get("graphsage_layers", 2)),
            dropout=float(cfg.get("graph_dropout", cfg.get("dropout", 0.2))),
            decoder_dropout=float(cfg.get("decoder_dropout", cfg.get("graph_decoder_dropout", cfg.get("graph_dropout", 0.2)))),
            use_residual=bool(cfg.get("use_residual", True)),
            use_layernorm=bool(cfg.get("use_layernorm", True)),
        ).to(device)

    @staticmethod
    def _pair_set(df: pd.DataFrame) -> set[tuple[int, int]]:
        if len(df) == 0 or "source_index" not in df.columns or "target_index" not in df.columns:
            return set()
        return set(zip(df["source_index"].astype(int), df["target_index"].astype(int)))

    def _pairs_for_relations(self, relation_ids: set[int]) -> set[tuple[int, int]]:
        if self.edge_index.numel() == 0:
            return set()
        edge_index_np = self.edge_index.detach().cpu().numpy()
        edge_type_np = self.edge_type.detach().cpu().numpy()
        mask = np.isin(edge_type_np, list(relation_ids))
        return set(zip(edge_index_np[0, mask].astype(int).tolist(), edge_index_np[1, mask].astype(int).tolist()))

    def _pseudo_weight_base_for_iteration(self, iteration: int | None) -> float:
        if iteration is None:
            return float(self.cfg.get("pseudo_weight", self.cfg.get("pseudo_weight_round1", 0.1)))
        if int(iteration) <= 0:
            return float(self.cfg.get("pseudo_weight_round1", self.cfg.get("pseudo_weight", 0.1)))
        if int(iteration) == 1:
            return float(self.cfg.get("pseudo_weight_round2", self.cfg.get("pseudo_weight", 0.2)))
        return float(
            self.cfg.get(
                "pseudo_weight_round3_plus",
                self.cfg.get("pseudo_weight_round3", self.cfg.get("pseudo_weight", 0.3)),
            )
        )

    def _prepare_pseudo_frame(self, pseudo_df: pd.DataFrame, iteration: int | None = None) -> pd.DataFrame:
        if len(pseudo_df) == 0:
            out = pseudo_df.copy()
            out["soft_target"] = pd.Series(dtype=float)
            out["sample_weight"] = pd.Series(dtype=float)
            out["kd_target"] = pd.Series(dtype=float)
            out["pseudo_weight_base_round"] = pd.Series(dtype=float)
            return out
        out = pseudo_df.copy()
        if "label" in out.columns:
            out["label"] = (out["label"].astype(float) >= 0.5).astype(int)
        else:
            out["label"] = 0

        if "combined_score" in out.columns:
            final_score = out["combined_score"].astype(float)
        elif "model_score" in out.columns:
            final_score = out["model_score"].astype(float)
        else:
            final_score = out["label"].astype(float)
        final_score_np = np.clip(final_score.to_numpy(dtype=float), 0.0, 1.0)
        soft = np.clip(
            final_score_np,
            float(self.cfg.get("pseudo_soft_target_min", 0.10)),
            float(self.cfg.get("pseudo_soft_target_max", 0.90)),
        )
        confidence = np.abs(soft - 0.5) * 2.0
        pseudo_weight = self._pseudo_weight_base_for_iteration(iteration)
        min_weight = float(self.cfg.get("pseudo_min_weight", 0.05))
        max_weight = float(self.cfg.get("pseudo_max_weight", 0.5))
        weights = np.clip(pseudo_weight * confidence, min_weight, max_weight)
        out["soft_target"] = soft.astype(np.float32)
        out["kd_target"] = soft.astype(np.float32)
        out["sample_weight"] = weights.astype(np.float32)
        out["label_confidence"] = confidence.astype(np.float32)
        out["pseudo_weight_base_round"] = float(pseudo_weight)
        return out

    def _build_training_graph_folds(self, train_df: pd.DataFrame) -> list[Dict[str, object]]:
        positives = train_df[train_df["label"].astype(int) == 1].copy().reset_index(drop=True)
        negatives = train_df[train_df["label"].astype(int) == 0].copy().reset_index(drop=True)
        requested_folds = int(self.cfg.get("graph_edge_masking_folds", 5))
        if len(positives) <= 0:
            requested_folds = 1
        folds = max(1, min(requested_folds, max(1, len(positives))))
        pos_splits = _split_frame(positives, folds)
        neg_splits = _split_frame(negatives, folds)
        out: list[Dict[str, object]] = []
        for fold_id in range(folds):
            pos_fold = pos_splits[fold_id]
            neg_fold = neg_splits[fold_id] if fold_id < len(neg_splits) else negatives.iloc[0:0].copy()
            supervised_fold = pd.concat([pos_fold, neg_fold], ignore_index=True)
            excluded = self._pair_set(pos_fold)
            edge_index_np, edge_type_np, stats = build_rel_graphsage_graph(
                similarity=self.data.similarity,
                train_labels=self.data.train_labels,
                num_nodes=self.data.num_concepts,
                cfg=self.cfg,
                val_labels=self.data.val_labels,
                test_labels=self.data.test_labels,
                excluded_prereq_pairs=excluded,
            )
            out.append(
                {
                    "fold_id": int(fold_id),
                    "train_df": supervised_fold,
                    "rank_df": self._rank_positive_frame(supervised_fold)[0],
                    "excluded_pairs": excluded,
                    "edge_index": torch.tensor(edge_index_np, dtype=torch.long, device=self.device),
                    "edge_type": torch.tensor(edge_type_np, dtype=torch.long, device=self.device),
                    "stats": stats,
                }
            )
        return out

    def _rank_positive_frame(self, train_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        if len(train_df) == 0:
            return train_df.copy(), 0
        df = train_df[train_df["label"].astype(int) == 1].copy()
        if len(df) == 0:
            return df, 0
        negative_type = df.get("negative_type", pd.Series(["none"] * len(df), index=df.index)).astype(str)
        is_gold = negative_type.eq("none")
        pseudo_rows = int((~is_gold).sum())
        if bool(self.cfg.get("rank_on_gold_only", True)):
            df = df[is_gold].copy()
        elif not bool(self.cfg.get("rank_on_pseudo", False)):
            df = df[is_gold].copy()
        if len(df) == 0:
            return df, pseudo_rows
        keep_idx = []
        for idx, row in df.iterrows():
            pair = (int(row.source_index), int(row.target_index))
            reverse_pair = (pair[1], pair[0])
            if pair[0] != pair[1] and reverse_pair not in self._gold_pos_pairs:
                keep_idx.append(idx)
        return df.loc[keep_idx].reset_index(drop=True), pseudo_rows

    @staticmethod
    def _supervised_pos_weight(train_df: pd.DataFrame, device: torch.device) -> torch.Tensor | None:
        if len(train_df) == 0:
            return None
        labels = train_df["label"].astype(int).to_numpy()
        pos = int((labels == 1).sum())
        neg = int((labels == 0).sum())
        if pos <= 0 or neg <= 0:
            return torch.tensor(1.0, dtype=torch.float32, device=device)
        return torch.tensor(float(neg) / float(pos), dtype=torch.float32, device=device)

    def _dynamic_leakage_guard(self, pseudo_df: pd.DataFrame) -> Dict[str, int]:
        pseudo_pairs = self._pair_set(pseudo_df)
        train_neg_pairs = self._pair_set(self.data.train_labels[self.data.train_labels["label"].astype(int) == 0])
        val_pairs = self._pair_set(self.data.val_labels)
        test_pairs = self._pair_set(self.data.test_labels)
        return {
            "val_forward_prereq_overlap": int(len(self._forward_prereq_pairs & val_pairs)),
            "test_forward_prereq_overlap": int(len(self._forward_prereq_pairs & test_pairs)),
            "negative_forward_prereq_overlap": int(len(self._forward_prereq_pairs & train_neg_pairs)),
            "pseudo_forward_prereq_overlap": int(len(self._forward_prereq_pairs & pseudo_pairs)),
        }

    def _augment_relational_graph_view(
        self,
        edge_dropout: float = 0.10,
        feature_dropout: float = 0.10,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_index = self.edge_index
        edge_type = self.edge_type
        x = self.x.clone()
        if edge_dropout > 0 and edge_type.numel() > 0:
            keep = torch.ones(edge_type.shape[0], dtype=torch.bool, device=edge_type.device)
            non_self = edge_type != REL_SELF_LOOP
            keep[non_self] = torch.rand(int(non_self.sum()), device=edge_type.device) >= float(edge_dropout)
            edge_index = edge_index[:, keep]
            edge_type = edge_type[keep]
        if feature_dropout > 0:
            keep_feat = (torch.rand_like(x) >= float(feature_dropout)).to(dtype=x.dtype)
            x = x * keep_feat
        return edge_index, edge_type, x

    def _encode_current(self) -> torch.Tensor:
        return self.model.encode(self.x, self.edge_index, self.edge_type)

    def _decode_logits(self, pairs_df: pd.DataFrame, batch_size: int = 4096) -> np.ndarray:
        if len(pairs_df) == 0:
            return np.asarray([], dtype=np.float32)
        self.model.eval()
        src_all = pairs_df["source_index"].astype(int).to_numpy()
        dst_all = pairs_df["target_index"].astype(int).to_numpy()
        outs = []
        with torch.no_grad():
            h = self._encode_current()
            for start in range(0, len(pairs_df), batch_size):
                src = torch.tensor(src_all[start:start + batch_size], dtype=torch.long, device=self.device)
                dst = torch.tensor(dst_all[start:start + batch_size], dtype=torch.long, device=self.device)
                outs.append(self.model.decode(h, src, dst).detach().cpu().numpy())
        return np.concatenate(outs).astype(np.float32)

    def _apply_calibration_np(self, logits: np.ndarray) -> np.ndarray:
        scaled = np.asarray(logits, dtype=np.float32) / max(float(self.calibration_temperature), _EPS) + float(self.calibration_bias)
        return (1.0 / (1.0 + np.exp(-scaled))).astype(np.float32)

    def fit_calibration(self, val_df: pd.DataFrame) -> Dict[str, float | str | bool]:
        logits_np = self.predict_logits(val_df)
        labels_np = val_df["label"].astype(float).to_numpy(dtype=np.float32)
        raw_probs = 1.0 / (1.0 + np.exp(-logits_np))
        if not bool(self.cfg.get("use_graph_calibration", True)) or len(val_df) == 0:
            self.calibration_temperature = 1.0
            self.calibration_bias = 0.0
            self.calibration_fitted = False
        else:
            logits = torch.tensor(logits_np, dtype=torch.float32, device=self.device)
            labels = torch.tensor(labels_np, dtype=torch.float32, device=self.device)
            log_temp = torch.tensor(
                math.log(max(float(self.cfg.get("graph_temperature_init", 1.0)), _EPS)),
                dtype=torch.float32,
                device=self.device,
                requires_grad=True,
            )
            bias = torch.tensor(0.0, dtype=torch.float32, device=self.device, requires_grad=True)
            min_t = float(self.cfg.get("graph_temperature_min", 0.5))
            max_t = float(self.cfg.get("graph_temperature_max", 10.0))
            optimizer = torch.optim.LBFGS([log_temp, bias], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")

            def closure():
                optimizer.zero_grad(set_to_none=True)
                temp = torch.clamp(torch.exp(log_temp), min=min_t, max=max_t)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits / temp + bias, labels)
                loss.backward()
                return loss

            try:
                optimizer.step(closure)
            except RuntimeError:
                adam = torch.optim.Adam([log_temp, bias], lr=0.05)
                for _ in range(200):
                    adam.zero_grad(set_to_none=True)
                    temp = torch.clamp(torch.exp(log_temp), min=min_t, max=max_t)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits / temp + bias, labels)
                    loss.backward()
                    adam.step()
            self.calibration_temperature = float(torch.clamp(torch.exp(log_temp), min=min_t, max=max_t).detach().cpu())
            self.calibration_bias = float(bias.detach().cpu())
            self.calibration_fitted = True
        cal_probs = self._apply_calibration_np(logits_np)
        metrics: Dict[str, float | str | bool] = {
            "graph_temperature": float(self.calibration_temperature),
            "graph_calibration_bias": float(self.calibration_bias),
            "graph_calibration_fitted": bool(self.calibration_fitted),
            "graph_brier_score": _brier_score(labels_np, cal_probs),
            "graph_ece": _ece_score(labels_np, cal_probs),
            **_score_stats(raw_probs, "raw_graph_score"),
            **_score_stats(cal_probs, "calibrated_graph_score"),
        }
        warning = bool(metrics["calibrated_graph_score_p90"] > 0.99 and metrics["calibrated_graph_score_p50"] < 0.05)
        metrics["graph_calibration_saturation_warning"] = warning
        if warning:
            metrics["graph_calibration_warning_message"] = "calibrated graph scores remain saturated: p90 > 0.99 and p50 < 0.05"
        self.calibration_metrics = metrics
        return metrics

    def fit(
        self,
        train_df: pd.DataFrame,
        pseudo_df: pd.DataFrame | None = None,
        epochs: int | None = None,
        iteration: int | None = None,
        val_df: pd.DataFrame | None = None,
    ) -> Dict[str, float]:
        if epochs is None:
            epochs = int(self.cfg.get("graph_epochs", 200))
        pseudo_df = pd.DataFrame() if pseudo_df is None else self._prepare_pseudo_frame(pseudo_df, iteration=iteration)
        val_df = pd.DataFrame() if val_df is None else val_df
        if len(train_df) == 0 and len(pseudo_df) == 0:
            return {"graph_loss": 0.0, "graph_model_type": "rel_graphsage", "epoch_history": []}

        training_folds = self._build_training_graph_folds(train_df) if len(train_df) else []
        pos_weight = self._supervised_pos_weight(train_df, self.device) if len(train_df) else None
        gold_pos_target = float(self.cfg.get("graph_gold_pos_target", 0.95))
        gold_neg_target = float(self.cfg.get("graph_gold_neg_target", 0.05))

        if len(pseudo_df):
            pseudo_src = torch.tensor(pseudo_df["source_index"].astype(int).to_numpy(), dtype=torch.long, device=self.device)
            pseudo_dst = torch.tensor(pseudo_df["target_index"].astype(int).to_numpy(), dtype=torch.long, device=self.device)
            pseudo_targets = torch.tensor(pseudo_df["soft_target"].astype(float).to_numpy(), dtype=torch.float32, device=self.device)
            pseudo_sample_weights = torch.tensor(pseudo_df["sample_weight"].astype(float).to_numpy(), dtype=torch.float32, device=self.device)
        else:
            pseudo_src = pseudo_dst = pseudo_targets = pseudo_sample_weights = None

        rank_df_full, pseudo_rank_rows = self._rank_positive_frame(train_df)
        if bool(self.cfg.get("rank_on_gold_only", True)) and bool(self.cfg.get("rank_on_pseudo", False)):
            raise ValueError("rank_on_gold_only=true but rank_on_pseudo=true would allow pseudo ranking rows")

        lr = float(self.cfg.get("graph_learning_rate", self.cfg.get("graph_lr", 5e-3)))
        weight_decay = float(self.cfg.get("graph_weight_decay", self.cfg.get("weight_decay", 5e-4)))
        kd_weight = float(self.cfg.get("graph_kd_weight", self.cfg.get("beta_kd", 0.05)))
        kd_temperature = float(self.cfg.get("kd_temperature", 3.0))
        kd_conf_threshold = float(self.cfg.get("kd_confidence_threshold", 0.2))
        rank_weight = float(self.cfg.get("graph_rank_weight", self.cfg.get("lambda_rank", 0.05)))
        rank_margin_prob = float(self.cfg.get("rank_margin_prob", self.cfg.get("graph_rank_margin", self.cfg.get("rank_margin", 0.25))))
        logit_cap = float(self.cfg.get("graph_logit_cap", 5.0))
        logit_penalty_weight = float(self.cfg.get("graph_logit_penalty_weight", 0.003))
        use_kd = bool(self.cfg.get("use_kd", True)) and kd_weight > 0 and len(pseudo_df) > 0
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        losses = []
        gold_losses = []
        pseudo_losses = []
        rank_losses = []
        kd_losses = []
        logit_penalty_losses = []
        epoch_history = []
        supervised_abs_logits_last: np.ndarray = np.asarray([], dtype=np.float32)
        kd_active_rows_last = 0
        kd_confidence_mean_last = 0.0
        mean_abs_gap_last = 0.0
        gold_bce = torch.tensor(0.0, device=self.device)
        pseudo_bce = torch.tensor(0.0, device=self.device)
        kd = torch.tensor(0.0, device=self.device)
        rank = torch.tensor(0.0, device=self.device)
        logit_penalty = torch.tensor(0.0, device=self.device)
        self.model.train()
        pbar = tqdm(range(epochs), desc="rel_graphsage graph training", leave=False)
        round_idx = int(iteration) if iteration is not None else 0
        for epoch in pbar:
            optimizer.zero_grad(set_to_none=True)
            fold_losses = []
            fold_gold_losses = []
            fold_pseudo_losses = []
            fold_kd_losses = []
            fold_rank_losses = []
            fold_logit_penalties = []
            epoch_abs_logits: list[np.ndarray] = []
            kd_active_rows_last = 0
            kd_confidence_mean_last = 0.0
            mean_abs_gap_last = 0.0

            active_folds = training_folds if training_folds else [
                {"edge_index": self.edge_index, "edge_type": self.edge_type, "train_df": pd.DataFrame(), "rank_df": pd.DataFrame()}
            ]
            for fold in active_folds:
                h = self.model.encode(self.x, fold["edge_index"], fold["edge_type"])
                fold_logits_for_penalty: list[torch.Tensor] = []

                fold_gold_bce = torch.tensor(0.0, device=self.device)
                fold_train_df = fold["train_df"]
                if len(fold_train_df):
                    gold_src, gold_dst, gold_hard_labels = make_pair_tensors(fold_train_df, self.device)
                    gold_logits = self.model.decode(h, gold_src, gold_dst)
                    gold_targets = torch.where(
                        gold_hard_labels >= 0.5,
                        torch.full_like(gold_hard_labels, gold_pos_target),
                        torch.full_like(gold_hard_labels, gold_neg_target),
                    )
                    if pos_weight is not None:
                        gold_weights = torch.where(
                            gold_hard_labels >= 0.5,
                            torch.full_like(gold_hard_labels, float(pos_weight.detach().cpu())),
                            torch.ones_like(gold_hard_labels),
                        )
                    else:
                        gold_weights = torch.ones_like(gold_hard_labels)
                    fold_gold_bce = _weighted_soft_bce(gold_logits, gold_targets, gold_weights)
                    fold_logits_for_penalty.append(gold_logits)

                fold_pseudo_bce = torch.tensor(0.0, device=self.device)
                pseudo_logits = None
                if pseudo_src is not None and pseudo_dst is not None and pseudo_targets is not None:
                    pseudo_logits = self.model.decode(h, pseudo_src, pseudo_dst)
                    fold_pseudo_bce = _weighted_soft_bce(pseudo_logits, pseudo_targets, pseudo_sample_weights)
                    fold_logits_for_penalty.append(pseudo_logits)

                fold_kd = torch.tensor(0.0, device=self.device)
                if use_kd and pseudo_logits is not None and pseudo_targets is not None:
                    teacher_conf = torch.abs(pseudo_targets - 0.5) * 2.0
                    kd_mask = teacher_conf >= kd_conf_threshold
                    kd_active_rows = int(kd_mask.sum().detach().cpu())
                    kd_active_rows_last += kd_active_rows
                    if kd_active_rows:
                        teacher_logits = torch.logit(torch.clamp(pseudo_targets, _EPS, 1.0 - _EPS))
                        teacher_soft = torch.sigmoid(teacher_logits / kd_temperature)
                        kd_vec = torch.nn.functional.binary_cross_entropy_with_logits(
                            pseudo_logits / kd_temperature, teacher_soft, reduction="none"
                        )
                        kd_weights = (
                            teacher_conf
                            if bool(self.cfg.get("teacher_confidence_weighted_kd", True))
                            else torch.ones_like(teacher_conf)
                        )
                        fold_kd = (kd_vec[kd_mask] * kd_weights[kd_mask]).mean()
                        kd_confidence_mean_last += float(teacher_conf[kd_mask].mean().detach().cpu())
                        student_prob = torch.sigmoid(pseudo_logits.detach())
                        mean_abs_gap_last += float(torch.abs(student_prob[kd_mask] - pseudo_targets[kd_mask]).mean().detach().cpu())

                fold_rank = torch.tensor(0.0, device=self.device)
                fold_rank_df = fold["rank_df"]
                if rank_weight > 0 and len(fold_rank_df):
                    rank_src = torch.tensor(fold_rank_df["source_index"].astype(int).to_numpy(), dtype=torch.long, device=self.device)
                    rank_dst = torch.tensor(fold_rank_df["target_index"].astype(int).to_numpy(), dtype=torch.long, device=self.device)
                    forward_logits = self.model.decode(h, rank_src, rank_dst)
                    reverse_logits = self.model.decode(h, rank_dst, rank_src)
                    fold_rank = reverse_ranking_loss(forward_logits, reverse_logits, margin=rank_margin_prob)

                fold_logit_penalty = torch.tensor(0.0, device=self.device)
                if fold_logits_for_penalty:
                    all_supervised_logits = torch.cat(fold_logits_for_penalty)
                    fold_logit_penalty = _logit_cap_penalty(all_supervised_logits, logit_cap)
                    epoch_abs_logits.append(torch.abs(all_supervised_logits.detach()).cpu().numpy())

                fold_loss = (
                    fold_gold_bce
                    + fold_pseudo_bce
                    + kd_weight * fold_kd
                    + rank_weight * fold_rank
                    + logit_penalty_weight * fold_logit_penalty
                )
                fold_losses.append(fold_loss)
                fold_gold_losses.append(fold_gold_bce)
                fold_pseudo_losses.append(fold_pseudo_bce)
                fold_kd_losses.append(fold_kd)
                fold_rank_losses.append(fold_rank)
                fold_logit_penalties.append(fold_logit_penalty)

            gold_bce = torch.stack(fold_gold_losses).mean() if fold_gold_losses else torch.tensor(0.0, device=self.device)
            pseudo_bce = torch.stack(fold_pseudo_losses).mean() if fold_pseudo_losses else torch.tensor(0.0, device=self.device)
            kd = torch.stack(fold_kd_losses).mean() if fold_kd_losses else torch.tensor(0.0, device=self.device)
            rank = torch.stack(fold_rank_losses).mean() if fold_rank_losses else torch.tensor(0.0, device=self.device)
            logit_penalty = torch.stack(fold_logit_penalties).mean() if fold_logit_penalties else torch.tensor(0.0, device=self.device)
            if epoch_abs_logits:
                supervised_abs_logits_last = np.concatenate(epoch_abs_logits).astype(np.float32)
            active_kd_folds = sum(1 for x in fold_kd_losses if float(x.detach().cpu()) > 0.0)
            if active_kd_folds:
                kd_active_rows_last = int(round(kd_active_rows_last / float(max(1, active_kd_folds))))
                kd_confidence_mean_last /= float(active_kd_folds)
                mean_abs_gap_last /= float(active_kd_folds)

            base_loss = torch.stack(fold_losses).mean() if fold_losses else torch.tensor(0.0, device=self.device)
            loss = base_loss
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            gold_losses.append(float(gold_bce.detach().cpu()))
            pseudo_losses.append(float(pseudo_bce.detach().cpu()))
            rank_losses.append(float(rank.detach().cpu()))
            kd_losses.append(float(kd.detach().cpu()))
            logit_penalty_losses.append(float(logit_penalty.detach().cpu()))
            if (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
                pbar.set_postfix(
                    loss=f"{loss_value:.4f}",
                    gold=f"{float(gold_bce.detach().cpu()):.4f}",
                    pseudo=f"{float(pseudo_bce.detach().cpu()):.4f}",
                    kd=f"{float(kd.detach().cpu()):.4f}",
                    rank=f"{float(rank.detach().cpu()):.4f}",
                    cap=f"{float(logit_penalty.detach().cpu()):.4f}",
                )
            val_bce_loss = 0.0
            if len(val_df):
                val_src = torch.tensor(val_df["source_index"].astype(int).to_numpy(), dtype=torch.long, device=self.device)
                val_dst = torch.tensor(val_df["target_index"].astype(int).to_numpy(), dtype=torch.long, device=self.device)
                val_labels = torch.tensor(val_df["label"].astype(float).to_numpy(), dtype=torch.float32, device=self.device)
                self.model.eval()
                with torch.no_grad():
                    h_val = self._encode_current()
                    val_logits = self.model.decode(h_val, val_src, val_dst)
                    val_bce_loss = float(
                        torch.nn.functional.binary_cross_entropy_with_logits(val_logits, val_labels).detach().cpu()
                    )
                self.model.train()
            epoch_history.append(
                {
                    "round": round_idx + 1,
                    "epoch": epoch + 1,
                    "global_epoch": round_idx * int(epochs) + epoch + 1,
                    "train_total_loss": float(loss_value),
                    "val_bce_loss": float(val_bce_loss),
                    "gold_bce": float(gold_bce.detach().cpu()),
                    "pseudo_soft_bce": float(pseudo_bce.detach().cpu()),
                    "kd_bce": float(kd.detach().cpu()),
                    "rank_loss": float(rank.detach().cpu()),
                    "logit_cap_penalty": float(logit_penalty.detach().cpu()),
                }
            )

        pos_weight_value = float(pos_weight.detach().cpu()) if pos_weight is not None else 1.0
        pseudo_target_values = pseudo_df["soft_target"].astype(float).to_numpy() if len(pseudo_df) else np.asarray([], dtype=float)
        pseudo_weight_values = pseudo_df["sample_weight"].astype(float).to_numpy() if len(pseudo_df) else np.asarray([], dtype=float)
        pseudo_weight_base_values = (
            pseudo_df["pseudo_weight_base_round"].astype(float).to_numpy() if len(pseudo_df) else np.asarray([], dtype=float)
        )
        fold_masked_prereq_edges = [int(f["stats"].get("excluded_prereq_positive_edges", 0)) for f in training_folds]
        fold_message_edges = [int(f["stats"].get("num_edges", 0)) for f in training_folds]
        metrics = {
            "graph_model_type": "rel_graphsage",
            "graph_loss": float(np.mean(losses)) if losses else 0.0,
            "graph_sup_bce": float((gold_bce + pseudo_bce).detach().cpu()),
            "graph_gold_bce": float(np.mean(gold_losses)) if gold_losses else 0.0,
            "graph_pseudo_soft_bce": float(np.mean(pseudo_losses)) if pseudo_losses else 0.0,
            "graph_kd_bce": float(np.mean(kd_losses)) if kd_losses else 0.0,
            "graph_rank_loss": float(np.mean(rank_losses)) if rank_losses else 0.0,
            "graph_rank_margin": rank_margin_prob,
            "graph_rank_margin_prob": rank_margin_prob,
            "graph_rank_weight": rank_weight,
            "graph_rank_rows": int(len(rank_df_full)),
            "graph_rank_rows_gold_only": int(len(rank_df_full)),
            "graph_rank_rows_pseudo_should_be_zero": 0 if bool(self.cfg.get("rank_on_gold_only", True)) else int(pseudo_rank_rows),
            "graph_logit_cap": float(logit_cap),
            "graph_logit_penalty_weight": float(logit_penalty_weight),
            "graph_logit_cap_penalty": float(np.mean(logit_penalty_losses)) if logit_penalty_losses else 0.0,
            "graph_logit_abs_p95": float(np.quantile(supervised_abs_logits_last, 0.95)) if supervised_abs_logits_last.size else 0.0,
            "graph_logit_abs_max": float(np.max(supervised_abs_logits_last)) if supervised_abs_logits_last.size else 0.0,
            "graph_gold_pos_target": float(gold_pos_target),
            "graph_gold_neg_target": float(gold_neg_target),
            "graph_kd_weight": kd_weight if use_kd else 0.0,
            "graph_kd_temperature": kd_temperature,
            "graph_kd_confidence_threshold": kd_conf_threshold,
            "graph_kd_active_rows": int(kd_active_rows_last),
            "graph_kd_confidence_mean": float(kd_confidence_mean_last),
            "mean_abs_student_teacher_prob_gap": float(mean_abs_gap_last),
            "graph_weight_decay": weight_decay,
            "graph_pseudo_rows": int(len(pseudo_df)),
            "graph_pseudo_supervised_rows": int(len(pseudo_df)),
            "graph_pseudo_target_min": float(np.min(pseudo_target_values)) if pseudo_target_values.size else 0.0,
            "graph_pseudo_target_mean": float(np.mean(pseudo_target_values)) if pseudo_target_values.size else 0.0,
            "graph_pseudo_target_max": float(np.max(pseudo_target_values)) if pseudo_target_values.size else 0.0,
            "graph_pseudo_weight_min": float(np.min(pseudo_weight_values)) if pseudo_weight_values.size else 0.0,
            "graph_pseudo_weight_mean": float(np.mean(pseudo_weight_values)) if pseudo_weight_values.size else 0.0,
            "graph_pseudo_weight_max": float(np.max(pseudo_weight_values)) if pseudo_weight_values.size else 0.0,
            "graph_pseudo_weight_base_round": float(np.mean(pseudo_weight_base_values)) if pseudo_weight_base_values.size else 0.0,
            "graph_supervised_rows_total": int(len(train_df) + len(pseudo_df)),
            "graph_pos_weight": pos_weight_value,
            "graph_edge_masking_folds": int(len(training_folds) if training_folds else 1),
            "graph_fold_masked_prereq_edges": fold_masked_prereq_edges,
            "graph_fold_message_edges": fold_message_edges,
            "graph_relation_counts": self.graph_stats.get("relation_counts", {}),
            "graph_semantic_selected_k": int(self.graph_stats.get("semantic_selected_k", 0)),
            "graph_semantic_lcc_coverage": float(self.graph_stats.get("semantic_lcc_coverage", 0.0)),
            "graph_semantic_fallback_used": bool(self.graph_stats.get("semantic_fallback_used", False)),
            "graph_leakage_guard": self._dynamic_leakage_guard(pseudo_df),
            "epoch_history": epoch_history,
        }
        if metrics["graph_rank_rows_pseudo_should_be_zero"] != 0:
            raise ValueError("Pseudo rows unexpectedly participated in ranking loss")
        return metrics

    def predict_logits(self, pairs_df: pd.DataFrame, batch_size: int = 4096) -> np.ndarray:
        return self._decode_logits(pairs_df, batch_size=batch_size)

    def predict_scores(self, pairs_df: pd.DataFrame, batch_size: int = 4096, calibrated: bool | None = None) -> np.ndarray:
        logits = self.predict_logits(pairs_df, batch_size=batch_size)
        use_cal = bool(self.cfg.get("use_graph_calibration", True)) if calibrated is None else bool(calibrated)
        if use_cal:
            return self._apply_calibration_np(logits)
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)

    def save_checkpoint(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "type": "GraphTrainer",
                "model_type": "RelGraphSAGEPrereqModel",
                "state_dict": self.model.state_dict(),
                "cfg": self.cfg,
                "graph_stats": self.graph_stats,
                "calibration": {
                    "temperature": float(self.calibration_temperature),
                    "bias": float(self.calibration_bias),
                    "fitted": bool(self.calibration_fitted),
                    "metrics": self.calibration_metrics,
                },
            },
            path,
        )

    @classmethod
    def load_checkpoint(cls, path: str | Path, cfg: Dict, data, device: torch.device) -> "GraphTrainer":
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing graph model checkpoint: {checkpoint_path}. Run Stage2 training first.")
        payload = torch.load(checkpoint_path, map_location=device)
        if str(payload.get("type", "")) != "GraphTrainer":
            raise ValueError(f"Unsupported graph checkpoint type {payload.get('type')!r} in {checkpoint_path}")
        checkpoint_model_type = str(payload.get("model_type", ""))
        if checkpoint_model_type in {"GATDirectedModel", "RGCNPrereqModel"}:
            raise RuntimeError(
                f"Graph checkpoint {checkpoint_path} was created by the old {checkpoint_model_type} graph model. "
                "Re-run Stage2 training to create a new RelGraphSAGEPrereqModel checkpoint."
            )
        saved_cfg = dict(payload.get("cfg", {}) or {})
        merged_cfg = dict(cfg)
        merged_cfg.update(saved_cfg)
        trainer = cls(merged_cfg, data, device)
        try:
            trainer.model.load_state_dict(payload["state_dict"])
        except RuntimeError as exc:
            raise RuntimeError(
                f"Graph checkpoint {checkpoint_path} is incompatible with the Rel-GraphSAGE graph model. "
                "Re-run Stage2 training to create a new graph checkpoint."
            ) from exc
        cal = dict(payload.get("calibration", {}) or {})
        trainer.calibration_temperature = float(cal.get("temperature", trainer.calibration_temperature))
        trainer.calibration_bias = float(cal.get("bias", 0.0))
        trainer.calibration_fitted = bool(cal.get("fitted", False))
        trainer.calibration_metrics = dict(cal.get("metrics", {}) or {})
        trainer.model.eval()
        return trainer

