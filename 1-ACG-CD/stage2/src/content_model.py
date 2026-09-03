from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm

from .utils import manual_sgd_step, sigmoid_np


class MockContentModel:
    """Deterministic debug scorer; no SciBERT download needed."""

    def __init__(self, similarity: np.ndarray | None = None):
        self.similarity = similarity
        self.bias = 0.0

    def fit(self, train_df: pd.DataFrame, profiles: Dict[int, str] | None = None, **kwargs) -> Dict[str, float]:
        labels = train_df["label"].astype(float).to_numpy() if len(train_df) else np.array([0.0])
        self.bias = float(labels.mean() - 0.5)
        epochs = int(kwargs.get("epochs", 1))
        iteration = kwargs.get("iteration")
        round_idx = int(iteration) if iteration is not None else 0
        history = [
            {
                "round": round_idx + 1,
                "epoch": epoch + 1,
                "global_epoch": round_idx * epochs + epoch + 1,
                "train_total_loss": 0.0,
                "val_bce_loss": 0.0,
            }
            for epoch in range(epochs)
        ]
        return {"content_loss": 0.0, "debug_mock": 1.0, "epoch_history": history}

    def predict_scores(self, pairs_df: pd.DataFrame, profiles: Dict[int, str] | None = None, batch_size: int = 256) -> np.ndarray:
        if len(pairs_df) == 0:
            return np.asarray([], dtype=np.float32)
        if self.similarity is not None:
            sims = np.asarray(
                [self.similarity[int(r.source_index), int(r.target_index)] for r in pairs_df.itertuples(index=False)],
                dtype=np.float32,
            )
            scores = 1.0 / (1.0 + np.exp(-4.0 * (sims - 0.5 + self.bias)))
            return scores.astype(np.float32)
        vals = []
        for r in pairs_df.itertuples(index=False):
            raw = ((int(r.source_index) * 997 + int(r.target_index) * 37) % 1000) / 1000.0
            vals.append(raw)
        return np.asarray(vals, dtype=np.float32)

    def save_checkpoint(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"type": "MockContentModel", "bias": self.bias}, path)


class TinyContentPairClassifier(nn.Module):
    """Small debug trainable pair classifier over source/target indices."""

    def __init__(self, num_concepts: int, embedding_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(num_concepts, embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        zi = self.embedding(src)
        zj = self.embedding(dst)
        feats = torch.cat([zi, zj, zi - zj, zi * zj], dim=-1)
        return self.classifier(feats).squeeze(-1)


class TinyContentModel:
    def __init__(self, num_concepts: int, device: torch.device, embedding_dim: int = 32):
        self.num_concepts = num_concepts
        self.embedding_dim = embedding_dim
        self.device = device
        self.model = TinyContentPairClassifier(num_concepts, embedding_dim).to(device)

    def fit(
        self,
        train_df: pd.DataFrame,
        profiles: Dict[int, str] | None = None,
        epochs: int = 1,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        gold_weight: float = 1.0,
        pseudo_weight: float = 0.3,
        val_df: pd.DataFrame | None = None,
        iteration: int | None = None,
        **kwargs,
    ) -> Dict[str, float]:
        if len(train_df) == 0:
            return {"content_loss": 0.0, "epoch_history": []}
        src = torch.tensor(train_df["source_index"].astype(int).to_numpy(), dtype=torch.long)
        dst = torch.tensor(train_df["target_index"].astype(int).to_numpy(), dtype=torch.long)
        y = torch.tensor(train_df["label"].astype(float).to_numpy(), dtype=torch.float32)
        neg_type = train_df.get("negative_type", pd.Series(["none"] * len(train_df))).astype(str)
        weights_np = np.where(neg_type.str.startswith("pseudo").to_numpy(), pseudo_weight, gold_weight).astype(np.float32)
        weights = torch.tensor(weights_np, dtype=torch.float32)
        losses = []
        epoch_history = []
        self.model.train()
        total_batches = math.ceil(len(src) / batch_size)
        round_idx = int(iteration) if iteration is not None else 0
        val_df = pd.DataFrame() if val_df is None else val_df
        for epoch in range(epochs):
            order = torch.randperm(len(src))
            epoch_losses = []
            pbar = tqdm(
                range(0, len(src), batch_size),
                total=total_batches,
                desc=f"content epoch {epoch + 1}/{epochs}",
                leave=False,
            )
            for start in pbar:
                idx = order[start : start + batch_size]
                logits = self.model(src[idx].to(self.device), dst[idx].to(self.device))
                loss_vec = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, y[idx].to(self.device), reduction="none"
                )
                sup_loss = (loss_vec * weights[idx].to(self.device)).mean()
                loss = sup_loss
                loss.backward()
                manual_sgd_step(self.model, learning_rate)
                loss_value = float(loss.detach().cpu())
                pbar.set_postfix(loss=f"{loss_value:.4f}")
                losses.append(loss_value)
                epoch_losses.append(loss_value)
            val_bce_loss = 0.0
            if len(val_df):
                val_src = torch.tensor(val_df["source_index"].astype(int).to_numpy(), dtype=torch.long, device=self.device)
                val_dst = torch.tensor(val_df["target_index"].astype(int).to_numpy(), dtype=torch.long, device=self.device)
                val_y = torch.tensor(val_df["label"].astype(float).to_numpy(), dtype=torch.float32, device=self.device)
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(val_src, val_dst)
                    val_bce_loss = float(torch.nn.functional.binary_cross_entropy_with_logits(val_logits, val_y).detach().cpu())
                self.model.train()
            epoch_history.append(
                {
                    "round": round_idx + 1,
                    "epoch": epoch + 1,
                    "global_epoch": round_idx * int(epochs) + epoch + 1,
                    "train_total_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
                    "val_bce_loss": float(val_bce_loss),
                }
            )
        return {
            "content_loss": float(np.mean(losses)) if losses else 0.0,
            "epoch_history": epoch_history,
        }

    def predict_scores(self, pairs_df: pd.DataFrame, profiles: Dict[int, str] | None = None, batch_size: int = 256) -> np.ndarray:
        if len(pairs_df) == 0:
            return np.asarray([], dtype=np.float32)
        src = torch.tensor(pairs_df["source_index"].astype(int).to_numpy(), dtype=torch.long)
        dst = torch.tensor(pairs_df["target_index"].astype(int).to_numpy(), dtype=torch.long)
        outs = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(src), batch_size):
                logits = self.model(src[start:start+batch_size].to(self.device), dst[start:start+batch_size].to(self.device))
                outs.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(outs).astype(np.float32)

    def save_checkpoint(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "type": "TinyContentModel",
                "state_dict": self.model.state_dict(),
                "num_concepts": self.num_concepts,
                "embedding_dim": self.embedding_dim,
            },
            path,
        )


class SciBERTCrossEncoder(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = int(self.encoder.config.hidden_size)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward_texts(self, texts_a: List[str], texts_b: List[str], max_length: int, device: torch.device) -> torch.Tensor:
        batch = self.tokenizer(
            texts_a,
            texts_b,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        out = self.encoder(**batch)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls).squeeze(-1)


class HFContentModel:
    """Real SciBERT/Transformer cross-encoder training path."""

    def __init__(self, model_name: str, device: torch.device, dropout: float = 0.1, max_length: int = 384):
        self.device = device
        self.model_name = model_name
        self.dropout = dropout
        self.max_length = max_length
        self.model = SciBERTCrossEncoder(model_name, dropout=dropout).to(device)

    @staticmethod
    def _texts(df: pd.DataFrame, profiles: Dict[int, str]) -> tuple[List[str], List[str]]:
        a = [
            "Candidate prerequisite: " + str(row.source) + ". " + profiles[int(row.source_index)]
            for row in df.itertuples(index=False)
        ]
        b = [
            "Target concept: " + str(row.target) + ". " + profiles[int(row.target_index)]
            for row in df.itertuples(index=False)
        ]
        return a, b

    def fit(
        self,
        train_df: pd.DataFrame,
        profiles: Dict[int, str],
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        gold_weight: float = 1.0,
        pseudo_weight: float = 0.3,
        seed: int = 42,
        val_df: pd.DataFrame | None = None,
        iteration: int | None = None,
        **kwargs,
    ) -> Dict[str, float]:
        labels = torch.tensor(train_df["label"].astype(float).to_numpy(), dtype=torch.float32)
        neg_type = train_df.get("negative_type", pd.Series(["none"] * len(train_df))).astype(str)
        weights_np = np.where(neg_type.str.startswith("pseudo").to_numpy(), pseudo_weight, gold_weight).astype(np.float32)
        weights = torch.tensor(weights_np, dtype=torch.float32)
        texts_a, texts_b = self._texts(train_df, profiles)
        losses = []
        epoch_history = []
        self.model.train()
        n = len(train_df)
        total_batches = math.ceil(n / batch_size)
        round_idx = int(iteration) if iteration is not None else 0
        val_df = pd.DataFrame() if val_df is None else val_df
        val_texts_a, val_texts_b = self._texts(val_df, profiles) if len(val_df) else ([], [])
        val_labels = (
            torch.tensor(val_df["label"].astype(float).to_numpy(), dtype=torch.float32)
            if len(val_df)
            else torch.tensor([], dtype=torch.float32)
        )
        for epoch in range(epochs):
            order = torch.randperm(n).tolist()
            epoch_losses = []
            pbar = tqdm(
                range(0, n, batch_size),
                total=total_batches,
                desc=f"content epoch {epoch + 1}/{epochs}",
                leave=False,
            )
            for start in pbar:
                idx = order[start:start+batch_size]
                logits = self.model.forward_texts([texts_a[i] for i in idx], [texts_b[i] for i in idx], self.max_length, self.device)
                loss_vec = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, labels[idx].to(self.device), reduction="none"
                )
                sup_loss = (loss_vec * weights[idx].to(self.device)).mean()
                loss = sup_loss
                loss.backward()
                manual_sgd_step(self.model, learning_rate)
                loss_value = float(loss.detach().cpu())
                pbar.set_postfix(loss=f"{loss_value:.4f}")
                losses.append(loss_value)
                epoch_losses.append(loss_value)
            val_bce_loss = 0.0
            if len(val_df):
                val_losses = []
                self.model.eval()
                with torch.no_grad():
                    for v_start in range(0, len(val_df), batch_size):
                        val_logits = self.model.forward_texts(
                            val_texts_a[v_start:v_start + batch_size],
                            val_texts_b[v_start:v_start + batch_size],
                            self.max_length,
                            self.device,
                        )
                        val_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            val_logits, val_labels[v_start:v_start + batch_size].to(self.device)
                        )
                        val_losses.append(float(val_loss.detach().cpu()))
                self.model.train()
                val_bce_loss = float(np.mean(val_losses)) if val_losses else 0.0
            epoch_history.append(
                {
                    "round": round_idx + 1,
                    "epoch": epoch + 1,
                    "global_epoch": round_idx * int(epochs) + epoch + 1,
                    "train_total_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
                    "val_bce_loss": float(val_bce_loss),
                }
            )
        return {
            "content_loss": float(np.mean(losses)) if losses else 0.0,
            "epoch_history": epoch_history,
        }

    def predict_scores(self, pairs_df: pd.DataFrame, profiles: Dict[int, str], batch_size: int = 64) -> np.ndarray:
        if len(pairs_df) == 0:
            return np.asarray([], dtype=np.float32)
        texts_a, texts_b = self._texts(pairs_df, profiles)
        outs = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(pairs_df), batch_size):
                logits = self.model.forward_texts(
                    texts_a[start:start+batch_size], texts_b[start:start+batch_size], self.max_length, self.device
                )
                outs.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(outs).astype(np.float32)

    def save_checkpoint(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "type": "HFContentModel",
                "state_dict": self.model.state_dict(),
                "model_name": self.model_name,
                "dropout": self.dropout,
                "max_length": self.max_length,
            },
            path,
        )


def build_content_model(cfg: Dict, data, device: torch.device):
    return HFContentModel(
        model_name=str(cfg["content_model_name"]),
        device=device,
        dropout=float(cfg.get("content_dropout", 0.1)),
        max_length=int(cfg.get("content_max_length", 384)),
    )


def load_content_model(checkpoint_path: str | Path, cfg: Dict, data, device: torch.device):
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing content model checkpoint: {path}. Run Stage2 training first.")
    payload = torch.load(path, map_location=device)
    model_type = str(payload.get("type", ""))
    if model_type == "HFContentModel":
        model = HFContentModel(
            model_name=str(payload.get("model_name", cfg.get("content_model_name", "allenai/scibert_scivocab_uncased"))),
            device=device,
            dropout=float(payload.get("dropout", cfg.get("content_dropout", 0.1))),
            max_length=int(payload.get("max_length", cfg.get("content_max_length", 384))),
        )
        model.model.load_state_dict(payload["state_dict"])
        return model
    raise ValueError(f"Unsupported content checkpoint type {model_type!r} in {path}")
