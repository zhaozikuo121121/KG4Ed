from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def fit_content_model(
    model,
    train_df: pd.DataFrame,
    profiles: Dict[int, str],
    cfg: Dict,
    val_df: pd.DataFrame | None = None,
    iteration: int | None = None,
) -> Dict[str, float]:
    epochs = int(cfg.get("content_epochs", 20))
    return model.fit(
        train_df=train_df,
        profiles=profiles,
        epochs=epochs,
        batch_size=int(cfg.get("content_batch_size", 8)),
        learning_rate=float(cfg.get("content_learning_rate", 2e-5)),
        gold_weight=float(cfg.get("gold_weight", 1.0)),
        pseudo_weight=float(cfg.get("pseudo_weight", 0.3)),
        seed=int(cfg.get("seed", 42)),
        val_df=val_df,
        iteration=iteration,
    )


def save_content_checkpoint(model, path: str | Path) -> None:
    model.save_checkpoint(path)
