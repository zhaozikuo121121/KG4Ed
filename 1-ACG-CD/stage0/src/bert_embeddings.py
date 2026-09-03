from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import torch


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _concept_embedding_text(row, profile: str) -> str:
    concept = str(row.concept)
    aliases = str(getattr(row, "aliases", "") or "")
    alias_text = ", ".join([item.strip() for item in aliases.split("::;") if item.strip()])
    parts = [f"Concept: {concept}"]
    if alias_text and alias_text != concept:
        parts.append(f"Aliases: {alias_text}")
    parts.append(f"Profile: {profile}")
    return "\n".join(parts)


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return summed / counts


def encode_concept_texts_with_bert(concepts: pd.DataFrame, profiles: Dict[int, str], cfg: Dict) -> tuple[np.ndarray, Dict[str, object]]:
    """Encode each Stage0 concept as BERT/Transformer mean-pooled text embedding.

    The text follows the roadmap requirement: concept name plus its LLM profile.
    This function intentionally uses ``transformers`` directly rather than
    ``sentence-transformers`` so the project does not need an extra runtime
    dependency beyond the existing requirements.
    """

    try:
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only in incomplete environments.
        raise RuntimeError("Stage0 BERT embeddings require the 'transformers' package.") from exc

    model_name = str(cfg.get("bert_model_name", "sentence-transformers/all-mpnet-base-v2"))
    batch_size = int(cfg.get("bert_batch_size", cfg.get("batch_size", 16)))
    max_length = int(cfg.get("bert_max_length", 256))
    device = _resolve_device(str(cfg.get("device", "auto")))

    ordered = concepts.sort_values("concept_id").reset_index(drop=True)
    texts = [
        _concept_embedding_text(row, profiles[int(row.concept_id)])
        for row in ordered.itertuples(index=False)
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    vectors: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            pooled = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            vectors.append(pooled.detach().cpu().numpy().astype(np.float32))

    embeddings = np.concatenate(vectors, axis=0) if vectors else np.zeros((0, 0), dtype=np.float32)
    stats: Dict[str, object] = {
        "embedding_backend": "bert",
        "bert_model_name": model_name,
        "bert_batch_size": batch_size,
        "bert_max_length": max_length,
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.shape[0] else 0,
        "embedding_text_format": "Concept + aliases + Profile",
        "device": str(device),
    }
    return embeddings, stats
