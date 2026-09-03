from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data_io import Concept, phrase_token, tokenize


@dataclass
class Vocabulary:
    token_to_id: Dict[str, int]
    id_to_token: List[str]
    counts: Dict[str, int]


class SGNSModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)
        bound = 0.5 / max(1, embedding_dim)
        nn.init.uniform_(self.in_embed.weight, -bound, bound)
        nn.init.zeros_(self.out_embed.weight)

    def forward(self, centers: torch.Tensor, contexts: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        center_vec = self.in_embed(centers)
        context_vec = self.out_embed(contexts)
        pos_score = torch.sum(center_vec * context_vec, dim=1)
        pos_loss = torch.nn.functional.logsigmoid(pos_score)

        neg_vec = self.out_embed(negatives)
        neg_score = torch.bmm(neg_vec, center_vec.unsqueeze(2)).squeeze(2)
        neg_loss = torch.nn.functional.logsigmoid(-neg_score).sum(dim=1)
        return -(pos_loss + neg_loss).mean()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_vocabulary(documents: Sequence[Sequence[str]], min_count: int = 1) -> Vocabulary:
    counts = Counter(tok for doc in documents for tok in doc)
    tokens = sorted([tok for tok, count in counts.items() if count >= min_count])
    token_to_id = {tok: idx for idx, tok in enumerate(tokens)}
    kept_counts = {tok: int(counts[tok]) for tok in tokens}
    return Vocabulary(token_to_id=token_to_id, id_to_token=tokens, counts=kept_counts)


def encode_documents(documents: Sequence[Sequence[str]], vocab: Vocabulary) -> List[np.ndarray]:
    encoded: List[np.ndarray] = []
    token_to_id = vocab.token_to_id
    for doc in documents:
        ids = [token_to_id[tok] for tok in doc if tok in token_to_id]
        if ids:
            encoded.append(np.asarray(ids, dtype=np.int64))
    return encoded


def build_skipgram_pairs(encoded_docs: Sequence[np.ndarray], window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    centers: List[np.ndarray] = []
    contexts: List[np.ndarray] = []
    for ids in encoded_docs:
        n = len(ids)
        if n <= 1:
            continue
        for offset in range(1, window_size + 1):
            if n <= offset:
                break
            left_centers = ids[offset:]
            left_contexts = ids[:-offset]
            right_centers = ids[:-offset]
            right_contexts = ids[offset:]
            centers.extend([left_centers, right_centers])
            contexts.extend([left_contexts, right_contexts])
    if not centers:
        raise ValueError("No skip-gram pairs could be built from the corpus.")
    return np.concatenate(centers).astype(np.int64), np.concatenate(contexts).astype(np.int64)


def negative_sampling_distribution(vocab: Vocabulary) -> torch.Tensor:
    counts = np.asarray([vocab.counts[token] for token in vocab.id_to_token], dtype=np.float64)
    probs = np.power(counts, 0.75)
    probs = probs / probs.sum()
    return torch.tensor(probs, dtype=torch.float32)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def train_phrase2vec(
    documents: Sequence[Sequence[str]],
    embedding_dim: int,
    window_size: int,
    negative_samples: int,
    epochs: int,
    seed: int,
    min_count: int = 1,
    batch_size: int = 8192,
    learning_rate: float = 0.002,
    max_pairs_per_epoch: int | None = None,
    device: str = "auto",
) -> Tuple[SGNSModel, Vocabulary, Dict[str, int | float | str]]:
    set_seed(seed)
    vocab = build_vocabulary(documents, min_count=min_count)
    encoded_docs = encode_documents(documents, vocab)
    centers, contexts = build_skipgram_pairs(encoded_docs, window_size=window_size)
    total_pairs = int(len(centers))
    effective_pairs = total_pairs if max_pairs_per_epoch in (None, 0) else min(total_pairs, int(max_pairs_per_epoch))

    dev = resolve_device(device)
    model = SGNSModel(vocab_size=len(vocab.id_to_token), embedding_dim=embedding_dim).to(dev)
    neg_probs = negative_sampling_distribution(vocab).to(dev)
    rng = np.random.default_rng(seed)

    model.train()
    for epoch in range(epochs):
        if effective_pairs < total_pairs:
            sample_idx = rng.choice(total_pairs, size=effective_pairs, replace=False)
        else:
            sample_idx = rng.permutation(total_pairs)
        epoch_centers = torch.from_numpy(centers[sample_idx])
        epoch_contexts = torch.from_numpy(contexts[sample_idx])
        dataset = TensorDataset(epoch_centers, epoch_contexts)
        generator = torch.Generator()
        generator.manual_seed(seed + epoch)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
        for batch_centers, batch_contexts in loader:
            batch_centers = batch_centers.to(dev, non_blocking=True)
            batch_contexts = batch_contexts.to(dev, non_blocking=True)
            negatives = torch.multinomial(
                neg_probs,
                num_samples=batch_centers.shape[0] * negative_samples,
                replacement=True,
            ).view(batch_centers.shape[0], negative_samples).to(dev)
            model.zero_grad(set_to_none=True)
            loss = model(batch_centers, batch_contexts, negatives)
            loss.backward()
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        param -= learning_rate * param.grad

    stats: Dict[str, int | float | str] = {
        "vocab_size": len(vocab.id_to_token),
        "skipgram_pairs_total": total_pairs,
        "skipgram_pairs_per_epoch": effective_pairs,
        "device": str(dev),
    }
    return model, vocab, stats


def deterministic_random_vector(text: str, dim: int, seed: int) -> np.ndarray:
    # Stable across Python runs, unlike built-in hash().
    import hashlib

    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()
    int_seed = int.from_bytes(digest[:8], "little", signed=False) % (2**32)
    rng = np.random.default_rng(int_seed)
    vec = rng.normal(0.0, 0.01, size=dim).astype(np.float32)
    return vec


def extract_concept_embeddings(
    model: SGNSModel,
    vocab: Vocabulary,
    concepts: Sequence[Concept],
    embedding_dim: int,
    seed: int,
) -> Tuple[np.ndarray, List[Dict[str, str]]]:
    weights = model.in_embed.weight.detach().cpu().numpy().astype(np.float32)
    rows: List[np.ndarray] = []
    sources: List[Dict[str, str]] = []

    for concept in concepts:
        canonical_token = phrase_token(concept.name)
        selected_token = None
        source = ""
        if canonical_token in vocab.token_to_id:
            selected_token = canonical_token
            source = "canonical_phrase"
        else:
            for alias in concept.aliases:
                alias_token = phrase_token(alias)
                if alias_token in vocab.token_to_id:
                    selected_token = alias_token
                    source = "alias_phrase"
                    break

        if selected_token is not None:
            vec = weights[vocab.token_to_id[selected_token]]
        else:
            word_ids = [vocab.token_to_id[tok] for tok in tokenize(concept.name) if tok in vocab.token_to_id]
            if word_ids:
                vec = weights[word_ids].mean(axis=0)
                selected_token = "+".join(vocab.id_to_token[idx] for idx in word_ids)
                source = "word_average"
            else:
                vec = deterministic_random_vector(concept.name, embedding_dim, seed)
                selected_token = "<deterministic_random>"
                source = "deterministic_random"

        rows.append(vec.astype(np.float32))
        sources.append({"embedding_source": source, "embedding_token": selected_token})

    embeddings = np.vstack(rows).astype(np.float32)
    return embeddings, sources


def save_model(path: str | Path, model: SGNSModel, vocab: Vocabulary, config: Dict) -> None:
    path = Path(path)
    payload = {
        "state_dict": model.state_dict(),
        "token_to_id": vocab.token_to_id,
        "id_to_token": vocab.id_to_token,
        "counts": vocab.counts,
        "config": config,
    }
    torch.save(payload, path)


def save_vocab(path: str | Path, vocab: Vocabulary) -> None:
    path = Path(path)
    payload = {
        "token_to_id": vocab.token_to_id,
        "id_to_token": vocab.id_to_token,
        "counts": vocab.counts,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

