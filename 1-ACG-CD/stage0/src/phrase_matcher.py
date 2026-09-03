from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from .data_io import Concept, phrase_token, tokenize


TrieNode = Dict[str, dict]


@dataclass
class PhraseCorpus:
    documents: List[List[str]]
    phrase_counts: Dict[str, int]
    raw_token_count: int
    matched_phrase_count: int


def build_alias_token_map(concepts: Sequence[Concept]) -> Dict[Tuple[str, ...], str]:
    alias_to_token: Dict[Tuple[str, ...], str] = {}
    # Long aliases should win during matching; duplicate aliases keep first concept deterministically.
    for concept in concepts:
        for alias in concept.aliases:
            toks = tuple(tokenize(alias))
            if toks and toks not in alias_to_token:
                alias_to_token[toks] = phrase_token(alias)
    return alias_to_token


def build_trie(alias_to_token: Dict[Tuple[str, ...], str]) -> dict:
    root: dict = {}
    for alias_tokens, token in alias_to_token.items():
        node = root
        for tok in alias_tokens:
            node = node.setdefault(tok, {})
        node["$"] = token
    return root


def replace_phrases(tokens: Sequence[str], trie: dict) -> Tuple[List[str], int]:
    output: List[str] = []
    matches = 0
    i = 0
    n = len(tokens)
    while i < n:
        node = trie
        best_token = None
        best_end = i
        j = i
        while j < n and tokens[j] in node:
            node = node[tokens[j]]
            j += 1
            if "$" in node:
                best_token = node["$"]
                best_end = j
        if best_token is not None:
            output.append(best_token)
            matches += 1
            i = best_end
        else:
            output.append(tokens[i])
            i += 1
    return output, matches


def build_phrase_corpus(texts: Iterable[str], concepts: Sequence[Concept]) -> PhraseCorpus:
    alias_map = build_alias_token_map(concepts)
    trie = build_trie(alias_map)
    docs: List[List[str]] = []
    phrase_counts: Counter[str] = Counter()
    raw_token_count = 0
    matched_phrase_count = 0

    for text in texts:
        raw_tokens = tokenize(text)
        raw_token_count += len(raw_tokens)
        phrase_tokens, matches = replace_phrases(raw_tokens, trie)
        matched_phrase_count += matches
        phrase_counts.update(tok for tok in phrase_tokens if tok.startswith("__phrase__"))
        if phrase_tokens:
            docs.append(phrase_tokens)

    return PhraseCorpus(
        documents=docs,
        phrase_counts=dict(phrase_counts),
        raw_token_count=raw_token_count,
        matched_phrase_count=matched_phrase_count,
    )
