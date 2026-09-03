from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .bert_embeddings import encode_concept_texts_with_bert
from .concept_profile import build_related_slide_texts, load_or_create_profiles
from .data_io import parse_label_stats, read_caption_texts, read_concepts
from .graph import build_initial_graph, cosine_similarity_matrix, edge_rows_from_adjacency


def _jsonable_config(cfg: Dict) -> Dict:
    serializable = {}
    for key, value in cfg.items():
        if isinstance(value, Path):
            serializable[key] = str(value)
        else:
            serializable[key] = value
    return serializable


def run_stage0(cfg: Dict) -> Dict:
    data_dir = Path(cfg["data_dir"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    concepts = read_concepts(data_dir / str(cfg["concept_file"]))
    caption_texts, caption_counts = read_caption_texts(data_dir, cfg.get("caption_files"))
    if not caption_texts:
        alias_texts = [" ".join(concept.aliases) for concept in concepts]
        chunk_size = 50
        caption_texts = [
            " . ".join(alias_texts[i : i + chunk_size])
            for i in range(0, len(alias_texts), chunk_size)
            if alias_texts[i : i + chunk_size]
        ]
        caption_counts["__concept_alias_fallback__"] = len(caption_texts)

    concept_rows: List[Dict] = []
    profile_cfg = cfg.get("profile", {}) if isinstance(cfg.get("profile"), dict) else {}
    related_slide_texts = build_related_slide_texts(
        concepts,
        caption_texts,
        max_snippets=int(profile_cfg.get("max_snippets", 3)),
        max_chars=int(profile_cfg.get("snippet_max_chars", 500)),
    )
    for concept in concepts:
        concept_rows.append(
            {
                "concept_id": concept.concept_id,
                "concept": concept.name,
                "aliases": "::;".join(concept.aliases),
                "slide_snippets": related_slide_texts.get(concept.concept_id, []),
                "embedding_source": "bert",
                "embedding_token": "",
                "embedding_model": str(cfg.get("bert_model_name", "sentence-transformers/all-mpnet-base-v2")),
            }
        )

    concepts_df = pd.DataFrame(concept_rows)
    profiles = load_or_create_profiles(concepts_df, cfg)
    embeddings, embedding_stats = encode_concept_texts_with_bert(concepts_df, profiles, cfg)
    similarity = cosine_similarity_matrix(embeddings)
    adjacency, graph_stats = build_initial_graph(
        similarity=similarity,
        k=int(cfg["k"]),
        fallback_k=int(cfg["fallback_k"]),
        min_lcc_coverage=float(cfg["min_lcc_coverage"]),
    )
    concepts_df["slide_snippets"] = concepts_df["slide_snippets"].apply(
        lambda snippets: json.dumps(snippets, ensure_ascii=False)
    )
    edges_df = pd.DataFrame(edge_rows_from_adjacency(adjacency, [c.name for c in concepts]))

    concepts_df.to_csv(output_dir / "concepts.csv", index=False, encoding="utf-8")
    edges_df.to_csv(output_dir / "edges_knn.csv", index=False, encoding="utf-8")
    np.save(output_dir / "concept_embeddings.npy", embeddings)
    np.save(output_dir / "similarity_matrix.npy", similarity)
    np.save(output_dir / "adjacency_matrix.npy", adjacency)

    ml_label_stats = parse_label_stats(data_dir / str(cfg["label_file"]))
    warnings = []
    if graph_stats["largest_connected_component_coverage"] < float(cfg["min_lcc_coverage"]):
        warnings.append(
            "Largest connected component coverage is below the configured threshold even after fallback k."
        )

    source_counts = concepts_df["embedding_source"].value_counts().to_dict()
    summary = {
        "stage": "stage0",
        "config": _jsonable_config(cfg),
        "concept_count": len(concepts),
        "caption_record_counts": caption_counts,
        "embedding_shape": list(embeddings.shape),
        "embedding_stats": embedding_stats,
        "embedding_source_counts": {str(k): int(v) for k, v in source_counts.items()},
        "graph_stats": graph_stats,
        "label_stats_for_validation_only": ml_label_stats,
        "leakage_note": f"{cfg['label_file']} labels are parsed only for validation statistics; Stage0 graph construction uses concept/profile text only.",
        "warnings": warnings,
        "outputs": {
            "concepts_csv": str(output_dir / "concepts.csv"),
            "concept_embeddings_npy": str(output_dir / "concept_embeddings.npy"),
            "similarity_matrix_npy": str(output_dir / "similarity_matrix.npy"),
            "adjacency_matrix_npy": str(output_dir / "adjacency_matrix.npy"),
            "edges_knn_csv": str(output_dir / "edges_knn.csv"),
            "concept_profiles_jsonl": str(
                Path(profile_cfg.get("cache_path", output_dir / "concept_profiles.jsonl"))
            ),
        },
        "concept_profile_count": len(profiles),
    }
    (output_dir / "stage0_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary

