from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "stage0"))

from src.data_io import parse_label_stats, read_concepts, tokenize
from src.bert_embeddings import _concept_embedding_text
from src.graph import build_initial_graph, cosine_similarity_matrix, edge_rows_from_adjacency
from src.phrase_matcher import build_phrase_corpus


class Stage0DataTests(unittest.TestCase):
    def test_real_mooc_concepts_and_label_stats(self):
        data_dir = ROOT / "data" / "MOOCML"
        concepts = read_concepts(data_dir / "MOOCML_concepts.csv")
        self.assertEqual(len(concepts), 244)
        stats = parse_label_stats(data_dir / "MOOCML_prerequisites.csv")
        self.assertEqual(stats["unique_directed_positive_edges"], 1735)
        self.assertEqual(stats["label_0"], 4977)


    def test_csv_concepts_support_aliases_and_quoted_newlines(self):
        with tempfile.TemporaryDirectory(prefix="stage0_csv_concepts_") as tmp:
            path = Path(tmp) / "concepts.csv"
            path.write_text(
                'Main Concept,Alias One\n"Best\n_worst_and_average_case"\n',
                encoding="utf-8",
            )
            concepts = read_concepts(path)
            self.assertEqual(concepts[0].name, "main concept")
            self.assertEqual(concepts[0].aliases, ("main concept", "alias one"))
            self.assertEqual(concepts[1].name, "best _worst_and_average_case")

    def test_csv_prerequisite_stats_use_a_to_b_direction(self):
        with tempfile.TemporaryDirectory(prefix="stage0_csv_labels_") as tmp:
            path = Path(tmp) / "labels.csv"
            path.write_text("A,B,1\nC,D,0\n", encoding="utf-8")
            stats = parse_label_stats(path)
            self.assertEqual(stats["label_1"], 1)
            self.assertEqual(stats["label_0"], 1)
            self.assertEqual(stats["unique_directed_positive_edges"], 1)

    def test_phrase_matching_uses_longest_aliases(self):
        data_dir = ROOT / "data" / "MOOCML"
        concepts = read_concepts(data_dir / "MOOCML_concepts.csv")[:20]
        corpus = build_phrase_corpus(["Back propagation uses an activation function."], concepts)
        doc = corpus.documents[0]
        self.assertIn("__phrase__back_propagation", doc)
        self.assertIn("__phrase__activation_function", doc)


class Stage0GraphTests(unittest.TestCase):
    def test_bert_embedding_text_uses_concept_aliases_and_profile(self):
        class Row:
            concept = "gradient descent"
            aliases = "gradient descent::;steepest descent"

        text = _concept_embedding_text(Row(), "Profile text.")
        self.assertIn("Concept: gradient descent", text)
        self.assertIn("Aliases: gradient descent, steepest descent", text)
        self.assertIn("Profile: Profile text.", text)

    def test_cosine_and_adjacency_properties(self):
        emb = np.asarray(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ],
            dtype=np.float32,
        )
        sim = cosine_similarity_matrix(emb)
        self.assertEqual(sim.shape, (4, 4))
        self.assertTrue(np.allclose(np.diag(sim), 1.0))
        adj, stats = build_initial_graph(sim, k=1, fallback_k=2, min_lcc_coverage=0.90)
        self.assertTrue(np.allclose(adj, adj.T))
        self.assertTrue(np.allclose(np.diag(adj), 0.0))
        self.assertGreaterEqual(stats["largest_connected_component_coverage"], 0.90)
        rows = edge_rows_from_adjacency(adj, ["a", "b", "c", "d"])
        for row in rows:
            self.assertAlmostEqual(row["weight"], float(adj[row["source_id"], row["target_id"]]))


if __name__ == "__main__":
    unittest.main()


