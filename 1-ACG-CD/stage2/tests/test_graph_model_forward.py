from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stage2"))

from src.data_loader import load_stage2_data
from src.graph_model import (
    NUM_RELATIONS,
    REL_PREREQ_BACKWARD,
    REL_PREREQ_FORWARD,
    REL_SEMANTIC_SIM,
    REL_SELF_LOOP,
    RelGraphSAGEPrereqModel,
    build_rel_graphsage_graph,
    reverse_ranking_loss,
)
from src.train_graph import GraphTrainer
from src.train_graph import _logit_cap_penalty


def _pairs(df: pd.DataFrame) -> set[tuple[int, int]]:
    if len(df) == 0:
        return set()
    return set(zip(df.source_index.astype(int), df.target_index.astype(int)))


class GraphModelForwardTests(unittest.TestCase):
    def test_rel_graphsage_graph_relations_use_only_r_train_gold_prereq_edges(self):
        data = load_stage2_data(ROOT / "stage0" / "outputs", ROOT / "stage1" / "outputs")
        original_disk = np.load(ROOT / "stage0" / "outputs" / "adjacency_matrix.npy")
        before = data.adjacency.copy()
        edge_index, edge_type, stats = build_rel_graphsage_graph(data.similarity, data.train_labels, data.num_concepts, {})

        self.assertEqual(edge_index.shape[0], 2)
        self.assertEqual(edge_type.shape[0], edge_index.shape[1])
        self.assertEqual(set(edge_type.tolist()), {0, 1, 2, 3})
        self.assertEqual(stats["num_relations"], NUM_RELATIONS)
        self.assertEqual(stats["relation_counts"]["self_loop"], data.num_concepts)
        self.assertTrue(np.allclose(data.adjacency, before))
        self.assertTrue(np.allclose(np.load(ROOT / "stage0" / "outputs" / "adjacency_matrix.npy"), original_disk))

        forward = set(zip(edge_index[0, edge_type == REL_PREREQ_FORWARD], edge_index[1, edge_type == REL_PREREQ_FORWARD]))
        backward = set(zip(edge_index[0, edge_type == REL_PREREQ_BACKWARD], edge_index[1, edge_type == REL_PREREQ_BACKWARD]))
        r_train = _pairs(data.train_labels[data.train_labels.label.astype(int) == 1])
        r_neg = _pairs(data.train_labels[data.train_labels.label.astype(int) == 0])
        self.assertEqual(forward, r_train)
        self.assertEqual(backward, {(dst, src) for src, dst in r_train})
        self.assertFalse(forward & _pairs(data.val_labels))
        self.assertFalse(forward & _pairs(data.test_labels))
        self.assertFalse(forward & r_neg)
        self.assertEqual(stats["leakage_guard"]["val_forward_prereq_overlap"], 0)
        self.assertEqual(stats["leakage_guard"]["test_forward_prereq_overlap"], 0)
        self.assertEqual(stats["leakage_guard"]["negative_forward_prereq_overlap"], 0)

        excluded_pair = next(iter(r_train))
        masked_edge_index, masked_edge_type, masked_stats = build_rel_graphsage_graph(
            data.similarity,
            data.train_labels,
            data.num_concepts,
            {},
            excluded_prereq_pairs={excluded_pair},
        )
        masked_forward = set(
            zip(
                masked_edge_index[0, masked_edge_type == REL_PREREQ_FORWARD],
                masked_edge_index[1, masked_edge_type == REL_PREREQ_FORWARD],
            )
        )
        masked_backward = set(
            zip(
                masked_edge_index[0, masked_edge_type == REL_PREREQ_BACKWARD],
                masked_edge_index[1, masked_edge_type == REL_PREREQ_BACKWARD],
            )
        )
        self.assertNotIn(excluded_pair, masked_forward)
        self.assertNotIn((excluded_pair[1], excluded_pair[0]), masked_backward)
        self.assertEqual(masked_stats["excluded_prereq_positive_edges"], 1)

    def test_forward_shape_directionality_and_raw_logits(self):
        data = load_stage2_data(ROOT / "stage0" / "outputs", ROOT / "stage1" / "outputs")
        edge_index, edge_type, _ = build_rel_graphsage_graph(data.similarity, data.train_labels, data.num_concepts, {})
        model = RelGraphSAGEPrereqModel(
            input_dim=data.concept_embeddings.shape[1],
            hidden_dim=16,
            num_relations=NUM_RELATIONS,
            graphsage_layers=1,
            dropout=0.0,
            decoder_dropout=0.0,
        )
        x = torch.tensor(data.concept_embeddings, dtype=torch.float32)
        ei = torch.tensor(edge_index, dtype=torch.long)
        et = torch.tensor(edge_type, dtype=torch.long)
        pairs = data.train_labels.head(10)
        src = torch.tensor(pairs.source_index.astype(int).to_numpy(), dtype=torch.long)
        dst = torch.tensor(pairs.target_index.astype(int).to_numpy(), dtype=torch.long)
        logits = model(x, ei, et, src, dst)
        reverse_logits = model(x, ei, et, dst, src)
        self.assertEqual(tuple(logits.shape), (10,))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(tuple(model.encode(x, ei, et).shape), (data.num_concepts, 32))
        self.assertFalse(torch.allclose(logits, reverse_logits))
        self.assertEqual(tuple(model.bilinear_diag.shape), (32,))
        self.assertFalse(hasattr(model, "decoder"))
        # Forward returns raw logits rather than sigmoid probabilities.
        self.assertFalse(torch.allclose(logits, torch.sigmoid(logits)))

    def test_reverse_ranking_loss_penalizes_bad_direction_order(self):
        forward = torch.tensor([0.0, 1.0])
        reverse = torch.tensor([0.3, 1.2])
        loss = reverse_ranking_loss(forward, reverse, margin=0.25)
        self.assertGreater(float(loss), 0.0)
        satisfied = reverse_ranking_loss(torch.tensor([2.0]), torch.tensor([0.0]), margin=0.25)
        self.assertAlmostEqual(float(satisfied), 0.0)
        # The margin is on probabilities, not raw logits.
        barely = reverse_ranking_loss(torch.tensor([2.0]), torch.tensor([0.0]), margin=0.5)
        self.assertGreater(float(barely), 0.0)

    def test_logit_cap_penalty_only_penalizes_excess_confidence(self):
        inside = _logit_cap_penalty(torch.tensor([-5.0, 0.0, 4.5]), cap=5.0)
        outside = _logit_cap_penalty(torch.tensor([-6.0, 0.0, 7.0]), cap=5.0)
        self.assertAlmostEqual(float(inside), 0.0)
        self.assertGreater(float(outside), 0.0)

    def test_graph_trainer_uses_pseudo_for_supervision_soft_kd_and_ranking(self):
        data = load_stage2_data(ROOT / "stage0" / "outputs", ROOT / "stage1" / "outputs")
        cfg = {
            "graph_hidden_dim": 16,
            "graph_dropout": 0.0,
            "decoder_dropout": 0.0,
            "graphsage_layers": 1,
            "graph_learning_rate": 0.005,
            "graph_weight_decay": 0.0005,
            "graph_kd_weight": 0.05,
            "kd_temperature": 3.0,
            "kd_confidence_threshold": 0.2,
            "rank_margin_prob": 0.25,
            "graph_rank_weight": 0.03,
            "graph_edge_masking_folds": 2,
            "graph_gold_pos_target": 0.95,
            "graph_gold_neg_target": 0.05,
            "graph_logit_cap": 5.0,
            "graph_logit_penalty_weight": 0.003,
            "pseudo_weight_round1": 0.1,
            "pseudo_min_weight": 0.05,
            "pseudo_max_weight": 0.5,
        }
        trainer = GraphTrainer(cfg, data, torch.device("cpu"))
        supervised = pd.concat(
            [
                data.train_labels[data.train_labels.label.astype(int) == 1].head(4),
                data.train_labels[data.train_labels.label.astype(int) == 0].head(4),
            ],
            ignore_index=True,
        )
        excluded = _pairs(data.train_labels) | _pairs(data.val_labels) | _pairs(data.test_labels)
        pseudo_rows = []
        for src_id in range(data.num_concepts):
            for dst_id in range(data.num_concepts):
                if src_id != dst_id and (src_id, dst_id) not in excluded:
                    pseudo_rows.append(
                        {
                            "source_id": src_id,
                            "target_id": dst_id,
                            "source_index": src_id,
                            "target_index": dst_id,
                            "source": data.id_to_name[src_id],
                            "target": data.id_to_name[dst_id],
                        }
                    )
                if len(pseudo_rows) == 4:
                    break
            if len(pseudo_rows) == 4:
                break
        pseudo = pd.DataFrame(pseudo_rows)
        pseudo["label"] = [1, 0, 1, 0]
        pseudo["teacher_model"] = "content"
        pseudo["pseudo_type"] = ["positive", "negative", "positive", "negative"]
        pseudo["model_score"] = [0.91, 0.18, 0.82, 0.25]
        pseudo["combined_score"] = [0.83, 0.27, 0.72, 0.31]
        pseudo["llm_score"] = ""
        pseudo["iteration"] = 0
        pseudo["label_confidence"] = [0.01, 0.02, 0.03, 0.04]
        metrics = trainer.fit(supervised, pseudo_df=pseudo, epochs=1, iteration=0)
        self.assertIn("graph_sup_bce", metrics)
        self.assertIn("graph_kd_bce", metrics)
        self.assertIn("graph_rank_loss", metrics)
        self.assertIn("graph_gold_bce", metrics)
        self.assertIn("graph_pseudo_soft_bce", metrics)
        self.assertIn("graph_logit_cap_penalty", metrics)
        self.assertNotIn("graph_kl", metrics)
        self.assertGreaterEqual(metrics["graph_kd_bce"], 0.0)
        self.assertEqual(metrics["graph_kd_weight"], 0.05)
        self.assertEqual(metrics["graph_rank_weight"], 0.03)
        self.assertEqual(metrics["graph_rank_margin_prob"], 0.25)
        self.assertEqual(metrics["graph_gold_pos_target"], 0.95)
        self.assertEqual(metrics["graph_gold_neg_target"], 0.05)
        self.assertEqual(metrics["graph_logit_cap"], 5.0)
        self.assertEqual(metrics["graph_edge_masking_folds"], 2)
        self.assertEqual(metrics["graph_pseudo_supervised_rows"], 4)
        self.assertEqual(metrics["graph_supervised_rows_total"], 12)
        self.assertGreater(metrics["graph_pseudo_weight_mean"], 0.0)
        self.assertLess(metrics["graph_pseudo_weight_mean"], 1.0)
        self.assertGreater(metrics["graph_pseudo_weight_min"], 0.0)
        self.assertLessEqual(metrics["graph_pseudo_weight_max"], 0.5)
        self.assertAlmostEqual(metrics["graph_pseudo_weight_base_round"], 0.1)
        self.assertEqual(metrics["graph_rank_rows_pseudo_should_be_zero"], 0)
        self.assertEqual(metrics["graph_leakage_guard"]["pseudo_forward_prereq_overlap"], 0)
        self.assertTrue(all(x >= 0 for x in metrics["graph_fold_masked_prereq_edges"]))

    def test_pseudo_kd_uses_soft_targets_and_confidence_from_teacher_probability(self):
        data = load_stage2_data(ROOT / "stage0" / "outputs", ROOT / "stage1" / "outputs")
        trainer = GraphTrainer(
            {
                "pseudo_weight_round1": 0.1,
                "pseudo_min_weight": 0.05,
                "pseudo_max_weight": 0.5,
                "pseudo_soft_target_min": 0.10,
                "pseudo_soft_target_max": 0.90,
            },
            data,
            torch.device("cpu"),
        )
        pseudo = pd.DataFrame(
            {
                "source_index": [0, 1, 2],
                "target_index": [1, 2, 3],
                "label": [0.83, 0.27, 1.0],
                "model_score": [0.99, 0.01, 0.66],
                "combined_score": [0.99, 0.01, 0.61],
                "label_confidence": [0.01, 0.02, 0.03],
            }
        )
        out = trainer._prepare_pseudo_frame(pseudo, iteration=0)
        self.assertEqual(out["label"].tolist(), [1, 0, 1])
        self.assertTrue(np.allclose(out["soft_target"].to_numpy(dtype=float), [0.90, 0.10, 0.61]))
        self.assertTrue(np.allclose(out["kd_target"].to_numpy(dtype=float), [0.90, 0.10, 0.61]))
        self.assertAlmostEqual(float(out.iloc[0].label_confidence), 0.80)
        self.assertAlmostEqual(float(out.iloc[1].label_confidence), 0.80)
        self.assertAlmostEqual(float(out.iloc[2].label_confidence), 0.22)
        self.assertTrue((out["sample_weight"] <= 0.5).all())
        self.assertTrue((out["sample_weight"] >= 0.05).all())
        self.assertTrue(np.allclose(out["sample_weight"].to_numpy(dtype=float), [0.08, 0.08, 0.05]))

    def test_calibrated_prediction_path_applies_temperature_and_bias(self):
        data = load_stage2_data(ROOT / "stage0" / "outputs", ROOT / "stage1" / "outputs")
        trainer = GraphTrainer({"graph_hidden_dim": 8, "graph_dropout": 0.0, "decoder_dropout": 0.0}, data, torch.device("cpu"))
        pairs = data.val_labels.head(12).copy()
        raw = trainer.predict_scores(pairs, calibrated=False)
        trainer.calibration_temperature = 2.0
        trainer.calibration_bias = 0.25
        trainer.calibration_fitted = True
        calibrated = trainer.predict_scores(pairs, calibrated=True)
        self.assertEqual(raw.shape, calibrated.shape)
        self.assertFalse(np.allclose(raw, calibrated))

    def test_ranking_skips_pairs_whose_reverse_is_gold_positive(self):
        data = load_stage2_data(ROOT / "stage0" / "outputs", ROOT / "stage1" / "outputs")
        cfg = {"graph_hidden_dim": 8, "graph_dropout": 0.0, "decoder_dropout": 0.0}
        trainer = GraphTrainer(cfg, data, torch.device("cpu"))
        rows = data.train_labels[data.train_labels.label.astype(int) == 1].head(1).copy()
        reverse = rows.copy()
        reverse["source_index"], reverse["target_index"] = rows["target_index"].values, rows["source_index"].values
        reverse["source_id"], reverse["target_id"] = rows["target_id"].values, rows["source_id"].values
        reverse["source"], reverse["target"] = rows["target"].values, rows["source"].values
        trainer._gold_pos_pairs.add((int(reverse.iloc[0].source_index), int(reverse.iloc[0].target_index)))
        rank_df, pseudo_rows = trainer._rank_positive_frame(pd.concat([rows, reverse], ignore_index=True))
        self.assertEqual(len(rank_df), 0)
        self.assertEqual(pseudo_rows, 0)


if __name__ == "__main__":
    unittest.main()

