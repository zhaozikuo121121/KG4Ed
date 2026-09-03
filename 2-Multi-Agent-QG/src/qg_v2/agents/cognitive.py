from __future__ import annotations


class CognitiveDifficultyAgent:
    """Rule-constrained Bloom and target difficulty controller."""

    def decide(self, blueprint: dict[str, object], structural_features: dict[str, object]) -> dict[str, object]:
        question_type = str(blueprint["question_type"])
        n_prereq = int(structural_features.get("num_direct_prereqs", 0))
        n_parts = int(structural_features.get("num_part_components", 0))
        has_chain = bool(structural_features.get("has_clean_2hop_chain", False))
        has_conf = bool(structural_features.get("has_strong_confusable", False))

        if question_type == "concept_discrimination":
            candidates = ["analysis", "evaluation"]
            selected = "analysis"
            bloom_reason = "This item compares confusable concept boundaries and fits the analysis level."
        elif question_type in {"prerequisite_dependency", "application"}:
            candidates = ["understanding", "application"]
            selected = "application" if question_type == "application" or has_chain else "understanding"
            bloom_reason = "This item uses prerequisite relations to judge dependencies and fits understanding or application."
        elif question_type in {"component_membership", "whole_decomposition"}:
            candidates = ["understanding", "analysis"]
            selected = "analysis" if n_parts > 1 else "understanding"
            bloom_reason = "This item uses a part-of structure to judge whole-part relations."
        elif question_type == "multi_hop_reasoning":
            candidates = ["application", "analysis"]
            selected = "analysis"
            bloom_reason = "This item follows a multi-hop prerequisite chain and requires relational analysis."
        else:
            candidates = ["remembering", "understanding"]
            selected = "understanding"
            bloom_reason = "This item focuses on a single concept definition and fits the understanding level."

        if has_chain or n_prereq >= 3 or n_parts >= 4 or (has_conf and question_type == "concept_discrimination"):
            difficulty = "hard" if has_chain or n_prereq >= 3 or n_parts >= 4 else "medium"
        elif has_conf or n_prereq > 0 or n_parts > 0:
            difficulty = "medium"
        else:
            difficulty = "easy"

        if difficulty == "easy":
            diff_reason = "The local subgraph is simple and mainly relies on one concept definition."
        elif difficulty == "medium":
            diff_reason = "The local subgraph has direct or confusable relations but needs no complex multi-hop reasoning."
        else:
            diff_reason = "The local subgraph has multi-hop chains, multiple prerequisites, or complex part structures, so difficulty is higher."

        return {
            "status": "ok",
            "bloom_candidates": candidates,
            "selected_bloom": selected,
            "bloom_reason": bloom_reason,
            "target_difficulty": difficulty,
            "difficulty_reason": diff_reason,
        }
