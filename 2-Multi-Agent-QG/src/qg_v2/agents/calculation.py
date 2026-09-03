from __future__ import annotations
import re
from typing import Any
from qg_v2.graph import KnowledgeGraph
from qg_v2.llm import LLMRouter

class CalculationPlanningAgent:
    """Plans open calculation questions from explicit dataset/profile formulas."""
    def __init__(self, graph: KnowledgeGraph, llm: LLMRouter) -> None:
        self.graph, self.llm = graph, llm

    def has_explicit_target_formula(self, target: str) -> bool:
        return bool(self.graph.concept_formulas(target) or (target.lower() == "precision" and self.graph.concept_profile(target)))

    def plan(self, target: str, activated_concepts: list[str], source_context: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        formulas = [item.to_dict() for item in self.graph.concept_formulas(target)]
        if target.lower() == "precision" and not formulas:
            formulas = [{"formula_id": "PROFILE-precision", "name": "precision", "expression": "TP / (TP + FP)", "variables": {"TP": "true positives", "FP": "false positives"}, "provenance": "dataset", "source_record_ids": ["profile:precision"]}]
        if not formulas:
            return None
        return {"status": "ok", "target_concept": target, "question_type": "calculation", "activated_subgraph": {"concepts": activated_concepts, "edges": []}, "formula_candidates": formulas, "selected_formula": formulas[0], "calculation_design": {"design_type": "direct_numeric_calculation"}, "intent": f"Apply the verified formula for {target}."}
