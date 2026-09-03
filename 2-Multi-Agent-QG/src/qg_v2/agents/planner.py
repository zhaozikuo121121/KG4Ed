from __future__ import annotations

from typing import Any

from qg_v2.graph import KnowledgeGraph


class PlannerAgent:
    """Deterministic graph planner.

    `plan()` preserves the historical single-plan behavior for internal fallback.
    `plan_all()` enumerates every feasible structural question type for one concept.
    """

    STRUCTURAL_TYPES = [
        "concept_discrimination",
        "prerequisite_dependency",
        "component_membership",
        "multi_hop_reasoning",
    ]

    def __init__(self, graph: KnowledgeGraph) -> None:
        self.graph = graph

    def plan(self, user_input: str, *, force_definition: bool = False) -> dict[str, object]:
        plans = self.plan_all(user_input, force_definition=force_definition)
        if plans.get("status") != "ok":
            return plans
        return plans["plans"][0]

    def plan_all(self, user_input: str, *, force_definition: bool = False) -> dict[str, Any]:
        resolved = self.graph.resolve(user_input)
        if resolved is None:
            return {"status": "return", "to": "orchestrator", "reason": "concept_not_found"}
        target = resolved[0] if isinstance(resolved, list) else resolved

        if force_definition:
            definition_plan = self._definition_plan(target)
            definition_plan["available_question_types"] = ["definition"]
            return {"status": "ok", "target_concept": target, "plans": [definition_plan], "available_question_types": ["definition"]}

        plans: list[dict[str, object]] = []
        concept_plan = self._concept_discrimination_plan(target)
        if concept_plan:
            plans.append(concept_plan)

        prereq_plan = self._prerequisite_plan(target)
        if prereq_plan:
            plans.append(prereq_plan)

        part_plan = self._part_plan(target)
        if part_plan:
            plans.append(part_plan)

        chain_plan = self._multi_hop_plan(target)
        if chain_plan:
            plans.append(chain_plan)

        if not plans:
            plans = [self._definition_plan(target)]

        available = [str(plan["question_type"]) for plan in plans]
        for plan in plans:
            plan["available_question_types"] = available
        return {"status": "ok", "target_concept": target, "plans": plans, "available_question_types": available}

    def _concept_discrimination_plan(self, target: str) -> dict[str, object] | None:
        conf = self.graph.confusables_of(target)
        if not conf:
            return None
        edges = conf[:6]
        concepts = self._unique([target] + [e.target for e in edges])
        return {
            "status": "ok",
            "target_concept": target,
            "question_type": "concept_discrimination",
            "activated_subgraph": {"concepts": concepts, "edges": [e.to_dict() for e in edges]},
            "intent": f"Assess whether the learner can distinguish {target} from confusable concepts.",
        }

    def _prerequisite_plan(self, target: str) -> dict[str, object] | None:
        prereqs = self.graph.prereqs_of(target)
        dependents = self.graph.dependents_of(target)
        if not (prereqs or dependents):
            return None
        edges = (prereqs or dependents)[:4]
        concepts = self._unique([target] + [e.source for e in edges] + [e.target for e in edges])
        return {
            "status": "ok",
            "target_concept": target,
            "question_type": "prerequisite_dependency",
            "activated_subgraph": {"concepts": concepts, "edges": [e.to_dict() for e in edges]},
            "intent": f"Assess whether the learner can determine prerequisite direction for {target}.",
        }

    def _part_plan(self, target: str) -> dict[str, object] | None:
        parts = self.graph.parts_of(target)
        wholes = self.graph.wholes_of(target)
        if not (parts or wholes):
            return None
        edges = (parts or wholes)[:4]
        question_type = "component_membership" if parts else "whole_decomposition"
        concepts = self._unique([target] + [e.source for e in edges] + [e.target for e in edges])
        return {
            "status": "ok",
            "target_concept": target,
            "question_type": question_type,
            "activated_subgraph": {"concepts": concepts, "edges": [e.to_dict() for e in edges]},
            "intent": f"Assess whether the learner understands part-of and whole relations for {target}.",
        }

    def _multi_hop_plan(self, target: str) -> dict[str, object] | None:
        chains = self.graph.clean_chains(target, max_hops=2)
        if not chains:
            return None
        path = chains[0]
        edges = []
        for left, right in zip(path, path[1:]):
            for edge in self.graph.prereqs_of(left) + self.graph.dependents_of(left):
                if {edge.source, edge.target} == {left, right}:
                    edges.append(edge.to_dict())
                    break
        return {
            "status": "ok",
            "target_concept": target,
            "question_type": "multi_hop_reasoning",
            "activated_subgraph": {"concepts": path, "edges": edges},
            "intent": f"Assess whether the learner can follow a prerequisite chain involving {target}.",
        }

    def _definition_plan(self, target: str) -> dict[str, object]:
        return {
            "status": "ok",
            "target_concept": target,
            "question_type": "definition",
            "activated_subgraph": {"concepts": [target], "edges": []},
            "intent": f"Assess whether the learner understands the basic meaning of {target}.",
        }

    @staticmethod
    def _unique(items: list[str]) -> list[str]:
        seen: set[str] = set()
        output: list[str] = []
        for item in items:
            if item not in seen:
                seen.add(item)
                output.append(item)
        return output
