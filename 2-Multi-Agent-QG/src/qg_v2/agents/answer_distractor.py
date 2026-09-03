from __future__ import annotations

import json
from typing import Any

from qg_v2.graph import KnowledgeGraph
from qg_v2.llm import LLMError, LLMRouter


class AnswerDistractorAgent:
    def __init__(self, graph: KnowledgeGraph, llm: LLMRouter) -> None:
        self.graph = graph
        self.llm = llm

    def plan(self, blueprint: dict[str, Any]) -> dict[str, Any]:
        if not self.llm.is_mock:
            try:
                return self._plan_with_llm(blueprint)
            except Exception as exc:  # keep CLI usable only when fallback is explicitly enabled
                if not self.llm.settings.allow_llm_fallback:
                    raise
                fallback = self._plan_deterministic(blueprint)
                fallback["llm_fallback_reason"] = str(exc)
                return fallback
        return self._plan_deterministic(blueprint)

    def _plan_with_llm(self, blueprint: dict[str, Any]) -> dict[str, Any]:
        target = str(blueprint["target_concept"])
        subgraph = blueprint.get("activated_subgraph", {})
        concepts = list(dict.fromkeys([target] + list(subgraph.get("concepts", []))))
        # Include extra candidates so the model can filter but not invent.
        candidate_edges = self._candidate_pool(target)
        candidate_names = [edge["concept"] for edge in candidate_edges]
        definitions = self.graph.concept_definitions(list(dict.fromkeys(concepts + candidate_names)))
        prompt = {
            "role": "user",
            "content": (
                "You plan the correct answer and distractors. Use only the supplied concepts, definitions, and edges. Return strict JSON.\n"
                "Edge semantics: prerequisite(X->Y) means X precedes Y; part-of(X->Y) means X is a component of Y; similar/confusable is symmetric.\n"
                "Choose correct_answer and three distractors. Do not write the final stem.\n"
                + json.dumps(
                    {
                        "instruction_version": blueprint.get("instruction_version"),
                        "question_type": blueprint.get("question_type"),
                        "intent": blueprint.get("intent"),
                        "target_concept": target,
                        "activated_subgraph": subgraph,
                        "concept_definitions": definitions,
                        "source_context": blueprint.get("source_context", []),
                        "target_difficulty": blueprint.get("target_difficulty"),
                        "candidate_pool": candidate_edges,
                        "output_schema": {
                            "status": "ok",
                            "correct_answer": {"concept": "<id>", "basis": "<definition or edge evidence>"},
                            "distractors": [
                                {
                                    "concept": "<id>",
                                    "source_relation": "similar/confusable|sibling-part-of|prereq-neighbor|same_topic|synthetic",
                                    "reason": "why it might be selected",
                                    "diff_from_correct": "difference from the correct answer",
                                    "exclusive": "why it is mutually exclusive",
                                }
                            ],
                        },
                    },
                    ensure_ascii=False,
                )
            ),
        }
        result = self.llm.complete_json("answer_distractor", [prompt], temperature=0.2)
        if result.get("status") != "ok" or len(result.get("distractors", [])) != 3:
            raise LLMError("answer_distractor returned invalid schema")
        return result

    def _plan_deterministic(self, blueprint: dict[str, Any]) -> dict[str, Any]:
        target = str(blueprint["target_concept"])
        qtype = str(blueprint["question_type"])
        if qtype == "prerequisite_dependency" and self.graph.prereqs_of(target):
            correct = self.graph.prereqs_of(target)[0].source
            basis = f"Prerequisite edge {correct} -> {target} means {correct} must be learned before {target}."
            forbidden = {e.source for e in self.graph.prereqs_of(target)} | {target}
        elif qtype in {"component_membership", "whole_decomposition"} and self.graph.parts_of(target):
            correct = self.graph.parts_of(target)[0].source
            basis = f"Part-of edge {correct} -> {target} means {correct} is a component of {target}."
            forbidden = {e.source for e in self.graph.parts_of(target)} | {target}
        else:
            correct = target
            profile = self.graph.concept_profile(target)
            basis = f"Definition of {target}: {profile[:240] if profile else 'the target concept registered in the graph.'}"
            forbidden = {target}

        distractors = self._choose_distractors(target, correct, forbidden)
        if len(distractors) < 3:
            return {"status": "return", "to": "planner", "reason": "cannot_form_exclusive_options"}
        return {
            "status": "ok",
            "correct_answer": {"concept": correct, "basis": basis},
            "distractors": distractors[:3],
        }


    def revise_with_feedback(self, blueprint: dict[str, Any], review_feedback: dict[str, Any]) -> dict[str, Any]:
        """Revise distractors after contextual review without changing the correct answer."""
        if not self.llm.is_mock:
            try:
                return self._revise_with_llm(blueprint, review_feedback)
            except Exception as exc:
                if not self.llm.settings.allow_llm_fallback:
                    raise
                fallback = self._revise_deterministic(blueprint, review_feedback)
                fallback["llm_fallback_reason"] = str(exc)
                return fallback
        return self._revise_deterministic(blueprint, review_feedback)

    def _revise_with_llm(self, blueprint: dict[str, Any], review_feedback: dict[str, Any]) -> dict[str, Any]:
        target = str(blueprint["target_concept"])
        current_answer_plan = blueprint.get("answer_plan", {})
        correct_answer = current_answer_plan.get("correct_answer") or current_answer_plan
        correct_concept = str(correct_answer.get("concept", target))
        candidate_edges = self._candidate_pool(target) + self._candidate_pool(correct_concept)
        candidate_names = [edge["concept"] for edge in candidate_edges]
        current_distractors = blueprint.get("distractor_plan", {}).get("distractors", [])
        current_names = [d.get("concept") for d in current_distractors]
        definitions = self.graph.concept_definitions(list(dict.fromkeys([target, correct_concept] + candidate_names + [str(n) for n in current_names if n])))
        prompt = {
            "role": "user",
            "content": (
                "Redesign distractors using contextual review feedback. Do not change correct_answer, target_concept, question_type, Bloom, or difficulty. "
                "Return three plausible distractors with the same answer type and semantic role as the correct answer. "
                "If candidates are insufficient, synthetic distractors are allowed but must record source_relation=synthetic and a reason. Return strict JSON.\n"
                + json.dumps(
                    {
                        "instruction_version": blueprint.get("instruction_version"),
                        "stem": blueprint.get("writer_output", {}).get("stem"),
                        "target_concept": target,
                        "question_type": blueprint.get("question_type"),
                        "selected_bloom": blueprint.get("selected_bloom"),
                        "target_difficulty": blueprint.get("target_difficulty"),
                        "correct_answer": correct_answer,
                        "current_distractors": current_distractors,
                        "review_feedback": review_feedback,
                        "candidate_pool": candidate_edges,
                        "concept_definitions": definitions,
                        "source_context": blueprint.get("source_context", []),
                        "output_schema": {
                            "status": "ok",
                            "correct_answer": correct_answer,
                            "distractors": [
                                {
                                    "concept": "<id>",
                                    "source_relation": "similar/confusable|sibling-part-of|prereq-neighbor|same_topic|synthetic",
                                    "reason": "why it might be selected in the current stem",
                                    "diff_from_correct": "difference from the correct answer",
                                    "exclusive": "why it is mutually exclusive",
                                    "review_adjusted": True,
                                    "review_adjustment_reason": "how review feedback was addressed",
                                }
                            ],
                        },
                    },
                    ensure_ascii=False,
                )
            ),
        }
        result = self.llm.complete_json("answer_distractor", [prompt], temperature=0.35)
        # Models occasionally return a valid JSON object but omit ``status``
        # or produce fewer than three options during a revision.  This is a
        # content/schema issue, not an API failure; recover locally so one
        # malformed revision cannot abort the whole generation run.
        distractors = result.get("distractors") if isinstance(result, dict) else None
        if isinstance(distractors, dict):
            # Some models wrap the list in an object such as
            # {"items": [...]} or {"options": [...]}.
            for key in ("items", "options", "choices"):
                if isinstance(distractors.get(key), list):
                    distractors = distractors[key]
                    break
        if isinstance(distractors, list):
            distractors = [item for item in distractors if isinstance(item, dict) and item.get("concept")]
        if (not isinstance(result, dict) or result.get("status") not in (None, "ok")
                or not isinstance(distractors, list) or len(distractors) != 3):
            fallback = self._revise_deterministic(blueprint, review_feedback)
            if fallback.get("status") == "ok":
                fallback["llm_fallback_reason"] = (
                    "revision response did not satisfy schema "
                    f"(status={result.get('status') if isinstance(result, dict) else None!r}, "
                    f"distractors={len(distractors) if isinstance(distractors, list) else 'invalid'})"
                )
                return fallback
            raise LLMError("answer_distractor revision returned invalid schema")
        result["distractors"] = distractors
        result["correct_answer"] = correct_answer
        for distractor in result["distractors"]:
            distractor.setdefault("review_adjusted", True)
            distractor.setdefault("review_adjustment_reason", "Redesigned from contextual distractor review feedback.")
        return result

    def _revise_deterministic(self, blueprint: dict[str, Any], review_feedback: dict[str, Any]) -> dict[str, Any]:
        target = str(blueprint["target_concept"])
        current_answer_plan = blueprint.get("answer_plan", {})
        correct_answer = current_answer_plan.get("correct_answer") or current_answer_plan
        correct = str(correct_answer.get("concept", target))
        weak_concepts = {
            str(item.get("concept"))
            for item in review_feedback.get("weak_options", [])
            if isinstance(item, dict) and item.get("concept")
        }
        forbidden = {target, correct} | weak_concepts
        constraints = [str(c) for c in review_feedback.get("suggested_constraints", [])]
        selected = self._choose_distractors_with_constraints(target, correct, forbidden, constraints)
        if len(selected) < 3:
            return {"status": "return", "to": "planner", "reason": "cannot_revise_contextual_distractors"}
        for distractor in selected[:3]:
            distractor["review_adjusted"] = distractor.get("concept") in weak_concepts or bool(weak_concepts)
            distractor["review_adjustment_reason"] = review_feedback.get("feedback_to_answer_distractor", "Replaced a weak distractor based on contextual review feedback.")
        return {"status": "ok", "correct_answer": correct_answer, "distractors": selected[:3]}

    def _choose_distractors_with_constraints(
        self, target: str, correct: str, forbidden: set[str], constraints: list[str]
    ) -> list[dict[str, str]]:
        selected: list[dict[str, str]] = []
        seen = set(forbidden)
        wants_method = any("method" in c.lower() or "technique" in c.lower() for c in constraints)
        wants_metric = any("metric" in c.lower() or "ratio" in c.lower() for c in constraints)
        candidate_sources = self._candidate_pool(correct) + self._candidate_pool(target)
        for item in candidate_sources:
            concept = item["concept"]
            if concept in seen or concept not in self.graph.concepts:
                continue
            if not self._matches_review_constraints(concept, wants_method=wants_method, wants_metric=wants_metric):
                continue
            seen.add(concept)
            selected.append(self._distractor_record(concept, item["source_relation"], correct))
            if len(selected) >= 3:
                return selected
        for concept, source_relation in self._synthetic_distractors(correct, wants_method=wants_method, wants_metric=wants_metric):
            if concept in seen:
                continue
            seen.add(concept)
            record = self._distractor_record(concept, source_relation, correct)
            record["reason"] = f"{concept} and {correct} share the same answer type and are harder to eliminate from context."
            selected.append(record)
            if len(selected) >= 3:
                return selected
        return selected

    @staticmethod
    def _matches_review_constraints(concept: str, *, wants_method: bool, wants_metric: bool) -> bool:
        lowered = concept.lower()
        if wants_method:
            if any(token in lowered for token in ["error", "loss", "cost", "set", "matrix", "accuracy", "precision", "recall", "score"]):
                return False
            return any(token in lowered for token in ["validation", "bootstrap", "hold", "method", "algorithm", "regression", "clustering", "learning", "approach"])
        if wants_metric:
            if any(token in lowered for token in ["set", "matrix", "dataset", "method", "algorithm"]):
                return False
            return any(token in lowered for token in ["accuracy", "precision", "recall", "f1", "score", "rate", "ratio", "error"])
        return True

    @staticmethod
    def _synthetic_distractors(correct: str, *, wants_method: bool, wants_metric: bool) -> list[tuple[str, str]]:
        if wants_method:
            return [
                ("hold-out validation", "synthetic"),
                ("bootstrap validation", "synthetic"),
                ("leave-one-out validation", "synthetic"),
            ]
        if wants_metric:
            return [
                ("classification accuracy", "synthetic"),
                ("recall", "synthetic"),
                ("F1 score", "synthetic"),
            ]
        return [
            (f"related alternative to {correct}", "synthetic"),
            (f"neighboring concept of {correct}", "synthetic"),
            (f"contrasting concept to {correct}", "synthetic"),
        ]

    def _candidate_pool(self, target: str) -> list[dict[str, Any]]:
        pool: list[dict[str, Any]] = []
        for edge in self.graph.confusables_of(target)[:12]:
            pool.append({"concept": edge.target, "source_relation": "similar/confusable", "edge": edge.to_dict()})
        for edge in self.graph.sibling_parts(target)[:8]:
            pool.append({"concept": edge.source, "source_relation": "sibling-part-of", "edge": edge.to_dict()})
        for edge in (self.graph.prereqs_of(target) + self.graph.dependents_of(target))[:12]:
            other = edge.source if edge.source != target else edge.target
            pool.append({"concept": other, "source_relation": "prereq-neighbor", "edge": edge.to_dict()})
        for edge in self.graph.similar_of(target)[:12]:
            pool.append({"concept": edge.target, "source_relation": "same_topic", "edge": edge.to_dict()})
        seen = set()
        unique = []
        for item in pool:
            if item["concept"] not in seen:
                seen.add(item["concept"])
                unique.append(item)
        return unique

    def _choose_distractors(self, target: str, correct: str, forbidden: set[str]) -> list[dict[str, str]]:
        selected: list[dict[str, str]] = []
        seen = set(forbidden)
        candidate_sources = self._candidate_pool(correct) + self._candidate_pool(target)
        for item in candidate_sources:
            concept = item["concept"]
            if concept in seen or concept not in self.graph.concepts:
                continue
            seen.add(concept)
            selected.append(self._distractor_record(concept, item["source_relation"], correct))
            if len(selected) >= 3:
                return selected
        for concept in self.graph.all_concept_names():
            if concept in seen:
                continue
            seen.add(concept)
            selected.append(self._distractor_record(concept, "same_topic", correct))
            if len(selected) >= 3:
                return selected
        return selected

    @staticmethod
    def _distractor_record(concept: str, source_relation: str, correct: str) -> dict[str, str]:
        return {
            "concept": concept,
            "source_relation": source_relation,
            "reason": f"{concept} is locally related to {correct} in the graph or topic and may be confused by learners.",
            "diff_from_correct": f"{concept} is not the requested answer {correct} or its specified relation.",
            "exclusive": f"Choosing {concept} would conflict with the definition or graph relation for {correct}.",
        }

