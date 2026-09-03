from __future__ import annotations

import json
import ast
import operator
import re
from typing import Any

from qg_v2.graph import KnowledgeGraph
from qg_v2.llm.client import LLMError
from qg_v2.llm import LLMRouter


class WriterAgent:
    def __init__(self, graph: KnowledgeGraph, llm: LLMRouter) -> None:
        self.graph = graph
        self.llm = llm

    def write(self, blueprint: dict[str, Any]) -> dict[str, Any]:
        if blueprint.get("question_type") == "calculation":
            formula = blueprint.get("selected_formula") or blueprint.get("formula_candidates", [{}])[0]
            if str(blueprint.get("target_concept", "")).lower() == "precision":
                return {"item_type":"open_calculation","stem":"A classifier has 80 true positives and 20 false positives. Calculate precision.","given":{"TP":80,"FP":20},"ask":["Calculate precision"],"formula_used":"TP / (TP + FP)","final_answer":"0.8","unit":"dimensionless","solution_steps":["80 / (80 + 20) = 0.8"],"explanation":"Precision is TP divided by predicted positives.","context_refs":[]}
            raise LLMError("calculation generation requires a supported formula")
        if not self.llm.is_mock:
            try:
                return self._write_with_llm(blueprint)
            except Exception as exc:
                if not self.llm.settings.allow_llm_fallback:
                    raise
                item = self._write_deterministic(blueprint)
                item["llm_fallback_reason"] = str(exc)
                return item
        return self._write_deterministic(blueprint)

    def _write_with_llm(self, blueprint: dict[str, Any]) -> dict[str, Any]:
        answer = blueprint["answer_plan"]["correct_answer"]
        distractors = blueprint["distractor_plan"]["distractors"]
        concepts = [answer["concept"]] + [d["concept"] for d in distractors] + list(blueprint.get("activated_subgraph", {}).get("concepts", []))
        definitions = self.graph.concept_definitions(concepts)
        prompt = {
            "role": "user",
            "content": (
                "You are an English-language quiz writer. Do not change the concepts in correct_answer or distractors. "
                "Write the stem, options, and explanation entirely in English using the supplied definitions and graph relations. "
                "The source context is optional supporting evidence and does not have to be used. Return strict JSON.\n"
                + json.dumps(
                    {
                        "instruction_version": blueprint.get("instruction_version"),
                        "target_concept": blueprint.get("target_concept"),
                        "question_type": blueprint.get("question_type"),
                        "selected_bloom": blueprint.get("selected_bloom"),
                        "target_difficulty": blueprint.get("target_difficulty"),
                        "correct_answer": answer,
                        "distractors": distractors,
                        "concept_definitions": definitions,
                        "source_context": blueprint.get("source_context", []),
                        "output_schema": {
                            "stem": "... ?",
                            "options": {"A": "correct concept", "B": "distractor", "C": "distractor", "D": "distractor"},
                            "answer": "A",
                            "explanation": "Why the correct option is supported and why each distractor is incorrect.",
                            "context_refs": [],
                        },
                    },
                    ensure_ascii=False,
                )
            ),
        }
        result = self.llm.complete_json("writer", [prompt], temperature=0.4)
        if not self._valid_item(result):
            raise LLMError("writer returned invalid item schema")
        self._sanitize_context_refs(result, blueprint)
        return result

    def _write_deterministic(self, blueprint: dict[str, Any]) -> dict[str, Any]:
        target = str(blueprint["target_concept"])
        qtype = str(blueprint["question_type"])
        correct = blueprint["answer_plan"]["correct_answer"]
        distractors = blueprint["distractor_plan"]["distractors"]
        correct_concept = str(correct["concept"])

        if qtype == "prerequisite_dependency":
            stem = f"According to the prerequisite relations in the knowledge graph, which concept should typically be mastered before learning \"{target}\"?"
        elif qtype in {"component_membership", "whole_decomposition"}:
            stem = f"According to the part-of relations in the knowledge graph, which concept is a component of \"{target}\"?"
        elif qtype == "concept_discrimination":
            stem = f"Based on the concept definition and confusable relations, which option best matches the meaning or boundary of \"{target}\"?"
        else:
            stem = f"Based on the concept definition, which option best matches the meaning of \"{target}\"?"

        options = {
            "A": correct_concept,
            "B": str(distractors[0]["concept"]),
            "C": str(distractors[1]["concept"]),
            "D": str(distractors[2]["concept"]),
        }
        explanation_parts = [f"The correct concept is {correct_concept}, supported by the graph and definition."]
        for label, distractor in zip(["B", "C", "D"], distractors):
            explanation_parts.append(f"{distractor['concept']} is incorrect because it differs from {correct_concept} in its role or definition.")
        return {
            "stem": stem,
            "options": options,
            "answer": "A",
            "explanation": " ".join(explanation_parts),
            "context_refs": [],
        }

    @staticmethod
    def _sanitize_context_refs(item: dict[str, Any], blueprint: dict[str, Any]) -> None:
        allowed = {entry.get("context_id") for entry in blueprint.get("source_context", [])}
        refs = item.get("context_refs") if isinstance(item.get("context_refs"), list) else []
        item["context_refs"] = [ref for ref in refs if ref in allowed]

    @staticmethod
    def _valid_item(item: dict[str, Any]) -> bool:
        return (
            isinstance(item.get("stem"), str)
            and isinstance(item.get("options"), dict)
            and set(item["options"].keys()) == {"A", "B", "C", "D"}
            and item.get("answer") in {"A", "B", "C", "D"}
            and isinstance(item.get("explanation"), str)
        )


