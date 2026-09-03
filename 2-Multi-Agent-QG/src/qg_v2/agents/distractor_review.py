from __future__ import annotations

import json
import re
from typing import Any

from qg_v2.graph import KnowledgeGraph
from qg_v2.llm import LLMRouter


REVIEW_CHECKS = [
    "answer_type_match",
    "surface_eliminability",
    "peer_set_consistency",
    "contextual_plausibility",
]


class DistractorReviewAgent:
    """Review whether distractors remain plausible in the concrete stem context."""

    def __init__(self, graph: KnowledgeGraph, llm: LLMRouter) -> None:
        self.graph = graph
        self.llm = llm

    def review(self, blueprint: dict[str, Any]) -> dict[str, Any]:
        if not self.llm.settings.enable_distractor_review:
            return self._disabled_result()
        if not self.llm.is_mock:
            try:
                return self._review_with_llm(blueprint)
            except Exception as exc:
                if not self.llm.settings.allow_llm_fallback:
                    raise
                result = self._review_deterministic(blueprint)
                result["llm_fallback_reason"] = str(exc)
                return result
        return self._review_deterministic(blueprint)

    @staticmethod
    def _disabled_result() -> dict[str, Any]:
        return {
            "passed": True,
            "checks": {key: True for key in REVIEW_CHECKS},
            "weak_options": [],
            "issue_type": None,
            "feedback_to_answer_distractor": "distractor review disabled",
            "suggested_constraints": [],
        }

    def _review_with_llm(self, blueprint: dict[str, Any]) -> dict[str, Any]:
        item = blueprint["writer_output"]
        answer_plan = blueprint.get("answer_plan", {})
        distractor_plan = blueprint.get("distractor_plan", {})
        concepts = list(item.get("options", {}).values()) + [
            d.get("concept", "") for d in distractor_plan.get("distractors", [])
        ]
        definitions = self.graph.concept_definitions([str(c) for c in concepts if c])
        prompt = {
            "role": "user",
            "content": (
                "Review distractors in the concrete stem context. Do not only check topical relatedness: each distractor must share the correct answer's type, semantic role, and constraint category, and must not be trivially eliminated by surface category. If the stem asks for a method, error values, datasets, matrices, and metric results are generally invalid peers. Return strict JSON.\n"
                + json.dumps(
                    {
                        "instruction_version": blueprint.get("instruction_version"),
                        "item": item,
                        "answer_plan": answer_plan,
                        "distractor_plan": distractor_plan,
                        "concept_definitions": definitions,
                        "source_context": blueprint.get("source_context", []),
                        "checks_required": REVIEW_CHECKS,
                        "output_schema": {
                            "passed": True,
                            "checks": {
                                "answer_type_match": True,
                                "surface_eliminability": True,
                                "peer_set_consistency": True,
                                "contextual_plausibility": True,
                            },
                            "weak_options": [
                                {
                                    "label": "B",
                                    "concept": "...",
                                    "reason": "why it can be eliminated from the current stem",
                                    "issue_type": "answer_type_mismatch|surface_eliminable|not_peer_set|not_contextually_plausible",
                                }
                            ],
                            "issue_type": "answer_type_mismatch|surface_eliminable|not_peer_set|not_contextually_plausible|null",
                            "feedback_to_answer_distractor": "redesign requirements for the answer/distractor planner",
                            "suggested_constraints": ["new distractors must be methods or techniques, not errors, sets, matrices, or metric results"],
                        },
                    },
                    ensure_ascii=False,
                )
            ),
        }
        result = self.llm.complete_json("distractor_review", [prompt], temperature=0.1)
        return self._normalize_result(result)

    def _review_deterministic(self, blueprint: dict[str, Any]) -> dict[str, Any]:
        item = blueprint.get("writer_output", {})
        options = item.get("options", {}) if isinstance(item, dict) else {}
        answer_label = item.get("answer", "A") if isinstance(item, dict) else "A"
        stem = str(item.get("stem", "")) if isinstance(item, dict) else ""
        expected_type = self._infer_expected_answer_type(stem)
        weak_options: list[dict[str, str]] = []

        for label, text in options.items():
            if label == answer_label:
                continue
            concept = self._option_to_concept(label, text, blueprint)
            actual_type = self._classify_concept_type(concept or str(text))
            reason = self._weak_reason(expected_type, actual_type, concept or str(text))
            if reason:
                weak_options.append(
                    {
                        "label": label,
                        "concept": concept or str(text),
                        "reason": reason,
                        "issue_type": "answer_type_mismatch" if expected_type else "surface_eliminable",
                    }
                )

        passed = not weak_options
        issue_type = weak_options[0]["issue_type"] if weak_options else None
        constraints = self._suggested_constraints(expected_type, stem)
        return {
            "passed": passed,
            "checks": {
                "answer_type_match": passed,
                "surface_eliminability": passed,
                "peer_set_consistency": passed,
                "contextual_plausibility": passed,
            },
            "weak_options": weak_options,
            "issue_type": issue_type,
            "feedback_to_answer_distractor": self._feedback_text(weak_options, constraints),
            "suggested_constraints": constraints,
        }

    def _normalize_result(self, result: dict[str, Any]) -> dict[str, Any]:
        checks_raw = result.get("checks") if isinstance(result.get("checks"), dict) else {}
        checks = {key: self._coerce_bool(checks_raw.get(key, result.get(key, True))) for key in REVIEW_CHECKS}
        weak_options = result.get("weak_options") if isinstance(result.get("weak_options"), list) else []
        passed_raw = result.get("passed")
        passed = self._coerce_bool(passed_raw) if passed_raw is not None else all(checks.values()) and not weak_options
        issue_type = result.get("issue_type") if isinstance(result.get("issue_type"), str) and result.get("issue_type") != "null" else None
        return {
            "passed": passed,
            "checks": checks,
            "weak_options": weak_options,
            "issue_type": issue_type,
            "feedback_to_answer_distractor": str(result.get("feedback_to_answer_distractor") or ""),
            "suggested_constraints": result.get("suggested_constraints") if isinstance(result.get("suggested_constraints"), list) else [],
            "raw_distractor_review_output": result,
        }

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "pass", "passed", "ok", "1"}
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, dict):
            for key in ("passed", "pass", "ok", "value"):
                if key in value:
                    return DistractorReviewAgent._coerce_bool(value[key])
        return False

    @staticmethod
    def _infer_expected_answer_type(stem: str) -> str | None:
        lowered = stem.lower()
        if re.search(r"procedure|method|technique|approach|algorithm", lowered):
            return "method"
        if re.search(r"metric|measure|score|ratio|rate", lowered):
            return "metric"
        if re.search(r"test set|validation set|dataset", lowered):
            return "dataset"
        if re.search(r"error|loss", lowered):
            return "error"
        return None

    @staticmethod
    def _classify_concept_type(concept: str) -> str | None:
        lowered = concept.lower()
        if re.search(r"error|loss|cost", lowered):
            return "error"
        if re.search(r"set|dataset|fold", lowered):
            return "dataset"
        if re.search(r"matrix", lowered):
            return "matrix"
        if re.search(r"accuracy|precision|recall|f1|score|auc|rate|ratio|metric", lowered):
            return "metric"
        if re.search(r"validation|cross validation|bootstrap|hold[- ]?out|algorithm|method|approach|technique|regression|clustering|model", lowered):
            return "method"
        return None

    @staticmethod
    def _weak_reason(expected_type: str | None, actual_type: str | None, concept: str) -> str:
        if expected_type == "method" and actual_type in {"error", "dataset", "matrix", "metric"}:
            return f"The stem asks for a method, but {concept} looks like {actual_type} and can be eliminated by category."
        if expected_type == "metric" and actual_type in {"dataset", "matrix", "method"}:
            return f"The stem asks for a metric or ratio, but {concept} is not a peer metric and is easy to eliminate."
        if expected_type and actual_type and expected_type != actual_type:
            return f"The expected answer type is {expected_type}, but {concept} looks like {actual_type}."
        return ""

    @staticmethod
    def _suggested_constraints(expected_type: str | None, stem: str) -> list[str]:
        if expected_type == "method":
            return ["Prefer other evaluation, validation, or resampling methods/techniques", "Avoid error values, datasets, matrices, and metric results"]
        if expected_type == "metric":
            return ["Prefer other metrics, ratios, or measures", "Avoid methods, datasets, and matrices"]
        return ["Distractors must share the correct answer's type and semantic role"]

    @staticmethod
    def _feedback_text(weak_options: list[dict[str, str]], constraints: list[str]) -> str:
        if not weak_options:
            return "Distractors passed the contextual review."
        weak = "; ".join(f"{w.get('label')}={w.get('concept')}: {w.get('reason')}" for w in weak_options)
        return f"These options are not sufficiently plausible in context: {weak}. Redesign requirements: {'; '.join(constraints)}."

    @staticmethod
    def _option_to_concept(label: str, option_text: Any, blueprint: dict[str, Any]) -> str:
        distractors = blueprint.get("distractor_plan", {}).get("distractors", [])
        label_to_index = {"B": 0, "C": 1, "D": 2}
        if label in label_to_index and label_to_index[label] < len(distractors):
            return str(distractors[label_to_index[label]].get("concept", option_text))
        if label == "A":
            return str(blueprint.get("answer_plan", {}).get("correct_answer", {}).get("concept", option_text))
        text = str(option_text)
        match = re.search(r"\(([^)]+)\)", text)
        return match.group(1) if match else text
