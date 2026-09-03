from __future__ import annotations

import json
from typing import Any

from qg_v2.graph import KnowledgeGraph
from qg_v2.llm import LLMRouter
from qg_v2.llm.client import LLMError

CHECK_KEYS = [
    "unique_answer",
    "answerable",
    "on_topic",
    "clear_language",
    "options_exclusive",
    "distractors_valid",
    "bloom_aligned",
    "graph_faithful",
    "explanation_supported",
]

CHECK_ALIASES = {
    "unique_answer": ["answer_correctness", "single_correct_answer", "correct_answer_unique"],
    "answerable": ["concept_definition_consistency", "concept_consistency", "answerability", "sufficient_information", "format_compliance"],
    "on_topic": ["blueprint_alignment", "topic_relevance", "target_alignment"],
    "clear_language": ["stem_clarity", "language_clarity", "clarity"],
    "options_exclusive": ["option_validity", "option_exclusivity", "option_independence", "option_count"],
    "distractors_valid": ["distractor_plausibility", "distractor_validity", "distractors_plausible"],
    "bloom_aligned": ["bloom_taxonomy_alignment", "bloom_alignment"],
    "graph_faithful": ["knowledge_graph_consistency", "graph_consistency", "kg_consistency"],
    "explanation_supported": ["explanation_quality", "explanation_accuracy", "explanation_consistency", "explanation_validity"],
}


class ValidatorAgent:
    def __init__(self, graph: KnowledgeGraph, llm: LLMRouter) -> None:
        self.graph = graph
        self.llm = llm

    def validate(self, blueprint: dict[str, Any]) -> dict[str, Any]:
        if not self.llm.is_mock:
            try:
                return self._validate_with_llm(blueprint)
            except Exception as exc:
                if not self.llm.settings.allow_llm_fallback:
                    raise
                result = self._validate_deterministic(blueprint)
                result["llm_fallback_reason"] = str(exc)
                return result
        return self._validate_deterministic(blueprint)

    def _validate_calculation(self, blueprint: dict[str, Any]) -> dict[str, Any]:
        if not self.llm.is_mock:
            return self._validate_calculation_with_llm(blueprint)
        item = blueprint.get("writer_output", {})
        formulas = blueprint.get("formula_candidates", [])
        selected = blueprint.get("selected_formula") or (formulas[0] if formulas else {})
        formula_used = str(item.get("formula_used", ""))
        given = item.get("given", {}) if isinstance(item, dict) else {}
        steps = item.get("solution_steps", []) if isinstance(item, dict) else []
        source_expressions = {self._compact_formula(str(f.get("expression", ""))) for f in formulas if isinstance(f, dict)}
        compact_used = self._compact_formula(formula_used)
        formula_supported = bool(formula_used and any(expr == compact_used or expr in compact_used for expr in source_expressions))
        expected_variables = set((selected.get("variables") or {}).keys()) if isinstance(selected, dict) else set()
        if "=" in str(selected.get("expression", "")):
            expected_variables.discard(str(selected.get("expression", "")).split("=", 1)[0].strip())
        variables_complete = bool(given) and all(value is not None for value in given.values()) and expected_variables.issubset(given)
        calculation_well_defined = True
        if "TP" in given and "FP" in given:
            try:
                calculation_well_defined = (float(given["TP"]) + float(given["FP"])) != 0
            except (TypeError, ValueError):
                calculation_well_defined = False
        answer_correct = True
        if "TP" in given and "FP" in given:
            try:
                expected = float(given["TP"]) / (float(given["TP"]) + float(given["FP"]))
                answer_correct = abs(float(str(item.get("final_answer", "")).replace("%", "")) - expected) < 1e-6
            except (TypeError, ValueError, ZeroDivisionError):
                answer_correct = False
        elif selected.get("expression"):
            try:
                from qg_v2.agents.writer import WriterAgent
                _, rhs = WriterAgent._split_formula(str(selected["expression"]))
                expected = WriterAgent._safe_numeric_eval(rhs, {key: float(value) for key, value in given.items()})
                answer_correct = abs(float(str(item.get("final_answer", "")).replace("%", "")) - expected) < 1e-6
            except (TypeError, ValueError, ZeroDivisionError, SyntaxError):
                answer_correct = False
        checks = {
            "formula_supported": formula_supported,
            "variables_complete": variables_complete,
            "calculation_well_defined": calculation_well_defined,
            "question_meaningful": isinstance(item.get("stem"), str) and len(item.get("stem", "")) > 20,
            "answer_correct": answer_correct,
            "solution_complete": bool(steps) and isinstance(item.get("explanation"), str),
            "graph_faithful": bool(blueprint.get("target_concept")) and bool(formulas),
            "self_contained": variables_complete and bool(item.get("ask")),
            "clear_language": bool(item.get("stem")) and not item.get("options"),
            "formula_provenance_recorded": selected.get("provenance") in {"dataset", "llm", "hybrid"},
            "units_consistent": isinstance(item.get("unit"), str) and bool(item.get("unit")),
        }
        passed = all(checks.values())
        return {
            "checks": checks,
            "solver_passed": True,
            "passed": passed,
            "to": None if passed else "writer",
            "reason": "" if passed else "calculation validation failed",
        }

    def _validate_calculation_with_llm(self, blueprint: dict[str, Any]) -> dict[str, Any]:
        item = blueprint.get("writer_output", {})
        selected = blueprint.get("selected_formula") or {}
        required = [
            "formula_supported",
            "variables_complete",
            "calculation_well_defined",
            "question_meaningful",
            "answer_correct",
            "solution_complete",
            "source_grounded",
            "self_contained",
            "clear_language",
            "formula_provenance_recorded",
            "units_consistent",
        ]
        prompt = {
            "role": "user",
            "content": (
                "Independently solve and validate this open calculation question. Check the selected verified formula, every input value, units, substitutions, final answer, "
                "and provenance. Source context is optional evidence. Return strict JSON with all required boolean checks.\n"
                + json.dumps({
                    "instruction_version": blueprint.get("instruction_version"),
                    "item": item,
                    "target_concept": blueprint.get("target_concept"),
                    "selected_formula": selected,
                    "formula_candidates": blueprint.get("formula_candidates", []),
                    "source_context": blueprint.get("source_context", []),
                    "required_checks": required,
                    "output_schema": {"checks": {key: True for key in required}, "computed_answer": "...", "passed": True, "reason": ""},
                }, ensure_ascii=False)
            ),
        }
        result = self.llm.complete_json("solver", [prompt], temperature=0.0)
        raw = result.get("checks") if isinstance(result.get("checks"), dict) else {}
        checks = {key: self._coerce_check_bool(raw.get(key, False)) for key in required}
        passed = all(checks.values())
        return {
            "checks": checks,
            "solver_passed": checks["answer_correct"],
            "solver_result": {"computed_answer": result.get("computed_answer")},
            "passed": passed,
            "to": None if passed else "writer",
            "reason": "" if passed else str(result.get("reason") or "calculation validation failed"),
            "raw_result": result,
        }

    @staticmethod
    def _compact_formula(value: str) -> str:
        return "".join(value.lower().split())

    def _validate_with_llm(self, blueprint: dict[str, Any]) -> dict[str, Any]:
        item = blueprint["writer_output"]
        concepts = list(item.get("options", {}).values()) + list(blueprint.get("activated_subgraph", {}).get("concepts", []))
        definitions = self.graph.concept_definitions(concepts)
        solver = self._solver_with_llm(item)
        prompt = {
            "role": "user",
            "content": (
                "You are a quality validator. Structurally validate this multiple-choice item without rewriting it. Return strict JSON.\n"
                "Include the nine boolean checks, solver_passed, passed, to, and reason.\n"
                + json.dumps(
                    {
                        "instruction_version": blueprint.get("instruction_version"),
                        "item": item,
                        "blueprint": {
                            "target_concept": blueprint.get("target_concept"),
                            "question_type": blueprint.get("question_type"),
                            "selected_bloom": blueprint.get("selected_bloom"),
                            "activated_subgraph": blueprint.get("activated_subgraph"),
                            "correct_answer": blueprint.get("answer_plan", {}).get("correct_answer"),
                            "distractors": blueprint.get("distractor_plan", {}).get("distractors"),
                        },
                        "concept_definitions": definitions,
                        "source_context": blueprint.get("source_context", []),
                        "solver_result": solver,
                        "routing": {
                            "unique_answer": "answer_distractor",
                            "distractors_valid/options_exclusive": "answer_distractor",
                            "answerable/clear_language/explanation_supported": "writer",
                            "on_topic": "planner",
                            "bloom_aligned": "writer",
                            "graph_faithful": "answer_distractor",
                        },
                    },
                    ensure_ascii=False,
                )
            ),
        }
        result = self.llm.complete_json("validator", [prompt], temperature=0.1)
        return self._normalize_llm_result(result, item)

    def _solver_with_llm(self, item: dict[str, Any]) -> dict[str, Any]:
        prompt = {
            "role": "user",
            "content": (
                "Act as a careful student. Answer using only the stem and options below; do not explain your reasoning. "
                "If multiple options could be correct or information is insufficient, set chosen to uncertain. Return JSON.\n"
                + json.dumps({"stem": item.get("stem"), "options": item.get("options")}, ensure_ascii=False)
            ),
        }
        return self.llm.complete_json("solver", [prompt], temperature=0.0)

    def _validate_deterministic(self, blueprint: dict[str, Any]) -> dict[str, Any]:
        item = blueprint.get("writer_output", {})
        options = item.get("options", {}) if isinstance(item, dict) else {}
        correct = blueprint.get("answer_plan", {}).get("correct_answer", {}).get("concept")
        answer_label = item.get("answer") if isinstance(item, dict) else None
        answer_text = options.get(answer_label) if isinstance(options, dict) else None

        schema_ok = isinstance(item.get("stem"), str) and set(options.keys()) == {"A", "B", "C", "D"} and answer_label in options
        unique_options = len(set(options.values())) == 4 if isinstance(options, dict) else False
        correct_matches = answer_text == correct or (isinstance(answer_text, str) and isinstance(correct, str) and correct in answer_text)
        explanation = item.get("explanation", "") if isinstance(item, dict) else ""

        checks = {
            "unique_answer": bool(schema_ok and unique_options and correct_matches),
            "answerable": bool(schema_ok and item.get("stem")),
            "on_topic": bool(blueprint.get("target_concept") and str(blueprint.get("target_concept")) in item.get("stem", "") or blueprint.get("question_type") == "prerequisite_dependency"),
            "clear_language": bool(schema_ok and item.get("stem", "").rstrip().endswith(("？", "?"))),
            "options_exclusive": bool(unique_options),
            "distractors_valid": bool(unique_options and len(blueprint.get("distractor_plan", {}).get("distractors", [])) == 3),
            "bloom_aligned": bool(blueprint.get("selected_bloom")),
            "graph_faithful": bool(correct_matches and blueprint.get("activated_subgraph") is not None),
            "explanation_supported": bool(isinstance(explanation, str) and str(correct or "") in explanation),
        }
        solver = {"chosen": answer_label if correct_matches else "uncertain", "confidence": 0.9 if correct_matches else 0.2}
        solver_passed = solver["chosen"] == answer_label and solver["confidence"] >= 0.5
        passed = all(checks.values()) and solver_passed
        to, reason = (None, "") if passed else self._route_failure(checks, solver_passed)
        return {
            "checks": checks,
            "solver_passed": solver_passed,
            "solver_result": solver,
            "passed": passed,
            "to": to,
            "reason": reason,
        }

    @staticmethod
    def _route_failure(checks: dict[str, bool], solver_passed: bool) -> tuple[str, str]:
        if not checks.get("on_topic", True):
            return "planner", "on_topic failed"
        if not checks.get("unique_answer", True):
            return "answer_distractor", "unique_answer failed"
        if not checks.get("options_exclusive", True) or not checks.get("distractors_valid", True):
            return "answer_distractor", "distractors/options failed"
        if not checks.get("answerable", True) or not checks.get("clear_language", True) or not checks.get("explanation_supported", True):
            return "writer", "language or explanation failed"
        if not checks.get("bloom_aligned", True):
            return "writer", "bloom_aligned failed"
        if not checks.get("graph_faithful", True):
            return "answer_distractor", "graph_faithful failed"
        if not solver_passed:
            return "writer", "solver failed"
        return "writer", "validation failed"

    def _normalize_llm_result(self, result: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
        """Normalize real LLM validator output to the internal strict schema.

        Qwen sometimes returns checks as {"passed": true, "reason": "..."} objects,
        or uses alternative field names. The orchestrator only needs booleans plus a
        route, so keep detailed raw output while enforcing stable public fields.
        """
        raw_checks = result.get("checks")
        if not isinstance(raw_checks, dict):
            raise LLMError(f"validator returned invalid checks schema: {json.dumps(result, ensure_ascii=False)[:1000]}")

        checks: dict[str, bool] = {}
        check_reasons: dict[str, str] = {}
        for key in CHECK_KEYS:
            value = self._get_check_value(raw_checks, key)
            if value is None:
                inferred = self._coerce_optional_bool(result.get("passed"))
                if inferred is None:
                    raise LLMError(f"validator missing check '{key}': {json.dumps(result, ensure_ascii=False)[:1000]}")
                value = inferred
            checks[key] = self._coerce_check_bool(value)
            reason = self._extract_check_reason(value)
            if reason:
                check_reasons[key] = reason

        answer_label = item.get("answer")
        solver_result = result.get("solver_result") if isinstance(result.get("solver_result"), dict) else {}
        solver_passed = self._coerce_optional_bool(result.get("solver_passed"))
        if solver_passed is None:
            chosen = solver_result.get("chosen")
            confidence = self._coerce_float(solver_result.get("confidence"), default=0.0)
            solver_passed = chosen == answer_label and confidence >= 0.5

        passed = all(checks.values()) and bool(solver_passed)
        to = result.get("to") if isinstance(result.get("to"), str) and result.get("to") else None
        reason = result.get("reason") if isinstance(result.get("reason"), str) else ""
        if not passed and not to:
            to, route_reason = self._route_failure(checks, bool(solver_passed))
            reason = reason or route_reason
        if passed:
            to = None
            reason = reason if isinstance(reason, str) else ""

        normalized = {
            "checks": checks,
            "check_reasons": check_reasons,
            "solver_passed": bool(solver_passed),
            "solver_result": solver_result,
            "passed": passed,
            "to": to,
            "reason": reason,
            "raw_validator_output": result,
        }
        return normalized

    @staticmethod
    def _get_check_value(raw_checks: dict[str, Any], key: str) -> Any:
        if key in raw_checks:
            return raw_checks[key]
        for alias in CHECK_ALIASES.get(key, []):
            if alias in raw_checks:
                return raw_checks[alias]
        return None

    @staticmethod
    def _coerce_check_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, dict):
            for field in ("passed", "pass", "ok", "result", "value"):
                if field in value:
                    return ValidatorAgent._coerce_check_bool(value[field])
            status = value.get("status")
            if isinstance(status, str):
                return status.strip().lower() in {"pass", "passed", "ok", "true", "yes"}
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "pass", "passed", "ok", "1"}
        if isinstance(value, (int, float)):
            return bool(value)
        return False

    @staticmethod
    def _extract_check_reason(value: Any) -> str:
        if isinstance(value, dict):
            for field in ("reason", "rationale", "explanation", "comment"):
                reason = value.get(field)
                if isinstance(reason, str):
                    return reason
        return ""

    @staticmethod
    def _coerce_optional_bool(value: Any) -> bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "pass", "passed", "ok", "1"}:
                return True
            if lowered in {"false", "no", "fail", "failed", "0"}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        return None

    @staticmethod
    def _coerce_float(value: Any, *, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default




