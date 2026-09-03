from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
import copy
import uuid
import random
import re

from qg_v2.agents import AnswerDistractorAgent, CalculationPlanningAgent, CognitiveDifficultyAgent, CourseScopeDistractorAgent, DistractorReviewAgent, PlannerAgent, ValidatorAgent, WriterAgent
from qg_v2.graph import KnowledgeGraph
from qg_v2.llm import LLMRouter


@dataclass
class Orchestrator:
    graph: KnowledgeGraph
    llm: LLMRouter
    max_attempts: int = 3
    progress_callback: Callable[[str], None] | None = None

    def __post_init__(self) -> None:
        self.planner = PlannerAgent(self.graph)
        self.cognitive = CognitiveDifficultyAgent()
        self.answer_distractor = AnswerDistractorAgent(self.graph, self.llm)
        self.calculation_planner = CalculationPlanningAgent(self.graph, self.llm)
        self.writer = WriterAgent(self.graph, self.llm)
        self.distractor_reviewer = DistractorReviewAgent(self.graph, self.llm)
        self.course_scope_reviewer = CourseScopeDistractorAgent(self.graph, self.llm)
        self.validator = ValidatorAgent(self.graph, self.llm)

    def _progress(self, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(message)

    def generate(self, user_input: str) -> dict[str, Any]:
        """Generate one item per feasible type under the selected instruction version."""
        instruction_version = self.llm.settings.instruction_version
        self._progress(f"Starting question-set generation: {user_input}")
        plan_result = self.planner.plan_all(user_input)
        if plan_result.get("status") != "ok":
            return {"status": "failed", "instruction_version": instruction_version, "reason": plan_result.get("reason", "planner_failed"), "planner_output": plan_result}

        target = str(plan_result.get("target_concept", user_input))
        related_concepts: list[str] = []
        for plan in plan_result.get("plans", []):
            related_concepts.extend(plan.get("activated_subgraph", {}).get("concepts", []))
        related_concepts = list(dict.fromkeys([target] + related_concepts))
        source_context = self.graph.select_source_context(target, related_concepts) if instruction_version == "multisource-v2" else []
        calculation_plan = self.calculation_planner.plan(target, related_concepts, source_context)
        if calculation_plan:
            plan_result.setdefault("plans", []).append(calculation_plan)
            plan_result["available_question_types"] = [str(p.get("question_type")) for p in plan_result["plans"]]

        plans = list(plan_result.get("plans", []))
        for plan in plans:
            plan["instruction_version"] = instruction_version
            plan["source_context"] = copy.deepcopy(source_context)
        available = [str(plan.get("question_type")) for plan in plans]
        for plan in plans:
            plan["available_question_types"] = available

        target = plan_result.get("target_concept", user_input)
        self._progress(f"Planner: found {len(plans)} feasible question types: {', '.join(available)}")

        items: list[dict[str, Any]] = []
        failed_items: list[dict[str, Any]] = []
        total = len(plans)
        for index, plan in enumerate(plans, start=1):
            question_type = plan.get("question_type")
            self._progress(f"Generating item {index}/{total}: {question_type}")
            result = self._generate_from_plan(
                user_input=user_input,
                plan=plan,
                question_type_index=index,
                total_question_types=total,
                available_question_types=available,
            )
            if result.get("status") == "ok":
                items.append({k: v for k, v in result.items() if k != "status"})
                self._progress(f"Item {index}/{total} complete: {question_type}")
            else:
                failed_items.append(
                    {
                        "question_type": question_type,
                        "reason": result.get("reason", "unknown_failure"),
                        "result": result,
                    }
                )
                self._progress(f"Item {index}/{total} failed: {question_type}; reason: {result.get('reason', 'unknown_failure')}")

        status = "ok" if items else "failed"
        summary = {
            "instruction_version": instruction_version,
            "target_concept": target,
            "generated_count": len(items),
            "failed_count": len(failed_items),
            "question_types": [item["metadata"].get("question_type") for item in items],
            "available_question_types": available,
        }
        if status == "ok":
            self._progress(f"Complete: generated {len(items)}/{total} items")
        else:
            self._progress("Failed: no question type was generated successfully")
        return {"status": status, "instruction_version": instruction_version, "items": items, "failed_items": failed_items, "summary": summary}

    def _generate_from_plan(
        self,
        *,
        user_input: str,
        plan: dict[str, Any],
        question_type_index: int,
        total_question_types: int,
        available_question_types: list[str],
    ) -> dict[str, Any]:
        last_blueprint: dict[str, Any] = {"user_input": user_input}
        attempts_total = self.max_attempts + 1
        question_type = str(plan.get("question_type"))
        for attempt in range(1, attempts_total + 1):
            self._progress(f"Item {question_type_index}/{total_question_types} ({question_type}), attempt {attempt}/{attempts_total}: initializing blueprint")
            blueprint: dict[str, Any] = {
                "user_input": user_input,
                "instruction_version": self.llm.settings.instruction_version,
                "attempt": attempt,
                "question_type_index": question_type_index,
                "total_question_types": total_question_types,
                "available_question_types": available_question_types,
            }
            blueprint.update({k: copy.deepcopy(v) for k, v in plan.items() if k != "status"})
            blueprint["question_type_index"] = question_type_index
            blueprint["total_question_types"] = total_question_types
            blueprint["available_question_types"] = available_question_types

            target = str(blueprint["target_concept"])
            self._progress(f"Item {question_type_index}/{total_question_types}: Cognitive/Difficulty selected Bloom level and target difficulty")
            features = self.graph.structural_features(target)
            cognition = self.cognitive.decide(blueprint, features)
            if cognition.get("status") != "ok":
                last_blueprint = blueprint | {"cognitive_output": cognition}
                continue
            blueprint.update({k: v for k, v in cognition.items() if k != "status"})
            blueprint["structural_features"] = features

            self._progress(f"Item {question_type_index}/{total_question_types}: Answer/Distractor planned the answer and distractors")
            answer_plan = self.answer_distractor.plan(blueprint)
            if answer_plan.get("status") != "ok":
                last_blueprint = blueprint | {"answer_plan_error": answer_plan}
                continue
            blueprint["answer_plan"] = {
                "correct_answer": answer_plan["correct_answer"],
                "distractors": answer_plan["distractors"],
            }
            blueprint["distractor_plan"] = {"distractors": answer_plan["distractors"]}

            self._progress(f"Item {question_type_index}/{total_question_types}: Writer generated the stem, options, and explanation")
            try:
                blueprint["writer_output"] = self.writer.write(blueprint)
            except Exception as exc:
                raise
            if question_type != "calculation":
                self._review_and_revise_distractors(blueprint)

            # Validate the exact item that will be published. Shuffling after
            # validation used to leave explanation/validation labels stale.
            self._shuffle_choice_options(blueprint["writer_output"])

            self._progress(f"Item {question_type_index}/{total_question_types}: Validator performed final quality checks")
            validation = self.validator.validate(blueprint)
            blueprint["validation_record"] = validation
            last_blueprint = blueprint
            if validation.get("passed"):
                return self._success_result(user_input, blueprint)

        return {
            "status": "failed",
            "reason": "max_attempts_exhausted",
            "item": last_blueprint.get("writer_output"),
            "metadata": self._metadata(last_blueprint),
            "blueprint": last_blueprint,
            "validation_record": last_blueprint.get("validation_record"),
        }

    def _success_result(self, user_input: str, blueprint: dict[str, Any]) -> dict[str, Any]:
        item = copy.deepcopy(blueprint["writer_output"])
        return {
            "status": "ok",
            "item": item,
            "metadata": self._metadata(blueprint),
            "blueprint": blueprint,
            "validation_record": blueprint["validation_record"],
        }

    @classmethod
    def _shuffle_choice_options(cls, item: dict[str, Any]) -> None:
        """Randomize options and keep every published label reference consistent."""
        options = item.get("options")
        if not isinstance(options, dict) or set(options) != {"A", "B", "C", "D"}:
            return
        answer = item.get("answer")
        labels = ["A", "B", "C", "D"]
        values = [options[label] for label in labels]
        random.SystemRandom().shuffle(values)
        item["options"] = dict(zip(labels, values))
        old_to_new = {old_label: labels[values.index(options[old_label])] for old_label in labels}
        item["answer"] = old_to_new.get(answer, answer)
        explanation = item.get("explanation")
        if isinstance(explanation, str):
            item["explanation"] = cls._remap_explanation_labels(explanation, old_to_new)

    @staticmethod
    def _remap_explanation_labels(explanation: str, old_to_new: dict[str, str]) -> str:
        """Remap explicit option references in one simultaneous pass.

        A callback is essential here: sequential replacements (A -> C, C -> B)
        can corrupt labels that have already been replaced.
        """
        reference = re.compile(
            r"(?P<prefix>\b(?:option|choice)\s*|"
            r"\b(?:correct\s+answer|answer)\s*(?:is|:)\s*|"
            r"correct\s+answer\s*(?:is|:)?\s*)"
            r"(?P<label>[A-D])(?![A-Za-z])",
            flags=re.IGNORECASE,
        )

        def replace(match: re.Match[str]) -> str:
            label = match.group("label")
            remapped = old_to_new.get(label.upper(), label.upper())
            return f"{match.group('prefix')}{remapped}"

        return reference.sub(replace, explanation)

    def _metadata(self, blueprint: dict[str, Any]) -> dict[str, Any]:
        subgraph = blueprint.get("activated_subgraph", {}) or {}
        answer_plan = blueprint.get("answer_plan", {}) or {}
        validation = blueprint.get("validation_record", {}) or {}
        return {
            "item_id": str(uuid.uuid4()),
            "instruction_version": blueprint.get("instruction_version", "legacy-v1"),
            "user_input": blueprint.get("user_input"),
            "target_concept": blueprint.get("target_concept"),
            "question_type": blueprint.get("question_type"),
            "base_question_type": blueprint.get("base_question_type"),
            "available_question_types": blueprint.get("available_question_types", []),
            "question_type_index": blueprint.get("question_type_index"),
            "total_question_types": blueprint.get("total_question_types"),
            "activated_concepts": subgraph.get("concepts", []),
            "activated_edges": subgraph.get("edges", []),
            "bloom_control": {
                "bloom_candidates": blueprint.get("bloom_candidates", []),
                "selected_bloom": blueprint.get("selected_bloom"),
                "reason": blueprint.get("bloom_reason"),
            },
            "difficulty_control": {
                "target_difficulty": blueprint.get("target_difficulty"),
                "reason": blueprint.get("difficulty_reason"),
            },
            "answer_plan": answer_plan.get("correct_answer"),
            "distractor_plan": self._published_distractor_plan(blueprint, answer_plan),
            "validation_result": validation.get("checks", {}),
            "distractor_review_result": blueprint.get("distractor_review_record"),
            "course_scope_distractor_result": blueprint.get("course_scope_distractor_record"),
            "source_context_control": {
                "provided_context_ids": [item.get("context_id") for item in blueprint.get("source_context", [])],
                "used_context_ids": list((blueprint.get("writer_output") or {}).get("context_refs", [])),
            },
        }

    @staticmethod
    def _published_distractor_plan(
        blueprint: dict[str, Any], answer_plan: dict[str, Any]
    ) -> dict[str, Any]:
        """Bind distractor metadata to labels in the final shuffled item."""
        options = (blueprint.get("writer_output") or {}).get("options") or {}
        if not isinstance(options, dict):
            options = {}
        result: dict[str, Any] = {}
        for distractor in answer_plan.get("distractors", []):
            concept = str(distractor.get("concept", ""))
            label = next((key for key, value in options.items() if str(value) == concept), None)
            if label is not None:
                result[f"option_{label}"] = distractor
        return result

    @staticmethod
    def _review_and_revise_distractors(self, blueprint: dict[str, Any]) -> None:
        self._progress("Contextual distractor review: checking plausibility in the current stem")
        if not self.llm.settings.enable_distractor_review:
            blueprint["distractor_review_record"] = {"passed": True, "attempts": [], "revision_count": 0, "disabled": True}
        else:
            attempts: list[dict[str, Any]] = []
            revision_count = 0
            max_revisions = max(0, int(self.llm.settings.distractor_review_max_revisions))
            while True:
                review = self.distractor_reviewer.review(blueprint)
                attempts.append(review)
                if review.get("passed"):
                    self._progress("Contextual distractor review: passed")
                    blueprint["distractor_review_record"] = {
                        "passed": True,
                        "attempts": attempts,
                        "revision_count": revision_count,
                    }
                    break
                if revision_count >= max_revisions:
                    self._progress("Contextual distractor review: failed; maximum redesign attempts reached")
                    blueprint["distractor_review_record"] = {
                        "passed": False,
                        "attempts": attempts,
                        "revision_count": revision_count,
                        "reason": review.get("feedback_to_answer_distractor") or review.get("issue_type"),
                    }
                    break
                self._progress(f"Contextual distractor review: weak distractors found; redesign {revision_count + 1}/{max_revisions}")
                revised = self.answer_distractor.revise_with_feedback(blueprint, review)
                revision_count += 1
                if revised.get("status") != "ok":
                    blueprint["distractor_review_record"] = {
                        "passed": False,
                        "attempts": attempts,
                        "revision_count": revision_count,
                        "reason": revised.get("reason", "distractor_revision_failed"),
                    }
                    break
                blueprint["answer_plan"] = {
                    "correct_answer": revised["correct_answer"],
                    "distractors": revised["distractors"],
                }
                blueprint["distractor_plan"] = {"distractors": revised["distractors"]}
                self._progress("Distractor redesign complete: Writer regenerated the item text")
                blueprint["writer_output"] = self.writer.write(blueprint)

        self._apply_course_scope_review(blueprint)

    def _apply_course_scope_review(self, blueprint: dict[str, Any]) -> None:
        self._progress("Course-scope distractor review: starting one-pass check")
        record = self.course_scope_reviewer.review(blueprint)
        replacements = record.get("replacements", []) if isinstance(record, dict) else []
        if replacements:
            distractors = blueprint.get("distractor_plan", {}).get("distractors", [])
            for replacement in replacements:
                index = replacement.get("distractor_index")
                if not isinstance(index, int) or not (0 <= index < len(distractors)):
                    continue
                distractor = distractors[index]
                distractor["concept"] = replacement["new_concept"]
                distractor["source_relation"] = "course_scope_replacement"
                distractor["course_scope_evidence"] = replacement.get("evidence", [])
                distractor["course_scope_reason"] = replacement.get("reason", "")
            blueprint["answer_plan"]["distractors"] = distractors
            blueprint["distractor_plan"]["distractors"] = distractors
            self._progress(f"Course-scope distractor review: replacing {len(replacements)} options and regenerating item text")
            blueprint["writer_output"] = self.writer.write(blueprint)
            record["writer_rewritten"] = True
        else:
            self._progress("Course-scope distractor review: no options need replacement")
            record["writer_rewritten"] = False
        blueprint["course_scope_distractor_record"] = record
