from __future__ import annotations

import json
import re
from typing import Any

from qg_v2.graph import KnowledgeGraph, normalize_name
from qg_v2.graph.store import EDGE_CONFUSABLE, EDGE_PART_OF, EDGE_PREREQUISITE
from qg_v2.llm import LLMRouter


class CourseScopeDistractorAgent:
    """Prefer course-grounded distractors without forcing weak replacements.

    Candidate retrieval happens once per item.  The model (when available)
    may choose only from that grounded pool; a deterministic suitability gate
    is applied before any replacement is accepted.
    """

    def __init__(self, graph: KnowledgeGraph, llm: LLMRouter) -> None:
        self.graph = graph
        self.llm = llm

    def review(self, blueprint: dict[str, Any]) -> dict[str, Any]:
        if not self.llm.settings.enable_course_scope_distractor_review:
            return {
                "enabled": False,
                "checked_once": False,
                "replacements": [],
                "unresolved_out_of_scope_distractors": [],
            }

        entries = self._distractor_entries(blueprint)
        out_of_scope = [entry for entry in entries if not entry["evidence"]]
        if not out_of_scope:
            return {
                "enabled": True,
                "checked_once": True,
                "candidate_count": 0,
                "replacements": [],
                "unresolved_out_of_scope_distractors": [],
            }

        candidates = self._candidate_pool(blueprint, entries)
        source_excerpts = self._source_excerpts(blueprint, candidates)
        decisions = self._llm_decisions(blueprint, out_of_scope, candidates, source_excerpts)
        if decisions is None:
            decisions = self._deterministic_decisions(blueprint, out_of_scope, candidates)

        replacements: list[dict[str, Any]] = []
        unresolved: list[dict[str, Any]] = []
        candidate_by_name = {normalize_name(item["concept"]): item for item in candidates}
        used = {normalize_name(entry["concept"]) for entry in entries}
        for entry in out_of_scope:
            decision = decisions.get(entry["distractor_index"])
            candidate_name = str((decision or {}).get("candidate") or "").strip()
            candidate = candidate_by_name.get(normalize_name(candidate_name))
            if (
                candidate is None
                and candidate_name
                and self._mentioned_in_excerpts(candidate_name, source_excerpts)
            ):
                evidence = self.graph.course_evidence(candidate_name)
                if evidence:
                    candidate = {
                        "concept": candidate_name,
                        "relation": "course-source-mention",
                        "score": 65,
                        "evidence": evidence,
                        "definition": "",
                    }
            if candidate and normalize_name(candidate["concept"]) not in used and self._suitable(
                blueprint, candidate, entry
            ):
                replacements.append(
                    {
                        "label": entry["label"],
                        "distractor_index": entry["distractor_index"],
                        "old_concept": entry["concept"],
                        "new_concept": candidate["concept"],
                        "reason": str((decision or {}).get("reason") or "The in-course candidate passed the fit check."),
                        "evidence": candidate["evidence"][:5],
                    }
                )
                used.add(normalize_name(candidate["concept"]))
            else:
                unresolved.append(
                    {
                        "label": entry["label"],
                        "distractor_index": entry["distractor_index"],
                        "concept": entry["concept"],
                        "reason": str((decision or {}).get("reason") or "No suitable in-course replacement was found."),
                    }
                )

        return {
            "enabled": True,
            "checked_once": True,
            "candidate_count": len(candidates),
            "out_of_scope_count": len(out_of_scope),
            "replacements": replacements,
            "unresolved_out_of_scope_distractors": unresolved,
        }

    def _distractor_entries(self, blueprint: dict[str, Any]) -> list[dict[str, Any]]:
        item = blueprint.get("writer_output") if isinstance(blueprint.get("writer_output"), dict) else {}
        options = item.get("options") if isinstance(item.get("options"), dict) else {}
        distractors = blueprint.get("distractor_plan", {}).get("distractors", [])
        entries: list[dict[str, Any]] = []
        for index, distractor in enumerate(distractors):
            if not isinstance(distractor, dict):
                continue
            concept = str(distractor.get("concept") or "").strip()
            label = self._label_for_concept(options, concept, index)
            if not label or label == item.get("answer"):
                continue
            entries.append(
                {
                    "label": label,
                    "distractor_index": index,
                    "concept": concept,
                    "option_text": str(options.get(label) or concept),
                    "evidence": self.graph.course_evidence(concept),
                }
            )
        return entries

    @staticmethod
    def _label_for_concept(options: dict[str, Any], concept: str, index: int) -> str:
        normalized = normalize_name(concept)
        for label, value in options.items():
            if normalize_name(str(value)) == normalized:
                return str(label)
        return {0: "B", 1: "C", 2: "D"}.get(index, "")

    def _candidate_pool(self, blueprint: dict[str, Any], entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        target = str(blueprint.get("target_concept") or "")
        correct = str(blueprint.get("answer_plan", {}).get("correct_answer", {}).get("concept") or target)
        excluded = {normalize_name(target), normalize_name(correct)} | {
            normalize_name(entry["concept"]) for entry in entries
        }
        relation_scores: dict[str, tuple[int, str]] = {}

        def add(names: list[str], score: int, relation: str) -> None:
            for name in names:
                key = normalize_name(name)
                if key and key not in excluded:
                    previous = relation_scores.get(key)
                    if previous is None or score > previous[0]:
                        relation_scores[key] = (score, relation)

        for concept in (target, correct):
            add([edge.target for edge in self.graph.confusables_of(concept)], 100, EDGE_CONFUSABLE)
            add([edge.source for edge in self.graph.confusables_of(concept)], 100, EDGE_CONFUSABLE)
            add([edge.source for edge in self.graph.sibling_parts(concept)], 85, "sibling-part-of")
            add([edge.source for edge in self.graph.prereqs_of(concept)], 75, EDGE_PREREQUISITE)
            add([edge.target for edge in self.graph.dependents_of(concept)], 75, EDGE_PREREQUISITE)
            add([edge.source for edge in self.graph.parts_of(concept)], 75, EDGE_PART_OF)
            add([edge.target for edge in self.graph.wholes_of(concept)], 75, EDGE_PART_OF)
            add([edge.target for edge in self.graph.similar_of(concept)], 65, "similarity")

        target_text = " ".join(
            [target, correct, self.graph.concept_profile(target), self.graph.concept_profile(correct)]
            + [str(item.get("evidence") or "") for item in blueprint.get("source_context", [])]
        )
        target_tokens = set(re.findall(r"[a-z0-9]+", normalize_name(target_text)))
        for name in self.graph.all_concept_names():
            key = normalize_name(name)
            if key in excluded or key in relation_scores:
                continue
            overlap = target_tokens & set(re.findall(r"[a-z0-9]+", normalize_name(name)))
            if overlap:
                relation_scores[key] = (20 + len(overlap), "course-text-overlap")

        ordered = sorted(relation_scores.items(), key=lambda item: (-item[1][0], item[0]))[:40]
        result: list[dict[str, Any]] = []
        for key, (score, relation) in ordered:
            concept = next((name for name in self.graph.all_concept_names() if normalize_name(name) == key), key)
            evidence = self.graph.course_evidence(concept)
            if evidence:
                result.append(
                    {
                        "concept": concept,
                        "relation": relation,
                        "score": score,
                        "evidence": evidence,
                        "definition": self.graph.concept_profile(concept)[:500],
                    }
                )
        return result

    def _source_excerpts(
        self, blueprint: dict[str, Any], candidates: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        target = str(blueprint.get("target_concept") or "")
        correct = str(blueprint.get("answer_plan", {}).get("correct_answer", {}).get("concept") or target)
        activated = blueprint.get("activated_subgraph", {}).get("concepts", [])
        ordered = list(
            dict.fromkeys(
                [target, correct]
                + [str(item) for item in activated]
                + [str(item["concept"]) for item in candidates[:15]]
            )
        )
        excerpts: list[dict[str, str]] = []
        for concept in ordered:
            profile = self.graph.concept_profile(concept)
            if profile:
                excerpts.append({"concept": concept, "source": "profile", "text": profile[:800]})
            for context in self.graph.concept_contexts(concept)[:3]:
                text = " ".join([context.evidence] + list(context.fields.values())).strip()
                if text:
                    excerpts.append({"concept": concept, "source": context.context_id, "text": text[:800]})
            if len(excerpts) >= 30:
                break
        return excerpts[:30]

    @staticmethod
    def _mentioned_in_excerpts(term: str, excerpts: list[dict[str, str]]) -> bool:
        normalized = normalize_name(term)
        if not normalized:
            return False
        phrase = re.compile(rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])", re.IGNORECASE)
        return any(phrase.search(normalize_name(item.get("text", ""))) for item in excerpts)

    def _llm_decisions(
        self,
        blueprint: dict[str, Any],
        out_of_scope: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
        source_excerpts: list[dict[str, str]],
    ) -> dict[int, dict[str, str]] | None:
        if self.llm.is_mock or (not candidates and not source_excerpts):
            return None
        prompt = {
            "role": "user",
            "content": (
                "You review course-scope distractor replacements once. Keep a distractor that already has in-course evidence. "
                "Replace only an out-of-course item when the candidate has explicit in-course evidence, the same answer type as the correct answer, and contextual plausibility. "
                "Choose from the candidate pool or terms appearing verbatim in source_excerpts; do not invent concepts. Keep the original when no suitable candidate exists. Return strict JSON.\n"
                + json.dumps(
                    {
                        "item": blueprint.get("writer_output", {}),
                        "target_concept": blueprint.get("target_concept"),
                        "out_of_scope_distractors": out_of_scope,
                        "candidate_pool": candidates,
                        "source_excerpts": source_excerpts,
                        "output_schema": {
                            "decisions": [
                                {
                                    "distractor_index": 0,
                                    "candidate": "exact candidate concept or empty string",
                                    "reason": "replacement or keep reason",
                                }
                            ]
                        },
                    },
                    ensure_ascii=False,
                )
            ),
        }
        try:
            result = self.llm.complete_json("course_scope_distractor", [prompt], temperature=0.1)
        except Exception:
            return None
        raw = result.get("decisions") if isinstance(result, dict) else None
        if not isinstance(raw, list):
            return None
        return {
            int(item["distractor_index"]): item
            for item in raw
            if isinstance(item, dict) and str(item.get("distractor_index", "")).isdigit()
        }

    def _deterministic_decisions(
        self,
        blueprint: dict[str, Any],
        out_of_scope: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
    ) -> dict[int, dict[str, str]]:
        decisions: dict[int, dict[str, str]] = {}
        for entry in out_of_scope:
            selected = next(
                (candidate for candidate in candidates if self._suitable(blueprint, candidate, entry)),
                None,
            )
            decisions[entry["distractor_index"]] = {
                "candidate": selected["concept"] if selected else "",
                "reason": "The in-course candidate passed the deterministic fit check." if selected else "No suitable in-course replacement was found.",
            }
        return decisions

    def _suitable(self, blueprint: dict[str, Any], candidate: dict[str, Any], entry: dict[str, Any]) -> bool:
        concept = str(candidate.get("concept") or "")
        if not concept or not candidate.get("evidence"):
            return False
        stem = str(blueprint.get("writer_output", {}).get("stem") or "").lower()
        expected = self._expected_type(stem)
        actual = self._concept_type(concept.lower())
        if expected and actual and expected != actual:
            return False
        if expected == "method" and actual in {"error", "dataset", "matrix", "metric"}:
            return False
        if expected == "metric" and actual in {"error", "dataset", "matrix", "method"}:
            return False
        return int(candidate.get("score") or 0) >= 65

    @staticmethod
    def _expected_type(stem: str) -> str | None:
        if re.search(r"method|technique|approach|algorithm", stem):
            return "method"
        if re.search(r"metric|measure|score|ratio|rate", stem):
            return "metric"
        if re.search(r"dataset|training set|validation set|test set", stem):
            return "dataset"
        if re.search(r"error|loss|cost", stem):
            return "error"
        return None

    @staticmethod
    def _concept_type(concept: str) -> str | None:
        if re.search(r"error|loss|cost", concept):
            return "error"
        if re.search(r"dataset|set|fold", concept):
            return "dataset"
        if re.search(r"matrix", concept):
            return "matrix"
        if re.search(r"accuracy|precision|recall|f1|score|auc|rate|ratio|metric", concept):
            return "metric"
        if re.search(r"algorithm|method|approach|technique|regression|clustering|model", concept):
            return "method"
        return None
