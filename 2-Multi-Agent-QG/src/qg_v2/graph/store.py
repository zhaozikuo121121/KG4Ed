from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
import csv
import json
import re
from typing import Any, Iterable


EDGE_PREREQUISITE = "prerequisite"
EDGE_PART_OF = "part-of"
EDGE_CONFUSABLE = "similar/confusable"
EDGE_SIMILARITY = "similarity"


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


@dataclass(slots=True)
class Concept:
    id: int
    name: str
    aliases: set[str] = field(default_factory=set)
    profile: str = ""

    @property
    def display_name(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class Edge:
    source: str
    target: str
    type: str
    metadata: str | None = None
    weight: float | None = None

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"source": self.source, "target": self.target, "type": self.type}
        if self.metadata is not None:
            data["metadata"] = self.metadata
        if self.weight is not None:
            data["weight"] = self.weight
        return data


@dataclass(frozen=True, slots=True)
class SourceContext:
    context_id: str
    concept: str
    evidence: str
    fields: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "concept": self.concept,
            "evidence": self.evidence,
            "source": dict(self.fields),
        }

@dataclass(frozen=True, slots=True)
class FormulaRecord:
    formula_id: str
    concept: str
    expression: str
    fields: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {"formula_id": self.formula_id, "name": self.concept, "expression": self.expression,
                "source_concept": self.concept, "provenance": "dataset", "source_record_ids": [self.formula_id],
                "source": dict(self.fields)}


class KnowledgeGraph:
    """In-memory three-relation graph plus optional similarity graph."""

    def __init__(self) -> None:
        self.concepts: dict[str, Concept] = {}
        self.alias_to_name: dict[str, str] = {}
        self.prereq_forward: dict[str, list[Edge]] = defaultdict(list)  # X -> Y, X is prereq of Y
        self.prereq_reverse: dict[str, list[Edge]] = defaultdict(list)  # Y -> X
        self.part_forward: dict[str, list[Edge]] = defaultdict(list)  # X -> Y, X part of Y
        self.part_reverse: dict[str, list[Edge]] = defaultdict(list)  # Y -> X
        self.confusable: dict[str, list[Edge]] = defaultdict(list)
        self.similar: dict[str, list[Edge]] = defaultdict(list)
        self.contexts: dict[str, list[SourceContext]] = defaultdict(list)
        self.formulas: dict[str, list[FormulaRecord]] = defaultdict(list)
        self.data_root: Path | None = None

    @classmethod
    def load(cls, data_dir: str | Path) -> "KnowledgeGraph":
        graph = cls()
        root = Path(data_dir)
        graph.data_root = root.resolve()
        # MOOC uses the original text-file layout; MLR/DGL use prefixed CSV files.
        graph._load_concepts(root / "concepts.txt", graph._fallback(root, "_concepts.csv"))
        graph._load_profiles(root / "concept_profiles.jsonl", graph._fallback(root, "_profiles.jsonl"))
        graph._load_prerequisites(root / "prerequisite.txt", graph._fallback(root, "_prerequisite.csv"))
        graph._load_partof(root / "partof.txt", graph._fallback(root, "_partof.csv"))
        graph._load_confusion(root / "confusion.txt", graph._fallback(root, "_confusion.csv"))
        graph._load_similarity_edges(
            root / "similarity" / "edges_knn.csv",
            graph._fallback(root, "_edges_knn.csv"),
        )
        graph._load_contexts(graph._fallback(root, "_concepts_metadata.csv"), root.name)
        graph._load_formulas(graph._fallback(root, "_formula.csv"), root.name)
        return graph

    @staticmethod
    def _fallback(root: Path, suffix: str) -> Path | None:
        matches = sorted(root.glob(f"*{suffix}"))
        return matches[0] if matches else None

    def _ensure_concept(self, raw_name: str, aliases: Iterable[str] = ()) -> str:
        name = raw_name.strip()
        if not name:
            raise ValueError("Empty concept name")
        norm = normalize_name(name)
        canonical = self.alias_to_name.get(norm)
        if canonical:
            concept = self.concepts[canonical]
            concept.aliases.update(a.strip() for a in aliases if a.strip())
            for alias in concept.aliases:
                self.alias_to_name[normalize_name(alias)] = canonical
            return canonical
        canonical = name
        concept = Concept(id=len(self.concepts), name=canonical, aliases=set(a.strip() for a in aliases if a.strip()))
        self.concepts[canonical] = concept
        self.alias_to_name[norm] = canonical
        for alias in concept.aliases:
            self.alias_to_name[normalize_name(alias)] = canonical
        return canonical

    def _canonical(self, raw_name: str) -> str | None:
        return self.alias_to_name.get(normalize_name(raw_name))

    def _load_concepts(self, path: Path, csv_path: Path | None = None) -> None:
        if path.exists():
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                parts = [p.strip() for p in raw_line.split("::;") if p.strip()]
                if parts:
                    self._ensure_concept(parts[0], aliases=parts[1:])
            return
        if csv_path and csv_path.exists():
            with csv_path.open("r", encoding="utf-8", newline="") as fh:
                for row in csv.reader(fh):
                    values = [value.strip() for value in row if value.strip()]
                    if values and values[0].lower() not in {"concept", "concept_name"}:
                        self._ensure_concept(values[0], aliases=values[1:])

    def _load_profiles(self, path: Path, jsonl_path: Path | None = None) -> None:
        source = path if path.exists() else jsonl_path
        if source is None or not source.exists():
            return
        with source.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                item = json.loads(line)
                name = str(item.get("concept_name", "")).strip()
                if not name:
                    continue
                canonical = self._canonical(name) or self._ensure_concept(name)
                self.concepts[canonical].profile = str(item.get("profile", "")).strip()

    def _load_prerequisites(self, path: Path, csv_path: Path | None = None) -> None:
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
            rows = ([p.strip() for p in re.split(r"\t+", line) if p.strip()] for line in lines if line.strip())
        elif csv_path and csv_path.exists():
            with csv_path.open("r", encoding="utf-8", newline="") as fh:
                rows = [[p.strip() for p in row if p.strip()] for row in csv.reader(fh)]
        else:
            return
        for parts in rows:
            if len(parts) < 2 or parts[0].lower() in {"source", "concept_1", "prerequisite"}:
                continue
            source = self._canonical(parts[0]) or self._ensure_concept(parts[0])
            target = self._canonical(parts[1]) or self._ensure_concept(parts[1])
            metadata = parts[2] if len(parts) > 2 else None
            edge = Edge(source=source, target=target, type=EDGE_PREREQUISITE, metadata=metadata)
            self.prereq_forward[source].append(edge)
            self.prereq_reverse[target].append(edge)

    def _load_partof(self, path: Path, csv_path: Path | None = None) -> None:
        if path.exists():
            rows = ((line.split(" part-of ", 1) if " part-of " in line else [])
                    for line in path.read_text(encoding="utf-8").splitlines())
        elif csv_path and csv_path.exists():
            with csv_path.open("r", encoding="utf-8", newline="") as fh:
                rows = [[row[0], row[1]] if len(row) >= 2 else [] for row in csv.reader(fh)]
        else:
            return
        for row in rows:
            if len(row) < 2 or row[0].strip().lower() in {"part_concept", "source"}:
                continue
            source_raw, target_raw = row[0].strip(), row[1].strip()
            source = self._canonical(source_raw) or self._ensure_concept(source_raw)
            target = self._canonical(target_raw) or self._ensure_concept(target_raw)
            edge = Edge(source=source, target=target, type=EDGE_PART_OF)
            self.part_forward[source].append(edge)
            self.part_reverse[target].append(edge)

    def _load_confusion(self, path: Path, csv_path: Path | None = None) -> None:
        if path.exists():
            rows = ((line.split("::", 1) if "::" in line else [])
                    for line in path.read_text(encoding="utf-8").splitlines())
        elif csv_path and csv_path.exists():
            with csv_path.open("r", encoding="utf-8", newline="") as fh:
                rows = [[row[0], row[1]] if len(row) >= 2 else [] for row in csv.reader(fh)]
        else:
            return
        for row in rows:
            if len(row) < 2 or row[0].strip().lower() in {"concept_1", "source"}:
                continue
            left_raw, right_raw = row[0].strip(), row[1].strip()
            if not left_raw or not right_raw:
                continue
            left = self._canonical(left_raw) or self._ensure_concept(left_raw)
            right = self._canonical(right_raw) or self._ensure_concept(right_raw)
            edge_lr = Edge(source=left, target=right, type=EDGE_CONFUSABLE)
            edge_rl = Edge(source=right, target=left, type=EDGE_CONFUSABLE)
            self.confusable[left].append(edge_lr)
            self.confusable[right].append(edge_rl)

    def _load_similarity_edges(self, path: Path, csv_path: Path | None = None) -> None:
        source = path if path.exists() else csv_path
        if source is None or not source.exists():
            return
        with source.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                src_raw = (row.get("source") or "").strip()
                tgt_raw = (row.get("target") or "").strip()
                if not src_raw or not tgt_raw:
                    continue
                src = self._canonical(src_raw) or self._ensure_concept(src_raw)
                tgt = self._canonical(tgt_raw) or self._ensure_concept(tgt_raw)
                try:
                    weight = float(row.get("weight") or 0.0)
                except ValueError:
                    weight = None
                edge = Edge(source=src, target=tgt, type=EDGE_SIMILARITY, weight=weight)
                reverse = Edge(source=tgt, target=src, type=EDGE_SIMILARITY, weight=weight)
                self.similar[src].append(edge)
                self.similar[tgt].append(reverse)

    def _load_contexts(self, path: Path | None, dataset: str) -> None:
        if path is None or not path.exists():
            return
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            for index, row in enumerate(csv.DictReader(fh), start=1):
                raw_name = str(row.get("concept") or row.get("concept_name") or "").strip()
                evidence = str(row.get("evidence") or row.get("context") or "").strip()
                if not raw_name or not evidence:
                    continue
                canonical = self._canonical(raw_name)
                if canonical is None:
                    continue
                fields = {str(key): str(value or "").strip() for key, value in row.items() if key}
                self.contexts[canonical].append(
                    SourceContext(f"{dataset}-META-{index:04d}", canonical, evidence, fields)
                )

    def _load_formulas(self, path: Path | None, dataset: str) -> None:
        if path is None or not path.exists(): return
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            for index, row in enumerate(csv.DictReader(fh), start=1):
                raw = str(row.get("concept") or row.get("concept_name") or "").strip()
                expression = str(row.get("latex") or row.get("formula") or row.get("expression") or "").strip()
                canonical = self._canonical(raw) if raw else None
                if canonical and expression:
                    self.formulas[canonical].append(FormulaRecord(str(row.get("formula_id") or f"{dataset}-FORMULA-{index:04d}"), canonical, expression, dict(row)))

    def resolve(self, user_input: str) -> str | list[str] | None:
        text = user_input.strip()
        if not text:
            return None
        direct = self._canonical(text)
        if direct:
            return direct
        norm = normalize_name(text)
        contains = [name for name in self.concepts if norm in normalize_name(name)]
        if len(contains) == 1:
            return contains[0]
        if contains:
            return sorted(contains, key=lambda n: (len(n), n))[:10]
        tokens = set(norm.split())
        if tokens:
            scored: list[tuple[int, str]] = []
            for name in self.concepts:
                overlap = len(tokens & set(normalize_name(name).split()))
                if overlap:
                    scored.append((overlap, name))
            if scored:
                scored.sort(key=lambda x: (-x[0], len(x[1]), x[1]))
                candidates = [name for _, name in scored[:10]]
                return candidates[0] if len(candidates) == 1 else candidates
        return None

    def concept_profile(self, concept: str) -> str:
        canonical = self._canonical(concept) or concept
        return self.concepts.get(canonical, Concept(-1, canonical)).profile

    def concept_definitions(self, concepts: Iterable[str]) -> dict[str, dict[str, str]]:
        result: dict[str, dict[str, str]] = {}
        for concept in concepts:
            canonical = self._canonical(concept) or concept
            result[canonical] = {"zh": canonical, "definition": self.concept_profile(canonical)}
        return result

    def course_evidence(self, term: str) -> list[dict[str, Any]]:
        """Return course-local evidence that a term is present in this dataset.

        Exact concepts and aliases are strongest.  For generated/surface-form
        distractors, also check profiles and metadata evidence using a bounded
        phrase match.  The method only reads the already-loaded dataset.
        """
        text = str(term or "").strip()
        if not text:
            return []
        canonical = self._canonical(text)
        if canonical:
            evidence: list[dict[str, Any]] = [
                {"kind": "concept", "concept": canonical, "source": "concepts"}
            ]
            profile = self.concept_profile(canonical)
            if profile:
                evidence.append(
                    {
                        "kind": "profile",
                        "concept": canonical,
                        "source": "profile",
                        "evidence": profile[:500],
                    }
                )
            for context in self.concept_contexts(canonical):
                evidence.append(
                    {
                        "kind": "metadata",
                        "concept": canonical,
                        "source": context.context_id,
                        "evidence": context.evidence[:500],
                    }
                )
            return evidence

        normalized = normalize_name(text)
        if not normalized:
            return []
        escaped = re.escape(normalized)
        phrase = re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", re.IGNORECASE)
        matches: list[dict[str, Any]] = []
        for concept, record in self.concepts.items():
            profile = record.profile
            if profile and phrase.search(normalize_name(profile)):
                matches.append(
                    {
                        "kind": "profile_mention",
                        "concept": concept,
                        "source": "profile",
                        "evidence": profile[:500],
                    }
                )
            for context in self.concept_contexts(concept):
                metadata_text = " ".join([context.evidence] + list(context.fields.values()))
                if phrase.search(normalize_name(metadata_text)):
                    matches.append(
                        {
                            "kind": "metadata_mention",
                            "concept": concept,
                            "source": context.context_id,
                            "evidence": metadata_text[:500],
                        }
                    )
        return matches

    def concept_contexts(self, concept: str) -> list[SourceContext]:
        canonical = self._canonical(concept) or concept
        return list(self.contexts.get(canonical, []))

    def concept_formulas(self, concept: str) -> list[FormulaRecord]:
        return list(self.formulas.get(self._canonical(concept) or concept, []))

    def select_source_context(self, target: str, concepts: Iterable[str]) -> list[dict[str, Any]]:
        ordered = list(dict.fromkeys([target] + [str(item) for item in concepts]))
        selected: list[SourceContext] = []
        for concept in ordered:
            limit = 5 if normalize_name(concept) == normalize_name(target) else 2
            selected.extend(self.concept_contexts(concept)[:limit])
            if len(selected) >= 15:
                break
        return [item.to_dict() for item in selected[:15]]

    def prereqs_of(self, concept: str) -> list[Edge]:
        return list(self.prereq_reverse.get(concept, []))

    def dependents_of(self, concept: str) -> list[Edge]:
        return list(self.prereq_forward.get(concept, []))

    def parts_of(self, concept: str) -> list[Edge]:
        return list(self.part_reverse.get(concept, []))

    def wholes_of(self, concept: str) -> list[Edge]:
        return list(self.part_forward.get(concept, []))

    def confusables_of(self, concept: str) -> list[Edge]:
        return list(self.confusable.get(concept, []))

    def similar_of(self, concept: str) -> list[Edge]:
        return sorted(self.similar.get(concept, []), key=lambda e: e.weight or 0.0, reverse=True)

    def neighbors(self, concept: str) -> list[Edge]:
        return self.prereqs_of(concept) + self.dependents_of(concept) + self.parts_of(concept) + self.wholes_of(concept) + self.confusables_of(concept)

    def clean_chains(self, concept: str, max_hops: int = 2) -> list[list[str]]:
        """Return simple prerequisite chains ending at or starting from concept, max_hops edges."""
        chains: list[list[str]] = []
        for direction in ("backward", "forward"):
            queue: deque[list[str]] = deque([[concept]])
            while queue:
                path = queue.popleft()
                if len(path) - 1 >= max_hops:
                    if len(path) > 2:
                        chains.append(path)
                    continue
                current = path[-1]
                edges = self.prereqs_of(current) if direction == "backward" else self.dependents_of(current)
                for edge in edges:
                    nxt = edge.source if direction == "backward" else edge.target
                    if nxt in path:
                        continue
                    new_path = path + [nxt]
                    if len(new_path) > 2:
                        chains.append(new_path)
                    queue.append(new_path)
        return chains

    def sibling_parts(self, concept: str) -> list[Edge]:
        siblings: list[Edge] = []
        seen: set[str] = set()
        for whole_edge in self.wholes_of(concept):
            whole = whole_edge.target
            for part_edge in self.parts_of(whole):
                if part_edge.source != concept and part_edge.source not in seen:
                    siblings.append(Edge(source=part_edge.source, target=whole, type="sibling-part-of"))
                    seen.add(part_edge.source)
        return siblings

    def structural_features(self, concept: str) -> dict[str, object]:
        return {
            "num_direct_prereqs": len(self.prereqs_of(concept)),
            "has_clean_2hop_chain": bool(self.clean_chains(concept, max_hops=2)),
            "num_part_components": len(self.parts_of(concept)),
            "has_strong_confusable": bool(self.confusables_of(concept)),
        }

    def all_concept_names(self) -> list[str]:
        return list(self.concepts.keys())
