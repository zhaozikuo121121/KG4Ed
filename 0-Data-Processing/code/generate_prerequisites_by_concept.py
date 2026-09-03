#!/usr/bin/env python3
"""Generate dataset-local prerequisite edges with one Qwen request per anchor concept.

This is an independent alternative to ``generate_prerequisites.py``. It reuses that module's
dataset loading, Qwen configuration, and append-only CSV publication, but keeps its own prompts
and checkpoints. Each unordered pair is assigned exactly once
to its lower-ID concept; the model may return a prerequisite edge in either direction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import generate_prerequisites as pairwise
import generate_profiles as profiles


PROGRAM_VERSION = "1.1.0"
PROMPT_VERSION = "prerequisite_relations_by_concept_names_v1"
STRATEGY_VERSION = "lower_id_anchor_all_remaining_v1"
PREREQUISITE_MODEL = pairwise.PREREQUISITE_MODEL
SCHEMA_REPAIR_ATTEMPTS = 2


PrerequisiteError = pairwise.PrerequisiteError
PrerequisiteValidationError = pairwise.PrerequisiteValidationError
DatasetSpec = pairwise.DatasetSpec
ConceptPair = pairwise.ConceptPair


@dataclass(frozen=True)
class AnchorTask:
    anchor_id: int
    pairs: tuple[ConceptPair, ...]

    @property
    def candidate_ids(self) -> tuple[int, ...]:
        return tuple(pair.right_id for pair in self.pairs)

    @property
    def pair_ids(self) -> tuple[int, ...]:
        return tuple(pair.pair_id for pair in self.pairs)


@dataclass(frozen=True)
class DirectedRelation:
    source_id: int
    target_id: int


@dataclass
class ConceptCentricData:
    spec: DatasetSpec
    concepts: list[profiles.Concept]
    pairs: list[ConceptPair]
    tasks: list[AnchorTask]
    input_hash: str


PREREQUISITE_BY_CONCEPT_SYSTEM_PROMPT = """You are a conservative annotator of direct prerequisite
relations among technical concepts from one educational dataset. The request contains one anchor
concept and a list of candidate concepts. Judge only relations between the anchor and each supplied
candidate. Never compare two candidates with each other and never introduce another concept.

Return only clear, direct, instructionally meaningful prerequisite relations. A source concept must
need to be learned before its target concept. Do not return uncertain, weak, indirect, transitive,
same-topic, same-level, synonymous, merely helpful, or directionally unclear relationships. Do not
infer a relation from input order, course co-occurrence, or because one name sounds broader or more
advanced. Only concept names are supplied. If a relation is not highly confident from the two names
and your general knowledge, omit it.

Return exactly one JSON object with exactly one field named "relations". Each relation must contain
only integer "source_concept_id" and "target_concept_id" fields. One endpoint must be the anchor and
the other must be a supplied candidate. Return each candidate at most once. Return an empty array
when no direct prerequisite relation is confirmed. Do not return Markdown, explanations, names,
labels, negative examples, or extra fields.
{"relations":[{"source_concept_id":12,"target_concept_id":37}]}"""


def _coerce_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise PrerequisiteValidationError(f"{field_name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return int(value.strip())
        except ValueError as exc:
            raise PrerequisiteValidationError(f"{field_name} must be an integer") from exc
    raise PrerequisiteValidationError(f"{field_name} must be an integer")


def group_anchor_tasks(pairs: Sequence[ConceptPair]) -> list[AnchorTask]:
    """Group retained pairs by their lower-ID endpoint without changing pair order or IDs."""

    grouped: dict[int, list[ConceptPair]] = {}
    seen_pair_ids: set[int] = set()
    for pair in pairs:
        if pair.left_id >= pair.right_id:
            raise PrerequisiteValidationError(
                f"Pair {pair.pair_id} is not a canonical lower-ID unordered pair"
            )
        if pair.pair_id in seen_pair_ids:
            raise PrerequisiteValidationError(f"Duplicate pair_id {pair.pair_id}")
        seen_pair_ids.add(pair.pair_id)
        grouped.setdefault(pair.left_id, []).append(pair)
    return [
        AnchorTask(anchor_id, tuple(grouped[anchor_id]))
        for anchor_id in sorted(grouped)
    ]


def _input_hash(
    spec: DatasetSpec,
    concepts: Sequence[profiles.Concept],
    pairs: Sequence[ConceptPair],
) -> str:
    payload = {
        "program_version": PROGRAM_VERSION,
        "prompt_version": PROMPT_VERSION,
        "strategy_version": STRATEGY_VERSION,
        "dataset": spec.name,
        "concepts": [
            {
                "concept_id": concept.concept_id,
                "concept_name": concept.name,
                "aliases": list(concept.aliases),
            }
            for concept in concepts
        ],
        "context": "canonical concept names only",
        "pairs": [
            {
                "pair_id": pair.pair_id,
                "left_id": pair.left_id,
                "right_id": pair.right_id,
            }
            for pair in pairs
        ],
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_dataset(spec: DatasetSpec) -> ConceptCentricData:
    base = pairwise.load_dataset(spec)
    tasks = group_anchor_tasks(base.pairs)
    return ConceptCentricData(
        spec=spec,
        concepts=base.concepts,
        pairs=base.pairs,
        tasks=tasks,
        input_hash=_input_hash(spec, base.concepts, base.pairs),
    )


def build_user_prompt(data: ConceptCentricData, task: AnchorTask) -> str:
    anchor = data.concepts[task.anchor_id]
    candidates = [
        {
            "concept_id": data.concepts[candidate_id].concept_id,
            "concept_name": data.concepts[candidate_id].name,
        }
        for candidate_id in task.candidate_ids
    ]
    return f"""Dataset: {data.spec.name}

Anchor concept:
{json.dumps({"concept_id": anchor.concept_id, "concept_name": anchor.name}, ensure_ascii=False, indent=2)}

Candidate concepts:
{json.dumps(candidates, ensure_ascii=False, indent=2)}

Identify only highly confident direct prerequisite relations between the anchor and individual
candidates. A relation may point from the anchor to a candidate or from a candidate to the anchor.
Do not compare candidates with each other. Omit every unconfirmed relation.

Return exactly this JSON shape, using IDs from the records above:
{{"relations":[{{"source_concept_id": {anchor.concept_id}, "target_concept_id": {task.candidate_ids[0]}}}]}}
Return {{"relations":[]}} if no relation is confirmed. Return no explanations or extra fields."""


def _validate_relation_list(
    raw_relations: Any,
    task: AnchorTask,
) -> list[DirectedRelation]:
    if not isinstance(raw_relations, list):
        raise PrerequisiteValidationError("Qwen response must contain a relations list")
    candidate_set = set(task.candidate_ids)
    by_candidate: dict[int, DirectedRelation] = {}
    for index, raw_item in enumerate(raw_relations):
        if not isinstance(raw_item, Mapping):
            raise PrerequisiteValidationError(f"relation {index} is not an object")
        if set(raw_item) != {"source_concept_id", "target_concept_id"}:
            raise PrerequisiteValidationError(
                f"relation {index} must contain only source_concept_id and target_concept_id"
            )
        source_id = _coerce_int(
            raw_item.get("source_concept_id"), f"relation {index} source_concept_id"
        )
        target_id = _coerce_int(
            raw_item.get("target_concept_id"), f"relation {index} target_concept_id"
        )
        if source_id == target_id:
            raise PrerequisiteValidationError(f"relation {index} is a self-loop")
        source_is_anchor = source_id == task.anchor_id
        target_is_anchor = target_id == task.anchor_id
        if source_is_anchor == target_is_anchor:
            raise PrerequisiteValidationError(
                f"relation {index} must contain the anchor exactly once"
            )
        candidate_id = target_id if source_is_anchor else source_id
        if candidate_id not in candidate_set:
            raise PrerequisiteValidationError(
                f"relation {index} contains unknown candidate_id {candidate_id}"
            )
        if candidate_id in by_candidate:
            raise PrerequisiteValidationError(
                f"candidate_id {candidate_id} appears more than once"
            )
        by_candidate[candidate_id] = DirectedRelation(source_id, target_id)
    return [
        by_candidate[candidate_id]
        for candidate_id in task.candidate_ids
        if candidate_id in by_candidate
    ]


def validate_response(value: Mapping[str, Any], task: AnchorTask) -> list[DirectedRelation]:
    if not isinstance(value, Mapping):
        raise PrerequisiteValidationError("Qwen response must be a JSON object")
    if set(value) != {"relations"}:
        raise PrerequisiteValidationError(
            'Qwen response must contain exactly one field named "relations"'
        )
    return _validate_relation_list(value.get("relations"), task)


def request_anchor_relations(
    client: profiles.QwenClient,
    data: ConceptCentricData,
    task: AnchorTask,
) -> tuple[list[DirectedRelation], list[dict[str, Any]], str]:
    prompt = build_user_prompt(data, task)
    current_prompt = prompt
    response_history: list[dict[str, Any]] = []
    last_error: Exception | None = None
    final_model = PREREQUISITE_MODEL
    for attempt in range(SCHEMA_REPAIR_ATTEMPTS + 1):
        completion = client.complete_json(PREREQUISITE_BY_CONCEPT_SYSTEM_PROMPT, current_prompt)
        final_model = completion.model
        response_history.append(
            {
                "stage": "initial" if attempt == 0 else f"schema_repair_{attempt}",
                "model": completion.model,
                "raw": completion.raw,
                "warnings": list(completion.warnings),
            }
        )
        try:
            return validate_response(completion.value, task), response_history, final_model
        except PrerequisiteValidationError as exc:
            last_error = exc
            if attempt >= SCHEMA_REPAIR_ATTEMPTS:
                break
            current_prompt = f"""Your previous JSON response failed local validation: {exc}

Return a corrected JSON object for the exact same anchor and candidates. Return only confirmed
direct prerequisite relations, use each candidate at most once, and return no commentary or extra
fields. The original request was:
{prompt}

Your previous response was:
{completion.raw}
"""
    raise PrerequisiteValidationError(
        f"Qwen concept-centric response failed validation after repairs: {last_error}"
    )


def checkpoint_dir(spec: DatasetSpec) -> Path:
    return spec.root / ".prerequisite_generation_by_concept"


def checkpoint_path(spec: DatasetSpec, anchor_id: int) -> Path:
    return checkpoint_dir(spec) / f"anchor_{anchor_id:05d}.json"


def _checkpoint_value(
    data: ConceptCentricData,
    task: AnchorTask,
    relations: Sequence[DirectedRelation],
    model: str,
    response_history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "version": PROGRAM_VERSION,
        "prompt_version": PROMPT_VERSION,
        "strategy_version": STRATEGY_VERSION,
        "dataset": data.spec.name,
        "input_hash": data.input_hash,
        "anchor_id": task.anchor_id,
        "candidate_ids": list(task.candidate_ids),
        "pair_ids": list(task.pair_ids),
        "relations": [
            {
                "source_concept_id": relation.source_id,
                "target_concept_id": relation.target_id,
            }
            for relation in relations
        ],
        "model": model,
        "raw_responses": list(response_history),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def valid_checkpoint(
    checkpoint: Mapping[str, Any] | None,
    data: ConceptCentricData,
    task: AnchorTask,
) -> bool:
    if not checkpoint:
        return False
    expected = {
        "version": PROGRAM_VERSION,
        "prompt_version": PROMPT_VERSION,
        "strategy_version": STRATEGY_VERSION,
        "dataset": data.spec.name,
        "input_hash": data.input_hash,
        "anchor_id": task.anchor_id,
        "candidate_ids": list(task.candidate_ids),
        "pair_ids": list(task.pair_ids),
        "model": PREREQUISITE_MODEL,
    }
    if any(checkpoint.get(key) != value for key, value in expected.items()):
        return False
    try:
        _validate_relation_list(checkpoint.get("relations"), task)
    except PrerequisiteValidationError:
        return False
    return True


def checkpoint_relations(
    checkpoint: Mapping[str, Any], task: AnchorTask
) -> list[DirectedRelation]:
    return _validate_relation_list(checkpoint.get("relations"), task)


def collect_checkpoint_status(
    data: ConceptCentricData,
) -> tuple[dict[int, list[DirectedRelation]], list[int], list[int]]:
    valid: dict[int, list[DirectedRelation]] = {}
    stale: list[int] = []
    missing: list[int] = []
    for task in data.tasks:
        path = checkpoint_path(data.spec, task.anchor_id)
        checkpoint = pairwise.read_checkpoint(path)
        if checkpoint is None:
            missing.append(task.anchor_id)
        elif valid_checkpoint(checkpoint, data, task):
            valid[task.anchor_id] = checkpoint_relations(checkpoint, task)
        else:
            stale.append(task.anchor_id)
    return valid, stale, missing


def _model_config(spec: DatasetSpec) -> profiles.ApiConfig:
    config = profiles.load_api_config(spec)  # type: ignore[arg-type]
    return replace(config, model=PREREQUISITE_MODEL, vision_model=PREREQUISITE_MODEL)


def publish_if_complete(data: ConceptCentricData) -> bool:
    incoming: list[tuple[str, str, int]] = []
    for task in data.tasks:
        checkpoint = pairwise.read_checkpoint(checkpoint_path(data.spec, task.anchor_id))
        if not valid_checkpoint(checkpoint, data, task):
            return False
        assert checkpoint is not None
        for relation in checkpoint_relations(checkpoint, task):
            incoming.append(
                (
                    data.concepts[relation.source_id].name,
                    data.concepts[relation.target_id].name,
                    1,
                )
            )

    stats = pairwise.append_public_rows(data.spec.output_path, data.concepts, incoming)
    print(
        f"{data.spec.name} concept-centric append summary: existing={stats.existing}, "
        f"added={stats.added}, duplicates={stats.duplicates}, conflicts={stats.conflicts}",
        flush=True,
    )
    return True


def process_dataset(
    data: ConceptCentricData,
    config: profiles.ApiConfig,
    *,
    force: bool = False,
    next_only: bool = False,
    dry_run: bool = False,
) -> bool:
    valid, stale, missing = collect_checkpoint_status(data)
    candidate_counts = [len(task.pairs) for task in data.tasks]
    candidate_range = (
        f"{min(candidate_counts)}-{max(candidate_counts)}" if candidate_counts else "0-0"
    )
    print(
        f"{data.spec.name}: concepts={len(data.concepts)}, pairs={len(data.pairs)}, "
        f"anchor_requests={len(data.tasks)}, "
        f"candidates_per_anchor={candidate_range}, checkpoints(valid/stale/missing)="
        f"{len(valid)}/{len(stale)}/{len(missing)}, "
        f"api_key={'configured' if config.api_key else 'missing'}, model={PREREQUISITE_MODEL}, "
        f"output={data.spec.output_path}",
        flush=True,
    )
    if dry_run:
        return False

    targets = [
        task
        for task in data.tasks
        if force
        or not valid_checkpoint(
            pairwise.read_checkpoint(checkpoint_path(data.spec, task.anchor_id)), data, task
        )
    ]
    if next_only:
        targets = targets[:1]
    if not targets:
        if publish_if_complete(data):
            print(f"Published {data.spec.output_path}", flush=True)
            return True
        print(
            f"{data.spec.name} is incomplete; existing public CSV was left unchanged.",
            file=sys.stderr,
            flush=True,
        )
        return False

    if not config.api_key:
        raise PrerequisiteError(f"QWEN_API_KEY is empty in {data.spec.env_path}")

    for completed, task in enumerate(targets, start=1):
        client = profiles.QwenClient(config)
        relations, response_history, model = request_anchor_relations(client, data, task)
        checkpoint = _checkpoint_value(data, task, relations, model, response_history)
        profiles.atomic_write_json(checkpoint_path(data.spec, task.anchor_id), checkpoint)
        print(
            f"[{completed}/{len(targets)}] {data.spec.name} anchor "
            f"{task.anchor_id + 1}/{len(data.concepts)} "
            f"({data.concepts[task.anchor_id].name!r}, candidates={len(task.pairs)}, "
            f"relations={len(relations)})",
            flush=True,
        )

    if publish_if_complete(data):
        print(f"Published {data.spec.output_path}", flush=True)
        return True
    print(
        f"{data.spec.name} is incomplete; existing public CSV was left unchanged.",
        file=sys.stderr,
        flush=True,
    )
    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate direct Qwen prerequisite edges with one request per anchor concept."
        )
    )
    parser.add_argument("--dataset", choices=["MLR", "DGL"], required=True)
    parser.add_argument(
        "--next-concept",
        action="store_true",
        help="Process only the next incomplete or stale anchor concept",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate every non-empty anchor concept",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and report work without calling Qwen or writing files",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.next_concept and args.force:
        parser.error("--next-concept and --force cannot be used together")
    try:
        spec = pairwise.DATASET_SPECS[args.dataset]
        data = load_dataset(spec)
        config = _model_config(spec)
        process_dataset(
            data,
            config,
            force=args.force,
            next_only=args.next_concept,
            dry_run=args.dry_run,
        )
        return 0
    except (PrerequisiteError, profiles.ProfileError, OSError, csv.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
