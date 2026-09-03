#!/usr/bin/env python3
"""Generate dataset-local prerequisite relations for MLR and DGL.

The concept universe is deliberately kept separate for each dataset.  Concepts are
paired within one dataset, sent to Qwen in restartable batches, and published as a
headerless ``source,target,label`` CSV only after every batch has been validated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import generate_profiles as profiles


PROGRAM_VERSION = "1.3.0"
PROMPT_VERSION = "prerequisite_relations_names_v2"
PREREQUISITE_MODEL = "qwen3.8-max"
DEFAULT_BATCH_SIZE = 50
DEFAULT_NEGATIVE_RATIO = 3
DEFAULT_RANDOM_SEED = 42
NO_POSITIVE_NEGATIVE_CAP = 1000
SCHEMA_REPAIR_ATTEMPTS = 2

ROOT_DIR = Path(__file__).resolve().parent.parent


class PrerequisiteError(RuntimeError):
    """Base error for prerequisite generation."""


class PrerequisiteValidationError(PrerequisiteError):
    """Input data or a model response failed validation."""


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path
    concepts_name: str
    profiles_name: str
    output_name: str

    @property
    def concepts_path(self) -> Path:
        return self.root / self.concepts_name

    @property
    def profiles_path(self) -> Path:
        return self.root / self.profiles_name

    @property
    def output_path(self) -> Path:
        return self.root / self.output_name

    @property
    def checkpoint_dir(self) -> Path:
        return self.root / ".prerequisite_generation"

    @property
    def env_path(self) -> Path:
        return self.root / ".env"


DATASET_SPECS: dict[str, DatasetSpec] = {
    "MLR": DatasetSpec(
        name="MLR",
        root=ROOT_DIR / "MLR",
        concepts_name="MLR_concepts.csv",
        profiles_name="MLR_profiles.jsonl",
        output_name="MLR_prerequisite.csv",
    ),
    "DGL": DatasetSpec(
        name="DGL",
        root=ROOT_DIR / "DGL",
        concepts_name="DGL_concepts.csv",
        profiles_name="DGL_profiles.jsonl",
        output_name="DGL_prerequisite.csv",
    ),
}


@dataclass(frozen=True)
class ConceptPair:
    pair_id: int
    left_id: int
    right_id: int


@dataclass(frozen=True)
class PairDecision:
    pair_id: int
    label: int


@dataclass(frozen=True)
class AppendStats:
    existing: int
    added: int
    duplicates: int
    conflicts: int


@dataclass
class DatasetData:
    spec: DatasetSpec
    concepts: list[profiles.Concept]
    # Kept as an empty compatibility field for callers that construct DatasetData directly.  The
    # prerequisite pipeline no longer loads or sends profile content.
    profiles_by_id: dict[int, str]
    pairs: list[ConceptPair]
    input_hash: str


PREREQUISITE_SYSTEM_PROMPT = """You are a conservative annotator of direct prerequisite relations among
technical concepts from one educational dataset.  Judge only the concept pairs supplied in the
current request; never import concepts or evidence from another dataset.

For every pair, use exactly one integer label:
- 1 means the left concept is a clear, direct, instructionally meaningful prerequisite for the right concept.
- -1 means the right concept is a clear, direct, instructionally meaningful prerequisite for the left concept.
- 0 means there is no confirmed direct prerequisite relation.

Use 0 for uncertainty, weak or indirect association, the same topic or level, synonyms, a merely
generic background relationship, an unclear direction, or a relation that would require transitive
closure.  Do not infer a prerequisite just because one concept sounds broader, more advanced, or
appears earlier in the input.  Do not use course co-occurrence as evidence.  This task supplies only
concept names; do not assume that omitted definitions or source passages exist.  If you are not
highly confident from the two names and your general knowledge, output 0.

Return exactly one JSON object and no Markdown, explanation, or concept names.  Its only required
field is an array named "decisions", with one object for every input pair:
{"decisions":[{"pair_id":0,"label":0}]}"""


def _clean(value: Any, max_length: int = 20_000) -> str:
    return profiles.clean_text(value, max_length=max_length)


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise PrerequisiteValidationError(f"Missing {description}: {path}")


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


def load_profiles(path: Path, concepts: Sequence[profiles.Concept]) -> dict[int, str]:
    """Load and strictly align generated profiles to canonical concept IDs."""

    _require_file(path, "profiles JSONL")
    expected = {concept.concept_id: concept for concept in concepts}
    loaded: dict[int, str] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise PrerequisiteValidationError(
                    f"Invalid JSON in {path}:{line_number}"
                ) from exc
            if not isinstance(row, Mapping):
                raise PrerequisiteValidationError(f"Profile row {path}:{line_number} is not an object")
            concept_id = _coerce_int(row.get("concept_id"), f"concept_id at {path}:{line_number}")
            concept = expected.get(concept_id)
            if concept is None:
                raise PrerequisiteValidationError(
                    f"Profile at {path}:{line_number} has unknown concept_id {concept_id}"
                )
            if concept_id in loaded:
                raise PrerequisiteValidationError(
                    f"Duplicate profile for concept_id {concept_id} at {path}:{line_number}"
                )
            name = _clean(row.get("concept_name"), max_length=500)
            if profiles.normalize_key(name) != profiles.normalize_key(concept.name):
                raise PrerequisiteValidationError(
                    f"Profile concept_name {name!r} does not match canonical concept {concept.name!r}"
                )
            profile_text = _clean(row.get("profile"), max_length=30_000)
            if not profile_text:
                raise PrerequisiteValidationError(
                    f"Empty profile for {concept.name!r} at {path}:{line_number}"
                )
            loaded[concept_id] = profile_text
    missing = sorted(set(expected) - set(loaded))
    if missing:
        names = ", ".join(expected[item].name for item in missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise PrerequisiteValidationError(
            f"{path} is missing {len(missing)} profiles: {names}{suffix}"
        )
    return loaded


def enumerate_pairs(concepts: Sequence[profiles.Concept]) -> list[ConceptPair]:
    pairs: list[ConceptPair] = []
    pair_id = 0
    for left_id in range(len(concepts)):
        for right_id in range(left_id + 1, len(concepts)):
            pairs.append(ConceptPair(pair_id, left_id, right_id))
            pair_id += 1
    return pairs


def _input_hash(
    spec: DatasetSpec,
    concepts: Sequence[profiles.Concept],
    profiles_by_id: Mapping[int, str],
    pairs: Sequence[ConceptPair],
) -> str:
    payload = {
        "program_version": PROGRAM_VERSION,
        "prompt_version": PROMPT_VERSION,
        "dataset": spec.name,
        "concepts": [
            {
                "concept_id": concept.concept_id,
                "concept_name": concept.name,
                "aliases": list(concept.aliases),
            }
            for concept in concepts
        ],
        "context": "canonical concept names only; profiles are not supplied",
        "pair_count": len(pairs),
        "pair_ids": [pair.pair_id for pair in pairs],
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_dataset(spec: DatasetSpec) -> DatasetData:
    concepts = profiles.load_concepts(spec.concepts_path)
    pairs = enumerate_pairs(concepts)
    return DatasetData(
        spec=spec,
        concepts=concepts,
        profiles_by_id={},
        pairs=pairs,
        input_hash=_input_hash(spec, concepts, {}, pairs),
    )


def _normalized_row(row: Sequence[str]) -> tuple[str, str, str]:
    """Normalize a prerequisite row for idempotent exact-row deduplication."""

    return (
        profiles.normalize_key(row[0]),
        profiles.normalize_key(row[1]),
        profiles.normalize_key(row[2]),
    )


def batched(items: Sequence[ConceptPair], batch_size: int) -> list[list[ConceptPair]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [list(items[start : start + batch_size]) for start in range(0, len(items), batch_size)]


def build_user_prompt(data: DatasetData, batch: Sequence[ConceptPair]) -> str:
    used_ids = sorted({item.left_id for item in batch} | {item.right_id for item in batch})
    concept_records = []
    for concept_id in used_ids:
        concept = data.concepts[concept_id]
        concept_records.append(
            {
                "concept_id": concept.concept_id,
                "concept_name": concept.name,
            }
        )
    pair_records = [
        {
            "pair_id": item.pair_id,
            "left_concept_id": item.left_id,
            "right_concept_id": item.right_id,
        }
        for item in batch
    ]
    return f"""Dataset: {data.spec.name}

The records below belong only to this dataset.  Use only the supplied concept names to classify each
pair.  A relation must be direct and instructionally meaningful; do not infer a transitive relation.

Concept records:
{json.dumps(concept_records, ensure_ascii=False, indent=2)}

Pairs to classify:
{json.dumps(pair_records, ensure_ascii=False, indent=2)}

Return one decision for every pair_id, in any order, using only integer labels -1, 0, or 1:
{{"decisions":[{{"pair_id": {batch[0].pair_id if batch else 0}, "label": 0}}]}}
Do not return names, explanations, or extra fields in the decision objects."""


def _validate_decisions(value: Mapping[str, Any], batch: Sequence[ConceptPair]) -> list[PairDecision]:
    if not isinstance(value, Mapping):
        raise PrerequisiteValidationError("Qwen response must be a JSON object")
    raw_decisions = value.get("decisions")
    if not isinstance(raw_decisions, list):
        raise PrerequisiteValidationError("Qwen response must contain a decisions list")
    expected_ids = [item.pair_id for item in batch]
    expected_set = set(expected_ids)
    seen: set[int] = set()
    decisions: list[PairDecision] = []
    for index, raw_item in enumerate(raw_decisions):
        if not isinstance(raw_item, Mapping):
            raise PrerequisiteValidationError(f"decision {index} is not an object")
        pair_id = _coerce_int(raw_item.get("pair_id"), f"decision {index} pair_id")
        if pair_id not in expected_set:
            raise PrerequisiteValidationError(f"unknown pair_id {pair_id}")
        if pair_id in seen:
            raise PrerequisiteValidationError(f"duplicate pair_id {pair_id}")
        label = _coerce_int(raw_item.get("label"), f"decision {index} label")
        if label not in {-1, 0, 1}:
            raise PrerequisiteValidationError(f"label must be -1, 0, or 1; got {label}")
        seen.add(pair_id)
        decisions.append(PairDecision(pair_id, label))
    missing = expected_set - seen
    if missing:
        raise PrerequisiteValidationError(
            f"missing decisions for pair_id(s): {', '.join(str(item) for item in sorted(missing))}"
        )
    by_id = {item.pair_id: item for item in decisions}
    return [by_id[item.pair_id] for item in batch]


def request_batch_decisions(
    client: profiles.QwenClient,
    data: DatasetData,
    batch: Sequence[ConceptPair],
) -> tuple[list[PairDecision], list[dict[str, Any]], str]:
    """Call Qwen and validate a batch, retrying only schema corrections."""

    prompt = build_user_prompt(data, batch)
    current_prompt = prompt
    response_history: list[dict[str, Any]] = []
    last_error: Exception | None = None
    final_model = PREREQUISITE_MODEL
    for attempt in range(SCHEMA_REPAIR_ATTEMPTS + 1):
        completion = client.complete_json(PREREQUISITE_SYSTEM_PROMPT, current_prompt)
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
            decisions = _validate_decisions(completion.value, batch)
            return decisions, response_history, final_model
        except PrerequisiteValidationError as exc:
            last_error = exc
            if attempt >= SCHEMA_REPAIR_ATTEMPTS:
                break
            current_prompt = f"""Your previous JSON response failed local validation: {exc}

Return a corrected JSON object for exactly the same input pairs.  Include every pair_id exactly
once, use only labels -1, 0, and 1, and return no commentary.  The original request was:
{prompt}

Your previous response was:
{completion.raw}
"""
    raise PrerequisiteValidationError(
        f"Qwen prerequisite response failed validation after repairs: {last_error}"
    )


def checkpoint_path(spec: DatasetSpec, batch_index: int) -> Path:
    return spec.checkpoint_dir / f"batch_{batch_index:05d}.json"


def read_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def valid_checkpoint(
    checkpoint: Mapping[str, Any] | None,
    data: DatasetData,
    batch_index: int,
    batch: Sequence[ConceptPair],
    batch_size: int,
) -> bool:
    if not checkpoint:
        return False
    if checkpoint.get("version") != PROGRAM_VERSION:
        return False
    if checkpoint.get("prompt_version") != PROMPT_VERSION:
        return False
    if checkpoint.get("dataset") != data.spec.name:
        return False
    if checkpoint.get("input_hash") != data.input_hash:
        return False
    if checkpoint.get("batch_index") != batch_index or checkpoint.get("batch_size") != batch_size:
        return False
    if checkpoint.get("model") != PREREQUISITE_MODEL:
        return False
    raw_pair_ids = checkpoint.get("pair_ids")
    if raw_pair_ids != [item.pair_id for item in batch]:
        return False
    try:
        _validate_decisions(checkpoint, batch)
    except PrerequisiteValidationError:
        return False
    return True


def checkpoint_decisions(checkpoint: Mapping[str, Any], batch: Sequence[ConceptPair]) -> list[PairDecision]:
    return _validate_decisions(checkpoint, batch)


def collect_valid_checkpoints(
    data: DatasetData,
    batches: Sequence[Sequence[ConceptPair]],
    batch_size: int,
) -> tuple[dict[int, list[PairDecision]], list[int], list[int]]:
    valid: dict[int, list[PairDecision]] = {}
    stale: list[int] = []
    missing: list[int] = []
    for batch_index, batch in enumerate(batches):
        path = checkpoint_path(data.spec, batch_index)
        checkpoint = read_checkpoint(path)
        if checkpoint is None:
            missing.append(batch_index)
        elif valid_checkpoint(checkpoint, data, batch_index, batch, batch_size):
            valid[batch_index] = checkpoint_decisions(checkpoint, batch)
        else:
            stale.append(batch_index)
    return valid, stale, missing


def _model_config(spec: DatasetSpec) -> profiles.ApiConfig:
    config = profiles.load_api_config(spec)  # type: ignore[arg-type]
    return replace(config, model=PREREQUISITE_MODEL, vision_model=PREREQUISITE_MODEL)


def _checkpoint_value(
    data: DatasetData,
    batch_index: int,
    batch: Sequence[ConceptPair],
    batch_size: int,
    decisions: Sequence[PairDecision],
    model: str,
    response_history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "version": PROGRAM_VERSION,
        "prompt_version": PROMPT_VERSION,
        "dataset": data.spec.name,
        "input_hash": data.input_hash,
        "batch_index": batch_index,
        "batch_size": batch_size,
        "pair_ids": [item.pair_id for item in batch],
        "decisions": [
            {"pair_id": item.pair_id, "label": item.label} for item in decisions
        ],
        "model": model,
        "raw_responses": list(response_history),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def _public_rows(
    data: DatasetData,
    decisions_by_id: Mapping[int, PairDecision],
    *,
    negative_ratio: int = DEFAULT_NEGATIVE_RATIO,
    seed: int = DEFAULT_RANDOM_SEED,
) -> list[tuple[str, str, int]]:
    positives: list[tuple[int, str, str, int]] = []
    negatives: list[tuple[int, str, str, int]] = []
    for pair in data.pairs:
        decision = decisions_by_id[pair.pair_id]
        left = data.concepts[pair.left_id].name
        right = data.concepts[pair.right_id].name
        if decision.label == 1:
            positives.append((pair.pair_id, left, right, 1))
        elif decision.label == -1:
            positives.append((pair.pair_id, right, left, 1))
        else:
            negatives.append((pair.pair_id, left, right, 0))

    if positives:
        sample_size = min(len(negatives), max(0, negative_ratio) * len(positives))
    else:
        sample_size = min(len(negatives), NO_POSITIVE_NEGATIVE_CAP)
    sampled = random.Random(seed).sample(negatives, sample_size)
    sampled.sort(key=lambda row: row[0])
    rows = positives + sampled
    return [(source, target, label) for _pair_id, source, target, label in rows]


def read_prerequisite_rows(path: Path) -> list[tuple[str, str, int]]:
    if not path.is_file():
        return []
    rows: list[tuple[str, str, int]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if not row or not any(cell.strip() for cell in row):
                continue
            if len(row) < 3:
                raise PrerequisiteValidationError(
                    f"Invalid prerequisite row at {path}:{line_number}; expected three columns"
                )
            source, target, raw_label = (cell.strip() for cell in row[:3])
            if not source or not target or raw_label not in {"0", "1"}:
                raise PrerequisiteValidationError(
                    f"Invalid prerequisite row at {path}:{line_number}: {row[:3]!r}"
                )
            rows.append((source, target, int(raw_label)))
    return rows


def _exact_public_row_key(row: tuple[str, str, int]) -> tuple[str, str, str]:
    return _normalized_row((row[0], row[1], str(row[2])))


def _resolved_public_row(
    row: tuple[str, str, int],
    resolver: profiles.ConceptResolver,
) -> tuple[tuple[int, int], tuple[Any, ...]] | None:
    source_id = resolver.resolve(row[0])
    target_id = resolver.resolve(row[1])
    if source_id is None or target_id is None:
        return None
    pair_key = (min(source_id, target_id), max(source_id, target_id))
    if row[2] == 0:
        semantic = (0,)
    else:
        semantic = (1, source_id, target_id)
    return pair_key, semantic


def append_public_rows(
    path: Path,
    concepts: Sequence[profiles.Concept],
    incoming: Sequence[tuple[str, str, int]],
) -> AppendStats:
    """Atomically append new, non-conflicting rows while preserving all existing rows."""

    existing = read_prerequisite_rows(path)
    resolver = profiles.ConceptResolver(concepts)
    exact_keys = {_exact_public_row_key(row) for row in existing}
    semantics_by_pair: dict[tuple[int, int], set[tuple[Any, ...]]] = {}
    for row in existing:
        resolved = _resolved_public_row(row, resolver)
        if resolved is not None:
            pair_key, semantic = resolved
            semantics_by_pair.setdefault(pair_key, set()).add(semantic)

    accepted: list[tuple[str, str, int]] = []
    duplicate_count = 0
    conflict_count = 0
    for row in incoming:
        exact_key = _exact_public_row_key(row)
        if exact_key in exact_keys:
            duplicate_count += 1
            continue
        resolved = _resolved_public_row(row, resolver)
        if resolved is not None:
            pair_key, semantic = resolved
            existing_semantics = semantics_by_pair.get(pair_key, set())
            if semantic in existing_semantics:
                duplicate_count += 1
                continue
            if existing_semantics:
                conflict_count += 1
                continue
            semantics_by_pair.setdefault(pair_key, set()).add(semantic)
        exact_keys.add(exact_key)
        accepted.append(row)

    if accepted or not path.is_file():
        output = io.StringIO(newline="")
        writer = csv.writer(output, lineterminator="\n")
        writer.writerows([*existing, *accepted])
        profiles.atomic_write_text(path, output.getvalue())
    return AppendStats(
        existing=len(existing),
        added=len(accepted),
        duplicates=duplicate_count,
        conflicts=conflict_count,
    )


def publish_if_complete(
    data: DatasetData,
    batches: Sequence[Sequence[ConceptPair]],
    batch_size: int,
    *,
    negative_ratio: int = DEFAULT_NEGATIVE_RATIO,
    seed: int = DEFAULT_RANDOM_SEED,
) -> bool:
    decisions_by_id: dict[int, PairDecision] = {}
    for batch_index, batch in enumerate(batches):
        checkpoint = read_checkpoint(checkpoint_path(data.spec, batch_index))
        if not valid_checkpoint(checkpoint, data, batch_index, batch, batch_size):
            return False
        assert checkpoint is not None
        for decision in checkpoint_decisions(checkpoint, batch):
            decisions_by_id[decision.pair_id] = decision
    if len(decisions_by_id) != len(data.pairs):
        return False
    incoming = _public_rows(
        data, decisions_by_id, negative_ratio=negative_ratio, seed=seed
    )
    append_stats = append_public_rows(data.spec.output_path, data.concepts, incoming)
    print(
        f"{data.spec.name} append summary: existing={append_stats.existing}, "
        f"added={append_stats.added}, duplicates={append_stats.duplicates}, "
        f"conflicts={append_stats.conflicts}",
        flush=True,
    )
    return True


def process_dataset(
    data: DatasetData,
    config: profiles.ApiConfig,
    *,
    batch_size: int,
    force: bool = False,
    next_only: bool = False,
    dry_run: bool = False,
    negative_ratio: int = DEFAULT_NEGATIVE_RATIO,
    seed: int = DEFAULT_RANDOM_SEED,
) -> bool:
    batches = batched(data.pairs, batch_size)
    valid, stale, missing = collect_valid_checkpoints(data, batches, batch_size)
    print(
        f"{data.spec.name}: concepts={len(data.concepts)}, pairs={len(data.pairs)}, "
        f"batches={len(batches)}, checkpoints(valid/stale/missing)="
        f"{len(valid)}/{len(stale)}/{len(missing)}, "
        f"api_key={'configured' if config.api_key else 'missing'}, model={PREREQUISITE_MODEL}, "
        f"output={data.spec.output_path}",
        flush=True,
    )
    if dry_run:
        return False

    targets = [
        index
        for index, batch in enumerate(batches)
        if force or not valid_checkpoint(
            read_checkpoint(checkpoint_path(data.spec, index)), data, index, batch, batch_size
        )
    ]
    if next_only:
        targets = targets[:1]
    if not targets:
        if publish_if_complete(
            data,
            batches,
            batch_size,
            negative_ratio=negative_ratio,
            seed=seed,
        ):
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

    for completed, batch_index in enumerate(targets, start=1):
        client = profiles.QwenClient(config)
        batch = batches[batch_index]
        decisions, response_history, model = request_batch_decisions(client, data, batch)
        checkpoint = _checkpoint_value(
            data, batch_index, batch, batch_size, decisions, model, response_history
        )
        profiles.atomic_write_json(checkpoint_path(data.spec, batch_index), checkpoint)
        print(
            f"[{completed}/{len(targets)}] {data.spec.name} batch {batch_index + 1}/{len(batches)} "
            f"(pairs {batch[0].pair_id}-{batch[-1].pair_id})",
            flush=True,
        )

    if publish_if_complete(
        data,
        batches,
        batch_size,
        negative_ratio=negative_ratio,
        seed=seed,
    ):
        print(f"Published {data.spec.output_path}", flush=True)
        return True
    print(
        f"{data.spec.name} is incomplete; existing public CSV was left unchanged.",
        file=sys.stderr,
        flush=True,
    )
    return False


def selected_specs(dataset_name: str) -> list[DatasetSpec]:
    if dataset_name == "all":
        return [DATASET_SPECS["MLR"], DATASET_SPECS["DGL"]]
    return [DATASET_SPECS[dataset_name]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate conservative Qwen prerequisite relations within MLR or DGL."
    )
    parser.add_argument("--dataset", choices=["MLR", "DGL", "all"], required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--negative-ratio", type=int, default=DEFAULT_NEGATIVE_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--next-batch", action="store_true", help="Process only the next incomplete batch")
    parser.add_argument("--force", action="store_true", help="Regenerate every batch in the selected dataset")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without calling Qwen")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.negative_ratio < 0:
        parser.error("--negative-ratio cannot be negative")
    if args.next_batch and args.force:
        parser.error("--next-batch and --force cannot be used together")
    try:
        for spec in selected_specs(args.dataset):
            data = load_dataset(spec)
            config = _model_config(spec)
            complete = process_dataset(
                data,
                config,
                batch_size=args.batch_size,
                force=args.force,
                next_only=args.next_batch,
                dry_run=args.dry_run,
                negative_ratio=args.negative_ratio,
                seed=args.seed,
            )
            # In --dataset all mode, never begin DGL while MLR is incomplete.
            # Dry-run is intentionally allowed to inspect both datasets without
            # making any changes or API calls.
            if args.dataset == "all" and spec.name == "MLR" and not complete and not args.dry_run:
                raise PrerequisiteError(
                    "MLR is incomplete; DGL was not started. Rerun MLR until all batches publish."
                )
        return 0
    except (PrerequisiteError, profiles.ProfileError, OSError, csv.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
