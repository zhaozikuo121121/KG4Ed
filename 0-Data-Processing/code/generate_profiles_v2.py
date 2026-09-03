from __future__ import annotations

"""Generate metadata-grounded MLR and DGL concept profiles with DeepSeek.

The script reads the canonical concept lists and metadata v2 CSV files, generates
one English profile per concept, and runs a second independent LLM review before
publishing complete JSONL outputs. Per-concept checkpoints make runs restartable.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


PROGRAM_VERSION = "2.0.0"
PROMPT_VERSION = "concept_profile_metadata_v2"
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_TIMEOUT_SECONDS = 1800
DEFAULT_MAX_RETRIES = 5
MAX_PROFILE_WORDS = 120
TARGET_MIN_PROFILE_WORDS = 50
RESPONSE_REPAIR_ATTEMPTS = 1

ROOT_DIR = Path(__file__).resolve().parent.parent
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
CONTEXT_DEPENDENT_RE = re.compile(
    r"\b(?:the|this|that)\s+(?:attached|provided|supplied|above|below)\s+"
    r"(?:context|evidence|excerpt|image|figure|diagram|page|slide)\b|"
    r"\b(?:on|in)\s+(?:this|the)\s+(?:page|slide)\b|"
    r"\b(?:evidence|excerpt)\s+(?:id|ids|number)\b",
    re.IGNORECASE,
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class ProfileV2Error(RuntimeError):
    """Base error for profile v2 generation."""


class DataValidationError(ProfileV2Error):
    """Input files do not satisfy the required schema."""


class ResponseValidationError(ProfileV2Error):
    """A model response cannot be safely checkpointed or published."""


class ApiError(ProfileV2Error):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path
    concepts_name: str
    metadata_name: str
    output_name: str

    @property
    def concepts_path(self) -> Path:
        return self.root / self.concepts_name

    @property
    def metadata_path(self) -> Path:
        return self.root / self.metadata_name

    @property
    def output_path(self) -> Path:
        return self.root / self.output_name

    @property
    def checkpoint_dir(self) -> Path:
        return self.root / ".profile_generation_v2"


DATASET_SPECS: dict[str, DatasetSpec] = {
    "MLR": DatasetSpec(
        name="MLR",
        root=ROOT_DIR / "MLR",
        concepts_name="MLR_concepts.csv",
        metadata_name="MLR_concepts_metadata_v2.csv",
        output_name="MLR_profiles_v2.jsonl",
    ),
    "DGL": DatasetSpec(
        name="DGL",
        root=ROOT_DIR / "DGL",
        concepts_name="DGL_concepts.csv",
        metadata_name="DGL_concepts_metadata_v2.csv",
        output_name="DGL_profiles_v2.jsonl",
    ),
}


@dataclass(frozen=True)
class Concept:
    concept_id: int
    name: str
    aliases: tuple[str, ...]


@dataclass
class DatasetData:
    spec: DatasetSpec
    concepts: list[Concept]
    evidence_by_id: dict[int, list[dict[str, str]]] = field(default_factory=dict)
    unmatched_metadata: int = 0


@dataclass(frozen=True)
class ProfileInput:
    dataset: str
    concept: Concept
    evidence: tuple[dict[str, str], ...]
    input_hash: str

    @property
    def evidence_ids(self) -> set[str]:
        return {item["evidence_id"] for item in self.evidence}


@dataclass(frozen=True)
class ApiConfig:
    api_key: str
    base_url: str
    model: str
    timeout: int
    max_retries: int


@dataclass(frozen=True)
class Completion:
    value: dict[str, Any]
    raw: str
    model: str


@dataclass(frozen=True)
class ValidatedDraft:
    profile: str
    claims: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ValidatedReview:
    final_profile: str
    sentence_reviews: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...]


Transport = Callable[[str, dict[str, Any], dict[str, str], int], dict[str, Any]]


def clean_text(value: Any, *, max_length: int = 20_000) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).replace("\x00", " ")
    return re.sub(r"\s+", " ", text).strip()[:max_length]


def normalize_key(value: Any) -> str:
    text = clean_text(value, max_length=500).casefold()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def stable_unique_strings(values: Iterable[Any]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for raw_value in values:
        value = clean_text(raw_value, max_length=500)
        key = value.casefold()
        if not value or key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


def english_word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def split_profile_sentences(profile: str) -> list[str]:
    return [clean_text(item) for item in SENTENCE_SPLIT_RE.split(profile) if clean_text(item)]


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            values[key] = value.strip().strip('"').strip("'")
    return values


def load_api_config(root: Path = ROOT_DIR) -> ApiConfig:
    file_values = read_env_file(root / ".env")

    def setting(name: str, default: str = "") -> str:
        if name in os.environ:
            return os.environ[name].strip()
        return file_values.get(name, default).strip()

    try:
        timeout = max(1, int(setting("DEEPSEEK_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS))))
    except ValueError as exc:
        raise DataValidationError("DEEPSEEK_TIMEOUT_SECONDS must be a positive integer") from exc
    try:
        max_retries = max(1, int(setting("DEEPSEEK_MAX_RETRIES", str(DEFAULT_MAX_RETRIES))))
    except ValueError as exc:
        raise DataValidationError("DEEPSEEK_MAX_RETRIES must be a positive integer") from exc
    return ApiConfig(
        api_key=setting("DEEPSEEK_API_KEY"),
        base_url=setting("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL) or DEFAULT_BASE_URL,
        model=setting("DEEPSEEK_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL,
        timeout=timeout,
        max_retries=max_retries,
    )


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise DataValidationError(f"Missing {description}: {path}")


def read_dict_rows(path: Path, required_fields: Sequence[str]) -> list[dict[str, str]]:
    require_file(path, "CSV file")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = [field for field in required_fields if field not in fieldnames]
        if missing:
            raise DataValidationError(f"{path} is missing columns: {', '.join(missing)}")
        return [dict(row) for row in reader]


def load_concepts(path: Path) -> list[Concept]:
    require_file(path, "concepts CSV")
    concepts: list[Concept] = []
    canonical_keys: dict[str, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.reader(handle), start=1):
            cells = stable_unique_strings(row)
            if not cells:
                continue
            canonical = cells[0]
            key = normalize_key(canonical)
            if not key:
                raise DataValidationError(f"Empty canonical concept at {path}:{row_number}")
            if key in canonical_keys:
                raise DataValidationError(
                    f"Duplicate canonical concept {canonical!r} at rows "
                    f"{canonical_keys[key]} and {row_number}"
                )
            canonical_keys[key] = row_number
            aliases = tuple(stable_unique_strings([canonical, *cells[1:]]))
            concepts.append(Concept(len(concepts), canonical, aliases))
    if not concepts:
        raise DataValidationError(f"No concepts found in {path}")
    return concepts


class ConceptResolver:
    def __init__(self, concepts: Sequence[Concept]):
        self.canonical = {normalize_key(item.name): item.concept_id for item in concepts}
        candidates: dict[str, set[int]] = {}
        for item in concepts:
            for alias in item.aliases:
                key = normalize_key(alias)
                if key:
                    candidates.setdefault(key, set()).add(item.concept_id)
        self.unique_aliases = {
            key: next(iter(ids)) for key, ids in candidates.items() if len(ids) == 1
        }

    def resolve(self, value: Any) -> int | None:
        key = normalize_key(value)
        if not key:
            return None
        return self.canonical.get(key, self.unique_aliases.get(key))


def load_dataset(spec: DatasetSpec) -> DatasetData:
    concepts = load_concepts(spec.concepts_path)
    resolver = ConceptResolver(concepts)
    data = DatasetData(spec=spec, concepts=concepts)
    rows = read_dict_rows(spec.metadata_path, ["concept", "evidence"])
    for order, row in enumerate(rows, start=1):
        concept_id = resolver.resolve(row.get("concept"))
        if concept_id is None:
            data.unmatched_metadata += 1
            continue
        evidence_text = clean_text(row.get("evidence"), max_length=1_200)
        if not evidence_text:
            continue
        item: dict[str, str] = {"evidence_id": f"{spec.name}-E{order:04d}"}
        for key, raw_value in row.items():
            if key == "concept":
                continue
            limit = 1_200 if key == "evidence" else 2_000
            value = clean_text(raw_value, max_length=limit)
            if value:
                item[key] = value
        data.evidence_by_id.setdefault(concept_id, []).append(item)

    if data.unmatched_metadata:
        raise DataValidationError(
            f"{spec.metadata_path} has {data.unmatched_metadata} metadata rows that do not map "
            "to a unique concept or alias"
        )
    missing = [item.name for item in concepts if not data.evidence_by_id.get(item.concept_id)]
    if missing:
        preview = ", ".join(repr(item) for item in missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise DataValidationError(
            f"{spec.name} has {len(missing)} concepts without non-empty v2 evidence: {preview}{suffix}"
        )
    return data


def build_profile_input(data: DatasetData, concept: Concept) -> ProfileInput:
    evidence = tuple(dict(item) for item in data.evidence_by_id[concept.concept_id])
    hash_payload = {
        "program_version": PROGRAM_VERSION,
        "prompt_version": PROMPT_VERSION,
        "dataset": data.spec.name,
        "concept_id": concept.concept_id,
        "concept_name": concept.name,
        "aliases": list(concept.aliases),
        "evidence": evidence,
    }
    serialized = json.dumps(hash_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    input_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return ProfileInput(data.spec.name, concept, evidence, input_hash)


def parse_json_object(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("DeepSeek response must be a JSON object")
    return value


def default_transport(
    endpoint: str, payload: dict[str, Any], headers: dict[str, str], timeout: int
) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ApiError(f"DeepSeek API HTTP {exc.code}: {body[:500]}", status=exc.code) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ApiError(f"DeepSeek API connection failed: {exc}") from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ApiError(f"DeepSeek API returned invalid JSON: {raw[:500]}") from exc
    if not isinstance(value, dict):
        raise ApiError("DeepSeek API returned a non-object response")
    return value


class DeepSeekClient:
    def __init__(
        self,
        config: ApiConfig,
        *,
        transport: Transport = default_transport,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        if not config.api_key:
            raise ProfileV2Error(
                "DEEPSEEK_API_KEY is empty; fill it in the project-root .env before generation"
            )
        self.config = config
        self.transport = transport
        self.sleeper = sleeper

    def complete_json(self, system_prompt: str, user_prompt: str) -> Completion:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        last_raw = ""
        for json_attempt in range(2):
            payload = {
                "model": self.config.model,
                "messages": messages,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            }
            response = self._request(payload)
            try:
                raw = response["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise ApiError(f"Unexpected DeepSeek response: {str(response)[:500]}") from exc
            if not isinstance(raw, str):
                raise ApiError("DeepSeek returned non-text message content")
            last_raw = raw
            try:
                return Completion(parse_json_object(raw), raw, self.config.model)
            except (json.JSONDecodeError, ValueError):
                if json_attempt == 0:
                    messages.extend(
                        [
                            {"role": "assistant", "content": raw},
                            {
                                "role": "user",
                                "content": "Return one corrected JSON object only, without Markdown.",
                            },
                        ]
                    )
        raise ResponseValidationError(
            f"DeepSeek returned invalid JSON after one repair request: {last_raw[:300]!r}"
        )

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        endpoint = f"{self.config.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        last_error: ApiError | None = None
        for attempt in range(self.config.max_retries):
            try:
                return self.transport(endpoint, payload, headers, self.config.timeout)
            except ApiError as exc:
                last_error = exc
                retryable = exc.status is None or exc.status == 429 or exc.status >= 500
                if not retryable or attempt == self.config.max_retries - 1:
                    raise
                self.sleeper(min(30.0, 2.0**attempt))
        raise last_error or ApiError("Unknown DeepSeek API failure")


GENERATOR_SYSTEM_PROMPT = """You create concise English glossary profiles from supplied metadata.
The metadata is the primary source. You may add at most one short sentence of generic background
context only when it is necessary for comprehension. General knowledge must never replace source
support. Never invent or import a named algorithm, application, dataset, metric, formula, numerical
claim, comparison, advantage, disadvantage, causal explanation, or example that is absent from the
metadata. Do not mention pages, slides, evidence IDs, excerpts, or supplied context in the profile.
Return exactly one JSON object and no Markdown."""


REVIEWER_SYSTEM_PROMPT = """You are an independent grounding reviewer for technical glossary profiles.
Treat the supplied metadata as the primary source. Check every final sentence. Preserve source-backed
meaning, remove exaggeration and mismatched claims, and rewrite unsupported details. At most one short
generic background sentence may remain when it only helps connect source-backed ideas. Background may
not introduce a named algorithm, application, dataset, metric, formula, number, performance assertion,
comparison, advantage, disadvantage, causal explanation, or new example. A shorter profile is better
than filling gaps. Return exactly one JSON object and no Markdown."""


def input_context(profile_input: ProfileInput) -> dict[str, Any]:
    return {
        "dataset": profile_input.dataset,
        "concept_name": profile_input.concept.name,
        "aliases": list(profile_input.concept.aliases),
        "metadata_v2": list(profile_input.evidence),
    }


def generator_prompt(profile_input: ProfileInput) -> str:
    context = input_context(profile_input)
    return f"""Write one standalone English profile for the concept below.

Requirements:
- Begin exactly with \"{profile_input.concept.name}:\".
- Target {TARGET_MIN_PROFILE_WORDS}-{MAX_PROFILE_WORDS} English words; never exceed {MAX_PROFILE_WORDS}.
- If metadata is limited, use fewer than {TARGET_MIN_PROFILE_WORDS} words instead of adding facts.
- Build the definition, role, mechanism, properties, and examples only when metadata supports them.
- Limited background is permitted only under the system rule and must be classified separately.
- Do not refer to metadata, evidence IDs, pages, slides, lectures, chapters, or source files.

For each factual claim, list the metadata evidence IDs that support it. A source_supported claim must
have at least one valid evidence ID. A background_context claim must have no evidence IDs, and there
may be at most one such claim.

Input context:
{json.dumps(context, ensure_ascii=False, indent=2)}

Return exactly this JSON shape:
{{
  "profile": "{profile_input.concept.name}: ...",
  "claims": [
    {{
      "claim": "one factual claim expressed in the profile",
      "evidence_ids": ["valid IDs from metadata_v2"],
      "classification": "source_supported or background_context"
    }}
  ],
  "warnings": ["uncertainties, or an empty list"]
}}"""


def reviewer_prompt(profile_input: ProfileInput, draft: Mapping[str, Any]) -> str:
    context = input_context(profile_input)
    return f"""Independently audit and, when necessary, revise the draft profile.

Rules for the final profile:
- Begin exactly with \"{profile_input.concept.name}:\".
- Target {TARGET_MIN_PROFILE_WORDS}-{MAX_PROFILE_WORDS} English words and never exceed {MAX_PROFILE_WORDS}.
- It may be shorter than {TARGET_MIN_PROFILE_WORDS} when source support is limited.
- Delete or rewrite unsupported, overstated, mismatched, or overly specific content.
- Do not mention metadata, evidence IDs, pages, slides, lectures, chapters, or source files.

Input context:
{json.dumps(context, ensure_ascii=False, indent=2)}

Draft JSON:
{json.dumps(dict(draft), ensure_ascii=False, indent=2)}

Return exactly this JSON shape. The sentence_reviews field is optional and may be a concise
summary of important grounding concerns; it does not need one item per sentence:
{{
  "final_profile": "{profile_input.concept.name}: ...",
  "sentence_reviews": [
    {{
      "sentence": "exact complete sentence from final_profile",
      "verdict": "source_supported or allowed_background",
      "evidence_ids": ["valid IDs from metadata_v2"],
      "reason": "brief grounding reason"
    }}
  ],
  "warnings": ["remaining uncertainties, or an empty list"]
}}"""


def repair_prompt(
    stage: str,
    profile_input: ProfileInput,
    previous: Mapping[str, Any],
    validation_error: str,
) -> str:
    shape = (
        "profile, claims, and warnings"
        if stage == "generation"
        else "final_profile, sentence_reviews, and warnings"
    )
    return f"""Correct the previous {stage} JSON so it passes every constraint.
Do not add factual content. Use only valid evidence IDs from the input context.
Required concept prefix: {profile_input.concept.name}:
Hard maximum: {MAX_PROFILE_WORDS} English words.
Required fields: {shape}.
Validation error: {validation_error}

Input context:
{json.dumps(input_context(profile_input), ensure_ascii=False, indent=2)}

Previous JSON:
{json.dumps(dict(previous), ensure_ascii=False, indent=2)}

Return exactly one corrected JSON object and no Markdown."""


def validate_string_list(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ResponseValidationError(f"{field_name} must be a list of strings")
    return tuple(stable_unique_strings(value))


def validate_profile_text(value: Any, profile_input: ProfileInput, field_name: str) -> str:
    if not isinstance(value, str):
        raise ResponseValidationError(f"{field_name} must be a string")
    profile = clean_text(value)
    required_prefix = f"{profile_input.concept.name}:"
    if not profile.casefold().startswith(required_prefix.casefold()):
        raise ResponseValidationError(f"{field_name} must begin with {required_prefix!r}")
    if english_word_count(profile) <= english_word_count(required_prefix):
        raise ResponseValidationError(f"{field_name} contains no explanation")
    words = english_word_count(profile)
    if words > MAX_PROFILE_WORDS:
        raise ResponseValidationError(
            f"{field_name} has {words} English words; maximum is {MAX_PROFILE_WORDS}"
        )
    if CONTEXT_DEPENDENT_RE.search(profile):
        raise ResponseValidationError(f"{field_name} contains source-dependent wording")
    return profile


def validate_evidence_ids(value: Any, profile_input: ProfileInput, field_name: str) -> tuple[str, ...]:
    evidence_ids = validate_string_list(value, field_name)
    unknown = set(evidence_ids) - profile_input.evidence_ids
    if unknown:
        raise ResponseValidationError(f"{field_name} contains unknown IDs: {sorted(unknown)}")
    return evidence_ids


def validate_draft(value: Mapping[str, Any], profile_input: ProfileInput) -> ValidatedDraft:
    profile = validate_profile_text(value.get("profile"), profile_input, "profile")
    raw_claims = value.get("claims")
    if not isinstance(raw_claims, list) or not raw_claims:
        raise ResponseValidationError("claims must be a non-empty list")
    claims: list[dict[str, Any]] = []
    background_count = 0
    for index, raw_claim in enumerate(raw_claims):
        if not isinstance(raw_claim, Mapping):
            raise ResponseValidationError(f"claims[{index}] must be an object")
        claim = clean_text(raw_claim.get("claim"), max_length=2_000)
        if not claim:
            raise ResponseValidationError(f"claims[{index}].claim must be non-empty")
        classification = clean_text(raw_claim.get("classification"), max_length=80)
        if classification not in {"source_supported", "background_context"}:
            raise ResponseValidationError(
                f"claims[{index}].classification must be source_supported or background_context"
            )
        evidence_ids = validate_evidence_ids(
            raw_claim.get("evidence_ids"), profile_input, f"claims[{index}].evidence_ids"
        )
        if classification == "source_supported" and not evidence_ids:
            raise ResponseValidationError(f"claims[{index}] needs at least one evidence ID")
        if classification == "background_context":
            background_count += 1
            if evidence_ids:
                raise ResponseValidationError(
                    f"claims[{index}] background_context must have no evidence IDs"
                )
        claims.append(
            {
                "claim": claim,
                "evidence_ids": list(evidence_ids),
                "classification": classification,
            }
        )
    if background_count > 1:
        raise ResponseValidationError("draft contains more than one background_context claim")
    warnings = validate_string_list(value.get("warnings"), "warnings")
    return ValidatedDraft(profile, tuple(claims), warnings)


def validate_review(value: Mapping[str, Any], profile_input: ProfileInput) -> ValidatedReview:
    profile = validate_profile_text(value.get("final_profile"), profile_input, "final_profile")
    raw_reviews = value.get("sentence_reviews")
    if not isinstance(raw_reviews, list):
        raw_reviews = []
    reviews: list[dict[str, Any]] = []
    for index, raw_review in enumerate(raw_reviews):
        if not isinstance(raw_review, Mapping):
            continue
        reviewed_sentence = clean_text(raw_review.get("sentence"), max_length=2_000)
        verdict = clean_text(raw_review.get("verdict"), max_length=80)
        evidence_ids_raw = raw_review.get("evidence_ids", [])
        evidence_ids = (
            tuple(item for item in evidence_ids_raw if isinstance(item, str) and item in profile_input.evidence_ids)
            if isinstance(evidence_ids_raw, list)
            else ()
        )
        reason = clean_text(raw_review.get("reason"), max_length=1_000)
        reviews.append(
            {
                "sentence": reviewed_sentence,
                "verdict": verdict,
                "evidence_ids": list(evidence_ids),
                "reason": reason or "Reviewed by the model.",
            }
        )
    raw_warnings = value.get("warnings", [])
    warnings = validate_string_list(raw_warnings if isinstance(raw_warnings, list) else [], "warnings")
    return ValidatedReview(profile, tuple(reviews), warnings)


def completion_with_constraint_repair(
    client: DeepSeekClient,
    *,
    stage: str,
    system_prompt: str,
    initial_prompt: str,
    profile_input: ProfileInput,
    validator: Callable[[Mapping[str, Any], ProfileInput], Any],
) -> tuple[Any, list[Completion]]:
    completions: list[Completion] = []
    prompt = initial_prompt
    for attempt in range(RESPONSE_REPAIR_ATTEMPTS + 1):
        completion = client.complete_json(system_prompt, prompt)
        completions.append(completion)
        try:
            return validator(completion.value, profile_input), completions
        except ResponseValidationError as exc:
            if attempt >= RESPONSE_REPAIR_ATTEMPTS:
                raise ResponseValidationError(
                    f"{stage} response failed validation after one repair request: {exc}"
                ) from exc
            prompt = repair_prompt(stage, profile_input, completion.value, str(exc))
    raise AssertionError("unreachable")


def generate_checkpoint(
    client: DeepSeekClient,
    profile_input: ProfileInput,
    stage_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if stage_callback:
        stage_callback("generating draft")
    draft, draft_completions = completion_with_constraint_repair(
        client,
        stage="generation",
        system_prompt=GENERATOR_SYSTEM_PROMPT,
        initial_prompt=generator_prompt(profile_input),
        profile_input=profile_input,
        validator=validate_draft,
    )
    draft_json = {
        "profile": draft.profile,
        "claims": list(draft.claims),
        "warnings": list(draft.warnings),
    }
    if stage_callback:
        stage_callback("reviewing grounding")
    review, review_completions = completion_with_constraint_repair(
        client,
        stage="review",
        system_prompt=REVIEWER_SYSTEM_PROMPT,
        initial_prompt=reviewer_prompt(profile_input, draft_json),
        profile_input=profile_input,
        validator=validate_review,
    )
    return {
        "version": PROGRAM_VERSION,
        "prompt_version": PROMPT_VERSION,
        "dataset": profile_input.dataset,
        "input_hash": profile_input.input_hash,
        "concept_id": profile_input.concept.concept_id,
        "concept_name": profile_input.concept.name,
        "aliases": list(profile_input.concept.aliases),
        "profile": review.final_profile,
        "word_count": english_word_count(review.final_profile),
        "model": review_completions[-1].model,
        "draft": draft_json,
        "review": {
            "final_profile": review.final_profile,
            "sentence_reviews": list(review.sentence_reviews),
            "warnings": list(review.warnings),
        },
        "raw_responses": [
            *[
                {"stage": "generation", "model": item.model, "raw": item.raw}
                for item in draft_completions
            ],
            *[
                {"stage": "review", "model": item.model, "raw": item.raw}
                for item in review_completions
            ],
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def checkpoint_path(spec: DatasetSpec, concept_id: int) -> Path:
    return spec.checkpoint_dir / f"concept_{concept_id:04d}.json"


def error_checkpoint_path(spec: DatasetSpec, concept_id: int) -> Path:
    return spec.checkpoint_dir / f"concept_{concept_id:04d}.error.json"


def read_checkpoint(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def checkpoint_is_valid(checkpoint: Mapping[str, Any] | None, profile_input: ProfileInput) -> bool:
    if not checkpoint:
        return False
    if checkpoint.get("version") != PROGRAM_VERSION:
        return False
    if checkpoint.get("prompt_version") != PROMPT_VERSION:
        return False
    if checkpoint.get("dataset") != profile_input.dataset:
        return False
    if checkpoint.get("input_hash") != profile_input.input_hash:
        return False
    if checkpoint.get("concept_id") != profile_input.concept.concept_id:
        return False
    if checkpoint.get("concept_name") != profile_input.concept.name:
        return False
    review = checkpoint.get("review")
    if not isinstance(review, Mapping):
        return False
    try:
        validated = validate_review(review, profile_input)
    except ResponseValidationError:
        return False
    if checkpoint.get("profile") != validated.final_profile:
        return False
    return isinstance(checkpoint.get("model"), str) and bool(checkpoint.get("model"))


def public_profile(checkpoint: Mapping[str, Any], concept: Concept) -> dict[str, Any]:
    return {
        "concept_id": concept.concept_id,
        "concept_name": concept.name,
        "aliases": list(concept.aliases),
        "profile": checkpoint["profile"],
        "model": checkpoint["model"],
        "prompt_version": checkpoint["prompt_version"],
    }


def checkpoint_status(data: DatasetData) -> tuple[list[int], list[int], list[int]]:
    valid: list[int] = []
    stale: list[int] = []
    missing: list[int] = []
    for concept in data.concepts:
        profile_input = build_profile_input(data, concept)
        checkpoint = read_checkpoint(checkpoint_path(data.spec, concept.concept_id))
        if checkpoint is None:
            missing.append(concept.concept_id)
        elif checkpoint_is_valid(checkpoint, profile_input):
            valid.append(concept.concept_id)
        else:
            stale.append(concept.concept_id)
    return valid, stale, missing


def determine_targets(
    datasets: Sequence[DatasetData],
    *,
    concept_id: int | None,
    force: bool,
    next_only: bool,
) -> list[tuple[DatasetData, Concept]]:
    targets: list[tuple[DatasetData, Concept]] = []
    for data in datasets:
        if concept_id is not None:
            if concept_id < 0 or concept_id >= len(data.concepts):
                raise DataValidationError(
                    f"concept ID {concept_id} is outside 0-{len(data.concepts) - 1} for {data.spec.name}"
                )
            candidates = [data.concepts[concept_id]]
        else:
            candidates = data.concepts
        for concept in candidates:
            profile_input = build_profile_input(data, concept)
            checkpoint = read_checkpoint(checkpoint_path(data.spec, concept.concept_id))
            if force or not checkpoint_is_valid(checkpoint, profile_input):
                targets.append((data, concept))
                if next_only:
                    return targets
    return targets


def publish_if_complete(data: DatasetData) -> bool:
    rows: list[str] = []
    for concept in data.concepts:
        profile_input = build_profile_input(data, concept)
        checkpoint = read_checkpoint(checkpoint_path(data.spec, concept.concept_id))
        if not checkpoint_is_valid(checkpoint, profile_input):
            return False
        rows.append(json.dumps(public_profile(checkpoint, concept), ensure_ascii=False))
    output = "\n".join(rows) + "\n"
    try:
        if data.spec.output_path.read_text(encoding="utf-8") == output:
            return True
    except FileNotFoundError:
        pass
    atomic_write_text(data.spec.output_path, output)
    return True


def dataset_report(data: DatasetData, config: ApiConfig) -> str:
    valid, stale, missing = checkpoint_status(data)
    evidence_count = sum(len(items) for items in data.evidence_by_id.values())
    return (
        f"{data.spec.name}: concepts={len(data.concepts)}, evidence={evidence_count}, "
        f"checkpoints(valid/stale/missing)={len(valid)}/{len(stale)}/{len(missing)}, "
        f"api_key={'configured' if config.api_key else 'missing'}, model={config.model}, "
        f"timeout={config.timeout}s"
    )


def selected_specs(dataset_name: str) -> list[DatasetSpec]:
    if dataset_name == "both":
        return [DATASET_SPECS["MLR"], DATASET_SPECS["DGL"]]
    return [DATASET_SPECS[dataset_name]]


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class ProgressReporter:
    """Compact terminal progress bar that also works in redirected consoles."""

    def __init__(
        self,
        total: int,
        *,
        completed: int = 0,
        skipped: int = 0,
        stream: Any = sys.stdout,
    ):
        self.total = max(1, total)
        self.completed = min(max(0, completed), self.total)
        self.skipped = max(0, skipped)
        self.stream = stream
        self.interactive = bool(getattr(stream, "isatty", lambda: False)())
        self.started = time.monotonic()
        self._rendered = False

    def _message(self, detail: str = "") -> str:
        ratio = self.completed / self.total
        filled = round(24 * ratio)
        bar = "#" * filled + "-" * (24 - filled)
        elapsed = time.monotonic() - self.started
        generated = max(0, self.completed - self.skipped)
        eta = 0.0 if not generated else elapsed / generated * (self.total - self.completed)
        text = (
            f"Progress [{bar}] {self.completed}/{self.total} ({ratio:.0%}) | "
            f"skipped {self.skipped} | elapsed {format_duration(elapsed)} | "
            f"ETA {format_duration(eta)}"
        )
        return f"{text} | {detail}" if detail else text

    def render(self, detail: str = "") -> None:
        message = self._message(detail)
        if self.interactive:
            self.stream.write("\r" + message[:220].ljust(220))
        else:
            self.stream.write(message + "\n")
        self.stream.flush()
        self._rendered = True

    def log(self, message: str) -> None:
        if self.interactive and self._rendered:
            self.stream.write("\r" + " " * 220 + "\r")
        self.stream.write(message + "\n")
        self.stream.flush()
        self._rendered = False

    def checkpointed(self, detail: str) -> None:
        self.completed = min(self.total, self.completed + 1)
        self.render(detail)

    def finish(self) -> None:
        if self.interactive and self._rendered:
            self.stream.write("\n")
            self.stream.flush()


def process_targets(
    targets: Sequence[tuple[DatasetData, Concept]], config: ApiConfig, progress: ProgressReporter
) -> None:
    client = DeepSeekClient(config)
    started = time.monotonic()
    total = len(targets)
    for index, (data, concept) in enumerate(targets, start=1):
        label = f"[{index}/{total}] {data.spec.name} concept {concept.concept_id}: {concept.name}"
        progress.log(f"{label} - starting")
        profile_input = build_profile_input(data, concept)

        def show_stage(stage: str) -> None:
            elapsed = format_duration(time.monotonic() - started)
            progress.render(f"{label} - {stage}; batch elapsed {elapsed}")

        try:
            checkpoint = generate_checkpoint(client, profile_input, show_stage)
        except Exception as exc:
            error = {
                "version": PROGRAM_VERSION,
                "prompt_version": PROMPT_VERSION,
                "dataset": data.spec.name,
                "input_hash": profile_input.input_hash,
                "concept_id": concept.concept_id,
                "concept_name": concept.name,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "failed_at": datetime.now(timezone.utc).isoformat(),
            }
            atomic_write_json(error_checkpoint_path(data.spec, concept.concept_id), error)
            progress.finish()
            print(f"{label} - FAILED: {exc}", file=sys.stderr, flush=True)
            raise
        atomic_write_json(checkpoint_path(data.spec, concept.concept_id), checkpoint)
        try:
            error_checkpoint_path(data.spec, concept.concept_id).unlink()
        except FileNotFoundError:
            pass
        elapsed = time.monotonic() - started
        average = elapsed / index
        eta = average * (total - index)
        progress.checkpointed(
            f"{label} - checkpointed; batch ETA {format_duration(eta)}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["MLR", "DGL", "both"], required=True)
    parser.add_argument("--dry-run", action="store_true", help="validate inputs without API calls or writes")
    parser.add_argument("--next", action="store_true", help="process only the next incomplete concept")
    parser.add_argument("--concept-id", type=int, help="process one zero-based concept ID")
    parser.add_argument("--force", action="store_true", help="regenerate the selected concept range")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.next and args.concept_id is not None:
        parser.error("--next and --concept-id cannot be used together")
    if args.next and args.force:
        parser.error("--next and --force cannot be used together; use --concept-id with --force")
    if args.dataset == "both" and args.concept_id is not None:
        parser.error("--concept-id requires --dataset MLR or --dataset DGL")

    started = time.monotonic()
    try:
        datasets = [load_dataset(spec) for spec in selected_specs(args.dataset)]
        config = load_api_config(ROOT_DIR)
        for data in datasets:
            print(dataset_report(data, config), flush=True)
        targets = determine_targets(
            datasets,
            concept_id=args.concept_id,
            force=args.force,
            next_only=args.next,
        )
        if args.dry_run:
            print(
                f"Dry run complete: {len(targets)} concept(s) need generation; no API calls or writes.",
                flush=True,
            )
            return 0
        if not config.api_key:
            raise ProfileV2Error(
                "DEEPSEEK_API_KEY is empty; fill it in the project-root .env before generation"
            )
        total_concepts = sum(len(data.concepts) for data in datasets)
        valid_checkpoints = sum(len(checkpoint_status(data)[0]) for data in datasets)
        pending_before_scope = total_concepts - valid_checkpoints
        if args.next or args.concept_id is not None:
            # A one-concept run reports progress for its explicitly selected scope,
            # while the resume line still describes the entire selected dataset.
            progress_total = max(1, len(targets))
            skipped = 0 if targets else progress_total
        else:
            progress_total = total_concepts
            skipped = 0 if args.force else valid_checkpoints
        progress = ProgressReporter(progress_total, completed=skipped, skipped=skipped)
        print(
            f"Resume status: total={total_concepts}, valid={valid_checkpoints}, "
            f"pending={pending_before_scope}, scheduled={len(targets)}.",
            flush=True,
        )
        progress.render("ready")
        if targets:
            process_targets(targets, config, progress)
        else:
            print("All selected checkpoints are current; no API calls needed.", flush=True)
        progress.finish()

        published: list[str] = []
        incomplete: list[str] = []
        for data in datasets:
            if publish_if_complete(data):
                published.append(str(data.spec.output_path))
            else:
                incomplete.append(data.spec.name)
        if published:
            print("Published: " + ", ".join(published), flush=True)
        if incomplete:
            print(
                "Not published because checkpoints are incomplete: " + ", ".join(incomplete),
                flush=True,
            )
        print(f"Finished in {format_duration(time.monotonic() - started)}.", flush=True)
        return 0
    except (ProfileV2Error, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
