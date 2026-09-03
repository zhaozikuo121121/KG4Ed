#!/usr/bin/env python3
"""Generate concept profiles for the MLR and DGL datasets.

The script is deliberately independent from both extraction programs.  It reads their
published CSV files, optionally supplies associated figures to Qwen through its
OpenAI-compatible chat-completions API, and stores one restartable checkpoint per
concept.  A public JSONL file is only replaced when every current concept has a valid
checkpoint.
"""

from __future__ import annotations

import argparse
import base64
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


PROGRAM_VERSION = "1.0.0"
PROMPT_VERSION = "concept_profile_assets_v1"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.8-max"
DEFAULT_TIMEOUT_SECONDS = 180
# The model is always instructed to stay within 150 words.  The slightly wider
# local acceptance limit avoids wasting an otherwise useful response for a small
# counting/tokenization discrepancy.
PROMPT_MAX_PROFILE_WORDS = 150
MAX_PROFILE_WORDS = 200
MAX_EVIDENCE = 3
MAX_FORMULAS = 3
MAX_VISUALS = 2
PROFILE_REPAIR_ATTEMPTS = 2

ROOT_DIR = Path(__file__).resolve().parent.parent
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
CONTEXT_DEPENDENT_RE = re.compile(
    r"\b(?:the|this|that)\s+(?:attached|provided|supplied|above|below)\s+"
    r"(?:image|figure|diagram|slide)\b|\b(?:on|in)\s+this\s+(?:page|slide)\b",
    re.IGNORECASE,
)


class ProfileError(RuntimeError):
    """Base error for profile generation."""


class DataValidationError(ProfileError):
    """Input files do not satisfy the expected schema."""


class ResponseValidationError(ProfileError):
    """A model response cannot be safely published."""


class ApiError(ProfileError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path
    concepts_name: str
    metadata_name: str
    formula_name: str
    graph_dir_name: str
    output_name: str

    @property
    def concepts_path(self) -> Path:
        return self.root / self.concepts_name

    @property
    def metadata_path(self) -> Path:
        return self.root / self.metadata_name

    @property
    def formula_path(self) -> Path:
        return self.root / self.formula_name

    @property
    def graph_dir(self) -> Path:
        return self.root / self.graph_dir_name

    @property
    def graph_index_path(self) -> Path:
        return self.graph_dir / "index.csv"

    @property
    def checkpoint_dir(self) -> Path:
        return self.root / ".profile_generation"

    @property
    def output_path(self) -> Path:
        return self.root / self.output_name

    @property
    def env_path(self) -> Path:
        return ROOT_DIR / ".env"


DATASET_SPECS: dict[str, DatasetSpec] = {
    "MLR": DatasetSpec(
        name="MLR",
        root=ROOT_DIR / "MLR",
        concepts_name="MLR_concepts.csv",
        metadata_name="MLR_concepts_metadata.csv",
        formula_name="MLR_formula.csv",
        graph_dir_name="MLR_graph",
        output_name="MLR_profiles.jsonl",
    ),
    "DGL": DatasetSpec(
        name="DGL",
        root=ROOT_DIR / "DGL",
        concepts_name="DGL_concepts.csv",
        metadata_name="DGL_concepts_metadata.csv",
        formula_name="DGL_formula.csv",
        graph_dir_name="DGL_graph",
        output_name="DGL_profiles.jsonl",
    ),
}


@dataclass(frozen=True)
class Concept:
    concept_id: int
    name: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class ImagePayload:
    mime_type: str
    data: bytes


@dataclass
class ProfileInput:
    dataset: str
    concept: Concept
    evidence: list[dict[str, Any]]
    formulas: list[dict[str, Any]]
    visuals: list[dict[str, Any]]
    images: list[ImagePayload]
    warnings: list[str]
    input_hash: str

    @property
    def formula_ids(self) -> set[str]:
        return {str(item["formula_id"]) for item in self.formulas}

    @property
    def visual_ids(self) -> set[str]:
        return {str(item["visual_id"]) for item in self.visuals}


@dataclass
class DatasetData:
    spec: DatasetSpec
    concepts: list[Concept]
    evidence_by_id: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    formulas_by_id: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    visuals_by_id: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    unmatched_metadata: int = 0
    unmatched_formulas: int = 0
    unmatched_visuals: int = 0


@dataclass(frozen=True)
class ApiConfig:
    api_key: str
    base_url: str
    model: str
    vision_model: str
    timeout: int
    max_retries: int


@dataclass(frozen=True)
class Completion:
    value: dict[str, Any]
    raw: str
    model: str
    used_images: bool
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValidatedProfile:
    profile: str
    used_formula_ids: tuple[str, ...]
    used_visual_ids: tuple[str, ...]
    warnings: tuple[str, ...]


Transport = Callable[[str, dict[str, Any], dict[str, str], int], dict[str, Any]]


def clean_text(value: Any, *, max_length: int = 4_000) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).replace("\x00", " ")
    return re.sub(r"\s+", " ", text).strip()[:max_length]


def normalize_key(value: Any) -> str:
    value = clean_text(value, max_length=500).casefold()
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


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
    if not path.exists():
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


def load_api_config(spec: DatasetSpec) -> ApiConfig:
    file_values = read_env_file(spec.env_path)

    def setting(name: str, default: str = "") -> str:
        if name in os.environ:
            return os.environ[name].strip()
        return file_values.get(name, default).strip()

    try:
        timeout = max(1, int(setting("QWEN_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS))))
    except ValueError:
        timeout = DEFAULT_TIMEOUT_SECONDS
    try:
        max_retries = max(1, int(setting("QWEN_MAX_RETRIES", "5")))
    except ValueError:
        max_retries = 5
    model = setting("QWEN_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
    return ApiConfig(
        api_key=setting("QWEN_API_KEY"),
        base_url=setting("QWEN_BASE_URL", DEFAULT_BASE_URL) or DEFAULT_BASE_URL,
        model=model,
        vision_model=setting("QWEN_VISION_MODEL", model) or model,
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
                previous = canonical_keys[key]
                raise DataValidationError(
                    f"Duplicate canonical concept {canonical!r} at rows {previous} and {row_number}"
                )
            canonical_keys[key] = row_number
            aliases = tuple(stable_unique_strings([canonical, *cells[1:]]))
            concepts.append(Concept(len(concepts), canonical, aliases))
    if not concepts:
        raise DataValidationError(f"No concepts found in {path}")
    return concepts


class ConceptResolver:
    def __init__(self, concepts: Sequence[Concept]):
        self.canonical: dict[str, int] = {normalize_key(item.name): item.concept_id for item in concepts}
        alias_candidates: dict[str, set[int]] = {}
        for item in concepts:
            for alias in item.aliases:
                alias_candidates.setdefault(normalize_key(alias), set()).add(item.concept_id)
        self.unique_aliases = {
            key: next(iter(ids)) for key, ids in alias_candidates.items() if key and len(ids) == 1
        }

    def resolve(self, value: Any) -> int | None:
        key = normalize_key(value)
        if not key:
            return None
        if key in self.canonical:
            return self.canonical[key]
        return self.unique_aliases.get(key)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def add_grouped(
    groups: dict[int, list[dict[str, Any]]], concept_id: int, item: dict[str, Any]
) -> None:
    groups.setdefault(concept_id, []).append(item)


def load_dataset(spec: DatasetSpec) -> DatasetData:
    concepts = load_concepts(spec.concepts_path)
    resolver = ConceptResolver(concepts)
    data = DatasetData(spec=spec, concepts=concepts)

    for order, row in enumerate(read_dict_rows(spec.metadata_path, ["concept", "evidence"])):
        concept_id = resolver.resolve(row.get("concept"))
        if concept_id is None:
            data.unmatched_metadata += 1
            continue
        evidence = clean_text(row.get("evidence"), max_length=1_200)
        if not evidence:
            continue
        item: dict[str, Any] = {
            "evidence": evidence,
            "source_type": clean_text(row.get("source_type"), max_length=80),
            "confidence": safe_float(row.get("confidence"), 1.0),
            "order": order,
        }
        for field_name in ("chapter", "chapter_title", "lecture", "pdf_file", "pdf_page", "book_page", "page_title"):
            if clean_text(row.get(field_name), max_length=300):
                item[field_name] = clean_text(row.get(field_name), max_length=300)
        add_grouped(data.evidence_by_id, concept_id, item)

    for order, row in enumerate(read_dict_rows(spec.formula_path, ["formula_id", "concept", "latex"])):
        concept_id = resolver.resolve(row.get("concept"))
        if concept_id is None:
            data.unmatched_formulas += 1
            continue
        latex = clean_text(row.get("latex"), max_length=2_000)
        if not latex:
            continue
        formula_id = clean_text(row.get("formula_id"), max_length=120)
        if not formula_id:
            formula_id = "FORMULA-" + hashlib.sha256(latex.encode("utf-8")).hexdigest()[:12].upper()
        item = {
            "formula_id": formula_id,
            "latex": latex,
            "label": clean_text(row.get("label") or row.get("equation_label"), max_length=500),
            "source_type": clean_text(row.get("source_type"), max_length=80),
            "confidence": safe_float(row.get("confidence"), 1.0),
            "order": order,
        }
        for field_name in ("chapter", "lecture", "pdf_file", "pdf_page", "book_page"):
            if clean_text(row.get(field_name), max_length=300):
                item[field_name] = clean_text(row.get(field_name), max_length=300)
        add_grouped(data.formulas_by_id, concept_id, item)

    for order, row in enumerate(read_dict_rows(spec.graph_index_path, ["concept", "file_name"])):
        concept_id = resolver.resolve(row.get("concept"))
        if concept_id is None:
            data.unmatched_visuals += 1
            continue
        file_name = clean_text(row.get("file_name"), max_length=500)
        if not file_name:
            continue
        visual_id = clean_text(row.get("visual_id"), max_length=120)
        if not visual_id:
            digest = hashlib.sha256(file_name.casefold().encode("utf-8")).hexdigest()[:12].upper()
            visual_id = f"{spec.name}-G-{digest}"
        item = {
            "visual_id": visual_id,
            "file_name": file_name,
            "path": spec.graph_dir / file_name,
            "kind": clean_text(row.get("kind") or "figure", max_length=200),
            "description": clean_text(row.get("description") or row.get("caption"), max_length=1_200),
            "figure_label": clean_text(row.get("figure_label"), max_length=300),
            "source_type": clean_text(row.get("source_type"), max_length=80),
            "confidence": safe_float(row.get("confidence"), 1.0),
            "order": order,
        }
        for field_name in ("chapter", "lecture", "pdf_file", "pdf_page", "book_page"):
            if clean_text(row.get(field_name), max_length=300):
                item[field_name] = clean_text(row.get(field_name), max_length=300)
        add_grouped(data.visuals_by_id, concept_id, item)
    return data


def distinct_ranked(
    rows: Sequence[dict[str, Any]], key_fields: Sequence[str], limit: int
) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: (-safe_float(row.get("confidence"), 1.0), int(row.get("order", 0))))
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for row in ranked:
        key = tuple(normalize_key(row.get(field, "")) for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
        if len(output) >= limit:
            break
    return output


def detect_image_mime(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def public_resource(row: Mapping[str, Any], excluded: Sequence[str] = ()) -> dict[str, Any]:
    # Confidence and source type help the model treat flattened handwriting conservatively.
    # Only local implementation details and filesystem paths stay out of the prompt/hash.
    excluded_fields = {"path", "order", *excluded}
    return {
        str(key): value
        for key, value in row.items()
        if key not in excluded_fields and value not in (None, "", [])
    }


def build_profile_input(data: DatasetData, concept: Concept, *, include_image_bytes: bool) -> ProfileInput:
    evidence_rows = distinct_ranked(
        data.evidence_by_id.get(concept.concept_id, []), ["evidence"], MAX_EVIDENCE
    )
    formula_rows = distinct_ranked(
        data.formulas_by_id.get(concept.concept_id, []), ["formula_id", "latex"], MAX_FORMULAS
    )
    visual_rows = distinct_ranked(
        data.visuals_by_id.get(concept.concept_id, []), ["visual_id", "file_name"], MAX_VISUALS
    )

    evidence = [public_resource(item) for item in evidence_rows]
    formulas = [public_resource(item) for item in formula_rows]
    visuals: list[dict[str, Any]] = []
    images: list[ImagePayload] = []
    warnings: list[str] = []
    visual_hashes: list[dict[str, Any]] = []
    for item in visual_rows:
        visual = public_resource(item)
        path = Path(item["path"])
        try:
            image_data = path.read_bytes()
        except OSError as exc:
            visual["image_available"] = False
            visual["image_slot"] = None
            warnings.append(f"Could not read {item['file_name']}: {exc}")
            visual_hashes.append({"visual_id": item["visual_id"], "sha256": "missing"})
        else:
            mime_type = detect_image_mime(image_data)
            digest = sha256_bytes(image_data)
            visual_hashes.append({"visual_id": item["visual_id"], "sha256": digest})
            if mime_type is None:
                visual["image_available"] = False
                visual["image_slot"] = None
                warnings.append(f"Unsupported or invalid image file: {item['file_name']}")
            else:
                visual["image_available"] = True
                visual["image_slot"] = len(images) + 1
                if include_image_bytes:
                    images.append(ImagePayload(mime_type, image_data))
                else:
                    # Preserve the same deterministic slot numbers while checking hashes.
                    images.append(ImagePayload(mime_type, b""))
        visuals.append(visual)

    hash_payload = {
        "program_version": PROGRAM_VERSION,
        "prompt_version": PROMPT_VERSION,
        "dataset": data.spec.name,
        "concept_id": concept.concept_id,
        "concept_name": concept.name,
        "aliases": list(concept.aliases),
        "evidence": evidence,
        "formulas": formulas,
        "visuals": visuals,
        "visual_hashes": visual_hashes,
    }
    serialized = json.dumps(hash_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    input_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    if not include_image_bytes:
        images = []
    return ProfileInput(
        dataset=data.spec.name,
        concept=concept,
        evidence=evidence,
        formulas=formulas,
        visuals=visuals,
        images=images,
        warnings=warnings,
        input_hash=input_hash,
    )


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
        raise ValueError("Expected one JSON object")
    return value


def extract_response_text(response: Mapping[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]  # type: ignore[index]
    except (KeyError, IndexError, TypeError) as exc:
        raise ApiError(f"Unexpected Qwen response shape: {str(response)[:500]}") from exc
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) for item in content if isinstance(item, Mapping)
        )
    raise ApiError(f"Unexpected Qwen message content: {type(content).__name__}")


def make_user_content(prompt: str, images: Sequence[ImagePayload]) -> Any:
    if not images:
        return prompt
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image in images:
        encoded = base64.b64encode(image.data).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image.mime_type};base64,{encoded}"},
            }
        )
    return content


class QwenClient:
    def __init__(
        self,
        config: ApiConfig,
        *,
        transport: Transport | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        if not config.api_key:
            raise ProfileError("QWEN_API_KEY is empty; fill the project-root .env file")
        self.config = config
        self.transport = transport or self._urllib_transport
        self.sleeper = sleeper

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        images: Sequence[ImagePayload] = (),
    ) -> Completion:
        active_images = list(images)
        model = self.config.vision_model if active_images else self.config.model
        warnings: list[str] = []
        last_raw = ""
        repair_attempt = 0
        while repair_attempt < 2:
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": make_user_content(user_prompt, active_images)},
            ]
            if repair_attempt:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was not valid JSON. Return one corrected JSON "
                            "object only, without Markdown fences or commentary."
                        ),
                    }
                )
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            }
            try:
                response = self._post(payload)
            except ApiError as exc:
                if active_images and exc.status in {400, 404, 415, 422}:
                    warnings.append(
                        "The vision endpoint rejected image input; generated from captions, formulas, "
                        "and text evidence instead."
                    )
                    active_images = []
                    model = self.config.model
                    repair_attempt = 0
                    continue
                raise
            last_raw = extract_response_text(response)
            try:
                value = parse_json_object(last_raw)
            except (json.JSONDecodeError, ValueError):
                repair_attempt += 1
                continue
            return Completion(value, last_raw, model, bool(active_images), tuple(warnings))
        raise ResponseValidationError(
            f"Qwen returned invalid JSON after repair retry: {last_raw[:300]!r}"
        )

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
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
                retryable = exc.status is None or exc.status == 429 or bool(exc.status and exc.status >= 500)
                if not retryable or attempt == self.config.max_retries - 1:
                    raise
                self.sleeper(min(30.0, 2.0**attempt))
        raise last_error or ApiError("Unknown Qwen API failure")

    @staticmethod
    def _urllib_transport(
        endpoint: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: int,
    ) -> dict[str, Any]:
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ApiError(f"Qwen API HTTP {exc.code}: {body[:500]}", status=exc.code) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ApiError(f"Qwen API connection failed: {exc}") from exc


PROFILE_SYSTEM_PROMPT = """You write precise English glossary profiles for technical concepts.
Use only the supplied concept aliases, source evidence, formulas, figure descriptions, and attached
images. Define the concept, explain its role or typical use, and mention only directly useful related
ideas. A formula or figure should be explained only when it materially improves understanding.
Never invent details, never guess unclear handwriting, and never refer to an attached image, a page,
a slide, a resource ID, or the supplied context in the finished profile. Keep the result standalone.
Return exactly one JSON object and no Markdown."""


def dataset_focus(dataset: str) -> str:
    if dataset == "DGL":
        return (
            "The source is an annotated graph deep learning course covering graph theory, knowledge "
            "graphs, graph representation learning, graph neural networks, and related methods."
        )
    return (
        "The source is a machine-learning textbook covering mathematical foundations, models, "
        "optimization, supervised and unsupervised learning, and related methods."
    )


def profile_user_prompt(profile_input: ProfileInput) -> str:
    context = {
        "dataset": profile_input.dataset,
        "concept_name": profile_input.concept.name,
        "aliases": list(profile_input.concept.aliases),
        "source_evidence": profile_input.evidence,
        "associated_formulas": profile_input.formulas,
        "associated_visuals": profile_input.visuals,
    }
    return f"""{dataset_focus(profile_input.dataset)}

Write one profile for the concept below. The profile must:
- begin exactly with \"{profile_input.concept.name}:\";
- target 80-130 English words and never exceed {PROMPT_MAX_PROFILE_WORDS} English words, including the name;
- be useful to a learner and remain understandable without any source page;
- explain supplied formulas or visuals only if they clarify the concept;
- avoid unsupported examples, generic filler, citations, page references, and resource IDs.

Input context:
{json.dumps(context, ensure_ascii=False, indent=2)}

Return exactly this JSON shape:
{{
  "profile": "{profile_input.concept.name}: ...",
  "used_formula_ids": ["only IDs actually used in the prose"],
  "used_visual_ids": ["only IDs actually used in the prose"],
  "warnings": ["uncertainties worth recording, or an empty list"]
}}
All three list fields are required. IDs must come from the input context."""


def repair_user_prompt(
    profile_input: ProfileInput,
    previous: Mapping[str, Any],
    validation_error: str,
) -> str:
    return f"""Revise the previous profile so it passes every constraint. Do not add new factual claims.
Required prefix: {profile_input.concept.name}:
Maximum length: {PROMPT_MAX_PROFILE_WORDS} English words.
Allowed formula IDs: {json.dumps(sorted(profile_input.formula_ids))}
Allowed visual IDs: {json.dumps(sorted(profile_input.visual_ids))}
Validation errors: {validation_error}

Previous JSON:
{json.dumps(dict(previous), ensure_ascii=False, indent=2)}

Return exactly one JSON object with profile, used_formula_ids, used_visual_ids, and warnings.
The finished profile must be standalone and must not mention an attached image, page, slide, or ID."""


def validate_string_list(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ResponseValidationError(f"{field_name} must be a list of strings")
    return tuple(stable_unique_strings(value))


def validate_profile_response(
    value: Mapping[str, Any], profile_input: ProfileInput
) -> ValidatedProfile:
    profile_value = value.get("profile")
    if not isinstance(profile_value, str):
        raise ResponseValidationError("profile must be a string")
    profile = clean_text(profile_value, max_length=20_000)
    required_prefix = f"{profile_input.concept.name}:"
    if not profile.casefold().startswith(required_prefix.casefold()):
        raise ResponseValidationError(f"profile must begin with {required_prefix!r}")
    word_count = english_word_count(profile)
    if word_count > MAX_PROFILE_WORDS:
        raise ResponseValidationError(
            f"profile has {word_count} English words; maximum is {MAX_PROFILE_WORDS}"
        )
    if word_count <= english_word_count(required_prefix):
        raise ResponseValidationError("profile contains no explanation")
    if CONTEXT_DEPENDENT_RE.search(profile):
        raise ResponseValidationError("profile contains a page-, slide-, or attachment-dependent phrase")

    formula_ids = validate_string_list(value.get("used_formula_ids"), "used_formula_ids")
    visual_ids = validate_string_list(value.get("used_visual_ids"), "used_visual_ids")
    warnings = validate_string_list(value.get("warnings"), "warnings")
    invalid_formulas = set(formula_ids) - profile_input.formula_ids
    invalid_visuals = set(visual_ids) - profile_input.visual_ids
    if invalid_formulas:
        raise ResponseValidationError(
            f"used_formula_ids contains unknown IDs: {sorted(invalid_formulas)}"
        )
    if invalid_visuals:
        raise ResponseValidationError(
            f"used_visual_ids contains unknown IDs: {sorted(invalid_visuals)}"
        )
    return ValidatedProfile(profile, formula_ids, visual_ids, warnings)


def generate_checkpoint(client: QwenClient, profile_input: ProfileInput) -> dict[str, Any]:
    prompt = profile_user_prompt(profile_input)
    completion = client.complete_json(PROFILE_SYSTEM_PROMPT, prompt, images=profile_input.images)
    response_history: list[dict[str, Any]] = []
    all_warnings = list(profile_input.warnings)
    final_completion = completion
    for attempt in range(PROFILE_REPAIR_ATTEMPTS + 1):
        response_history.append(
            {
                "stage": "initial" if attempt == 0 else f"content_repair_{attempt}",
                "model": final_completion.model,
                "used_images": final_completion.used_images,
                "raw": final_completion.raw,
            }
        )
        all_warnings.extend(final_completion.warnings)
        try:
            validated = validate_profile_response(final_completion.value, profile_input)
            break
        except ResponseValidationError as exc:
            if attempt >= PROFILE_REPAIR_ATTEMPTS:
                raise ResponseValidationError(
                    f"Profile for {profile_input.concept.name!r} failed validation after repairs: {exc}"
                ) from exc
            repair_prompt = repair_user_prompt(profile_input, final_completion.value, str(exc))
            final_completion = client.complete_json(PROFILE_SYSTEM_PROMPT, repair_prompt)
    else:  # pragma: no cover - loop always breaks or raises
        raise AssertionError("unreachable")

    all_warnings.extend(validated.warnings)
    warnings = stable_unique_strings(all_warnings)
    return {
        "version": PROGRAM_VERSION,
        "prompt_version": PROMPT_VERSION,
        "dataset": profile_input.dataset,
        "input_hash": profile_input.input_hash,
        "concept_id": profile_input.concept.concept_id,
        "concept_name": profile_input.concept.name,
        "aliases": list(profile_input.concept.aliases),
        "profile": validated.profile,
        "word_count": english_word_count(validated.profile),
        "model": final_completion.model,
        "used_formula_ids": list(validated.used_formula_ids),
        "used_visual_ids": list(validated.used_visual_ids),
        "warnings": warnings,
        "raw_responses": response_history,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def checkpoint_path(spec: DatasetSpec, concept_id: int) -> Path:
    return spec.checkpoint_dir / f"concept_{concept_id:04d}.json"


def read_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def checkpoint_is_valid(checkpoint: Mapping[str, Any] | None, profile_input: ProfileInput) -> bool:
    if not checkpoint:
        return False
    if checkpoint.get("version") != PROGRAM_VERSION or checkpoint.get("prompt_version") != PROMPT_VERSION:
        return False
    if checkpoint.get("dataset") != profile_input.dataset:
        return False
    if checkpoint.get("input_hash") != profile_input.input_hash:
        return False
    if checkpoint.get("concept_id") != profile_input.concept.concept_id:
        return False
    if checkpoint.get("concept_name") != profile_input.concept.name:
        return False
    try:
        validate_profile_response(checkpoint, profile_input)
    except ResponseValidationError:
        return False
    return isinstance(checkpoint.get("model"), str) and bool(checkpoint.get("model"))


def public_profile(checkpoint: Mapping[str, Any], concept: Concept) -> dict[str, Any]:
    return {
        "concept_id": concept.concept_id,
        "concept_name": concept.name,
        "aliases": list(concept.aliases),
        "profile": checkpoint["profile"],
        "model": checkpoint["model"],
        "prompt_version": PROMPT_VERSION,
    }


def collect_checkpoint_status(data: DatasetData) -> tuple[list[int], list[int], list[int]]:
    valid: list[int] = []
    stale: list[int] = []
    missing: list[int] = []
    for concept in data.concepts:
        profile_input = build_profile_input(data, concept, include_image_bytes=False)
        path = checkpoint_path(data.spec, concept.concept_id)
        checkpoint = read_checkpoint(path)
        if checkpoint is None:
            missing.append(concept.concept_id)
        elif checkpoint_is_valid(checkpoint, profile_input):
            valid.append(concept.concept_id)
        else:
            stale.append(concept.concept_id)
    return valid, stale, missing


def publish_if_complete(data: DatasetData) -> bool:
    lines: list[str] = []
    for concept in data.concepts:
        profile_input = build_profile_input(data, concept, include_image_bytes=False)
        checkpoint = read_checkpoint(checkpoint_path(data.spec, concept.concept_id))
        if not checkpoint_is_valid(checkpoint, profile_input):
            return False
        lines.append(json.dumps(public_profile(checkpoint, concept), ensure_ascii=False))
    atomic_write_text(data.spec.output_path, "\n".join(lines) + "\n")
    return True


def dataset_report(data: DatasetData, config: ApiConfig) -> str:
    valid, stale, missing = collect_checkpoint_status(data)
    formula_relations = sum(len(items) for items in data.formulas_by_id.values())
    visual_relations = sum(len(items) for items in data.visuals_by_id.values())
    evidence_relations = sum(len(items) for items in data.evidence_by_id.values())
    missing_images = 0
    for rows in data.visuals_by_id.values():
        missing_images += sum(not Path(row["path"]).is_file() for row in rows)
    return (
        f"{data.spec.name}: concepts={len(data.concepts)}, evidence={evidence_relations}, "
        f"formula_relations={formula_relations}, visual_relations={visual_relations}, "
        f"checkpoints(valid/stale/missing)={len(valid)}/{len(stale)}/{len(missing)}, "
        f"missing_images={missing_images}, api_key={'configured' if config.api_key else 'missing'}, "
        f"model={config.model}, vision_model={config.vision_model}, "
        f"unmatched_rows={data.unmatched_metadata + data.unmatched_formulas + data.unmatched_visuals}"
    )


def selected_specs(dataset_name: str) -> list[DatasetSpec]:
    if dataset_name == "all":
        return [DATASET_SPECS["MLR"], DATASET_SPECS["DGL"]]
    return [DATASET_SPECS[dataset_name]]


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
            profile_input = build_profile_input(data, concept, include_image_bytes=False)
            checkpoint = read_checkpoint(checkpoint_path(data.spec, concept.concept_id))
            if force or not checkpoint_is_valid(checkpoint, profile_input):
                targets.append((data, concept))
                if next_only:
                    return targets
    return targets


def process_targets(
    targets: Sequence[tuple[DatasetData, Concept]], configs: Mapping[str, ApiConfig]
) -> None:
    clients: dict[str, QwenClient] = {}
    total = len(targets)
    for index, (data, concept) in enumerate(targets, start=1):
        config = configs[data.spec.name]
        if not config.api_key:
            raise ProfileError(f"QWEN_API_KEY is empty in {data.spec.env_path}")
        if data.spec.name not in clients:
            clients[data.spec.name] = QwenClient(config)
        client = clients[data.spec.name]
        print(
            f"[{index}/{total}] {data.spec.name} concept {concept.concept_id}: {concept.name}",
            flush=True,
        )
        profile_input = build_profile_input(data, concept, include_image_bytes=True)
        checkpoint = generate_checkpoint(client, profile_input)
        atomic_write_json(checkpoint_path(data.spec, concept.concept_id), checkpoint)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate asset-aware Qwen profiles for MLR and DGL concepts."
    )
    parser.add_argument("--dataset", choices=["MLR", "DGL", "all"], required=True)
    parser.add_argument("--next", action="store_true", help="Process only the next incomplete concept")
    parser.add_argument("--concept-id", type=int, help="Process one zero-based concept ID")
    parser.add_argument("--force", action="store_true", help="Regenerate the selected concept range")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without calling Qwen")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.next and args.concept_id is not None:
        parser.error("--next and --concept-id cannot be used together")
    if args.next and args.force:
        parser.error("--next and --force cannot be used together; use --concept-id with --force")
    if args.dataset == "all" and args.concept_id is not None:
        parser.error("--concept-id requires --dataset MLR or --dataset DGL")

    try:
        datasets = [load_dataset(spec) for spec in selected_specs(args.dataset)]
        configs = {data.spec.name: load_api_config(data.spec) for data in datasets}
        for data in datasets:
            print(dataset_report(data, configs[data.spec.name]))
        targets = determine_targets(
            datasets,
            concept_id=args.concept_id,
            force=args.force,
            next_only=args.next,
        )
        if args.dry_run:
            if targets:
                names = ", ".join(
                    f"{data.spec.name}:{concept.concept_id} {concept.name}" for data, concept in targets[:5]
                )
                suffix = " ..." if len(targets) > 5 else ""
                print(f"Dry run: {len(targets)} concept(s) would be processed: {names}{suffix}")
            else:
                print("Dry run: no concepts need processing.")
            return 0

        if targets:
            process_targets(targets, configs)
        else:
            print("No concepts need generation.")
        for data in datasets:
            if publish_if_complete(data):
                print(f"Published {len(data.concepts)} profiles to {data.spec.output_path}")
            else:
                print(
                    f"{data.spec.name} is not complete; existing public JSONL was left unchanged.",
                    file=sys.stderr,
                )
        return 0
    except (ProfileError, OSError, csv.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
