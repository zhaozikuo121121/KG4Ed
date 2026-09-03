#!/usr/bin/env python3
"""Extract graph-learning concepts and assets from annotated DGL lecture PDFs.

One invocation processes one annotated lecture PDF. Each physical page is an
independent vision-LLM unit and is committed separately so interrupted runs can
resume without repeating completed pages.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyMuPDF is required. Install dependencies with: "
        "python -m pip install -r requirements.txt"
    ) from exc


PROGRAM_VERSION = "1.0.0"
EXPECTED_PAGE_WIDTH = 792.0
EXPECTED_PAGE_HEIGHT = 629.0
PAGE_SIZE_TOLERANCE = 1.0
STATE_DIR_NAME = ".dgl_extraction"
BODY_BOX = (38.0 / 792.0, 55.0 / 629.0, 774.0 / 792.0, 580.0 / 629.0)
OVERVIEW_SCALE = 2.0
TILE_SCALE = 3.0
ASSET_SCALE = 3.0
DEFAULT_TIMEOUT_SECONDS = 180
MIN_VISUAL_AREA = 0.002
MAX_VISUAL_AREA = 0.68

SOURCE_TYPES = {"printed", "handwritten", "mixed"}
VISUAL_KINDS = {
    "diagram",
    "plot",
    "matrix",
    "architecture",
    "graph example",
    "table",
    "other",
}
CONCEPT_CONFIDENCE = {"printed": 0.70, "mixed": 0.80, "handwritten": 0.90}
FORMULA_CONFIDENCE = {"printed": 0.82, "mixed": 0.87, "handwritten": 0.92}
VISUAL_CONFIDENCE = {"printed": 0.78, "mixed": 0.82, "handwritten": 0.90}


@dataclass(frozen=True)
class Lecture:
    number: int
    folder: str
    pdf_name: str
    expected_pages: int

    @property
    def relative_path(self) -> Path:
        return Path(self.folder) / self.pdf_name


LECTURES: tuple[Lecture, ...] = (
    Lecture(1, "DGL_Lecture_1", "DGL_Lecture_1_Annotated.pdf", 3),
    Lecture(2, "DGL_Lecture_2", "DGL_Lecture_2_Annotated.pdf", 5),
    Lecture(3, "DGL_Lecture_3", "DGL_Lecture_3_Annotated.pdf", 4),
    Lecture(4, "DGL_Lecture_4", "DGL_Lecture_4_Annotated.pdf", 4),
    Lecture(5, "DGL_Lecture_5", "DGL_Lecture_5_Annotated.pdf", 3),
    Lecture(6, "DGL_Lecture_6", "DGL_Lecture_6_Annotated.pdf", 4),
)


@dataclass(frozen=True)
class NormalizedBox:
    """A box in the DGL body coordinate system, normalized to 0..1000."""

    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def from_value(cls, value: Any) -> "NormalizedBox":
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ValueError("bbox must contain exactly four coordinates")
        values = [float(item) for item in value]
        if not all(math.isfinite(item) for item in values):
            raise ValueError("bbox contains a non-finite coordinate")
        box = cls(*values)
        box.validate()
        return box

    def validate(self) -> None:
        if not (0 <= self.x0 < self.x1 <= 1000 and 0 <= self.y0 < self.y1 <= 1000):
            raise ValueError(f"invalid normalized bbox: {self.as_list()}")

    @property
    def area_fraction(self) -> float:
        return ((self.x1 - self.x0) / 1000.0) * ((self.y1 - self.y0) / 1000.0)

    def as_list(self) -> list[float]:
        return [round(self.x0, 2), round(self.y0, 2), round(self.x1, 2), round(self.y1, 2)]

    def iou(self, other: "NormalizedBox") -> float:
        ix0, iy0 = max(self.x0, other.x0), max(self.y0, other.y0)
        ix1, iy1 = min(self.x1, other.x1), min(self.y1, other.y1)
        intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        left_area = (self.x1 - self.x0) * (self.y1 - self.y0)
        right_area = (other.x1 - other.x0) * (other.y1 - other.y0)
        union = left_area + right_area - intersection
        return intersection / union if union else 0.0


@dataclass(frozen=True)
class Tile:
    name: str
    x0: float
    y0: float
    x1: float
    y1: float

    def local_to_body(self, box: NormalizedBox) -> NormalizedBox:
        width, height = self.x1 - self.x0, self.y1 - self.y0
        return NormalizedBox(
            (self.x0 + (box.x0 / 1000.0) * width) * 1000.0,
            (self.y0 + (box.y0 / 1000.0) * height) * 1000.0,
            (self.x0 + (box.x1 / 1000.0) * width) * 1000.0,
            (self.y0 + (box.y1 / 1000.0) * height) * 1000.0,
        )


TILES: tuple[Tile, ...] = (
    Tile("top-left", 0.00, 0.00, 0.55, 0.55),
    Tile("top-right", 0.45, 0.00, 1.00, 0.55),
    Tile("bottom-left", 0.00, 0.45, 0.55, 1.00),
    Tile("bottom-right", 0.45, 0.45, 1.00, 1.00),
)


@dataclass
class ConceptCandidate:
    canonical: str
    aliases: list[str]
    evidence: str
    source_type: str
    confidence: float
    bbox: list[float]

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "ConceptCandidate":
        canonical = clean_concept_name(data.get("canonical") or data.get("concept") or "")
        raw_aliases = data.get("aliases") or []
        if isinstance(raw_aliases, str):
            raw_aliases = [raw_aliases]
        aliases = stable_unique(
            [clean_concept_name(str(item)) for item in raw_aliases if str(item).strip()],
            key=normalize_key,
        )
        aliases = [item for item in aliases if normalize_key(item) != normalize_key(canonical)]
        source_type = normalize_source_type(data.get("source_type"))
        candidate = cls(
            canonical=canonical,
            aliases=aliases,
            evidence=clean_text(str(data.get("evidence") or ""), max_length=800),
            source_type=source_type,
            confidence=safe_float(data.get("confidence"), 0.0),
            bbox=NormalizedBox.from_value(data.get("bbox")).as_list(),
        )
        candidate.validate()
        return candidate

    def validate(self) -> None:
        if not valid_concept_name(self.canonical):
            raise ValueError(f"invalid concept name: {self.canonical!r}")
        if not self.evidence:
            raise ValueError("concept evidence is required")
        if self.source_type not in SOURCE_TYPES:
            raise ValueError(f"invalid source type: {self.source_type}")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        NormalizedBox.from_value(self.bbox)


@dataclass
class FormulaCandidate:
    latex: str
    related_concepts: list[str]
    source_type: str
    confidence: float
    bbox: list[float]
    label: str = ""

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "FormulaCandidate":
        raw_related = data.get("related_concepts") or []
        if isinstance(raw_related, str):
            raw_related = [raw_related]
        candidate = cls(
            latex=clean_latex(str(data.get("latex") or "")),
            related_concepts=stable_unique(
                [clean_concept_name(str(item)) for item in raw_related if str(item).strip()],
                key=normalize_key,
            ),
            source_type=normalize_source_type(data.get("source_type")),
            confidence=safe_float(data.get("confidence"), 0.0),
            bbox=NormalizedBox.from_value(data.get("bbox")).as_list(),
            label=clean_text(str(data.get("label") or ""), max_length=100),
        )
        candidate.validate()
        return candidate

    def validate(self) -> None:
        if not valid_latex(self.latex):
            raise ValueError(f"invalid or uncertain LaTeX: {self.latex!r}")
        if not self.related_concepts:
            raise ValueError("formula must be related to at least one concept")
        if self.source_type not in SOURCE_TYPES or not 0 <= self.confidence <= 1:
            raise ValueError("invalid formula source/confidence")
        NormalizedBox.from_value(self.bbox)


@dataclass
class VisualCandidate:
    kind: str
    description: str
    related_concepts: list[str]
    source_type: str
    confidence: float
    bbox: list[float]

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "VisualCandidate":
        raw_related = data.get("related_concepts") or []
        if isinstance(raw_related, str):
            raw_related = [raw_related]
        kind = clean_text(str(data.get("kind") or "other"), max_length=40).casefold()
        if kind not in VISUAL_KINDS:
            kind = "other"
        candidate = cls(
            kind=kind,
            description=clean_text(str(data.get("description") or ""), max_length=600),
            related_concepts=stable_unique(
                [clean_concept_name(str(item)) for item in raw_related if str(item).strip()],
                key=normalize_key,
            ),
            source_type=normalize_source_type(data.get("source_type")),
            confidence=safe_float(data.get("confidence"), 0.0),
            bbox=NormalizedBox.from_value(data.get("bbox")).as_list(),
        )
        candidate.validate()
        return candidate

    def validate(self) -> None:
        if not self.description or not self.related_concepts:
            raise ValueError("visual needs a description and at least one related concept")
        if self.source_type not in SOURCE_TYPES or not 0 <= self.confidence <= 1:
            raise ValueError("invalid visual source/confidence")
        box = NormalizedBox.from_value(self.bbox)
        if not MIN_VISUAL_AREA <= box.area_fraction <= MAX_VISUAL_AREA:
            raise ValueError(f"visual bbox has unsafe area: {box.area_fraction:.4f}")


@dataclass
class PageResult:
    version: str
    lecture: int
    pdf_file: str
    pdf_page: int
    page_title: str
    summary: str
    key_topics: list[str]
    model: str
    completed_at: str
    body_bbox: list[float]
    concepts: list[dict[str, Any]]
    formulas: list[dict[str, Any]]
    visuals: list[dict[str, Any]]
    uncertain_handwriting: list[dict[str, Any]] = field(default_factory=list)
    raw_responses: list[dict[str, Any]] = field(default_factory=list)
    dedupe_decisions: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_path(cls, path: Path) -> "PageResult":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


class ExtractionError(RuntimeError):
    """Raised when a page cannot be safely extracted or committed."""


class ApiError(ExtractionError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def clean_text(value: str, max_length: int = 1000) -> str:
    value = unicodedata.normalize("NFKC", value).replace("\x00", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value[:max_length]


def clean_concept_name(value: Any) -> str:
    value = unicodedata.normalize("NFKC", str(value))
    value = value.replace("–", "-").replace("—", "-").replace("−", "-")
    value = value.casefold().strip().strip(".,;:!?\"'`()[]{}")
    value = re.sub(r"\s*-\s*", "-", value)
    return re.sub(r"\s+", " ", value)[:120]


def normalize_key(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", " ", clean_concept_name(value))
    tokens: list[str] = []
    for token in value.split():
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith("es") and not token.endswith("ses"):
            token = token[:-2]
        elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.append(token)
    return " ".join(tokens)


def fuzzy_equivalent(left: str, right: str) -> bool:
    left_key, right_key = normalize_key(left), normalize_key(right)
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True
    if re.findall(r"\d+", left_key) != re.findall(r"\d+", right_key):
        return False
    left_tokens, right_tokens = set(left_key.split()), set(right_key.split())
    union = left_tokens | right_tokens
    jaccard = len(left_tokens & right_tokens) / len(union) if union else 0.0
    ratio = SequenceMatcher(None, left_key, right_key).ratio()
    if ratio >= 0.90 and jaccard >= 0.80:
        return True
    left_ordered, right_ordered = left_key.split(), right_key.split()
    if len(left_ordered) != len(right_ordered):
        return False
    ratios = [
        SequenceMatcher(None, left_token, right_token).ratio()
        for left_token, right_token in zip(left_ordered, right_ordered)
    ]
    return bool(ratios) and min(ratios) >= 0.85 and sum(ratios) / len(ratios) >= 0.92


def valid_concept_name(value: str) -> bool:
    if not value or len(value) > 100 or not re.search(r"[a-z]", value):
        return False
    if len(value.split()) > 10:
        return False
    blocked = {
        "dgl",
        "dgl lecture",
        "basira lab",
        "youtube",
        "github",
        "special thanks",
        "recall box",
        "papers with code",
    }
    key = normalize_key(value)
    return not any(normalize_key(item) in key for item in blocked)


def clean_latex(value: str) -> str:
    return value.strip().strip("$").strip()


def latex_key(value: str) -> str:
    return re.sub(r"\s+", "", clean_latex(value))


def valid_latex(value: str) -> bool:
    if not value or len(value) > 1000:
        return False
    lowered = value.casefold()
    if any(marker in lowered for marker in ["?", "illegible", "unclear", "unknown", "cannot read"]):
        return False
    pairs = [("{", "}"), ("[", "]"), ("(", ")")]
    return all(value.count(left) == value.count(right) for left, right in pairs)


def normalize_source_type(value: Any) -> str:
    return str(value or "").strip().casefold()


def stable_unique(items: Iterable[Any], key: Callable[[Any], Any]) -> list[Any]:
    seen: set[Any] = set()
    output: list[Any] = []
    for item in items:
        marker = key(item)
        if marker in seen:
            continue
        seen.add(marker)
        output.append(item)
    return output


def slugify(value: str, max_length: int = 60) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return (value or "item")[:max_length].rstrip("_")


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


def atomic_write_csv(path: Path, rows: Iterable[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"").strip("'")
        if key:
            os.environ.setdefault(key, value)


class AnnotatedLecturePdf:
    def __init__(self, root: Path, lecture: Lecture):
        self.root = Path(root)
        self.lecture = lecture
        self.path = self.root / lecture.relative_path
        if "_Annotated.pdf" not in self.path.name or "_Clean.pdf" in self.path.name:
            raise ExtractionError(f"Refusing non-Annotated PDF: {self.path}")
        if not self.path.exists():
            raise ExtractionError(f"Annotated PDF not found: {self.path}")
        self.document = fitz.open(self.path)
        self.validate()

    def __enter__(self) -> "AnnotatedLecturePdf":
        return self

    def __exit__(self, *_: Any) -> None:
        self.document.close()

    def validate(self) -> None:
        if len(self.document) != self.lecture.expected_pages:
            raise ExtractionError(
                f"Lecture {self.lecture.number} expected {self.lecture.expected_pages} pages, "
                f"found {len(self.document)}"
            )
        for index, page in enumerate(self.document, start=1):
            if (
                abs(page.rect.width - EXPECTED_PAGE_WIDTH) > PAGE_SIZE_TOLERANCE
                or abs(page.rect.height - EXPECTED_PAGE_HEIGHT) > PAGE_SIZE_TOLERANCE
            ):
                raise ExtractionError(
                    f"Lecture {self.lecture.number} page {index} has unexpected size "
                    f"{page.rect.width:.1f}x{page.rect.height:.1f}"
                )

    def body_rect(self, pdf_page: int) -> Any:
        page = self.document[pdf_page - 1]
        x0, y0, x1, y1 = BODY_BOX
        return fitz.Rect(
            page.rect.x0 + x0 * page.rect.width,
            page.rect.y0 + y0 * page.rect.height,
            page.rect.x0 + x1 * page.rect.width,
            page.rect.y0 + y1 * page.rect.height,
        )

    def tile_rect(self, pdf_page: int, tile: Tile) -> Any:
        body = self.body_rect(pdf_page)
        return fitz.Rect(
            body.x0 + tile.x0 * body.width,
            body.y0 + tile.y0 * body.height,
            body.x0 + tile.x1 * body.width,
            body.y0 + tile.y1 * body.height,
        )

    def body_box_to_page_rect(
        self, pdf_page: int, box: NormalizedBox, padding: float = 8.0
    ) -> Any:
        body = self.body_rect(pdf_page)
        rect = fitz.Rect(
            body.x0 + box.x0 / 1000.0 * body.width,
            body.y0 + box.y0 / 1000.0 * body.height,
            body.x0 + box.x1 / 1000.0 * body.width,
            body.y0 + box.y1 / 1000.0 * body.height,
        )
        padded = fitz.Rect(rect.x0 - padding, rect.y0 - padding, rect.x1 + padding, rect.y1 + padding)
        return padded & body

    def body_text(self, pdf_page: int) -> str:
        page = self.document[pdf_page - 1]
        return clean_text(page.get_text("text", clip=self.body_rect(pdf_page), sort=True), 20_000)

    def tile_text(self, pdf_page: int, tile: Tile) -> str:
        page = self.document[pdf_page - 1]
        return clean_text(page.get_text("text", clip=self.tile_rect(pdf_page, tile), sort=True), 8_000)

    def render_body_jpeg(self, pdf_page: int, scale: float = OVERVIEW_SCALE) -> bytes:
        page = self.document[pdf_page - 1]
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(scale, scale),
            clip=self.body_rect(pdf_page),
            alpha=False,
            annots=True,
        )
        return pixmap.tobytes("jpeg", jpg_quality=84)

    def render_tile_jpeg(self, pdf_page: int, tile: Tile, scale: float = TILE_SCALE) -> bytes:
        page = self.document[pdf_page - 1]
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(scale, scale),
            clip=self.tile_rect(pdf_page, tile),
            alpha=False,
            annots=True,
        )
        return pixmap.tobytes("jpeg", jpg_quality=88)

    def crop_visual_png(self, pdf_page: int, box: NormalizedBox) -> bytes:
        if not MIN_VISUAL_AREA <= box.area_fraction <= MAX_VISUAL_AREA:
            raise ExtractionError(f"unsafe visual bbox area: {box.area_fraction:.4f}")
        page = self.document[pdf_page - 1]
        clip = self.body_box_to_page_rect(pdf_page, box, padding=8.0)
        if clip.is_empty or clip.get_area() < 600:
            raise ExtractionError("visual crop is empty or too small")
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(ASSET_SCALE, ASSET_SCALE),
            clip=clip,
            alpha=False,
            annots=True,
        )
        return pixmap.tobytes("png")


Transport = Callable[[str, dict[str, Any], dict[str, str], int], dict[str, Any]]


class QwenVisionClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        vision_model: str,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = 5,
        transport: Transport | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        if not api_key:
            raise ExtractionError("QWEN_API_KEY is empty; fill the project-root .env before extraction")
        if not vision_model:
            raise ExtractionError("QWEN_VISION_MODEL is required for flattened handwriting")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.vision_model = vision_model
        self.timeout = timeout
        self.max_retries = max_retries
        self.transport = transport or self._urllib_transport
        self.sleeper = sleeper

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        images: Sequence[bytes],
    ) -> tuple[dict[str, Any], str]:
        if not images:
            raise ExtractionError("DGL extraction requires at least one page image")
        last_raw = ""
        for repair_attempt in range(2):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": make_vision_content(user_prompt, images)},
            ]
            if repair_attempt:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Return one corrected JSON object only. Do not use Markdown fences or commentary."
                        ),
                    }
                )
            payload = {
                "model": self.vision_model,
                "messages": messages,
                "temperature": 0.05,
                "response_format": {"type": "json_object"},
            }
            response = self._post(payload)
            last_raw = extract_response_text(response)
            try:
                return parse_json_object(last_raw), last_raw
            except (json.JSONDecodeError, ValueError):
                continue
        raise ExtractionError(f"Qwen returned invalid JSON after repair retry: {last_raw[:300]!r}")

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        endpoint = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_error: ApiError | None = None
        for attempt in range(self.max_retries):
            try:
                return self.transport(endpoint, payload, headers, self.timeout)
            except ApiError as exc:
                last_error = exc
                if exc.status in {400, 404, 415, 422}:
                    raise ExtractionError(
                        "The configured QWEN_VISION_MODEL rejected image input; "
                        "DGL extraction will not fall back to text-only mode"
                    ) from exc
                retryable = exc.status is None or exc.status == 429 or (exc.status and exc.status >= 500)
                if not retryable or attempt == self.max_retries - 1:
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


def make_vision_content(prompt: str, images: Sequence[bytes]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image in images:
        encoded = base64.b64encode(image).decode("ascii")
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}
        )
    return content


def extract_response_text(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ExtractionError(f"Unexpected Qwen response shape: {str(response)[:500]}") from exc
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    raise ExtractionError(f"Unexpected Qwen message content: {type(content).__name__}")


def parse_json_object(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(raw[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


TILE_SYSTEM_PROMPT = """You are a meticulous multimodal analyst of dense, hand-annotated
Deep Graph Learning lecture boards. Analyze only the supplied cropped body tile. Extract
technically useful graph theory, knowledge graph, graph representation learning, GCN/GNN,
message passing, pooling, sampling, normalization, permutation, graph generation, and directly
supporting machine-learning concepts. Printed headings, printed body text, diagram labels, and
clearly legible handwriting are valid evidence.

Never guess handwriting. If any symbol or word is ambiguous, put it in uncertain_handwriting
instead of concepts or formulas. Do not extract author names, citations, URLs, brands, sponsors,
isolated variables, decorative marks, or generic words. A formula must be transcribed faithfully
to LaTeX and linked only to concepts it directly expresses. A visual resource must be a meaningful
diagram, plot, matrix, architecture, graph example, or table, not a heading or decoration.
All bbox values are relative to this tile and must use [x0,y0,x1,y1] coordinates from 0 to 1000.
Visual bboxes must tightly include the complete diagram and its labels or attached handwritten
notes without clipping them, while excluding unrelated neighboring panels.
Return valid JSON only."""


PAGE_SYSTEM_PROMPT = """You consolidate one complete page of a dense, hand-annotated Deep
Graph Learning lecture board. Use the overview image, clipped PDF text, and tile candidates.
Resolve overlap duplicates and retain only technically useful, well-supported content. Treat
handwriting conservatively: do not guess ambiguous words or mathematical symbols. A
handwritten-only item requires very high confidence; otherwise move it to uncertain_handwriting.

Canonical concept names must be concise English lowercase noun phrases. Merge spelling,
singular/plural, abbreviations, and true synonyms, but do not merge distinct technical ideas.
When a page concept is synonymous with an existing global concept, reuse that exact canonical
name. Every formula and visual must reference one or more concepts returned for this page.
Do not fabricate relationships merely because items share a page. Produce an English page title,
a concise 2-4 sentence English summary, and a short English key_topics list. Empty concept,
formula, or visual lists are valid. Return valid JSON only."""


TILE_JSON_SHAPE = """{
  "concepts": [{
    "canonical": "english lowercase concept",
    "aliases": ["alias"],
    "evidence": "short exact transcription or precise visual evidence",
    "source_type": "printed|handwritten|mixed",
    "confidence": 0.0,
    "bbox": [0, 0, 1000, 1000]
  }],
  "formulas": [{
    "latex": "LaTeX without dollar delimiters",
    "related_concepts": ["concept canonical name"],
    "source_type": "printed|handwritten|mixed",
    "confidence": 0.0,
    "bbox": [0, 0, 1000, 1000],
    "label": "optional equation label"
  }],
  "visuals": [{
    "kind": "diagram|plot|matrix|architecture|graph example|table|other",
    "description": "specific description of the useful visual",
    "related_concepts": ["concept canonical name"],
    "source_type": "printed|handwritten|mixed",
    "confidence": 0.0,
    "bbox": [0, 0, 1000, 1000]
  }],
  "uncertain_handwriting": [{
    "transcription": "best-effort text",
    "reason": "why it is uncertain",
    "bbox": [0, 0, 1000, 1000]
  }]
}"""


PAGE_JSON_SHAPE = """{
  "page_title": "concise English page title",
  "summary": "2-4 sentence English summary",
  "key_topics": ["english key topic"],
  "concepts": [{
    "canonical": "english lowercase concept",
    "aliases": ["alias"],
    "evidence": "short exact transcription or precise visual evidence",
    "source_type": "printed|handwritten|mixed",
    "confidence": 0.0,
    "bbox": [0, 0, 1000, 1000]
  }],
  "formulas": [{
    "latex": "LaTeX without dollar delimiters",
    "related_concepts": ["returned concept canonical name"],
    "source_type": "printed|handwritten|mixed",
    "confidence": 0.0,
    "bbox": [0, 0, 1000, 1000],
    "label": "optional equation label"
  }],
  "visuals": [{
    "kind": "diagram|plot|matrix|architecture|graph example|table|other",
    "description": "specific description of the useful visual",
    "related_concepts": ["returned concept canonical name"],
    "source_type": "printed|handwritten|mixed",
    "confidence": 0.0,
    "bbox": [0, 0, 1000, 1000]
  }],
  "uncertain_handwriting": [{
    "transcription": "best-effort text",
    "reason": "why it is uncertain",
    "bbox": [0, 0, 1000, 1000]
  }]
}"""


def tile_prompt(lecture: Lecture, pdf_page: int, tile: Tile, clipped_text: str) -> str:
    return f"""Analyze Lecture {lecture.number}, physical PDF page {pdf_page}, tile {tile.name}.
The image contains only the page body; course branding and footer content were cropped out.
Use the clipped PDF text only as support for printed content. It does not reliably contain
flattened handwriting. Return this exact JSON shape:

{TILE_JSON_SHAPE}

CLIPPED PRINTED TEXT:
{clipped_text or '[no reliable text extracted]'}
"""


def page_prompt(
    lecture: Lecture,
    pdf_page: int,
    body_text: str,
    tile_candidates: dict[str, Any],
    existing_concepts: Sequence[Sequence[str]],
) -> str:
    existing = [
        {"canonical": row[0], "aliases": list(row[1:])} for row in existing_concepts if row
    ]
    return f"""Consolidate Lecture {lecture.number}, physical PDF page {pdf_page}.
All candidate bbox coordinates below have already been converted to the full body coordinate
system, normalized from 0 to 1000. The supplied image is the full page body overview.

Existing global DGL concepts (reuse exact canonical names for true synonyms):
{json.dumps(existing, ensure_ascii=False)}

Tile candidates:
{json.dumps(tile_candidates, ensure_ascii=False)}

Clipped printed text:
{body_text or '[no reliable text extracted]'}

Return this exact JSON shape with all bbox values relative to the full body overview:
{PAGE_JSON_SHAPE}
"""


def transform_mapping_bbox(data: dict[str, Any], tile: Tile) -> dict[str, Any]:
    transformed = dict(data)
    local = NormalizedBox.from_value(data.get("bbox"))
    transformed["bbox"] = tile.local_to_body(local).as_list()
    return transformed


def parse_uncertain_items(items: Any, tile: Tile | None = None) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    output: list[dict[str, Any]] = []
    for raw in items:
        if not isinstance(raw, dict):
            continue
        try:
            box = NormalizedBox.from_value(raw.get("bbox"))
            if tile is not None:
                box = tile.local_to_body(box)
        except (TypeError, ValueError):
            continue
        output.append(
            {
                "transcription": clean_text(str(raw.get("transcription") or ""), 300),
                "reason": clean_text(str(raw.get("reason") or "ambiguous handwriting"), 300),
                "bbox": box.as_list(),
            }
        )
    return output


def parse_tile_response(payload: dict[str, Any], tile: Tile) -> dict[str, Any]:
    concepts: list[dict[str, Any]] = []
    formulas: list[dict[str, Any]] = []
    visuals: list[dict[str, Any]] = []
    for raw in payload.get("concepts", []):
        if not isinstance(raw, dict):
            continue
        try:
            concepts.append(asdict(ConceptCandidate.from_mapping(transform_mapping_bbox(raw, tile))))
        except (TypeError, ValueError):
            continue
    for raw in payload.get("formulas", []):
        if not isinstance(raw, dict):
            continue
        try:
            formulas.append(asdict(FormulaCandidate.from_mapping(transform_mapping_bbox(raw, tile))))
        except (TypeError, ValueError):
            continue
    for raw in payload.get("visuals", []):
        if not isinstance(raw, dict):
            continue
        try:
            visuals.append(asdict(VisualCandidate.from_mapping(transform_mapping_bbox(raw, tile))))
        except (TypeError, ValueError):
            continue
    return {
        "concepts": concepts,
        "formulas": formulas,
        "visuals": visuals,
        "uncertain_handwriting": parse_uncertain_items(payload.get("uncertain_handwriting"), tile),
    }


def merge_tile_candidates(tile_payloads: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    concepts: list[dict[str, Any]] = []
    formulas: list[dict[str, Any]] = []
    visuals: list[dict[str, Any]] = []
    uncertain: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []

    for payload in tile_payloads:
        concepts.extend(payload["concepts"])
        formulas.extend(payload["formulas"])
        visuals.extend(payload["visuals"])
        uncertain.extend(payload["uncertain_handwriting"])

    merged_concepts: list[dict[str, Any]] = []
    for candidate in sorted(concepts, key=lambda item: -safe_float(item.get("confidence"), 0.0)):
        existing = next(
            (
                item
                for item in merged_concepts
                if any(
                    fuzzy_equivalent(left, right)
                    for left in [candidate["canonical"], *candidate.get("aliases", [])]
                    for right in [item["canonical"], *item.get("aliases", [])]
                )
            ),
            None,
        )
        if existing is None:
            merged_concepts.append(candidate)
            continue
        existing["aliases"] = stable_unique(
            [*existing.get("aliases", []), candidate["canonical"], *candidate.get("aliases", [])],
            key=normalize_key,
        )
        decisions.append(
            {"action": "tile_concept_merge", "from": candidate["canonical"], "into": existing["canonical"]}
        )

    merged_formulas: list[dict[str, Any]] = []
    for candidate in formulas:
        box = NormalizedBox.from_value(candidate["bbox"])
        existing = next(
            (
                item
                for item in merged_formulas
                if latex_key(item["latex"]) == latex_key(candidate["latex"])
                and NormalizedBox.from_value(item["bbox"]).iou(box) >= 0.25
            ),
            None,
        )
        if existing is None:
            merged_formulas.append(candidate)
            continue
        existing["related_concepts"] = stable_unique(
            [*existing["related_concepts"], *candidate["related_concepts"]], key=normalize_key
        )
        if candidate["confidence"] > existing["confidence"]:
            existing.update({key: candidate[key] for key in ["confidence", "bbox", "source_type", "label"]})

    merged_visuals: list[dict[str, Any]] = []
    for candidate in visuals:
        box = NormalizedBox.from_value(candidate["bbox"])
        existing = next(
            (
                item
                for item in merged_visuals
                if item["kind"] == candidate["kind"]
                and NormalizedBox.from_value(item["bbox"]).iou(box) >= 0.55
            ),
            None,
        )
        if existing is None:
            merged_visuals.append(candidate)
            continue
        existing["related_concepts"] = stable_unique(
            [*existing["related_concepts"], *candidate["related_concepts"]], key=normalize_key
        )
        if candidate["confidence"] > existing["confidence"]:
            existing.update(
                {key: candidate[key] for key in ["confidence", "bbox", "source_type", "description"]}
            )

    return (
        {
            "concepts": merged_concepts,
            "formulas": merged_formulas,
            "visuals": merged_visuals,
            "uncertain_handwriting": uncertain,
        },
        decisions,
    )


def confidence_passes(source_type: str, confidence: float, thresholds: dict[str, float]) -> bool:
    return confidence >= thresholds[source_type]


def parse_page_response(payload: dict[str, Any]) -> tuple[
    str,
    str,
    list[str],
    list[ConceptCandidate],
    list[FormulaCandidate],
    list[VisualCandidate],
    list[dict[str, Any]],
    list[str],
]:
    warnings: list[str] = []
    concepts: list[ConceptCandidate] = []
    for raw in payload.get("concepts", []):
        if not isinstance(raw, dict):
            continue
        try:
            item = ConceptCandidate.from_mapping(raw)
        except (TypeError, ValueError):
            continue
        if confidence_passes(item.source_type, item.confidence, CONCEPT_CONFIDENCE):
            concepts.append(item)
        else:
            warnings.append(f"Dropped low-confidence concept: {item.canonical}")
    concepts, concept_decisions = merge_concepts_locally(concepts)
    warnings.extend(
        f"Merged duplicate concept {item['from']} into {item['into']}" for item in concept_decisions
    )

    concept_map = {normalize_key(item.canonical): item.canonical for item in concepts}
    for item in concepts:
        for alias in item.aliases:
            concept_map.setdefault(normalize_key(alias), item.canonical)

    formulas: list[FormulaCandidate] = []
    for raw in payload.get("formulas", []):
        if not isinstance(raw, dict):
            continue
        try:
            item = FormulaCandidate.from_mapping(raw)
        except (TypeError, ValueError):
            continue
        related = stable_unique(
            [concept_map[key] for name in item.related_concepts if (key := normalize_key(name)) in concept_map],
            key=normalize_key,
        )
        if not related:
            continue
        item.related_concepts = related
        if confidence_passes(item.source_type, item.confidence, FORMULA_CONFIDENCE):
            formulas.append(item)
        else:
            warnings.append(f"Dropped low-confidence formula: {item.latex[:80]}")

    visuals: list[VisualCandidate] = []
    for raw in payload.get("visuals", []):
        if not isinstance(raw, dict):
            continue
        try:
            item = VisualCandidate.from_mapping(raw)
        except (TypeError, ValueError):
            continue
        related = stable_unique(
            [concept_map[key] for name in item.related_concepts if (key := normalize_key(name)) in concept_map],
            key=normalize_key,
        )
        if not related:
            continue
        item.related_concepts = related
        if confidence_passes(item.source_type, item.confidence, VISUAL_CONFIDENCE):
            visuals.append(item)
        else:
            warnings.append(f"Dropped low-confidence visual: {item.description[:80]}")

    formulas = dedupe_formulas(formulas)
    visuals = dedupe_visuals(visuals)
    title = clean_text(str(payload.get("page_title") or "Untitled DGL lecture page"), 200)
    summary = clean_text(str(payload.get("summary") or ""), 2000)
    key_topics_raw = payload.get("key_topics") or []
    if isinstance(key_topics_raw, str):
        key_topics_raw = [key_topics_raw]
    key_topics = stable_unique(
        [clean_text(str(item), 120) for item in key_topics_raw if str(item).strip()], key=str.casefold
    )
    uncertain = parse_uncertain_items(payload.get("uncertain_handwriting"))
    if not summary:
        raise ExtractionError("Page consolidation returned no English summary")
    return title, summary, key_topics, concepts, formulas, visuals, uncertain, warnings


def merge_concepts_locally(
    concepts: Sequence[ConceptCandidate],
) -> tuple[list[ConceptCandidate], list[dict[str, Any]]]:
    merged: list[ConceptCandidate] = []
    decisions: list[dict[str, Any]] = []
    for candidate in sorted(concepts, key=lambda item: -item.confidence):
        target = next(
            (
                item
                for item in merged
                if any(
                    fuzzy_equivalent(left, right)
                    for left in [candidate.canonical, *candidate.aliases]
                    for right in [item.canonical, *item.aliases]
                )
            ),
            None,
        )
        if target is None:
            merged.append(candidate)
            continue
        target.aliases = [
            alias
            for alias in stable_unique(
                [*target.aliases, candidate.canonical, *candidate.aliases], key=normalize_key
            )
            if normalize_key(alias) != normalize_key(target.canonical)
        ]
        decisions.append({"action": "page_concept_merge", "from": candidate.canonical, "into": target.canonical})
    merged.sort(key=lambda item: (NormalizedBox.from_value(item.bbox).y0, NormalizedBox.from_value(item.bbox).x0))
    return merged, decisions


def dedupe_formulas(items: Sequence[FormulaCandidate]) -> list[FormulaCandidate]:
    output: list[FormulaCandidate] = []
    for item in items:
        box = NormalizedBox.from_value(item.bbox)
        existing = next(
            (
                candidate
                for candidate in output
                if latex_key(candidate.latex) == latex_key(item.latex)
                and NormalizedBox.from_value(candidate.bbox).iou(box) >= 0.25
            ),
            None,
        )
        if existing is None:
            output.append(item)
        else:
            existing.related_concepts = stable_unique(
                [*existing.related_concepts, *item.related_concepts], key=normalize_key
            )
    return output


def dedupe_visuals(items: Sequence[VisualCandidate]) -> list[VisualCandidate]:
    output: list[VisualCandidate] = []
    for item in items:
        box = NormalizedBox.from_value(item.bbox)
        existing = next(
            (
                candidate
                for candidate in output
                if candidate.kind == item.kind and NormalizedBox.from_value(candidate.bbox).iou(box) >= 0.65
            ),
            None,
        )
        if existing is None:
            output.append(item)
        else:
            existing.related_concepts = stable_unique(
                [*existing.related_concepts, *item.related_concepts], key=normalize_key
            )
    return output


def extract_page(
    pdf: AnnotatedLecturePdf,
    client: QwenVisionClient,
    pdf_page: int,
    existing_concepts: Sequence[Sequence[str]],
) -> PageResult:
    tile_payloads: list[dict[str, Any]] = []
    raw_responses: list[dict[str, Any]] = []
    for tile in TILES:
        payload, raw = client.complete_json(
            TILE_SYSTEM_PROMPT,
            tile_prompt(pdf.lecture, pdf_page, tile, pdf.tile_text(pdf_page, tile)),
            [pdf.render_tile_jpeg(pdf_page, tile)],
        )
        parsed = parse_tile_response(payload, tile)
        tile_payloads.append(parsed)
        raw_responses.append(
            {"stage": "tile", "tile": tile.name, "parsed": payload, "raw": raw}
        )

    merged, dedupe_decisions = merge_tile_candidates(tile_payloads)
    payload, raw = client.complete_json(
        PAGE_SYSTEM_PROMPT,
        page_prompt(
            pdf.lecture,
            pdf_page,
            pdf.body_text(pdf_page),
            merged,
            existing_concepts,
        ),
        [pdf.render_body_jpeg(pdf_page)],
    )
    raw_responses.append({"stage": "page_consolidation", "parsed": payload, "raw": raw})
    title, summary, topics, concepts, formulas, visuals, uncertain, warnings = parse_page_response(payload)
    uncertain = [*merged["uncertain_handwriting"], *uncertain]
    uncertain = stable_unique(
        uncertain,
        key=lambda item: (
            clean_text(item.get("transcription", "")).casefold(),
            tuple(item.get("bbox", [])),
        ),
    )
    body = pdf.body_rect(pdf_page)
    result = PageResult(
        version=PROGRAM_VERSION,
        lecture=pdf.lecture.number,
        pdf_file=pdf.path.name,
        pdf_page=pdf_page,
        page_title=title,
        summary=summary,
        key_topics=topics,
        model=client.vision_model,
        completed_at=now_utc(),
        body_bbox=[round(value, 2) for value in body],
        concepts=[asdict(item) for item in concepts],
        formulas=[asdict(item) for item in formulas],
        visuals=[asdict(item) for item in visuals],
        uncertain_handwriting=uncertain,
        raw_responses=raw_responses,
        dedupe_decisions=dedupe_decisions,
        warnings=warnings,
    )
    validate_page_result(result)
    return result


def checkpoint_path(state_dir: Path, lecture: int, pdf_page: int) -> Path:
    return state_dir / f"lecture_{lecture:02d}_page_{pdf_page:03d}.json"


def load_results(
    state_dir: Path,
    excluding: tuple[int, int] | None = None,
) -> list[PageResult]:
    if not state_dir.exists():
        return []
    results: list[PageResult] = []
    for path in sorted(state_dir.glob("lecture_[0-9][0-9]_page_[0-9][0-9][0-9].json")):
        result = PageResult.from_path(path)
        if excluding is None or (result.lecture, result.pdf_page) != excluding:
            results.append(result)
    return sorted(results, key=lambda item: (item.lecture, item.pdf_page))


def validate_page_result(result: PageResult) -> None:
    if not 1 <= result.lecture <= len(LECTURES):
        raise ExtractionError(f"Invalid lecture number: {result.lecture}")
    lecture = LECTURES[result.lecture - 1]
    if result.pdf_file != lecture.pdf_name:
        raise ExtractionError(f"Unexpected PDF file in checkpoint: {result.pdf_file}")
    if not 1 <= result.pdf_page <= lecture.expected_pages:
        raise ExtractionError(f"Invalid page {result.pdf_page} for Lecture {result.lecture}")
    if not result.page_title or not result.summary:
        raise ExtractionError("Page title and summary are required")

    concepts: list[ConceptCandidate] = []
    for raw in result.concepts:
        try:
            concept = ConceptCandidate.from_mapping(raw)
        except (TypeError, ValueError) as exc:
            raise ExtractionError(f"Invalid concept checkpoint entry: {exc}") from exc
        if not confidence_passes(concept.source_type, concept.confidence, CONCEPT_CONFIDENCE):
            raise ExtractionError(f"Low-confidence concept remained in result: {concept.canonical}")
        concepts.append(concept)
    merged, _ = merge_concepts_locally(concepts)
    if len(merged) != len(concepts):
        raise ExtractionError("Page result contains duplicate concepts")
    allowed: set[str] = set()
    for concept in concepts:
        allowed.add(normalize_key(concept.canonical))
        allowed.update(normalize_key(alias) for alias in concept.aliases)

    for raw in result.formulas:
        try:
            formula = FormulaCandidate.from_mapping(raw)
        except (TypeError, ValueError) as exc:
            raise ExtractionError(f"Invalid formula checkpoint entry: {exc}") from exc
        if not confidence_passes(formula.source_type, formula.confidence, FORMULA_CONFIDENCE):
            raise ExtractionError("Low-confidence formula remained in result")
        if any(normalize_key(name) not in allowed for name in formula.related_concepts):
            raise ExtractionError("Formula references a concept not retained on this page")

    for raw in result.visuals:
        try:
            visual = VisualCandidate.from_mapping(raw)
        except (TypeError, ValueError) as exc:
            raise ExtractionError(f"Invalid visual checkpoint entry: {exc}") from exc
        if not confidence_passes(visual.source_type, visual.confidence, VISUAL_CONFIDENCE):
            raise ExtractionError("Low-confidence visual remained in result")
        if any(normalize_key(name) not in allowed for name in visual.related_concepts):
            raise ExtractionError("Visual references a concept not retained on this page")


def canonical_for(name: str, registry: list[dict[str, Any]]) -> str:
    for item in registry:
        if any(fuzzy_equivalent(name, candidate) for candidate in [item["canonical"], *item["aliases"]]):
            return str(item["canonical"])
    return clean_concept_name(name)


def bbox_csv(value: Sequence[float]) -> str:
    return json.dumps([round(float(item), 2) for item in value], separators=(",", ":"))


def aggregate_results(results: Sequence[PageResult]) -> dict[str, Any]:
    registry: list[dict[str, Any]] = []
    metadata: list[list[Any]] = []
    formulas: list[dict[str, Any]] = []
    visuals: list[dict[str, Any]] = []
    summaries: list[list[Any]] = []

    for result in sorted(results, key=lambda item: (item.lecture, item.pdf_page)):
        local_map: dict[str, str] = {}
        for raw in result.concepts:
            concept = ConceptCandidate.from_mapping(raw)
            canonical = canonical_for(concept.canonical, registry)
            local_map[normalize_key(concept.canonical)] = canonical
            for alias in concept.aliases:
                local_map[normalize_key(alias)] = canonical
            existing = next((item for item in registry if item["canonical"] == canonical), None)
            if existing is None:
                existing = {
                    "canonical": canonical,
                    "aliases": [],
                    "first": (result.lecture, result.pdf_page, *concept.bbox[:2]),
                }
                registry.append(existing)
            aliases = [*existing["aliases"], *concept.aliases]
            if normalize_key(concept.canonical) != normalize_key(canonical):
                aliases.append(concept.canonical)
            existing["aliases"] = [
                alias
                for alias in stable_unique(aliases, key=normalize_key)
                if normalize_key(alias) != normalize_key(canonical)
            ]
            metadata.append(
                [
                    canonical,
                    result.lecture,
                    result.pdf_file,
                    result.pdf_page,
                    result.page_title,
                    concept.evidence,
                    concept.source_type,
                    f"{concept.confidence:.3f}",
                    bbox_csv(concept.bbox),
                ]
            )

        for raw in result.formulas:
            formula = FormulaCandidate.from_mapping(raw)
            related = stable_unique(
                [
                    local_map.get(normalize_key(name), canonical_for(name, registry))
                    for name in formula.related_concepts
                ],
                key=normalize_key,
            )
            formulas.append(
                {
                    "lecture": result.lecture,
                    "pdf_file": result.pdf_file,
                    "pdf_page": result.pdf_page,
                    "latex": formula.latex,
                    "related_concepts": related,
                    "source_type": formula.source_type,
                    "confidence": formula.confidence,
                    "bbox": formula.bbox,
                    "label": formula.label,
                }
            )

        for raw in result.visuals:
            visual = VisualCandidate.from_mapping(raw)
            related = stable_unique(
                [
                    local_map.get(normalize_key(name), canonical_for(name, registry))
                    for name in visual.related_concepts
                ],
                key=normalize_key,
            )
            visuals.append(
                {
                    "lecture": result.lecture,
                    "pdf_file": result.pdf_file,
                    "pdf_page": result.pdf_page,
                    "kind": visual.kind,
                    "description": visual.description,
                    "related_concepts": related,
                    "source_type": visual.source_type,
                    "confidence": visual.confidence,
                    "bbox": visual.bbox,
                }
            )

        summaries.append(
            [
                result.lecture,
                result.pdf_file,
                result.pdf_page,
                result.page_title,
                result.summary,
                "|".join(result.key_topics),
            ]
        )

    registry.sort(key=lambda item: item["first"])
    concept_rows = [[item["canonical"], *item["aliases"]] for item in registry]
    metadata = stable_unique(
        metadata,
        key=lambda row: (normalize_key(str(row[0])), row[1], row[3], str(row[5]), str(row[8])),
    )

    formula_entities: list[dict[str, Any]] = []
    for item in sorted(
        formulas,
        key=lambda value: (
            value["lecture"],
            value["pdf_page"],
            value["bbox"][1],
            value["bbox"][0],
            latex_key(value["latex"]),
        ),
    ):
        box = NormalizedBox.from_value(item["bbox"])
        existing = next(
            (
                candidate
                for candidate in formula_entities
                if candidate["lecture"] == item["lecture"]
                and candidate["pdf_page"] == item["pdf_page"]
                and latex_key(candidate["latex"]) == latex_key(item["latex"])
                and NormalizedBox.from_value(candidate["bbox"]).iou(box) >= 0.25
            ),
            None,
        )
        if existing is None:
            formula_entities.append(dict(item))
        else:
            existing["related_concepts"] = stable_unique(
                [*existing["related_concepts"], *item["related_concepts"]], key=normalize_key
            )
    for index, item in enumerate(formula_entities, start=1):
        item["formula_id"] = f"DGL-F{index:04d}"

    visual_entities: list[dict[str, Any]] = []
    for item in sorted(
        visuals,
        key=lambda value: (
            value["lecture"],
            value["pdf_page"],
            value["bbox"][1],
            value["bbox"][0],
            value["kind"],
        ),
    ):
        box = NormalizedBox.from_value(item["bbox"])
        existing = next(
            (
                candidate
                for candidate in visual_entities
                if candidate["lecture"] == item["lecture"]
                and candidate["pdf_page"] == item["pdf_page"]
                and NormalizedBox.from_value(candidate["bbox"]).iou(box) >= 0.65
            ),
            None,
        )
        if existing is None:
            visual_entities.append(dict(item))
        else:
            existing["related_concepts"] = stable_unique(
                [*existing["related_concepts"], *item["related_concepts"]], key=normalize_key
            )
    per_page_counter: dict[tuple[int, int], int] = {}
    for index, item in enumerate(visual_entities, start=1):
        key = (item["lecture"], item["pdf_page"])
        per_page_counter[key] = per_page_counter.get(key, 0) + 1
        item["visual_id"] = f"DGL-G{index:04d}"
        item["page_visual_index"] = per_page_counter[key]

    return {
        "concept_rows": concept_rows,
        "metadata_rows": metadata,
        "formula_entities": formula_entities,
        "visual_entities": visual_entities,
        "summary_rows": summaries,
    }


def read_existing_concept_rows(results: Sequence[PageResult]) -> list[list[str]]:
    return [list(row) for row in aggregate_results(results)["concept_rows"]]


def unique_file_name(directory: Path, requested: str) -> str:
    path = directory / requested
    if not path.exists():
        return requested
    stem, suffix = path.stem, path.suffix
    index = 2
    while (directory / f"{stem}_{index}{suffix}").exists():
        index += 1
    return f"{stem}_{index}{suffix}"


def materialize_visuals(
    root: Path,
    state_dir: Path,
    visual_entities: Sequence[dict[str, Any]],
) -> tuple[list[list[Any]], list[dict[str, Any]]]:
    graph_dir = root / "DGL_graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = state_dir / "generated_graph_files.json"
    previous_files: set[str] = set()
    if manifest_path.exists():
        try:
            previous_files = set(json.loads(manifest_path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, TypeError):
            previous_files = set()

    hash_to_name: dict[str, str] = {}
    for path in graph_dir.glob("*.png"):
        try:
            hash_to_name.setdefault(hashlib.sha256(path.read_bytes()).hexdigest(), path.name)
        except OSError:
            continue

    index_rows: list[list[Any]] = []
    warnings: list[dict[str, Any]] = []
    referenced_files: set[str] = set()
    open_pdfs: dict[int, AnnotatedLecturePdf] = {}
    try:
        for item in visual_entities:
            lecture_number = int(item["lecture"])
            if lecture_number not in open_pdfs:
                open_pdfs[lecture_number] = AnnotatedLecturePdf(root, LECTURES[lecture_number - 1])
            pdf = open_pdfs[lecture_number]
            try:
                png = pdf.crop_visual_png(
                    int(item["pdf_page"]), NormalizedBox.from_value(item["bbox"])
                )
            except (ExtractionError, ValueError) as exc:
                warnings.append(
                    {
                        "visual_id": item["visual_id"],
                        "lecture": lecture_number,
                        "pdf_page": item["pdf_page"],
                        "error": str(exc),
                    }
                )
                continue
            digest = hashlib.sha256(png).hexdigest()
            file_name = hash_to_name.get(digest)
            if file_name is None:
                primary = item["related_concepts"][0]
                base = (
                    f"l{lecture_number:02d}_p{int(item['pdf_page']):02d}_"
                    f"{slugify(primary, 48)}_{int(item['page_visual_index']):02d}.png"
                )
                file_name = unique_file_name(graph_dir, base)
                target = graph_dir / file_name
                fd, temp_name = tempfile.mkstemp(prefix=".visual.", suffix=".png", dir=graph_dir)
                try:
                    with os.fdopen(fd, "wb") as handle:
                        handle.write(png)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temp_name, target)
                except Exception:
                    try:
                        os.unlink(temp_name)
                    except FileNotFoundError:
                        pass
                    raise
                hash_to_name[digest] = file_name
            referenced_files.add(file_name)
            for concept in item["related_concepts"]:
                index_rows.append(
                    [
                        item["visual_id"],
                        concept,
                        file_name,
                        item["lecture"],
                        item["pdf_file"],
                        item["pdf_page"],
                        item["kind"],
                        item["description"],
                        item["source_type"],
                        f"{item['confidence']:.3f}",
                        bbox_csv(item["bbox"]),
                    ]
                )
    finally:
        for pdf in open_pdfs.values():
            pdf.document.close()

    for stale_name in previous_files - referenced_files:
        stale_path = (graph_dir / stale_name).resolve()
        if stale_path.parent == graph_dir.resolve() and stale_path.suffix.casefold() == ".png":
            stale_path.unlink(missing_ok=True)
    atomic_write_json(manifest_path, sorted(referenced_files))
    atomic_write_json(state_dir / "graph_warnings.json", warnings)
    return index_rows, warnings


def rebuild_outputs(root: Path, state_dir: Path, results: Sequence[PageResult]) -> None:
    aggregate = aggregate_results(results)
    graph_rows, _ = materialize_visuals(root, state_dir, aggregate["visual_entities"])

    formula_rows: list[list[Any]] = []
    for item in aggregate["formula_entities"]:
        for concept in item["related_concepts"]:
            formula_rows.append(
                [
                    item["formula_id"],
                    concept,
                    item["latex"],
                    item["lecture"],
                    item["pdf_file"],
                    item["pdf_page"],
                    item["source_type"],
                    f"{item['confidence']:.3f}",
                    bbox_csv(item["bbox"]),
                    item["label"],
                ]
            )

    atomic_write_csv(root / "DGL_concepts.csv", aggregate["concept_rows"])
    atomic_write_csv(
        root / "DGL_concepts_metadata.csv",
        [
            [
                "concept",
                "lecture",
                "pdf_file",
                "pdf_page",
                "page_title",
                "evidence",
                "source_type",
                "confidence",
                "bbox",
            ],
            *aggregate["metadata_rows"],
        ],
    )
    atomic_write_csv(
        root / "DGL_formula.csv",
        [
            [
                "formula_id",
                "concept",
                "latex",
                "lecture",
                "pdf_file",
                "pdf_page",
                "source_type",
                "confidence",
                "bbox",
                "label",
            ],
            *formula_rows,
        ],
    )
    atomic_write_csv(
        root / "DGL_graph" / "index.csv",
        [
            [
                "visual_id",
                "concept",
                "file_name",
                "lecture",
                "pdf_file",
                "pdf_page",
                "kind",
                "description",
                "source_type",
                "confidence",
                "bbox",
            ],
            *graph_rows,
        ],
    )
    atomic_write_csv(
        root / "DGL_page_summaries.csv",
        [
            ["lecture", "pdf_file", "pdf_page", "page_title", "summary", "key_topics"],
            *aggregate["summary_rows"],
        ],
    )


def commit_page(root: Path, state_dir: Path, result: PageResult) -> None:
    validate_page_result(result)
    current = load_results(state_dir, excluding=(result.lecture, result.pdf_page))
    combined = sorted([*current, result], key=lambda item: (item.lecture, item.pdf_page))
    state_dir.mkdir(parents=True, exist_ok=True)
    rebuild_outputs(root, state_dir, combined)
    atomic_write_json(checkpoint_path(state_dir, result.lecture, result.pdf_page), asdict(result))
    atomic_write_json(
        state_dir / "progress.json",
        {
            "version": PROGRAM_VERSION,
            "completed_pages": [
                {"lecture": item.lecture, "pdf_page": item.pdf_page} for item in combined
            ],
            "updated_at": now_utc(),
        },
    )


def lecture_is_complete(state_dir: Path, lecture: Lecture) -> bool:
    completed = {
        result.pdf_page for result in load_results(state_dir) if result.lecture == lecture.number
    }
    return completed == set(range(1, lecture.expected_pages + 1))


def next_incomplete_lecture(state_dir: Path) -> Lecture | None:
    return next((lecture for lecture in LECTURES if not lecture_is_complete(state_dir, lecture)), None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract one annotated DGL lecture PDF with Qwen vision. "
            "Pages are processed and committed sequentially."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lecture", type=int, choices=range(1, 7), metavar="N")
    group.add_argument("--next", action="store_true", help="process the next incomplete lecture PDF")
    group.add_argument("--list-lectures", action="store_true", help="show Annotated PDF page counts")
    parser.add_argument("--dry-run", action="store_true", help="validate crops and tiles without API calls")
    parser.add_argument("--force", action="store_true", help="replace every page of the selected lecture")
    return parser


def print_lectures() -> None:
    print("lecture\tannotated_pdf\tpages")
    for lecture in LECTURES:
        print(f"{lecture.number}\t{lecture.relative_path}\t{lecture.expected_pages}")


def dry_run(pdf: AnnotatedLecturePdf) -> None:
    print(
        f"Lecture {pdf.lecture.number}: {pdf.path.name} "
        f"({pdf.lecture.expected_pages} physical pages)"
    )
    print(
        "Body crop: normalized "
        f"[{BODY_BOX[0]:.4f},{BODY_BOX[1]:.4f},{BODY_BOX[2]:.4f},{BODY_BOX[3]:.4f}]"
    )
    for page_number in range(1, pdf.lecture.expected_pages + 1):
        body = pdf.body_rect(page_number)
        text_chars = len(pdf.body_text(page_number))
        tile_sizes = []
        for tile in TILES:
            rect = pdf.tile_rect(page_number, tile)
            tile_sizes.append(f"{tile.name}:{rect.width:.0f}x{rect.height:.0f}")
        print(
            f"Page {page_number}: body={body.width:.0f}x{body.height:.0f}, "
            f"text_chars={text_chars}, tiles=4 ({', '.join(tile_sizes)})"
        )
    print("API calls: skipped (--dry-run)")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parent.parent
    root = project_root / "DGL"
    load_env_file(project_root / ".env")
    if args.list_lectures:
        if args.dry_run or args.force:
            parser.error("--list-lectures cannot be combined with --dry-run or --force")
        print_lectures()
        return 0

    state_dir = root / STATE_DIR_NAME
    if args.next:
        if args.force:
            parser.error("--force is not meaningful with --next")
        lecture = next_incomplete_lecture(state_dir)
        if lecture is None:
            print("All six annotated lecture PDFs are complete.")
            return 0
    else:
        lecture = LECTURES[int(args.lecture) - 1]

    with AnnotatedLecturePdf(root, lecture) as pdf:
        if args.dry_run:
            dry_run(pdf)
            return 0

        api_key = os.getenv("QWEN_API_KEY", "").strip()
        base_url = os.getenv(
            "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ).strip()
        model = os.getenv("QWEN_MODEL", "qwen3.8-max").strip()
        vision_model = os.getenv("QWEN_VISION_MODEL", model).strip()
        try:
            timeout = int(os.getenv("QWEN_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)))
        except ValueError:
            timeout = DEFAULT_TIMEOUT_SECONDS
        pages_to_process = [
            page_number
            for page_number in range(1, lecture.expected_pages + 1)
            if args.force or not checkpoint_path(state_dir, lecture.number, page_number).exists()
        ]
        if not pages_to_process:
            print(f"Lecture {lecture.number} is already complete. Use --force to reprocess it.")
            return 0
        client = QwenVisionClient(api_key, base_url, model, vision_model, timeout=timeout)
        for page_number in pages_to_process:
            reference_results = load_results(
                state_dir,
                excluding=(lecture.number, page_number),
            )
            existing_rows = read_existing_concept_rows(reference_results)
            print(
                f"Extracting Lecture {lecture.number} page {page_number}/"
                f"{lecture.expected_pages} with four vision tiles..."
            )
            result = extract_page(pdf, client, page_number, existing_rows)
            commit_page(root, state_dir, result)
            print(
                f"Committed page {page_number}: {len(result.concepts)} concepts, "
                f"{len(result.formulas)} formulas, {len(result.visuals)} visuals."
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExtractionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
