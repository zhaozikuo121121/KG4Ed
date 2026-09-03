#!/usr/bin/env python3
"""Extract chapter-scoped ML concepts, formulas, and figures from the MLR PDF.

The program intentionally processes exactly one chapter per invocation.  It uses
Qwen through an OpenAI-compatible ``/chat/completions`` endpoint, keeps an
auditable checkpoint per chapter, and rebuilds all public CSV outputs from those
checkpoints after every successful run.
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
except ImportError as exc:  # pragma: no cover - exercised only without dependency
    raise SystemExit(
        "PyMuPDF is required. Install dependencies with: "
        "python -m pip install -r requirements.txt"
    ) from exc


PROGRAM_VERSION = "1.0.0"
EXPECTED_PDF_PAGES = 622
DEFAULT_PDF_NAME = "Machine Learning Refined, 2nd edition.pdf"
STATE_DIR_NAME = ".mlr_extraction"
MIN_CONCEPTS_PER_CHAPTER = 20
MAX_CONCEPTS_PER_CHAPTER = 100
MAX_REFILL_ROUNDS = 5
MIN_RELEVANCE = 0.65
MIN_ASSET_CONFIDENCE = 0.80
DEFAULT_CHUNK_CHARS = 18_000
DEFAULT_TIMEOUT_SECONDS = 120
BOOK_PAGE_OFFSET = 8


class ConsoleProgress:
    """Small dependency-free progress bar for interactive and redirected output."""

    def __init__(self, enabled: bool = True, stream: Any = None, width: int = 28):
        self.enabled = enabled
        self.stream = stream or sys.stderr
        self.width = width
        self.label = ""
        self.total = 0
        self.current = 0
        self._active = False
        self._last_bucket = -1
        self._interactive = bool(getattr(self.stream, "isatty", lambda: False)())

    def __enter__(self) -> "ConsoleProgress":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def start(self, label: str, total: int) -> None:
        self.close()
        self.label = label
        self.total = max(0, int(total))
        self.current = 0
        self._active = True
        self._last_bucket = -1
        self._render(force=True)
        if self.total == 0:
            self._active = False

    def update(self, current: int) -> None:
        if not self._active:
            return
        self.current = min(max(0, int(current)), self.total)
        self._render(force=self.current >= self.total)
        if self.current >= self.total:
            self._active = False

    def advance(self, amount: int = 1) -> None:
        self.update(self.current + amount)

    def finish(self) -> None:
        if self._active:
            self.update(self.total)

    def close(self) -> None:
        if self.enabled and self._active and self._interactive:
            self.stream.write("\n")
            self.stream.flush()
        self._active = False

    def _render(self, force: bool = False) -> None:
        if not self.enabled:
            return
        ratio = self.current / self.total if self.total else 1.0
        bucket = int(ratio * 10)
        if not self._interactive and not force and bucket == self._last_bucket:
            return
        self._last_bucket = bucket
        filled = min(self.width, int(ratio * self.width))
        bar = "#" * filled + "-" * (self.width - filled)
        line = f"{self.label:<24} [{bar}] {ratio:6.1%} ({self.current}/{self.total})"
        if self._interactive:
            self.stream.write(f"\r{line}")
            if self.current >= self.total:
                self.stream.write("\n")
        else:
            self.stream.write(f"{line}\n")
        self.stream.flush()


@dataclass(frozen=True)
class Chapter:
    number: int
    title: str
    pdf_start: int
    pdf_end: int
    book_start: int
    book_end: int

    def contains_pdf_page(self, page_number: int) -> bool:
        return self.pdf_start <= page_number <= self.pdf_end

    def book_page(self, pdf_page: int) -> int:
        if not self.contains_pdf_page(pdf_page):
            raise ValueError(f"PDF page {pdf_page} is outside chapter {self.number}")
        return pdf_page - BOOK_PAGE_OFFSET


CHAPTERS: tuple[Chapter, ...] = (
    Chapter(1, "Introduction to Machine Learning", 20, 37, 12, 29),
    Chapter(2, "Zero order optimization techniques", 41, 64, 33, 56),
    Chapter(3, "First order optimization techniques", 65, 101, 57, 93),
    Chapter(4, "Second order optimization techniques", 102, 124, 94, 116),
    Chapter(5, "Linear regression", 127, 153, 119, 145),
    Chapter(6, "Linear Two-class Classification", 154, 203, 146, 195),
    Chapter(7, "Linear Multi-class Classification", 204, 238, 196, 230),
    Chapter(8, "Linear Unsupervised Learning", 239, 268, 231, 260),
    Chapter(9, "Feature Engineering and Selection", 269, 305, 261, 297),
    Chapter(10, "Principles of Nonlinear Feature Engineering", 309, 338, 301, 330),
    Chapter(11, "Principles of Feature Learning", 339, 422, 331, 414),
    Chapter(12, "Kernel methods", 423, 442, 415, 434),
    Chapter(13, "Fully-connected neural networks", 443, 482, 435, 474),
    Chapter(14, "Tree-based learners", 483, 510, 475, 502),
)

EXPECTED_CHAPTER_HEADINGS = {
    1: ("1 introduction to machine", "learning"),
    2: ("2 zero order optimization", "techniques"),
    3: ("3 first order optimization", "techniques"),
    4: ("4 second order optimization", "techniques"),
    5: ("5 linear regression",),
    6: ("6 linear two-class classification",),
    7: ("7 linear multi-class classification",),
    8: ("8 linear unsupervised learning",),
    9: ("9 feature engineering and", "selection"),
    10: ("10 principles of nonlinear feature", "engineering"),
    11: ("11 principles of feature learning",),
    12: ("12 kernel methods",),
    13: ("13 fully-connected neural networks",),
    14: ("14 tree-based learners",),
}


@dataclass
class ConceptCandidate:
    canonical: str
    aliases: list[str]
    evidence_page: int
    evidence: str
    relevance: float
    anchor_page: int

    @classmethod
    def from_mapping(cls, data: dict[str, Any], chapter: Chapter) -> "ConceptCandidate":
        canonical = clean_concept_name(data.get("canonical") or data.get("concept") or "")
        raw_aliases = data.get("aliases") or []
        if isinstance(raw_aliases, str):
            raw_aliases = [raw_aliases]
        aliases = [clean_concept_name(str(item)) for item in raw_aliases]
        aliases = [item for item in aliases if item and normalize_key(item) != normalize_key(canonical)]
        aliases = stable_unique(aliases, key=normalize_key)

        evidence_page = safe_int(data.get("evidence_page") or data.get("pdf_page"), 0)
        anchor_page = safe_int(data.get("anchor_page"), evidence_page)
        relevance = safe_float(data.get("relevance"), 0.0)
        evidence = clean_evidence(str(data.get("evidence") or ""))
        candidate = cls(canonical, aliases, evidence_page, evidence, relevance, anchor_page)
        candidate.validate(chapter)
        return candidate

    def validate(self, chapter: Chapter) -> None:
        if not valid_concept_name(self.canonical):
            raise ValueError(f"invalid concept name: {self.canonical!r}")
        if not chapter.contains_pdf_page(self.evidence_page):
            raise ValueError(f"evidence page {self.evidence_page} is outside chapter {chapter.number}")
        if not chapter.contains_pdf_page(self.anchor_page):
            raise ValueError(f"anchor page {self.anchor_page} is outside chapter {chapter.number}")
        if not self.evidence:
            raise ValueError(f"concept {self.canonical!r} has no evidence")
        if not 0.0 <= self.relevance <= 1.0:
            raise ValueError(f"relevance must be between 0 and 1: {self.relevance}")


@dataclass
class FormulaRecord:
    concept: str
    latex: str
    pdf_page: int
    equation_label: str = ""
    confidence: float = 0.0


@dataclass
class FigureRecord:
    concept: str
    pdf_page: int
    figure_label: str
    caption: str
    confidence: float = 0.0


@dataclass
class ChapterResult:
    version: str
    chapter: int
    chapter_title: str
    pdf_start: int
    pdf_end: int
    model: str
    completed_at: str
    concepts: list[dict[str, Any]]
    formulas: list[dict[str, Any]]
    figures: list[dict[str, Any]]
    raw_responses: list[dict[str, Any]] = field(default_factory=list)
    dedupe_decisions: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_path(cls, path: Path) -> "ChapterResult":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


class ExtractionError(RuntimeError):
    """Raised when a chapter cannot be safely committed."""


class ApiError(ExtractionError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def clean_evidence(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\x00", " ")).strip()[:800]


def clean_concept_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("–", "-").replace("—", "-").replace("−", "-")
    value = value.casefold().strip().strip(".,;:!?\"'`()[]{}")
    value = re.sub(r"\s*[-]\s*", "-", value)
    value = re.sub(r"\s+", " ", value)
    return value[:120]


def normalize_key(value: str) -> str:
    value = clean_concept_name(value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    tokens = value.split()
    normalized_tokens: list[str] = []
    for token in tokens:
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith("es") and not token.endswith("ses"):
            token = token[:-2]
        elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        normalized_tokens.append(token)
    return " ".join(normalized_tokens)


def valid_concept_name(value: str) -> bool:
    if not value or len(value) > 100:
        return False
    if not re.search(r"[a-z]", value):
        return False
    if len(value.split()) > 10:
        return False
    return not bool(re.fullmatch(r"(?:chapter|section|example|exercise)\s+\d+(?:\.\d+)*", value))


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


def fuzzy_equivalent(left: str, right: str) -> bool:
    left_key, right_key = normalize_key(left), normalize_key(right)
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True
    # Numbers commonly encode a real technical distinction (L1/L2, type 1/2,
    # one-vs-two, etc.). Character similarity must never erase that distinction.
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
    token_ratios = [
        SequenceMatcher(None, left_token, right_token).ratio()
        for left_token, right_token in zip(left_ordered, right_ordered)
    ]
    return bool(token_ratios) and min(token_ratios) >= 0.85 and sum(token_ratios) / len(token_ratios) >= 0.92


def latex_key(value: str) -> str:
    value = value.strip().strip("$")
    return re.sub(r"\s+", "", value)


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
    """Load a small dotenv-compatible file without overriding real environment variables."""
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


class PdfBook:
    def __init__(
        self,
        path: Path,
        validate: bool = True,
        progress: ConsoleProgress | None = None,
    ):
        self.path = Path(path)
        self.progress = progress
        if not self.path.exists():
            raise ExtractionError(f"PDF not found: {self.path}")
        self.document = fitz.open(self.path)
        if validate:
            self.validate_edition()

    def close(self) -> None:
        self.document.close()

    def __enter__(self) -> "PdfBook":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def validate_edition(self) -> None:
        if len(self.document) != EXPECTED_PDF_PAGES:
            raise ExtractionError(
                f"Unexpected PDF edition: expected {EXPECTED_PDF_PAGES} pages, "
                f"found {len(self.document)}"
            )
        if self.progress:
            self.progress.start("Validating PDF", len(CHAPTERS) + 1)
        for index, chapter in enumerate(CHAPTERS, start=1):
            text = self.document[chapter.pdf_start - 1].get_text("text", sort=True)
            normalized = re.sub(r"\s+", " ", text.casefold())
            expected_parts = EXPECTED_CHAPTER_HEADINGS[chapter.number]
            if not all(part in normalized for part in expected_parts):
                raise ExtractionError(
                    f"Chapter {chapter.number} heading was not found on PDF page "
                    f"{chapter.pdf_start}; the page map may not match this edition"
                )
            if self.progress:
                self.progress.update(index)
        appendix_text = self.document[510].get_text("text", sort=True).casefold()
        if "part iv" not in appendix_text or "appendices" not in appendix_text:
            raise ExtractionError("Appendix boundary was not found on PDF page 511")
        if self.progress:
            self.progress.finish()

    def page_text(self, pdf_page: int, strip_margins: bool = True) -> str:
        page = self.document[pdf_page - 1]
        if not strip_margins:
            return clean_pdf_text(page.get_text("text", sort=True))
        pieces: list[str] = []
        for block in page.get_text("blocks", sort=True):
            x0, y0, x1, y1, text = block[:5]
            del x0, x1
            if y1 < 78 or y0 > page.rect.height - 58:
                continue
            cleaned = clean_pdf_text(str(text))
            if cleaned:
                pieces.append(cleaned)
        return "\n".join(pieces)

    def chapter_pages(self, chapter: Chapter) -> list[tuple[int, str]]:
        page_numbers = range(chapter.pdf_start, chapter.pdf_end + 1)
        if self.progress:
            self.progress.start("Reading PDF pages", len(page_numbers))
        pages: list[tuple[int, str]] = []
        for index, page in enumerate(page_numbers, start=1):
            pages.append((page, self.page_text(page)))
            if self.progress:
                self.progress.update(index)
        return pages

    def chapter_chunks(
        self,
        chapter: Chapter,
        max_chars: int = DEFAULT_CHUNK_CHARS,
        overlap_pages: int = 1,
        pages: Sequence[tuple[int, str]] | None = None,
    ) -> list[str]:
        pages = list(pages) if pages is not None else self.chapter_pages(chapter)
        chunks: list[str] = []
        start = 0
        while start < len(pages):
            selected: list[tuple[int, str]] = []
            total = 0
            cursor = start
            while cursor < len(pages):
                page_number, page_text = pages[cursor]
                marked = f"[PDF_PAGE {page_number} | BOOK_PAGE {page_number - BOOK_PAGE_OFFSET}]\n{page_text}"
                if selected and total + len(marked) > max_chars:
                    break
                selected.append((page_number, marked))
                total += len(marked)
                cursor += 1
            chunks.append("\n\n".join(marked for _, marked in selected))
            if cursor >= len(pages):
                break
            next_start = cursor - min(overlap_pages, max(0, len(selected) - 1))
            start = next_start if next_start > start else cursor
        return chunks

    def context_window(self, chapter: Chapter, anchor_page: int, radius: int = 1) -> list[int]:
        if not chapter.contains_pdf_page(anchor_page):
            raise ValueError(f"anchor page {anchor_page} is outside chapter {chapter.number}")
        return list(
            range(max(chapter.pdf_start, anchor_page - radius), min(chapter.pdf_end, anchor_page + radius) + 1)
        )

    def render_page_jpeg(self, pdf_page: int, scale: float = 1.5, quality: int = 78) -> bytes:
        page = self.document[pdf_page - 1]
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        return pixmap.tobytes("jpeg", jpg_quality=quality)

    def crop_figure_png(
        self,
        pdf_page: int,
        figure_label: str,
        caption_hint: str = "",
        scale: float = 2.0,
    ) -> bytes:
        page = self.document[pdf_page - 1]
        clip = locate_figure_clip(page, figure_label, caption_hint)
        if clip is None:
            raise ExtractionError(
                f"Could not reliably locate {figure_label or 'figure'} on PDF page {pdf_page}"
            )
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False)
        return pixmap.tobytes("png")


def clean_pdf_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def locate_figure_clip(page: Any, figure_label: str, caption_hint: str = "") -> Any | None:
    """Return a conservative figure+caption clip, never a whole-page fallback."""
    page_dict = page.get_text("dict", sort=True)
    text_blocks = [block for block in page_dict.get("blocks", []) if block.get("type") == 0]
    image_blocks = [block for block in page_dict.get("blocks", []) if block.get("type") == 1]
    label_key = re.sub(r"\s+", " ", figure_label.casefold()).strip()
    hint_key = re.sub(r"\s+", " ", caption_hint.casefold()).strip()[:80]

    caption_block: dict[str, Any] | None = None
    for block in text_blocks:
        text = block_text(block)
        normalized = re.sub(r"\s+", " ", text.casefold())
        if label_key and label_key in normalized:
            caption_block = block
            break
        if hint_key and len(hint_key) >= 12 and hint_key in normalized:
            caption_block = block
            break
    if caption_block is None:
        return None

    caption_rect = fitz.Rect(caption_block["bbox"])
    candidates: list[Any] = []
    for block in image_blocks:
        rect = fitz.Rect(block["bbox"])
        vertical_gap = caption_rect.y0 - rect.y1
        horizontal_overlap = max(0.0, min(rect.x1, caption_rect.x1) - max(rect.x0, caption_rect.x0))
        if -8 <= vertical_gap <= page.rect.height * 0.48 and horizontal_overlap > 10:
            candidates.append(rect)

    if candidates:
        nearest_gap = min(abs(caption_rect.y0 - rect.y1) for rect in candidates)
        selected = [rect for rect in candidates if abs(caption_rect.y0 - rect.y1) <= nearest_gap + 35]
        clip = caption_rect
        for rect in selected:
            clip |= rect
    else:
        drawing_rects: list[Any] = []
        for drawing in page.get_drawings():
            rect = fitz.Rect(drawing.get("rect"))
            if rect.is_empty or rect.get_area() < 25:
                continue
            if rect.y1 <= caption_rect.y0 + 8 and caption_rect.y0 - rect.y1 <= page.rect.height * 0.42:
                drawing_rects.append(rect)
        if drawing_rects:
            clip = caption_rect
            for rect in drawing_rects:
                clip |= rect
        else:
            # Vector-heavy figures are sometimes represented as text/drawings that
            # PyMuPDF cannot group. Use a bounded region above the caption, never a
            # full page, so the output remains a figure crop rather than a page dump.
            height = min(page.rect.height * 0.42, 330.0)
            clip = fitz.Rect(
                max(page.rect.x0 + 35, caption_rect.x0 - 35),
                max(page.rect.y0 + 65, caption_rect.y0 - height),
                min(page.rect.x1 - 35, caption_rect.x1 + 35),
                min(page.rect.y1 - 45, caption_rect.y1 + 8),
            )

    clip = fitz.Rect(clip.x0 - 8, clip.y0 - 8, clip.x1 + 8, clip.y1 + 8) & page.rect
    if clip.is_empty or clip.get_area() <= 1_000:
        return None
    if clip.get_area() > page.rect.get_area() * 0.72:
        return None
    return clip


def block_text(block: dict[str, Any]) -> str:
    lines: list[str] = []
    for line in block.get("lines", []):
        lines.append("".join(span.get("text", "") for span in line.get("spans", [])))
    return "\n".join(lines)


Transport = Callable[[str, dict[str, Any], dict[str, str], int], dict[str, Any]]


class QwenClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        vision_model: str | None = None,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = 5,
        transport: Transport | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        if not api_key:
            raise ExtractionError("QWEN_API_KEY is empty; fill the project-root .env before extraction")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.vision_model = vision_model or model
        self.timeout = timeout
        self.max_retries = max_retries
        self.transport = transport or self._urllib_transport
        self.sleeper = sleeper
        self.vision_available = True

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        images: Sequence[bytes] | None = None,
        use_vision_model: bool = False,
    ) -> tuple[dict[str, Any], str, bool]:
        requested_images = list(images or [])
        send_images = requested_images if self.vision_available else []
        model = self.vision_model if use_vision_model else self.model
        last_raw = ""
        repair_attempt = 0
        while repair_attempt < 2:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": make_user_content(user_prompt, send_images)},
            ]
            if repair_attempt:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was not valid JSON. Return one valid JSON object only, "
                            "without Markdown fences or commentary."
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
                if send_images and exc.status in {400, 404, 415, 422}:
                    self.vision_available = False
                    send_images = []
                    continue
                raise
            last_raw = extract_response_text(response)
            try:
                return parse_json_object(last_raw), last_raw, bool(send_images)
            except (json.JSONDecodeError, ValueError):
                repair_attempt += 1
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
                retryable = exc.status is None or exc.status == 429 or (exc.status and exc.status >= 500)
                if not retryable or attempt == self.max_retries - 1:
                    raise
                retry_after = min(30.0, 2.0**attempt)
                self.sleeper(retry_after)
        raise last_error or ApiError("Unknown API failure")

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


def make_user_content(prompt: str, images: Sequence[bytes]) -> Any:
    if not images:
        return prompt
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
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(raw[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


CONCEPT_SYSTEM_PROMPT = """You extract useful machine-learning concepts from a textbook.
Use only evidence explicitly present in the supplied chapter pages. Prefer methods, models,
learning paradigms, loss/objective functions, optimization algorithms, evaluation metrics,
feature engineering, regularization, cross-validation, and mathematical ideas that are
operationally important to machine learning. Exclude people, datasets, application-specific
entities, isolated variables, generic English words, exercise-only objects, and appendix
references. Names must be concise English lowercase noun phrases, at a granularity similar
to a machine-learning glossary. Do not invent concepts or page numbers."""


def concept_user_prompt(
    chapter: Chapter,
    text_chunk: str,
    supplemental: bool = False,
    retained_concepts: Sequence[str] = (),
    target_count: int = 0,
) -> str:
    instruction = (
        "Find additional valid concepts that a prior pass may have missed. Avoid obvious headline terms "
        "and focus on useful supporting technical concepts."
        if supplemental
        else "Extract the useful concepts in these pages."
    )
    retained_instruction = ""
    if retained_concepts:
        retained_instruction = f"""
The following concepts are already retained and must not be returned again:
{json.dumps(list(retained_concepts), ensure_ascii=False)}
Find at least {max(1, target_count)} additional distinct concepts if the supplied text supports them.
"""
    return f"""{instruction}
Chapter {chapter.number}: {chapter.title}
Allowed PDF pages: {chapter.pdf_start}-{chapter.pdf_end}.
{retained_instruction}

Return exactly this JSON shape:
{{
  "concepts": [
    {{
      "canonical": "english lowercase concept",
      "aliases": ["alias"],
      "evidence_page": 20,
      "anchor_page": 20,
      "evidence": "short verbatim supporting excerpt",
      "relevance": 0.0
    }}
  ]
}}
Relevance is from 0 to 1. Return an empty list if no useful concept is present.

TEXT:
{text_chunk}
"""


CONSOLIDATION_SYSTEM_PROMPT = """You consolidate a machine-learning concept glossary.
Merge singular/plural variants, spelling variants, abbreviations, and true synonyms. Do not
merge technically distinct concepts. Preserve only candidates supported by evidence. If a
candidate is synonymous with an existing global concept, use the existing canonical name.
Return valid JSON only."""


ASSET_SYSTEM_PROMPT = """You identify only strong, explicit relationships between supplied
machine-learning concepts and equations or figures on nearby textbook pages. Convert equations
faithfully to LaTeX. Do not infer an equation or figure merely because it is in the same chapter.
Every association must use a supplied concept and an allowed PDF page. Return valid JSON only."""


def parse_concept_response(
    payload: dict[str, Any],
    chapter: Chapter,
    allowed_pages: Sequence[int] | None = None,
) -> list[ConceptCandidate]:
    raw_concepts = payload.get("concepts")
    if not isinstance(raw_concepts, list):
        raise ExtractionError("Qwen concept response is missing a 'concepts' list")
    page_set = set(allowed_pages) if allowed_pages is not None else None
    output: list[ConceptCandidate] = []
    for item in raw_concepts:
        if not isinstance(item, dict):
            continue
        try:
            candidate = ConceptCandidate.from_mapping(item, chapter)
        except ValueError:
            continue
        if (
            candidate.relevance >= MIN_RELEVANCE
            and (page_set is None or candidate.evidence_page in page_set)
            and (page_set is None or candidate.anchor_page in page_set)
        ):
            output.append(candidate)
    return output


def pages_marked_in_chunk(text_chunk: str) -> list[int]:
    return [int(value) for value in re.findall(r"\[PDF_PAGE\s+(\d+)\s+\|", text_chunk)]


def evidence_is_supported(book: PdfBook, candidate: ConceptCandidate) -> bool:
    """Check that the claimed excerpt is actually grounded on its evidence page."""
    evidence_tokens = re.findall(r"[a-z0-9]+", clean_pdf_text(candidate.evidence).casefold())
    page_tokens = re.findall(
        r"[a-z0-9]+", clean_pdf_text(book.page_text(candidate.evidence_page, False)).casefold()
    )
    if len(evidence_tokens) < 3 or not page_tokens:
        return False
    evidence_joined = " ".join(evidence_tokens)
    page_joined = " ".join(page_tokens)
    if evidence_joined in page_joined:
        return True
    evidence_set = set(evidence_tokens)
    coverage = len(evidence_set & set(page_tokens)) / len(evidence_set)
    return coverage >= 0.78


def merge_candidates_locally(
    candidates: Sequence[ConceptCandidate],
) -> tuple[list[ConceptCandidate], list[dict[str, Any]]]:
    ordered = sorted(candidates, key=lambda item: (-item.relevance, item.evidence_page, item.canonical))
    merged: list[ConceptCandidate] = []
    decisions: list[dict[str, Any]] = []
    for candidate in ordered:
        target: ConceptCandidate | None = None
        candidate_names = [candidate.canonical, *candidate.aliases]
        for existing in merged:
            existing_names = [existing.canonical, *existing.aliases]
            if any(fuzzy_equivalent(left, right) for left in candidate_names for right in existing_names):
                target = existing
                break
        if target is None:
            merged.append(candidate)
            continue
        aliases = [*target.aliases, candidate.canonical, *candidate.aliases]
        target.aliases = [
            alias
            for alias in stable_unique(aliases, key=normalize_key)
            if normalize_key(alias) != normalize_key(target.canonical)
        ]
        decisions.append(
            {"action": "local_merge", "from": candidate.canonical, "into": target.canonical}
        )
    merged.sort(key=lambda item: (item.evidence_page, -item.relevance, item.canonical))
    return merged, decisions


def consolidate_with_llm(
    client: QwenClient,
    chapter: Chapter,
    candidates: Sequence[ConceptCandidate],
    existing_rows: Sequence[Sequence[str]],
) -> tuple[list[ConceptCandidate], dict[str, Any], str]:
    existing_payload = [
        {"canonical": row[0], "aliases": list(row[1:])} for row in existing_rows if row
    ]
    candidate_payload = [asdict(candidate) for candidate in candidates]
    prompt = f"""Consolidate candidates for Chapter {chapter.number}: {chapter.title}.
The existing global concepts are from earlier completed chapters. Reuse an existing canonical
name only for a true synonym.

Existing global concepts:
{json.dumps(existing_payload, ensure_ascii=False)}

Chapter candidates:
{json.dumps(candidate_payload, ensure_ascii=False)}

Return {{"concepts": [...]}} using the exact candidate schema. Keep between 20 and 100 concepts
when the evidence supports that many. Preserve real evidence pages and excerpts.
"""
    payload, raw, _ = client.complete_json(CONSOLIDATION_SYSTEM_PROMPT, prompt)
    return parse_concept_response(payload, chapter), payload, raw


def select_chapter_concepts(candidates: Sequence[ConceptCandidate]) -> list[ConceptCandidate]:
    merged, _ = merge_candidates_locally(candidates)
    if len(merged) < MIN_CONCEPTS_PER_CHAPTER:
        raise ExtractionError(
            f"Chapter has only {len(merged)} valid concepts; at least {MIN_CONCEPTS_PER_CHAPTER} are required"
        )
    if len(merged) > MAX_CONCEPTS_PER_CHAPTER:
        ranked = sorted(
            merged,
            key=lambda item: (-item.relevance, -len(item.evidence), item.evidence_page, item.canonical),
        )[:MAX_CONCEPTS_PER_CHAPTER]
        merged = sorted(ranked, key=lambda item: (item.evidence_page, item.canonical))
    return merged


def parse_asset_response(
    payload: dict[str, Any],
    chapter: Chapter,
    allowed_concepts: Sequence[str],
    allowed_pages: Sequence[int] | None = None,
) -> tuple[list[FormulaRecord], list[FigureRecord]]:
    concept_map = {normalize_key(name): name for name in allowed_concepts}
    page_set = set(allowed_pages) if allowed_pages is not None else set(
        range(chapter.pdf_start, chapter.pdf_end + 1)
    )
    formulas: list[FormulaRecord] = []
    for item in payload.get("formulas", []):
        if not isinstance(item, dict):
            continue
        concept = concept_map.get(normalize_key(str(item.get("concept", ""))))
        page = safe_int(item.get("pdf_page"), 0)
        confidence = safe_float(item.get("confidence"), 0.0)
        latex = str(item.get("latex") or "").strip().strip("$").strip()
        if concept and page in page_set and confidence >= MIN_ASSET_CONFIDENCE and latex:
            formulas.append(
                FormulaRecord(concept, latex, page, str(item.get("equation_label") or "").strip(), confidence)
            )
    figures: list[FigureRecord] = []
    for item in payload.get("figures", []):
        if not isinstance(item, dict):
            continue
        concept = concept_map.get(normalize_key(str(item.get("concept", ""))))
        page = safe_int(item.get("pdf_page"), 0)
        confidence = safe_float(item.get("confidence"), 0.0)
        label = str(item.get("figure_label") or "").strip()
        caption = clean_evidence(str(item.get("caption") or ""))
        if (
            concept
            and page in page_set
            and confidence >= MIN_ASSET_CONFIDENCE
            and label
            and caption
        ):
            figures.append(FigureRecord(concept, page, label, caption, confidence))
    formulas = stable_unique(
        formulas, key=lambda item: (normalize_key(item.concept), latex_key(item.latex), item.pdf_page)
    )
    figures = stable_unique(
        figures,
        key=lambda item: (normalize_key(item.concept), item.pdf_page, normalize_key(item.figure_label)),
    )
    return formulas, figures


def extract_assets(
    book: PdfBook,
    client: QwenClient,
    chapter: Chapter,
    concepts: Sequence[ConceptCandidate],
    raw_responses: list[dict[str, Any]],
    warnings: list[str],
    progress: ConsoleProgress | None = None,
) -> tuple[list[FormulaRecord], list[FigureRecord]]:
    by_anchor: dict[int, list[str]] = {}
    for concept in concepts:
        by_anchor.setdefault(concept.anchor_page, []).append(concept.canonical)
    all_formulas: list[FormulaRecord] = []
    all_figures: list[FigureRecord] = []
    warned_fallback = False
    anchors = sorted(by_anchor.items())
    if progress:
        progress.start("Analyzing assets", len(anchors))
    for anchor_index, (anchor_page, names) in enumerate(anchors, start=1):
        pages = book.context_window(chapter, anchor_page, radius=1)
        text = "\n\n".join(
            f"[PDF_PAGE {page} | BOOK_PAGE {page - BOOK_PAGE_OFFSET}]\n{book.page_text(page, False)}"
            for page in pages
        )
        prompt = f"""Concepts anchored on PDF page {anchor_page}:
{json.dumps(names, ensure_ascii=False)}
Allowed context pages: {pages}

Return exactly:
{{
  "formulas": [{{"concept":"...","latex":"...","pdf_page":1,"equation_label":"","confidence":0.0}}],
  "figures": [{{"concept":"...","pdf_page":1,"figure_label":"Figure 1.1","caption":"verbatim caption","confidence":0.0}}]
}}
Only return strong direct associations. Empty lists are valid.

PAGE TEXT:
{text}
"""
        images = [book.render_page_jpeg(page) for page in pages]
        payload, raw, used_vision = client.complete_json(
            ASSET_SYSTEM_PROMPT,
            prompt,
            images=images,
            use_vision_model=True,
        )
        if images and not used_vision and not warned_fallback:
            warnings.append(
                "Vision input was rejected by the configured model; asset analysis continued in text-only mode."
            )
            warned_fallback = True
        raw_responses.append(
            {"stage": "assets", "anchor_page": anchor_page, "parsed": payload, "raw": raw}
        )
        formulas, figures = parse_asset_response(payload, chapter, names, allowed_pages=pages)
        all_formulas.extend(formulas)
        all_figures.extend(figures)
        if progress:
            progress.update(anchor_index)
    return (
        stable_unique(
            all_formulas,
            key=lambda item: (normalize_key(item.concept), latex_key(item.latex), item.pdf_page),
        ),
        stable_unique(
            all_figures,
            key=lambda item: (normalize_key(item.concept), item.pdf_page, normalize_key(item.figure_label)),
        ),
    )


def extract_chapter(
    book: PdfBook,
    client: QwenClient,
    chapter: Chapter,
    existing_rows: Sequence[Sequence[str]],
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    progress: ConsoleProgress | None = None,
) -> ChapterResult:
    raw_responses: list[dict[str, Any]] = []
    warnings: list[str] = []
    candidates: list[ConceptCandidate] = []
    chunks = book.chapter_chunks(chapter, max_chars=chunk_chars, overlap_pages=1)
    if progress:
        progress.start("Extracting concepts", len(chunks))
    for index, chunk in enumerate(chunks, start=1):
        payload, raw, _ = client.complete_json(
            CONCEPT_SYSTEM_PROMPT,
            concept_user_prompt(chapter, chunk),
        )
        raw_responses.append({"stage": "concepts", "chunk": index, "parsed": payload, "raw": raw})
        candidates.extend(
            parse_concept_response(payload, chapter, allowed_pages=pages_marked_in_chunk(chunk))
        )
        if progress:
            progress.update(index)

    locally_merged, decisions = merge_candidates_locally(candidates)
    if len(locally_merged) < MIN_CONCEPTS_PER_CHAPTER:
        # Keep the same bounded chunks during the supplemental pass so even the
        # longest chapter never becomes one oversized request.
        if progress:
            progress.start("Supplementing concepts", len(chunks))
        for index, chunk in enumerate(chunks, start=1):
            payload, raw, _ = client.complete_json(
                CONCEPT_SYSTEM_PROMPT,
                concept_user_prompt(chapter, chunk, supplemental=True),
            )
            raw_responses.append(
                {"stage": "supplemental", "chunk": index, "parsed": payload, "raw": raw}
            )
            candidates.extend(
                parse_concept_response(payload, chapter, allowed_pages=pages_marked_in_chunk(chunk))
            )
            locally_merged, extra_decisions = merge_candidates_locally(candidates)
            decisions.extend(extra_decisions)
            if progress:
                progress.update(index)
            if len(locally_merged) >= MIN_CONCEPTS_PER_CHAPTER:
                break
        if progress:
            progress.finish()

    if progress:
        progress.start("Consolidating concepts", 1)
    consolidated, parsed, raw = consolidate_with_llm(
        client, chapter, locally_merged, existing_rows
    )
    if progress:
        progress.finish()
    raw_responses.append({"stage": "consolidation", "parsed": parsed, "raw": raw})
    if progress:
        progress.start("Checking evidence", len(consolidated))
    grounded: list[ConceptCandidate] = []
    for index, candidate in enumerate(consolidated, start=1):
        if evidence_is_supported(book, candidate):
            grounded.append(candidate)
        if progress:
            progress.update(index)
    rejected = len(consolidated) - len(grounded)
    if rejected:
        warnings.append(f"Rejected {rejected} concepts whose quoted evidence was not found on the claimed page.")
    grounded, grounded_decisions = merge_candidates_locally(grounded)
    decisions.extend(grounded_decisions)

    # Consolidation and evidence grounding can reduce a seemingly sufficient
    # candidate set below the chapter minimum. Keep every already-grounded
    # concept and run targeted refill passes that explicitly exclude it.
    refill_round = 0
    while len(grounded) < MIN_CONCEPTS_PER_CHAPTER and refill_round < MAX_REFILL_ROUNDS:
        refill_round += 1
        missing = MIN_CONCEPTS_PER_CHAPTER - len(grounded)
        retained_names = [item.canonical for item in grounded]
        refill_candidates: list[ConceptCandidate] = []
        if progress:
            progress.start(f"Refilling concepts {refill_round}/{MAX_REFILL_ROUNDS}", len(chunks))
        for index, chunk in enumerate(chunks, start=1):
            payload, raw, _ = client.complete_json(
                CONCEPT_SYSTEM_PROMPT,
                concept_user_prompt(
                    chapter,
                    chunk,
                    supplemental=True,
                    retained_concepts=retained_names,
                    target_count=missing,
                ),
            )
            raw_responses.append(
                {
                    "stage": "refill",
                    "round": refill_round,
                    "chunk": index,
                    "parsed": payload,
                    "raw": raw,
                }
            )
            refill_candidates.extend(
                parse_concept_response(payload, chapter, allowed_pages=pages_marked_in_chunk(chunk))
            )
            if progress:
                progress.update(index)

        refill_candidates, refill_decisions = merge_candidates_locally(refill_candidates)
        decisions.extend(refill_decisions)
        if not refill_candidates:
            continue
        if progress:
            progress.start("Consolidating refill", 1)
        refill_consolidated, parsed, raw = consolidate_with_llm(
            client, chapter, refill_candidates, existing_rows
        )
        raw_responses.append(
            {
                "stage": "refill_consolidation",
                "round": refill_round,
                "parsed": parsed,
                "raw": raw,
            }
        )
        if progress:
            progress.finish()
            progress.start("Checking refill evidence", len(refill_consolidated))
        refill_grounded: list[ConceptCandidate] = []
        for index, candidate in enumerate(refill_consolidated, start=1):
            if evidence_is_supported(book, candidate):
                refill_grounded.append(candidate)
            if progress:
                progress.update(index)
        refill_rejected = len(refill_consolidated) - len(refill_grounded)
        if refill_rejected:
            warnings.append(
                f"Refill round {refill_round} rejected {refill_rejected} concepts whose evidence was not found."
            )
        previous_count = len(grounded)
        grounded, refill_merge_decisions = merge_candidates_locally([*grounded, *refill_grounded])
        decisions.extend(refill_merge_decisions)
        if len(grounded) == previous_count:
            warnings.append(f"Refill round {refill_round} found no new grounded concepts.")

    if len(grounded) < MIN_CONCEPTS_PER_CHAPTER:
        raise ExtractionError(
            f"Chapter still has only {len(grounded)} valid concepts after "
            f"{MAX_REFILL_ROUNDS} refill rounds; at least {MIN_CONCEPTS_PER_CHAPTER} are required"
        )
    final_concepts = select_chapter_concepts(grounded)
    formulas, figures = extract_assets(
        book, client, chapter, final_concepts, raw_responses, warnings, progress
    )
    return ChapterResult(
        version=PROGRAM_VERSION,
        chapter=chapter.number,
        chapter_title=chapter.title,
        pdf_start=chapter.pdf_start,
        pdf_end=chapter.pdf_end,
        model=client.model,
        completed_at=now_utc(),
        concepts=[asdict(item) for item in final_concepts],
        formulas=[asdict(item) for item in formulas],
        figures=[asdict(item) for item in figures],
        raw_responses=raw_responses,
        dedupe_decisions=decisions,
        warnings=warnings,
    )


def state_path(state_dir: Path, chapter_number: int) -> Path:
    return state_dir / f"chapter_{chapter_number:02d}.json"


def load_results(state_dir: Path, excluding: int | None = None) -> list[ChapterResult]:
    results: list[ChapterResult] = []
    if not state_dir.exists():
        return results
    for path in sorted(state_dir.glob("chapter_[0-9][0-9].json")):
        result = ChapterResult.from_path(path)
        if excluding is None or result.chapter != excluding:
            results.append(result)
    return sorted(results, key=lambda item: item.chapter)


def read_existing_concept_rows(results: Sequence[ChapterResult]) -> list[list[str]]:
    rows, _, _, _ = aggregate_results(results)
    return [list(row) for row in rows]


def canonical_for(name: str, registry: list[dict[str, Any]]) -> str:
    for item in registry:
        if any(fuzzy_equivalent(name, candidate) for candidate in [item["canonical"], *item["aliases"]]):
            return str(item["canonical"])
    return clean_concept_name(name)


def aggregate_results(
    results: Sequence[ChapterResult],
) -> tuple[list[list[str]], list[list[Any]], list[list[Any]], list[dict[str, Any]]]:
    registry: list[dict[str, Any]] = []
    metadata: list[list[Any]] = []
    formulas_raw: list[dict[str, Any]] = []
    figures_raw: list[dict[str, Any]] = []

    for result in sorted(results, key=lambda item: item.chapter):
        chapter = CHAPTERS[result.chapter - 1]
        local_map: dict[str, str] = {}
        for raw_concept in result.concepts:
            candidate = ConceptCandidate.from_mapping(raw_concept, chapter)
            canonical = canonical_for(candidate.canonical, registry)
            local_map[normalize_key(candidate.canonical)] = canonical
            existing = next((item for item in registry if item["canonical"] == canonical), None)
            if existing is None:
                existing = {"canonical": canonical, "aliases": [], "first": (result.chapter, candidate.evidence_page)}
                registry.append(existing)
            aliases = [*existing["aliases"], *candidate.aliases]
            if normalize_key(candidate.canonical) != normalize_key(canonical):
                aliases.append(candidate.canonical)
            existing["aliases"] = [
                alias
                for alias in stable_unique(aliases, key=normalize_key)
                if normalize_key(alias) != normalize_key(canonical)
            ]
            metadata.append(
                [
                    canonical,
                    result.chapter,
                    result.chapter_title,
                    candidate.evidence_page,
                    chapter.book_page(candidate.evidence_page),
                    candidate.evidence,
                ]
            )
        for raw_formula in result.formulas:
            formula = FormulaRecord(**raw_formula)
            canonical = local_map.get(normalize_key(formula.concept), canonical_for(formula.concept, registry))
            formulas_raw.append(
                {
                    "concept": canonical,
                    "latex": formula.latex.strip().strip("$").strip(),
                    "chapter": result.chapter,
                    "pdf_page": formula.pdf_page,
                    "book_page": formula.pdf_page - BOOK_PAGE_OFFSET,
                    "equation_label": formula.equation_label,
                }
            )
        for raw_figure in result.figures:
            figure = FigureRecord(**raw_figure)
            canonical = local_map.get(normalize_key(figure.concept), canonical_for(figure.concept, registry))
            figures_raw.append(
                {
                    "concept": canonical,
                    "chapter": result.chapter,
                    "pdf_page": figure.pdf_page,
                    "book_page": figure.pdf_page - BOOK_PAGE_OFFSET,
                    "figure_label": figure.figure_label,
                    "caption": figure.caption,
                }
            )

    registry.sort(key=lambda item: item["first"])
    concept_rows = [[item["canonical"], *item["aliases"]] for item in registry]
    metadata = stable_unique(
        metadata,
        key=lambda row: (normalize_key(str(row[0])), row[1], row[3], str(row[5])),
    )
    formulas_raw = stable_unique(
        sorted(
            formulas_raw,
            key=lambda item: (item["chapter"], item["pdf_page"], normalize_key(item["concept"]), latex_key(item["latex"])),
        ),
        key=lambda item: (normalize_key(item["concept"]), latex_key(item["latex"]), item["pdf_page"]),
    )
    formula_rows: list[list[Any]] = []
    for index, item in enumerate(formulas_raw, start=1):
        formula_rows.append(
            [
                f"F{index:04d}",
                item["concept"],
                item["latex"],
                item["chapter"],
                item["pdf_page"],
                item["book_page"],
                item["equation_label"],
            ]
        )
    figures_raw = stable_unique(
        sorted(
            figures_raw,
            key=lambda item: (item["chapter"], item["pdf_page"], normalize_key(item["figure_label"]), normalize_key(item["concept"])),
        ),
        key=lambda item: (normalize_key(item["concept"]), item["pdf_page"], normalize_key(item["figure_label"])),
    )
    return concept_rows, metadata, formula_rows, figures_raw


def materialize_figures(
    root: Path,
    pdf_path: Path,
    state_dir: Path,
    figures: Sequence[dict[str, Any]],
    book: PdfBook | None = None,
) -> list[list[Any]]:
    graph_dir = root / "MLR_graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = state_dir / "generated_graph_files.json"
    previous_files: set[str] = set()
    if manifest_path.exists():
        try:
            previous_files = set(json.loads(manifest_path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, TypeError):
            previous_files = set()

    owns_book = book is None and bool(figures)
    if book is None and figures:
        book = PdfBook(pdf_path)
    hash_to_name: dict[str, str] = {}
    for existing_file in sorted(graph_dir.glob("*.png")):
        try:
            digest = hashlib.sha256(existing_file.read_bytes()).hexdigest()
            hash_to_name.setdefault(digest, existing_file.name)
        except OSError:
            continue

    index_rows: list[list[Any]] = []
    crop_warnings: list[dict[str, Any]] = []
    referenced_files: set[str] = set()
    crop_cache: dict[tuple[int, str, str], tuple[bytes, str]] = {}
    try:
        for item in figures:
            cache_key = (
                int(item["pdf_page"]),
                normalize_key(str(item["figure_label"])),
                clean_evidence(str(item["caption"])),
            )
            try:
                if cache_key not in crop_cache:
                    if book is None:  # Defensive: only possible with an inconsistent caller.
                        raise ExtractionError("A PDF is required to materialize figure crops")
                    png = book.crop_figure_png(
                        int(item["pdf_page"]), str(item["figure_label"]), str(item["caption"])
                    )
                    crop_cache[cache_key] = (png, hashlib.sha256(png).hexdigest())
                png, digest = crop_cache[cache_key]
            except ExtractionError as exc:
                print(f"warning: {exc}", file=sys.stderr)
                crop_warnings.append(
                    {
                        "concept": item["concept"],
                        "chapter": item["chapter"],
                        "pdf_page": item["pdf_page"],
                        "figure_label": item["figure_label"],
                        "error": str(exc),
                    }
                )
                continue

            file_name = hash_to_name.get(digest)
            if file_name is None:
                label_slug = slugify(str(item["figure_label"]), 24)
                concept_slug = slugify(str(item["concept"]), 48)
                base = (
                    f"ch{int(item['chapter']):02d}_{concept_slug}_{label_slug}_"
                    f"p{int(item['pdf_page'])}.png"
                )
                file_name = unique_file_name(graph_dir, base)
                target = graph_dir / file_name
                fd, temp_name = tempfile.mkstemp(prefix=".figure.", suffix=".png", dir=graph_dir)
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
            index_rows.append(
                [
                    item["concept"],
                    file_name,
                    item["chapter"],
                    item["pdf_page"],
                    item["book_page"],
                    item["figure_label"],
                    item["caption"],
                ]
            )
    finally:
        if owns_book and book is not None:
            book.close()

    # Delete only files recorded in our prior manifest; never touch user-added PNGs.
    for stale_name in previous_files - referenced_files:
        stale_path = (graph_dir / stale_name).resolve()
        if stale_path.parent == graph_dir.resolve() and stale_path.suffix.casefold() == ".png":
            stale_path.unlink(missing_ok=True)
    atomic_write_json(manifest_path, sorted(referenced_files))
    atomic_write_json(state_dir / "graph_warnings.json", crop_warnings)
    return index_rows


def unique_file_name(directory: Path, requested: str) -> str:
    path = directory / requested
    if not path.exists():
        return requested
    stem, suffix = path.stem, path.suffix
    index = 2
    while (directory / f"{stem}_{index}{suffix}").exists():
        index += 1
    return f"{stem}_{index}{suffix}"


def rebuild_outputs(
    root: Path,
    pdf_path: Path,
    state_dir: Path,
    results: Sequence[ChapterResult],
    book: PdfBook | None = None,
) -> None:
    concept_rows, metadata, formulas, figures = aggregate_results(results)
    index_rows = materialize_figures(root, pdf_path, state_dir, figures, book=book)

    atomic_write_csv(root / "MLR_concepts.csv", concept_rows)
    atomic_write_csv(
        root / "MLR_concepts_metadata.csv",
        [["concept", "chapter", "chapter_title", "pdf_page", "book_page", "evidence"], *metadata],
    )
    atomic_write_csv(
        root / "MLR_formula.csv",
        [["formula_id", "concept", "latex", "chapter", "pdf_page", "book_page", "equation_label"], *formulas],
    )
    atomic_write_csv(
        root / "MLR_graph" / "index.csv",
        [["concept", "file_name", "chapter", "pdf_page", "book_page", "figure_label", "caption"], *index_rows],
    )


def commit_result(
    root: Path,
    pdf_path: Path,
    state_dir: Path,
    result: ChapterResult,
    book: PdfBook,
) -> None:
    validate_result(result)
    current = load_results(state_dir, excluding=result.chapter)
    combined = sorted([*current, result], key=lambda item: item.chapter)
    state_dir.mkdir(parents=True, exist_ok=True)
    rebuild_outputs(root, pdf_path, state_dir, combined, book=book)
    atomic_write_json(state_path(state_dir, result.chapter), asdict(result))
    atomic_write_json(
        state_dir / "progress.json",
        {
            "version": PROGRAM_VERSION,
            "completed_chapters": [item.chapter for item in combined],
            "updated_at": now_utc(),
        },
    )


def validate_result(result: ChapterResult) -> None:
    if not 1 <= result.chapter <= 14:
        raise ExtractionError(f"Invalid chapter result: {result.chapter}")
    chapter = CHAPTERS[result.chapter - 1]
    concepts = [ConceptCandidate.from_mapping(item, chapter) for item in result.concepts]
    selected = select_chapter_concepts(concepts)
    if len(selected) != len(concepts):
        raise ExtractionError("Chapter checkpoint still contains duplicate or excess concepts")
    allowed = {normalize_key(item.canonical) for item in concepts}
    for raw_formula in result.formulas:
        formula = FormulaRecord(**raw_formula)
        if normalize_key(formula.concept) not in allowed or not chapter.contains_pdf_page(formula.pdf_page):
            raise ExtractionError("Formula references an invalid concept or page")
    for raw_figure in result.figures:
        figure = FigureRecord(**raw_figure)
        if normalize_key(figure.concept) not in allowed or not chapter.contains_pdf_page(figure.pdf_page):
            raise ExtractionError("Figure references an invalid concept or page")


def next_incomplete_chapter(state_dir: Path) -> int | None:
    completed = {result.chapter for result in load_results(state_dir)}
    return next((chapter.number for chapter in CHAPTERS if chapter.number not in completed), None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract one MLR textbook chapter with Qwen. Exactly one chapter is processed per run."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--chapter", type=int, choices=range(1, 15), metavar="N")
    group.add_argument("--next", action="store_true", help="process the next incomplete chapter")
    group.add_argument("--list-chapters", action="store_true", help="show the fixed chapter page map")
    parser.add_argument("--dry-run", action="store_true", help="validate and inspect without calling Qwen")
    parser.add_argument("--force", action="store_true", help="replace an existing chapter checkpoint")
    parser.add_argument("--pdf", type=Path, help="override the default PDF path")
    parser.add_argument("--no-progress", action="store_true", help="disable progress bars")
    parser.add_argument("--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS, help=argparse.SUPPRESS)
    return parser


def print_chapters() -> None:
    print("chapter\ttitle\tpdf_pages\tbook_pages")
    for chapter in CHAPTERS:
        print(
            f"{chapter.number}\t{chapter.title}\t"
            f"{chapter.pdf_start}-{chapter.pdf_end}\t{chapter.book_start}-{chapter.book_end}"
        )


def dry_run(
    book: PdfBook,
    chapter: Chapter,
    chunk_chars: int,
    progress: ConsoleProgress | None = None,
) -> None:
    pages = book.chapter_pages(chapter)
    chunks = book.chapter_chunks(
        chapter, max_chars=chunk_chars, overlap_pages=1, pages=pages
    )
    nonempty = sum(bool(text.strip()) for _, text in pages)
    figure_captions = 0
    if progress:
        progress.start("Scanning captions", len(pages))
    for index, (page_number, _) in enumerate(pages, start=1):
        text = book.page_text(page_number, False)
        figure_captions += len(re.findall(r"\bFigure\s+\d+\.\d+", text, flags=re.IGNORECASE))
        if progress:
            progress.update(index)
    print(f"PDF validation: OK ({len(book.document)} pages)")
    print(f"Chapter {chapter.number}: {chapter.title}")
    print(f"PDF pages: {chapter.pdf_start}-{chapter.pdf_end}")
    print(f"Book pages: {chapter.book_start}-{chapter.book_end}")
    print(f"Non-empty chapter pages: {nonempty}/{len(pages)}")
    print(f"Text chunks: {len(chunks)} (max {chunk_chars} characters, one-page overlap)")
    print(f"Figure-caption occurrences: {figure_captions}")
    print("API call: skipped (--dry-run)")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.chunk_chars < 1_000:
        parser.error("--chunk-chars must be at least 1000")
    project_root = Path(__file__).resolve().parent.parent
    script_dir = project_root / "MLR"
    load_env_file(project_root / ".env")
    if args.list_chapters:
        if args.force or args.dry_run:
            parser.error("--list-chapters cannot be combined with --force or --dry-run")
        print_chapters()
        return 0

    state_dir = script_dir / STATE_DIR_NAME
    if args.next:
        if args.force:
            parser.error("--force is not meaningful with --next")
        chapter_number = next_incomplete_chapter(state_dir)
        if chapter_number is None:
            print("All 14 chapters are complete.")
            return 0
    else:
        chapter_number = int(args.chapter)
    chapter = CHAPTERS[chapter_number - 1]
    checkpoint = state_path(state_dir, chapter_number)
    if checkpoint.exists() and not args.force and not args.dry_run:
        raise ExtractionError(
            f"Chapter {chapter_number} is already complete. Use --force to replace its checkpoint."
        )

    pdf_path = (args.pdf or (script_dir / DEFAULT_PDF_NAME)).resolve()
    with ConsoleProgress(enabled=not args.no_progress) as progress, PdfBook(
        pdf_path, progress=progress
    ) as book:
        if args.dry_run:
            dry_run(book, chapter, args.chunk_chars, progress)
            return 0

        api_key = os.getenv("QWEN_API_KEY", "").strip()
        base_url = os.getenv(
            "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ).strip()
        model = os.getenv("QWEN_MODEL", "qwen3.8-max").strip()
        vision_model = os.getenv("QWEN_VISION_MODEL", model).strip()
        timeout = safe_int(os.getenv("QWEN_TIMEOUT_SECONDS"), DEFAULT_TIMEOUT_SECONDS)
        client = QwenClient(api_key, base_url, model, vision_model, timeout=timeout)

        stored_results = load_results(state_dir, excluding=chapter_number)
        reference_results = (
            stored_results
            if args.force
            else [item for item in stored_results if item.chapter < chapter_number]
        )
        existing_rows = read_existing_concept_rows(reference_results)
        print(
            f"Extracting Chapter {chapter.number}: {chapter.title} "
            f"(PDF pages {chapter.pdf_start}-{chapter.pdf_end})"
        )
        result = extract_chapter(
            book,
            client,
            chapter,
            existing_rows,
            chunk_chars=args.chunk_chars,
            progress=progress,
        )
        progress.start("Writing outputs", 1)
        commit_result(script_dir, pdf_path, state_dir, result, book)
        progress.finish()
        print(
            f"Completed Chapter {chapter.number}: {len(result.concepts)} concepts, "
            f"{len(result.formulas)} formulas, {len(result.figures)} figure associations."
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExtractionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
