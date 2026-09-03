#!/usr/bin/env python3
"""Audit and, when justified, strengthen concept evidence in metadata v2 files.

The original metadata files are read-only inputs.  Only the *_metadata_v2.csv files
are written, and only when a grounded KEEP/APPEND/REPLACE/ADD_ROW/MIGRATE decision
can be applied safely.
"""

from __future__ import annotations

import argparse
import base64
import csv
import importlib.util
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import fitz  # type: ignore
except ImportError:  # pragma: no cover - dry-run and pure helpers do not need PyMuPDF
    fitz = None  # type: ignore


ROOT = Path(__file__).resolve().parent.parent
MAX_EVIDENCE_CHARS = 1200
MIN_LLM_CONFIDENCE = 0.80
VALID_ACTIONS = {"KEEP", "APPEND", "REPLACE", "ADD_ROW", "MIGRATE", "UNCHANGED_UNCERTAIN"}
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
PREFER_BASELINE_LOCATORS = {
    ("sequential refinement", "2", "48", "40"),
    ("unconstrained optimization", "5", "130", "122"),
    ("softmax cost", "6", "173", "165"),
    ("margin computation via normal vector", "6", "185", "177"),
    ("cluster assignment", "8", "254", "246"),
    ("least squares cost function", "10", "329", "321"),
    ("recursively defined trees", "14", "503", "495"),
}


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def require_fitz() -> Any:
    if fitz is None:
        raise RuntimeError("PyMuPDF is required for source-grounded auditing; install the dataset requirements first")
    return fitz


class ConsoleProgress:
    """Minimal terminal progress display that also remains readable in redirected logs."""

    def __init__(self, label: str, total: int, stream: Any = None):
        self.label = label
        self.total = max(total, 1)
        self.stream = stream or sys.stderr
        self.current = 0
        self.started_at = time.monotonic()
        self.interactive = bool(getattr(self.stream, "isatty", lambda: False)())

    def start(self) -> None:
        self._render("")

    def update(self, current: int, detail: str = "") -> None:
        self.current = min(max(current, 0), self.total)
        self._render(detail)

    def message(self, text: str) -> None:
        if self.interactive:
            self.stream.write("\r" + " " * 160 + "\r")
        self.stream.write(text + "\n")
        self.stream.flush()

    def finish(self) -> None:
        self.current = self.total
        self._render("done")
        if self.interactive:
            self.stream.write("\n")
            self.stream.flush()

    def _render(self, detail: str) -> None:
        elapsed = int(time.monotonic() - self.started_at)
        percent = int(self.current * 100 / self.total)
        text = f"[{self.label}] {self.current}/{self.total} ({percent}%) elapsed={elapsed}s"
        if detail:
            text += f" — {detail}"
        if self.interactive:
            self.stream.write("\r" + text[:160].ljust(160))
        else:
            self.stream.write(text + "\n")
        self.stream.flush()


def load_env_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def parse_json_object(raw: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
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


class DeepSeekClient:
    """Small OpenAI-compatible DeepSeek client used by the evidence auditor."""

    def __init__(self, api_key: str, base_url: str, model: str, timeout: int = 180, max_retries: int = 5):
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is empty; copy .env.example to .env and fill the real key")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    def complete_json(self, system_prompt: str, user_prompt: str, images: Sequence[bytes] = ()) -> tuple[dict[str, Any], str]:
        content: Any = user_prompt
        if images:
            content = [{"type": "text", "text": user_prompt}]
            for raw in images:
                encoded = base64.b64encode(raw).decode("ascii")
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})
        last_raw = ""
        for repair_attempt in range(2):
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ]
            if repair_attempt:
                messages.append({"role": "user", "content": "Return one corrected JSON object only, without Markdown."})
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            }
            response = self._post(payload)
            try:
                last_raw = response["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError(f"Unexpected DeepSeek response: {str(response)[:500]}") from exc
            if not isinstance(last_raw, str):
                raise RuntimeError("DeepSeek returned non-text message content")
            try:
                return parse_json_object(last_raw), last_raw
            except (json.JSONDecodeError, ValueError):
                continue
        raise RuntimeError(f"DeepSeek returned invalid JSON after repair retry: {last_raw[:300]!r}")

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"DeepSeek API HTTP {exc.code}: {body[:500]}")
                retryable = exc.code == 429 or exc.code >= 500
                if not retryable or attempt == self.max_retries - 1:
                    raise last_error from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = RuntimeError(f"DeepSeek API connection failed: {exc}")
                if attempt == self.max_retries - 1:
                    raise last_error from exc
            time.sleep(min(30.0, 2.0**attempt))
        raise last_error or RuntimeError("Unknown Qwen API failure")


def clean_text(value: Any, limit: int = 20000) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).replace("\x00", " ")
    return re.sub(r"\s+", " ", text).strip()[:limit]


def norm(value: Any) -> str:
    text = clean_text(value, 5000).casefold()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def tokens(value: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+", norm(value))


def normalized_match(value: Any) -> str:
    return " ".join(tokens(value))


def append_sentences(old: str, new: str) -> str:
    """Join evidence with one period and avoid repeated identical sentences."""
    old = clean_text(old, MAX_EVIDENCE_CHARS)
    new = clean_text(new, MAX_EVIDENCE_CHARS)
    if not new:
        return old
    old_parts = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+", old) if part.strip()]
    new_parts = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+", new) if part.strip()]
    seen = {normalized_match(part) for part in old_parts}
    additions = [part for part in new_parts if normalized_match(part) not in seen]
    if not additions:
        return old
    left = old.rstrip()
    left = re.sub(r"[.!?。！？]+$", "", left).rstrip()
    right = ". ".join(re.sub(r"[.!?。！？]+$", "", item).strip() for item in additions)
    result = f"{left}. {right}" if left else right
    return result


def evidence_is_duplicate(old: str, new: str) -> bool:
    candidate = normalized_match(new)
    return bool(candidate) and candidate in normalized_match(old)


def evidence_quality(row: dict[str, str], aliases: dict[str, list[str]] | None = None) -> float:
    """Heuristic quality score used only to reconcile an existing v2 with its baseline."""
    evidence = clean_text(row.get("evidence"), MAX_EVIDENCE_CHARS)
    concept = norm(row.get("concept"))
    normalized = norm(evidence)
    length = len(evidence)
    score = 0.0
    if length >= 160:
        score += 2.0
    elif length >= 60:
        score += 1.5
    elif length >= 35:
        score += 0.75
    elif length < 30:
        score -= 2.0
    if evidence.endswith((".", "!", "?", "。", "！", "？")):
        score += 0.25
    names = (aliases or {}).get(concept, [row.get("concept", "")])
    if any(norm(name) and norm(name) in normalized for name in names):
        score += 1.5
    if re.search(r"\b(is|are|called|referred|defined|means|method|algorithm|problem|function|process|consists|involves|used|wherein|because|therefore)\b", evidence.casefold()):
        score += 1.0
    if any(marker in evidence for marker in ["offthe", "off-", "dwd", "̊", "PP max", "firstorder", "w⋆", "≤g", "≥g", "P X", "g (w)", "max 0", "CCT", "VDVT", "∂w2", "ω ω"]):
        score -= 1.5
    words = re.findall(r"\b\w+\b", evidence.casefold())
    for size in range(5, 16):
        if any(words[index : index + size] == words[index + size : index + 2 * size] for index in range(len(words) - 2 * size + 1)):
            score -= 3.0
            break
    return round(score, 3)


def addable_new_occurrence(row: dict[str, str], aliases: dict[str, list[str]]) -> bool:
    """Reject v2-only additions that are merely a title, isolated formula, or OCR fragment."""
    evidence = clean_text(row.get("evidence"), MAX_EVIDENCE_CHARS)
    if len(evidence) < 35 or evidence_quality(row, aliases) < 3.5:
        return False
    concept = norm(row.get("concept"))
    normalized = norm(evidence)
    names = aliases.get(concept, [row.get("concept", "")])
    has_name = any(norm(name) and norm(name) in normalized for name in names)
    has_explanation = bool(re.search(r"\b(is|are|called|referred|defined|means|method|algorithm|problem|function|process|consists|involves|used|wherein|because|therefore)\b", evidence.casefold()))
    formula_like = sum(evidence.count(marker) for marker in ["=", "≤", "≥", "∑", "∂", "∇", "λ", "ω", "⋆"]) >= 2
    repeated = evidence_quality(row, aliases) <= 0
    weak_endings = {"form", "cost", "called", "referred", "where", "as", "of", "the", "a", "an", "is", "are", "to", "from", "for", "with", "and"}
    last_word = re.findall(r"[A-Za-z]+", evidence.casefold())[-1] if re.findall(r"[A-Za-z]+", evidence.casefold()) else ""
    heading_fragment = bool(re.match(r"^\s*\d+(?:\.\d+)+\s+[^.]{0,80}calculations?\b", evidence.casefold()))
    damaged_formula = any(marker in evidence for marker in ["= =", "β b +", "̊x", "̊f"])
    return (
        (has_name or has_explanation)
        and last_word not in weak_endings
        and not (formula_like and not has_explanation)
        and not heading_fragment
        and not damaged_formula
        and not repeated
    )


def reconcile_v2(root: Path, dataset: str) -> dict[str, int]:
    """Conservatively combine the original metadata with the current v2 in-place."""
    folder = root / dataset
    original_path = folder / f"{dataset}_concepts_metadata.csv"
    v2_path = folder / f"{dataset}_concepts_metadata_v2.csv"
    headers, original = read_rows(original_path)
    v2_headers, revised = read_rows(v2_path)
    if headers != v2_headers:
        raise RuntimeError(f"{dataset} original and v2 metadata schemas differ")
    aliases = load_concept_aliases(folder / f"{dataset}_concepts.csv")
    locator = lambda row: (row.get("concept", ""), row.get("chapter", row.get("lecture", "")), row.get("pdf_page", ""), row.get("book_page", ""))
    revised_by_locator: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in revised:
        revised_by_locator.setdefault(locator(row), []).append(row)

    output: list[dict[str, str]] = []
    replaced = 0
    restored = 0
    deduped = 0
    for baseline in original:
        candidates = revised_by_locator.get(locator(baseline), [])
        best = max(candidates, key=lambda row: evidence_quality(row, aliases), default=None)
        if locator(baseline) in PREFER_BASELINE_LOCATORS:
            best = None
        if best is not None and evidence_quality(best, aliases) > evidence_quality(baseline, aliases):
            chosen = dict(best)
            replaced += 1
        else:
            chosen = dict(baseline)
            if best is None:
                restored += 1
        output.append(chosen)

    existing_keys = {locator(row) for row in output}
    added = 0
    for key, candidates in revised_by_locator.items():
        if key in existing_keys:
            if len(candidates) > 1:
                deduped += len(candidates) - 1
            continue
        best = max(candidates, key=lambda row: evidence_quality(row, aliases))
        if addable_new_occurrence(best, aliases):
            output.append(dict(best))
            existing_keys.add(key)
            added += 1
        else:
            deduped += len(candidates)

    # Remove exact duplicates introduced by v2 while preserving first occurrence/order.
    unique: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    for row in output:
        marker = tuple(row.get(header, "") for header in headers)
        if marker in seen:
            deduped += 1
            continue
        seen.add(marker)
        unique.append(row)
    if unique != revised:
        atomic_write_rows(v2_path, headers, unique)
    print(f"[{dataset}] reconciliation: original={len(original)} v2_before={len(revised)} v2_after={len(unique)} replaced={replaced} restored={restored} added={added} deduped={deduped}")
    return {"original": len(original), "before": len(revised), "after": len(unique), "replaced": replaced, "restored": restored, "added": added, "deduped": deduped}


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Missing CSV header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def atomic_write_rows(path: Path, headers: Sequence[str], rows: Sequence[dict[str, str]]) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(headers), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def load_concept_aliases(path: Path) -> dict[str, list[str]]:
    aliases: dict[str, list[str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if not row or not clean_text(row[0]):
                continue
            canonical = clean_text(row[0], 200)
            aliases[norm(canonical)] = [canonical, *[clean_text(item, 200) for item in row[1:] if clean_text(item)]]
    return aliases


@dataclass
class Passage:
    page: int
    text: str
    source_type: str = "printed"
    bbox: list[float] | None = None


class MLRSource:
    def __init__(self, root: Path):
        require_fitz()
        self.mod = load_module("mlr_extract_for_audit", root / "code" / "extract_mlr.py")
        self.pdf = self.mod.PdfBook(root / "MLR" / "Machine Learning Refined, 2nd edition.pdf", validate=False)
        self.chapters = self.mod.CHAPTERS
        self.pages = {page: self.pdf.page_text(page, False) for page in range(1, len(self.pdf.document) + 1)}

    def close(self) -> None:
        self.pdf.close()

    def context(self, page: int, radius: int = 1) -> str:
        return "\n".join(self.pages.get(item, "") for item in range(max(1, page - radius), min(len(self.pages), page + radius) + 1))

    def passages(self, concept: str, aliases: Sequence[str], preferred_page: int | None = None, limit: int = 6) -> list[Passage]:
        needles = [norm(concept), *[norm(item) for item in aliases]]
        hits: list[Passage] = []
        for page, text in self.pages.items():
            lowered = norm(text)
            if not any(needle and needle in lowered for needle in needles):
                continue
            raw = text
            position = min((raw.casefold().find(item.casefold()) for item in [concept, *aliases] if item.casefold() in raw.casefold()), default=0)
            start = max(0, position - 700)
            hits.append(Passage(page, raw[start : start + 1800]))
        if preferred_page is not None:
            hits.sort(key=lambda item: (abs(item.page - preferred_page), item.page))
        return hits[:limit]

    def page_meta(self, page: int) -> dict[str, str]:
        for chapter in self.chapters:
            if chapter.pdf_start <= page <= chapter.pdf_end:
                return {"chapter": str(chapter.number), "chapter_title": chapter.title, "pdf_page": str(page), "book_page": str(page - self.mod.BOOK_PAGE_OFFSET)}
        return {"pdf_page": str(page), "book_page": str(page - self.mod.BOOK_PAGE_OFFSET)}

    def grounded(self, quote: str, page: int) -> bool:
        if len(tokens(quote)) < 3:
            return False
        source = normalized_match(self.pages.get(page, ""))
        candidate = normalized_match(quote)
        if candidate and candidate in source:
            return True
        wanted = set(tokens(quote))
        available = set(tokens(self.pages.get(page, "")))
        return bool(wanted) and len(wanted & available) / len(wanted) >= 0.90


class DGLSource:
    def __init__(self, root: Path):
        require_fitz()
        self.mod = load_module("dgl_extract_for_audit", root / "code" / "extract_dgl.py")
        self.lectures = {lecture.number: lecture for lecture in self.mod.LECTURES}
        dataset_root = root / "DGL"
        self.pdfs = {number: self.mod.AnnotatedLecturePdf(dataset_root, lecture) for number, lecture in self.lectures.items()}
        self.summaries: dict[tuple[int, int], str] = {}
        summary_path = root / "DGL" / "DGL_page_summaries.csv"
        if summary_path.exists():
            _, rows = read_rows(summary_path)
            for row in rows:
                self.summaries[(int(row["lecture"]), int(row["pdf_page"]))] = clean_text(row.get("page_title"), 200)

    def close(self) -> None:
        for pdf in self.pdfs.values():
            pdf.__exit__(None, None, None)

    def page_text(self, lecture: int, page: int) -> str:
        return self.pdfs[lecture].body_text(page)

    def page_image(self, lecture: int, page: int) -> bytes:
        return self.pdfs[lecture].render_body_jpeg(page)

    def crop_image(self, lecture: int, page: int, bbox: Sequence[float]) -> bytes:
        try:
            return self.pdfs[lecture].crop_visual_png(page, self.mod.NormalizedBox.from_value(bbox))
        except Exception:
            return b""

    def passages(self, lecture: int, page: int, concept: str, aliases: Sequence[str], limit: int = 5) -> list[Passage]:
        needles = [concept, *aliases]
        hits: list[Passage] = []
        pages = list(range(1, self.lectures[lecture].expected_pages + 1))
        pages.sort(key=lambda item: (abs(item - page), item))
        for candidate_page in pages:
            text = self.page_text(lecture, candidate_page)
            for needle in needles:
                position = text.casefold().find(needle.casefold())
                if position >= 0:
                    hits.append(Passage(candidate_page, text[max(0, position - 500) : position + 1400]))
                    break
            if len(hits) >= limit:
                break
        return hits

    def grounded(self, quote: str, lecture: int, page: int, source_type: str) -> bool:
        if source_type != "handwritten":
            source = normalized_match(self.page_text(lecture, page))
            candidate = normalized_match(quote)
            if len(tokens(quote)) >= 3 and candidate and candidate in source:
                return True
            if len(tokens(quote)) >= 3:
                wanted, available = set(tokens(quote)), set(tokens(self.page_text(lecture, page)))
                return len(wanted & available) / len(wanted) >= 0.90
        return bool(clean_text(quote))

    def meta(self, lecture: int, page: int) -> dict[str, str]:
        return {"lecture": str(lecture), "pdf_file": self.lectures[lecture].pdf_name, "pdf_page": str(page), "page_title": self.summaries.get((lecture, page), "")}


def prompt_for(group: Sequence[dict[str, Any]], passages: dict[str, list[Passage]], dataset: str) -> str:
    source_rules = (
        "For MLR, evidence must be a verbatim contiguous excerpt from the supplied PDF text."
        if dataset == "MLR"
        else "For DGL, printed evidence must be grounded in slide text; handwritten or visual evidence must be grounded in the supplied image and bbox."
    )
    return f"""Audit the supplied concept metadata rows against the source material.
{source_rules}

Classify evidence internally as mention or concept. A mention (acronym, heading, isolated formula,
caption fragment, or name) proves only that a concept appears. A concept evidence must define,
explain, characterize, formulate, or clearly use the canonical concept. Do not invent or paraphrase
source text. If evidence is reasonable but the concept lacks a core definition/role/mechanism in
the supplied source passages, append a concise missing excerpt or return ADD_ROW for another page.
If current evidence is wrong, return REPLACE or MIGRATE. If uncertain, return UNCHANGED_UNCERTAIN.
Return one decision for every row_id. Keep additions short and self-contained; do not repeat existing
sentences. Return JSON only.

ROWS:
{json.dumps(group, ensure_ascii=False)}

SOURCE PASSAGES:
{json.dumps({key: [vars(item) for item in value] for key, value in passages.items()}, ensure_ascii=False)}

JSON shape:
{{"decisions":[{{"row_id":"...","action":"KEEP|APPEND|REPLACE|ADD_ROW|MIGRATE|UNCHANGED_UNCERTAIN",
"evidence":"exact excerpt when needed","evidence_type":"mention|concept","confidence":0.0,
"target":{{"page":0,"lecture":0,"bbox":[0,0,0,0],"source_type":"printed|handwritten|mixed"}},"reason":"..."}}]}}"""


def apply_candidate(row: dict[str, str], candidate: dict[str, Any], dataset: str, source: Any, action: str) -> bool:
    quote = clean_text(candidate.get("evidence"), MAX_EVIDENCE_CHARS)
    if not quote:
        return False
    target = candidate.get("target") if isinstance(candidate.get("target"), dict) else {}
    confidence = float(candidate.get("confidence", 0.0) or 0.0)
    if confidence < MIN_LLM_CONFIDENCE:
        return False
    page = int(target.get("page") or row.get("pdf_page") or 0)
    lecture = int(target.get("lecture") or row.get("lecture") or 0)
    source_type = clean_text(target.get("source_type") or row.get("source_type") or "printed", 30)
    if dataset == "MLR":
        if not source.grounded(quote, page):
            return False
        target_meta = source.page_meta(page)
    else:
        if not source.grounded(quote, lecture, page, source_type):
            return False
        target_meta = source.meta(lecture, page)
        bbox = target.get("bbox") or row.get("bbox")
        if dataset == "DGL" and not target.get("bbox"):
            return False
        if bbox:
            try:
                source.mod.NormalizedBox.from_value(bbox)
            except (TypeError, ValueError):
                return False
            target_meta["bbox"] = "[" + ",".join(str(round(float(x), 2)) for x in bbox) + "]"
        target_meta["source_type"] = source_type
        target_meta["confidence"] = f"{confidence:.3f}"

    if action == "APPEND":
        if dataset == "MLR" and page != int(row.get("pdf_page") or 0):
            return False
        if dataset == "DGL" and (lecture != int(row.get("lecture") or 0) or page != int(row.get("pdf_page") or 0)):
            return False
        if evidence_is_duplicate(row.get("evidence", ""), quote):
            return False
        joined = append_sentences(row.get("evidence", ""), quote)
        if len(joined) > MAX_EVIDENCE_CHARS:
            return False
        row["evidence"] = joined
        if dataset == "DGL" and target_meta.get("bbox") and row.get("bbox"):
            row["bbox"] = union_bbox(row["bbox"], target_meta["bbox"])
            if row.get("source_type") != source_type:
                row["source_type"] = "mixed"
            row["confidence"] = f"{min(float(row.get('confidence') or 0), confidence):.3f}"
        return True

    row["evidence"] = quote
    row.update({key: value for key, value in target_meta.items() if value != ""})
    return True


def union_bbox(left: str, right: str) -> str:
    def parse(value: str) -> list[float]:
        return [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", value)]
    a, b = parse(left), parse(right)
    if len(a) != 4 or len(b) != 4:
        return left
    return "[" + ",".join(str(round(x, 2)) for x in [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]) + "]"


def run_dataset(root: Path, dataset: str, apply: bool, dry_run: bool, client: Any | None) -> dict[str, int]:
    folder = root / dataset
    metadata = folder / f"{dataset}_concepts_metadata_v2.csv"
    concepts_path = folder / f"{dataset}_concepts.csv"
    print(f"[{dataset}] Reading metadata v2 and concept aliases…")
    headers, rows = read_rows(metadata)
    aliases = load_concept_aliases(concepts_path)
    if dry_run:
        groups = {(str(row.get("chapter") or row.get("lecture") or ""), str(row.get("pdf_page") or "")) for row in rows}
        print(f"{dataset}: {len(rows)} metadata rows, {len(groups)} source-page groups; API calls skipped")
        return {"rows": len(rows), **{action: 0 for action in ["KEEP", "APPEND", "REPLACE", "ADD_ROW", "MIGRATE", "UNCHANGED_UNCERTAIN"]}}
    print(f"[{dataset}] Opening source {'PDF' if dataset == 'MLR' else 'slides'}…")
    source = MLRSource(root) if dataset == "MLR" else DGLSource(root)
    try:
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for index, row in enumerate(rows):
            key = (str(row.get("chapter") or row.get("lecture") or ""), str(row.get("pdf_page") or ""))
            groups.setdefault(key, []).append({"row_id": str(index), **row})
        counts = {action: 0 for action in ["KEEP", "APPEND", "REPLACE", "ADD_ROW", "MIGRATE", "UNCHANGED_UNCERTAIN"]}
        additions: list[dict[str, str]] = []
        changed = False
        if client is None:
            raise RuntimeError(f"No LLM client configured for {dataset}")
        progress = ConsoleProgress(f"{dataset} DeepSeek audit", len(groups))
        progress.start()
        for group_index, group_rows in enumerate(groups.values(), start=1):
            location = f"page {group_rows[0].get('pdf_page', '?')}"
            if dataset == "DGL":
                location = f"lecture {group_rows[0].get('lecture', '?')}, {location}"
            progress.update(group_index - 1, f"auditing {location}")
            passages: dict[str, list[Passage]] = {}
            all_occurrences: dict[str, list[dict[str, str]]] = {}
            for candidate_index, candidate_row in enumerate(rows):
                key = norm(candidate_row.get("concept"))
                all_occurrences.setdefault(key, []).append(
                    {"row_id": str(candidate_index), "page": candidate_row.get("pdf_page", ""), "evidence": candidate_row.get("evidence", "")}
                )
            for item in group_rows:
                concept = item.get("concept", "")
                names = aliases.get(norm(concept), [concept])
                item["same_concept_occurrences"] = all_occurrences.get(norm(concept), [])
                if dataset == "MLR":
                    passages[item["row_id"]] = source.passages(concept, names[1:], preferred_page=int(item.get("pdf_page") or 0))
                else:
                    passages[item["row_id"]] = source.passages(int(item.get("lecture") or 0), int(item.get("pdf_page") or 0), concept, names[1:])
            payload = prompt_for(group_rows, passages, dataset)
            if dataset == "MLR":
                response, _ = client.complete_json("You are a source-grounded evidence auditor. Return JSON only.", payload)
            else:
                images = [source.page_image(int(group_rows[0].get("lecture") or 0), int(group_rows[0].get("pdf_page") or 0))]
                response, _ = client.complete_json("You are a source-grounded multimodal evidence auditor. Return JSON only.", payload, images)
            decisions = response.get("decisions", []) if isinstance(response, dict) else []
            by_id = {str(item.get("row_id")): item for item in decisions if isinstance(item, dict)}
            for item in group_rows:
                action = str(by_id.get(item["row_id"], {}).get("action") or "UNCHANGED_UNCERTAIN").upper()
                if action not in VALID_ACTIONS:
                    action = "UNCHANGED_UNCERTAIN"
                counts[action] += 1
                decision = by_id.get(item["row_id"], {})
                if not apply and action != "KEEP":
                    progress.message(
                        f"{dataset} row={item['row_id']} action={action} "
                        f"concept={item.get('concept', '')!r} reason={clean_text(decision.get('reason'), 500)}"
                    )
                if action in {"APPEND", "REPLACE", "MIGRATE"}:
                    if apply_candidate(rows[int(item["row_id"])], by_id[item["row_id"]], dataset, source, action):
                        changed = True
                elif action == "ADD_ROW":
                    candidate = by_id.get(item["row_id"], {})
                    new_row = dict(rows[int(item["row_id"])])
                    if apply_candidate(new_row, candidate, dataset, source, "REPLACE"):
                        additions.append(new_row)
                        changed = True
            progress.update(group_index, f"audited {location}")
        progress.finish()
        if additions:
            print(f"[{dataset}] Adding {len(additions)} grounded cross-page evidence rows…")
            rows.extend(additions)
        if apply and changed:
            print(f"[{dataset}] Writing updated metadata v2 atomically…")
            atomic_write_rows(metadata, headers, rows)
        print(f"{dataset}: rows={len(rows)} actions={json.dumps(counts, ensure_ascii=False)} changed={changed} written={bool(apply and changed)}")
        return {"rows": len(rows), **counts}
    finally:
        source.close()


def build_client(dataset: str, root: Path) -> Any:
    values = load_env_values(root / ".env")
    del dataset
    key = os.getenv("DEEPSEEK_API_KEY", values.get("DEEPSEEK_API_KEY", "")).strip()
    base = os.getenv("DEEPSEEK_BASE_URL", values.get("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL)).strip()
    model = os.getenv("DEEPSEEK_MODEL", values.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)).strip()
    timeout = int(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", values.get("DEEPSEEK_TIMEOUT_SECONDS", "180")))
    return DeepSeekClient(key, base, model, timeout=timeout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["MLR", "DGL", "both"], default="both")
    parser.add_argument("--dry-run", action="store_true", help="inspect v2 files without API calls or writes")
    parser.add_argument("--apply", action="store_true", help="write only changed rows to metadata v2")
    parser.add_argument("--reconcile-v2", action="store_true", help="merge original metadata strengths into existing v2 without API calls")
    args = parser.parse_args()
    if sum(bool(value) for value in [args.dry_run, args.apply, args.reconcile_v2]) > 1:
        parser.error("--dry-run, --apply and --reconcile-v2 are mutually exclusive")
    datasets = ["MLR", "DGL"] if args.dataset == "both" else [args.dataset]
    if args.reconcile_v2:
        for dataset in datasets:
            reconcile_v2(ROOT, dataset)
        return 0
    for dataset in datasets:
        client = None if args.dry_run else build_client(dataset, ROOT)
        run_dataset(ROOT, dataset, args.apply, args.dry_run, client)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
