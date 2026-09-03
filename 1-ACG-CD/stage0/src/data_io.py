from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)?")


@dataclass(frozen=True)
class Concept:
    concept_id: int
    name: str
    aliases: Tuple[str, ...]


def clean_field(value: object) -> str:
    return " ".join(str(value).strip().lower().split())


def _dedupe_preserve_order(values: Iterable[str]) -> Tuple[str, ...]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        item = clean_field(value)
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return tuple(out)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("\u200b", " ")
    text = text.replace("_", " ")
    return text


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(normalize_text(text))


def phrase_token(phrase: str) -> str:
    toks = tokenize(phrase)
    return "__phrase__" + "_".join(toks)


def read_concepts(path: str | Path) -> List[Concept]:
    path = Path(path)
    concepts: List[Concept] = []
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.reader(f):
                aliases = _dedupe_preserve_order(cell for cell in row if str(cell).strip())
                if not aliases:
                    continue
                concepts.append(Concept(concept_id=len(concepts), name=aliases[0], aliases=aliases))
        return concepts

    for line in path.read_text(encoding="utf-8-sig").splitlines():
        aliases = _dedupe_preserve_order(p for p in line.split("::;") if p.strip())
        if not aliases:
            continue
        concepts.append(Concept(concept_id=len(concepts), name=aliases[0], aliases=aliases))
    return concepts


def iter_caption_records(paths: Sequence[str | Path]) -> Iterable[Dict]:
    for path in paths:
        with Path(path).open("r", encoding="utf-8-sig") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc


def _read_caption_csv(path: Path) -> Tuple[List[str], int]:
    texts: List[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        if not sample.strip():
            return texts, 0
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return texts, 0
        fieldnames = [str(name) for name in reader.fieldnames if name is not None]
        preferred = [name for name in ["title", "category", "text"] if name in fieldnames]
        for row in reader:
            values = []
            for key in (preferred or fieldnames):
                value = row.get(key, "")
                if value is not None and str(value).strip():
                    values.append(str(value).strip())
            if values:
                texts.append(" ".join(values))
    return texts, len(texts)


def read_caption_texts(data_dir: str | Path, caption_files: Sequence[str | Path] | None = None) -> Tuple[List[str], Dict[str, int]]:
    data_dir = Path(data_dir)
    if caption_files is None:
        caption_files = ["captions.jsonl"]
    files = [Path(file) if Path(file).is_absolute() else data_dir / file for file in caption_files]
    texts: List[str] = []
    per_file: Dict[str, int] = {}
    for file in files:
        if not file.exists():
            per_file[file.name] = 0
            continue
        if file.suffix.lower() == ".csv":
            csv_texts, count = _read_caption_csv(file)
            texts.extend(csv_texts)
            per_file[file.name] = count
            continue
        count = 0
        for obj in iter_caption_records([file]):
            title = str(obj.get("title", ""))
            text = str(obj.get("text", ""))
            category = " ".join(str(x) for x in obj.get("category", [])) if isinstance(obj.get("category"), list) else ""
            texts.append(" ".join(part for part in [title, category, text] if part))
            count += 1
        per_file[file.name] = count
    return texts, per_file


def parse_label_stats(path: str | Path) -> Dict[str, int]:
    path = Path(path)
    counts: Counter[str] = Counter()
    directed_positive_edges = []
    rows = 0
    invalid_rows = 0

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.reader(f):
                if not row or not any(str(cell).strip() for cell in row):
                    continue
                if len(row) != 3:
                    invalid_rows += 1
                    continue
                a, b, label = (clean_field(cell) for cell in row)
                if not a or not b or label not in {"0", "1"}:
                    invalid_rows += 1
                    continue
                rows += 1
                counts[label] += 1
                if label == "1":
                    directed_positive_edges.append((a, b))
        return {
            "rows": rows,
            "label_1": counts.get("1", 0),
            "label_0": counts.get("0", 0),
            "invalid_rows": invalid_rows,
            "directed_positive_edges": len(directed_positive_edges),
            "unique_directed_positive_edges": len(set(directed_positive_edges)),
        }

    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            parts = [clean_field(p) for p in re.split(r"\t+", line.strip()) if p.strip()]
            if len(parts) < 3:
                invalid_rows += 1
                continue
            a, b, label = parts[:3]
            rows += 1
            counts[label] += 1
            if label == "1-":
                directed_positive_edges.append((a, b))
            elif label == "-1":
                directed_positive_edges.append((b, a))
    return {
        "rows": rows,
        "label_1-": counts.get("1-", 0),
        "label_-1": counts.get("-1", 0),
        "label_-": counts.get("-", 0),
        "invalid_rows": invalid_rows,
        "directed_positive_edges": len(directed_positive_edges),
        "unique_directed_positive_edges": len(set(directed_positive_edges)),
    }
