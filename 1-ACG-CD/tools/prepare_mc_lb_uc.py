from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    ("mc", "MOOCML", "MOOCML_concepts.csv", "MOOCML_prerequisites.csv", "stage0/outputs_moocml"),
    ("lb", "LectureBank", "LectureBank_concepts.csv", "LectureBank_prerequisites.csv", "stage0/outputs_lecturebank"),
    ("uc", "UniversityCourse", "UniversityCourse_concepts.csv", "UniversityCourse_prerequisites.csv", "stage0/outputs_universitycourse"),
]


def clean(value: object) -> str:
    return " ".join(str(value).strip().lower().split())


def main() -> None:
    out = ROOT / "data/MC_LB_UC"
    out.mkdir(parents=True, exist_ok=True)
    concepts_rows: list[list[str]] = []
    labels: list[tuple[str, str, str]] = []
    for prefix, data_name, concept_name, label_name, stage0_rel in SOURCES:
        concept_path = ROOT / "data" / data_name / concept_name
        if not concept_path.exists():
            raise FileNotFoundError(concept_path)
        name_map: dict[str, str] = {}
        with concept_path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.reader(handle):
                aliases = [clean(cell) for cell in row if clean(cell)]
                if not aliases:
                    continue
                renamed = [f"{prefix}:{alias}" for alias in aliases]
                for alias, renamed_alias in zip(aliases, renamed):
                    name_map[alias] = renamed_alias
                concepts_rows.append(renamed)
        label_path = ROOT / "data" / data_name / label_name
        with label_path.open(encoding="utf-8-sig", newline="") as handle:
            if label_path.suffix.lower() == ".csv":
                rows = csv.reader(handle)
                for row in rows:
                    if len(row) != 3:
                        continue
                    a, b, value = (clean(x) for x in row)
                    if a in name_map and b in name_map and value in {"0", "1"}:
                        labels.append((name_map[a], name_map[b], value))
            else:
                for raw in handle:
                    parts = [clean(x) for x in raw.rstrip().split("\t") if clean(x)]
                    if len(parts) < 3:
                        continue
                    a, b, value = parts[:3]
                    if a in name_map and b in name_map:
                        labels.append((name_map[a], name_map[b], "1" if value == "1-" else "0"))
    with (out / "MC_LB_UC_concepts.csv").open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(concepts_rows)
    with (out / "MC_LB_UC_prerequisites.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(labels)
    pd.DataFrame({"dataset": [p for p, *_ in SOURCES], "concept_count": [sum(1 for row in concepts_rows if row[0].startswith(f"{p}:")) for p, *_ in SOURCES]}).to_csv(out / "manifest.csv", index=False)
    print(f"wrote {len(concepts_rows)} concepts and {len(labels)} labels to {out}")


if __name__ == "__main__":
    main()
