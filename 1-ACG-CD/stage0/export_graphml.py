from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

THIS_DIR = Path(__file__).resolve().parent
GRAPHML_NS = "http://graphml.graphdrawing.org/xmlns"
ET.register_namespace("", GRAPHML_NS)
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env_vars(value: str) -> str:
    def repl(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        return os.environ.get(name, default if default is not None else "")
    return _ENV_PATTERN.sub(repl, value)


def resolve_project_root(project_root_text: str | None) -> Path:
    raw = project_root_text or "${PROJECT_ROOT:-.}"
    root = Path(expand_env_vars(raw)).expanduser().resolve()
    if not (root / "stage0").exists():
        raise FileNotFoundError(
            f"Invalid project_root={root}. Run from the project root, set PROJECT_ROOT, "
            "or pass --project-root /path/to/KG4Ed."
        )
    return root


def _resolve_path(path_text: str | None, default: Path, project_root: Path) -> Path:
    p = Path(path_text) if path_text else default
    if not p.is_absolute():
        p = project_root / p
    return p.expanduser().resolve()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _add_key(root: ET.Element, key_id: str, for_: str, attr_name: str, attr_type: str) -> None:
    ET.SubElement(root, f"{{{GRAPHML_NS}}}key", {"id": key_id, "for": for_, "attr.name": attr_name, "attr.type": attr_type})


def _add_data(parent: ET.Element, key: str, value: object) -> None:
    data = ET.SubElement(parent, f"{{{GRAPHML_NS}}}data", {"key": key})
    data.text = "" if value is None else str(value)


def export_graphml(outputs_dir: Path, graphml_path: Path) -> Path:
    concepts = _read_csv(outputs_dir / "concepts.csv")
    edges = _read_csv(outputs_dir / "edges_knn.csv")
    root = ET.Element(f"{{{GRAPHML_NS}}}graphml")
    _add_key(root, "d0", "node", "concept_id", "int")
    _add_key(root, "d1", "node", "concept", "string")
    _add_key(root, "d2", "node", "aliases", "string")
    _add_key(root, "d3", "node", "embedding_source", "string")
    _add_key(root, "d4", "node", "embedding_token", "string")
    _add_key(root, "d5", "edge", "weight", "double")
    _add_key(root, "d6", "edge", "source_concept", "string")
    _add_key(root, "d7", "edge", "target_concept", "string")
    graph = ET.SubElement(root, f"{{{GRAPHML_NS}}}graph", {"id": "stage0_initial_weak_semantic_graph", "edgedefault": "undirected"})
    seen_node_ids: set[str] = set()
    for row in concepts:
        node_id = str(row["concept_id"])
        seen_node_ids.add(node_id)
        node = ET.SubElement(graph, f"{{{GRAPHML_NS}}}node", {"id": f"n{node_id}"})
        _add_data(node, "d0", node_id)
        _add_data(node, "d1", row.get("concept", ""))
        _add_data(node, "d2", row.get("aliases", ""))
        _add_data(node, "d3", row.get("embedding_source", ""))
        _add_data(node, "d4", row.get("embedding_token", ""))
    for edge_idx, row in enumerate(edges):
        source_id = str(row["source_id"])
        target_id = str(row["target_id"])
        if source_id not in seen_node_ids or target_id not in seen_node_ids:
            raise ValueError(f"Edge references unknown node: {row}")
        edge = ET.SubElement(graph, f"{{{GRAPHML_NS}}}edge", {"id": f"e{edge_idx}", "source": f"n{source_id}", "target": f"n{target_id}"})
        _add_data(edge, "d5", row.get("weight", ""))
        _add_data(edge, "d6", row.get("source", ""))
        _add_data(edge, "d7", row.get("target", ""))
    graphml_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    tree.write(graphml_path, encoding="utf-8", xml_declaration=True)
    return graphml_path


DATASET_OUTPUT_SUFFIXES = {
    "moocml": "_moocml",
    "lecturebank": "_lecturebank",
    "universitycourse": "_universitycourse",
}


def _stage0_outputs_for_dataset(dataset_name: str) -> Path:
    key = dataset_name.lower()
    if key not in DATASET_OUTPUT_SUFFIXES:
        raise ValueError(
            f"Unknown dataset_name={dataset_name!r}. Expected one of {sorted(DATASET_OUTPUT_SUFFIXES)}."
        )
    return Path(f"stage0/outputs{DATASET_OUTPUT_SUFFIXES[key]}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Stage0 concepts.csv and edges_knn.csv to GraphML.")
    parser.add_argument("--project-root", default=None, help="Project root. Overrides PROJECT_ROOT env.")
    parser.add_argument("--dataset-name", choices=["moocml", "lecturebank", "universitycourse"], default="moocml", help="Dataset preset used when --outputs-dir/--out are omitted.")
    parser.add_argument("--outputs-dir", default=None, help="Stage0 outputs directory containing concepts.csv and edges_knn.csv.")
    parser.add_argument("--out", default=None, help="Destination .graphml file path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = resolve_project_root(args.project_root)
    default_outputs = _stage0_outputs_for_dataset(args.dataset_name)
    outputs_dir = _resolve_path(args.outputs_dir, default_outputs, project_root)
    graphml_path = _resolve_path(args.out, default_outputs / "initial_graph.graphml", project_root)
    print(f"GraphML written to: {export_graphml(outputs_dir, graphml_path)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
