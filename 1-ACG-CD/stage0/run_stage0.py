from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `python stage0/run_stage0.py` work without installing the package.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from src.config import apply_cli_overrides, load_config
from src.pipeline import run_stage0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage0: BERT concept/profile embeddings, cosine similarity, and kNN weak graph.")
    parser.add_argument("--config", default=str(THIS_DIR / "config" / "config.yaml"), help="Path to Stage0 config YAML.")
    parser.add_argument("--project-root", default=None, help="Project root. Overrides config project_root and PROJECT_ROOT env.")
    parser.add_argument("--dataset-name", choices=["moocml", "lecturebank", "universitycourse", "mlr", "dgl", "mc_lb_uc"], default=None, help="Dataset preset to use; overrides config dataset_name.")
    parser.add_argument("--epochs", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--max-pairs-per-epoch",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--output-dir", default=None, help="Override output directory; relative paths resolve under stage0/.")
    parser.add_argument("--device", default=None, help="Override torch device: auto, cpu, cuda, cuda:0, ...")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(args.config, project_root_override=args.project_root)
    cfg = apply_cli_overrides(cfg, args)
    if cfg.get("max_pairs_per_epoch") == 0:
        cfg["max_pairs_per_epoch"] = None
    summary = run_stage0(cfg)
    print(json.dumps({
        "stage": summary["stage"],
        "concept_count": summary["concept_count"],
        "embedding_shape": summary["embedding_shape"],
        "graph_stats": summary["graph_stats"],
        "output_dir": cfg["output_dir"],
        "warnings": summary["warnings"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

