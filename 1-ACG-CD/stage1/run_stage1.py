from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from src.config import apply_cli_overrides, load_config
from src.pipeline import run_stage1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage1: split gold positives and construct initial negatives.")
    parser.add_argument("--config", default=str(THIS_DIR / "config" / "config.yaml"), help="Path to Stage1 config YAML.")
    parser.add_argument("--project-root", default=None, help="Project root. Overrides config project_root and PROJECT_ROOT env.")
    parser.add_argument("--dataset-name", choices=["moocml", "lecturebank", "universitycourse", "mlr", "dgl", "mc_lb_uc"], default=None, help="Dataset preset to use; overrides config dataset_name.")
    parser.add_argument("--train-ratio", type=float, choices=[0.15, 0.30, 0.60], default=None, help="Training positive-label ratio: 0.15, 0.30, or 0.60.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--output-dir", default=None, help="Override output directory; relative paths resolve under stage1/.")
    parser.add_argument("--reverse-negative-count", type=int, default=None, help="Override reverse negative count.")
    parser.add_argument("--hard-negative-count", type=int, default=None, help="Override hard negative count.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = apply_cli_overrides(load_config(args.config, project_root_override=args.project_root), args)
    summary = run_stage1(cfg)
    print(
        json.dumps(
            {
                "stage": summary["stage"],
                "concept_count": summary["concept_count"],
                "split_counts": summary["split_counts"],
                "output_dir": cfg["output_dir"],
                "leakage_guard": summary["leakage_guard"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

