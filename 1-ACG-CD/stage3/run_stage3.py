from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
STAGE2_DIR = PROJECT_ROOT / "stage2"
for path in [str(STAGE2_DIR), str(THIS_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from inference import apply_cli_overrides, load_stage3_config, run_stage3
from src.utils import resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage3 inference on Dtest.")
    parser.add_argument("--config", default=str(THIS_DIR / "config" / "config.yaml"), help="Path to Stage3 config YAML.")
    parser.add_argument("--project-root", default=None, help="Project root. Overrides config project_root and PROJECT_ROOT env.")
    parser.add_argument("--dataset-name", choices=["moocml", "lecturebank", "universitycourse", "mlr", "dgl", "mc_lb_uc"], default=None, help="Dataset preset to use; overrides config dataset_name.")
    parser.add_argument("--device", default=None, help="Override device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--output-dir", default=None, help="Override Stage3 output directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = apply_cli_overrides(load_stage3_config(args.config, project_root_override=args.project_root), args)
    device = resolve_device(str(cfg.get("device", "auto")))
    summary = run_stage3(cfg, device)
    print(
        json.dumps(
            {
                "precision": summary["test_metrics"]["precision"],
                "recall": summary["test_metrics"]["recall"],
                "f1": summary["test_metrics"]["f1"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
