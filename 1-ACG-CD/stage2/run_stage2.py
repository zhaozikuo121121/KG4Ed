from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from src.akd_loop import run_akd
from src.utils import apply_cli_overrides, load_config, resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage2 AKD training/fusion pipeline.")
    parser.add_argument("--config", default=str(THIS_DIR / "config.yaml"), help="Path to Stage2 config YAML.")
    parser.add_argument("--project-root", default=None, help="Project root. Overrides config project_root and PROJECT_ROOT env.")
    parser.add_argument("--dataset-name", choices=["moocml", "lecturebank", "universitycourse", "mlr", "dgl", "mc_lb_uc"], required=True, help="Dataset preset used to resolve input, output, and checkpoint directories.")
    parser.add_argument("--device", default=None, help="Override device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--output-dir", default=None, help="Override output directory; relative paths resolve under stage2/.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = apply_cli_overrides(load_config(args.config, project_root_override=args.project_root), args)
    device = resolve_device(str(cfg.get("device", "auto")))
    summary = run_akd(cfg, device)
    print(json.dumps({
        "stage": summary["stage"],
        "rounds_run": summary["rounds_run"],
        "selected_alpha": summary["fusion_params"]["selected_alpha"],
        "selected_threshold": summary["fusion_params"]["selected_threshold"],
        "stage2_outputs_dir": cfg["stage2_outputs_dir"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

