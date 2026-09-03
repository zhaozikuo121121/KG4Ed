from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASETS = ("moocml", "lecturebank", "universitycourse", "mlr", "dgl", "mc_lb_uc")
TRANSFER_CONFIGS = (
    "config_moocml_to_lecturebank.yaml",
    "config_moocml_to_universitycourse.yaml",
    "config_lecturebank_to_moocml.yaml",
    "config_lecturebank_to_universitycourse.yaml",
    "config_mc_lb_uc_to_mlr.yaml",
    "config_mc_lb_uc_to_dgl.yaml",
)


def build_commands(device: str) -> list[list[str]]:
    commands: list[list[str]] = [[sys.executable, "tools/prepare_mc_lb_uc.py"]]
    for dataset in DATASETS:
        commands.append([sys.executable, f"stage0/run_stage0.py", "--config", f"stage0/config/config_{dataset}.yaml", "--device", device])
    for dataset in DATASETS:
        for ratio in ("0.15", "0.30", "0.60"):
            commands.append([sys.executable, "stage1/run_stage1.py", "--config", f"stage1/config/config_{dataset}.yaml", "--train-ratio", ratio])
    for dataset in DATASETS:
        commands.append([sys.executable, "stage2/run_stage2.py", "--config", "stage2/config.yaml", "--dataset-name", dataset, "--device", device])
    for dataset in DATASETS:
        commands.append([sys.executable, "stage3/run_stage3.py", "--config", f"stage3/config/config_{dataset}.yaml", "--device", device])
    for config_name in TRANSFER_CONFIGS:
        commands.append([sys.executable, "stage3/run_stage3.py", "--config", f"stage3/config/{config_name}", "--device", device])
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description="Run every registered Stage0-Stage3 training and evaluation command.")
    parser.add_argument("--device", default="auto", help="Device forwarded to Stage0, Stage2 and Stage3.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue after a failed command.")
    args = parser.parse_args()
    commands = build_commands(args.device)
    failures = 0
    for index, command in enumerate(commands, start=1):
        display = " ".join(command)
        print(f"[{index}/{len(commands)}] {display}", flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode != 0:
            failures += 1
            if not args.continue_on_error:
                raise SystemExit(result.returncode)
    if failures:
        raise SystemExit(f"{failures} command(s) failed")


if __name__ == "__main__":
    main()
