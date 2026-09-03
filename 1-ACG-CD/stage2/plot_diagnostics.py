from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from src.diagnostic_plots import generate_diagnostic_plots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Stage2 diagnostic plots from saved CSV histories.")
    parser.add_argument(
        "--output-dir",
        default=str(THIS_DIR / "outputs"),
        help="Stage2 output directory containing training_history_content.csv, training_history_graph.csv, and akd_round_diagnostics.csv.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    content_path = output_dir / "training_history_content.csv"
    graph_path = output_dir / "training_history_graph.csv"
    rounds_path = output_dir / "akd_round_diagnostics.csv"
    missing = [str(p) for p in [content_path, graph_path, rounds_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing diagnostic CSV file(s). Stage2 must finish at least once with the new logging code. Missing: "
            + ", ".join(missing)
        )
    plot_paths = generate_diagnostic_plots(
        output_dir=output_dir,
        content_history=pd.read_csv(content_path),
        graph_history=pd.read_csv(graph_path),
        round_diagnostics=pd.read_csv(rounds_path),
    )
    print("Generated diagnostic plots:")
    for path in plot_paths:
        print(path)


if __name__ == "__main__":
    main()
