from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pandas as pd


def _import_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _ensure_plot_dir(output_dir: str | Path) -> Path:
    plot_dir = Path(output_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def _write_placeholder_png(path: Path, width: int = 640, height: int = 360) -> None:
    """Write a minimal valid PNG if matplotlib is unavailable in a smoke-test env."""
    raw_rows = []
    for _ in range(height):
        raw_rows.append(b"\x00" + b"\xff\xff\xff" * width)
    raw = b"".join(raw_rows)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def _round_boundaries(history: pd.DataFrame) -> list[tuple[float, int]]:
    if history.empty or "round" not in history.columns or "global_epoch" not in history.columns:
        return []
    out: list[tuple[float, int]] = []
    grouped = history.groupby("round")["global_epoch"].max().sort_index()
    for round_value, end_epoch in grouped.iloc[:-1].items():
        out.append((float(end_epoch), int(round_value)))
    return out


def _annotate_rounds(ax, history: pd.DataFrame) -> None:
    if history.empty or "round" not in history.columns or "global_epoch" not in history.columns:
        return
    ymax = ax.get_ylim()[1]
    for x, _round_value in _round_boundaries(history):
        ax.axvline(x=x, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    for round_value, group in history.groupby("round", sort=True):
        lo = float(group["global_epoch"].min())
        hi = float(group["global_epoch"].max())
        ax.text((lo + hi) / 2.0, ymax, f"Round {int(round_value)}", ha="center", va="top", fontsize=9)


def _plot_train_val_loss(history: pd.DataFrame, title: str, output_path: Path) -> None:
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(11, 5))
    if not history.empty:
        ax.plot(history["global_epoch"], history["train_total_loss"], label="train loss", color="#1f77b4")
        ax.plot(history["global_epoch"], history["val_bce_loss"], label="val loss", color="#ff7f0e")
        _annotate_rounds(ax, history)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_graph_components(history: pd.DataFrame, output_path: Path) -> None:
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(12, 5.5))
    components = [
        "gold_bce",
        "pseudo_soft_bce",
        "kd_bce",
        "rank_loss",
        "logit_cap_penalty",
    ]
    if not history.empty:
        for col in components:
            if col in history.columns:
                ax.plot(history["global_epoch"], history[col], label=col)
        _annotate_rounds(ax, history)
    ax.set_title("Model 2 (Graph) loss components")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss component value")
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_auc_by_round(diagnostics: pd.DataFrame, output_path: Path) -> None:
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(8, 5))
    if not diagnostics.empty:
        ax.plot(diagnostics["round"], diagnostics["content_val_auc"], marker="o", label="Model 1 content val AUC")
        ax.plot(diagnostics["round"], diagnostics["graph_val_auc"], marker="o", label="Model 2 graph val AUC")
        ax.set_xticks(diagnostics["round"].astype(int).tolist())
    ax.set_title("Validation AUC by AKD round")
    ax.set_xlabel("AKD Round")
    ax.set_ylabel("Val AUC")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_pseudo_quantity_quality(diagnostics: pd.DataFrame, output_path: Path) -> None:
    plt = _import_pyplot()
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    if not diagnostics.empty:
        ax1.bar(
            diagnostics["round"],
            diagnostics["r_syn_total"],
            alpha=0.35,
            color="#1f77b4",
            label="R_syn_pos + R_syn_neg",
        )
        ax2.plot(
            diagnostics["round"],
            diagnostics["pseudo_label_confidence_mean"],
            marker="o",
            color="#d62728",
            label="mean pseudo confidence",
        )
        if "pseudo_llm_score_mean" in diagnostics.columns and diagnostics["pseudo_llm_score_mean"].notna().any():
            ax2.plot(
                diagnostics["round"],
                diagnostics["pseudo_llm_score_mean"],
                marker="s",
                color="#2ca02c",
                label="mean LLM score",
            )
        ax1.set_xticks(diagnostics["round"].astype(int).tolist())
    ax1.set_title("Pseudo-label quantity and quality by AKD round")
    ax1.set_xlabel("AKD Round")
    ax1.set_ylabel("Pseudo-label count")
    ax2.set_ylabel("Quality / confidence")
    ax2.set_ylim(0.0, 1.0)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax1.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def generate_diagnostic_plots(
    output_dir: str | Path,
    content_history: pd.DataFrame,
    graph_history: pd.DataFrame,
    round_diagnostics: pd.DataFrame,
) -> list[str]:
    plot_dir = _ensure_plot_dir(output_dir)
    paths = {
        "content_train_val_loss": plot_dir / "content_train_val_loss.png",
        "graph_train_val_loss": plot_dir / "graph_train_val_loss.png",
        "graph_loss_components": plot_dir / "graph_loss_components.png",
        "akd_val_auc_by_round": plot_dir / "akd_val_auc_by_round.png",
        "pseudo_label_quantity_quality": plot_dir / "pseudo_label_quantity_quality.png",
    }
    try:
        _plot_train_val_loss(content_history, "Model 1 (Content) train/val loss", paths["content_train_val_loss"])
        _plot_train_val_loss(graph_history, "Model 2 (Graph) train/val loss", paths["graph_train_val_loss"])
        _plot_graph_components(graph_history, paths["graph_loss_components"])
        _plot_auc_by_round(round_diagnostics, paths["akd_val_auc_by_round"])
        _plot_pseudo_quantity_quality(round_diagnostics, paths["pseudo_label_quantity_quality"])
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        for path in paths.values():
            _write_placeholder_png(path)
    return [str(path) for path in paths.values()]
