"""Regenerate the CD-diagram figures used on the docs home page.

Produces a light-mode and a dark-mode PNG from the same synthetic
BenchmarkReport so the two figures share layout/data exactly.

Run:  ../.venv/bin/python docs/_static/_gen_cd_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from graphnetz.benchmark import BenchmarkReport  # noqa: E402

# Per-task accuracy means engineered to reproduce mean Friedman ranks of
# GCN=1.67, GraphSAGE=2.00, GraphTransformer=2.67, GAT=3.67 across three
# tasks (matches the figure that was previously committed to the repo).
TASK_MEANS = {
    "task1": {"GCN": 0.90, "GraphSAGE": 0.80, "GraphTransformer": 0.70, "GAT": 0.60},
    "task2": {"GCN": 0.90, "GraphSAGE": 0.80, "GAT": 0.70, "GraphTransformer": 0.60},
    "task3": {"GraphTransformer": 0.90, "GraphSAGE": 0.80, "GCN": 0.70, "GAT": 0.60},
}


def _histories():
    histories = {}
    for task, model_means in TASK_MEANS.items():
        histories[task] = {model: [{"test_acc": [acc]} for _ in range(5)] for model, acc in model_means.items()}
    return histories


def _save_light(out: Path) -> None:
    report = BenchmarkReport(seeds=(0, 1, 2, 3, 4), histories=_histories())
    fig, _ = report.plot_critical_difference(alpha=0.05)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _recolor_for_dark(fig: plt.Figure) -> None:
    """Walk the figure and swap dark on light for cream on dark."""
    fg = "#EAE0D5"
    muted = "#aaa49a"

    # Transparent canvas so the surrounding page colour (whatever Furo's
    # current --color-background-primary happens to be) shows through.
    fig.patch.set_alpha(0.0)
    for ax in fig.get_axes():
        ax.patch.set_alpha(0.0)
        for spine in ax.spines.values():
            spine.set_edgecolor(fg)
        ax.tick_params(colors=fg, which="both")
        ax.xaxis.label.set_color(fg)
        ax.yaxis.label.set_color(fg)
        ax.title.set_color(fg)
        for text in ax.texts:
            current = mpl.colors.to_hex(text.get_color()).lower()
            # Re-color labels: keep the per-model accent colors from the
            # original cycle, swap any near-black text to the foreground
            # cream, swap any near-grey text (Friedman header, axis caption)
            # to the muted cream.
            if current in {"#22333b", "#0a0908", "#000000"}:
                text.set_color(fg)
            elif current in {"#666666", "#7f7f7f", "#808080"}:
                text.set_color(muted)
        for line in ax.lines:
            current = mpl.colors.to_hex(line.get_color()).lower()
            if current in {"#22333b", "#0a0908", "#000000"}:
                line.set_color(fg)
            elif current in {"#666666", "#7f7f7f", "#808080"}:
                line.set_color(muted)


def _save_dark(out: Path) -> None:
    report = BenchmarkReport(seeds=(0, 1, 2, 3, 4), histories=_histories())
    fig, _ = report.plot_critical_difference(alpha=0.05)
    _recolor_for_dark(fig)
    fig.savefig(
        out,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
        transparent=True,
    )
    plt.close(fig)


if __name__ == "__main__":
    static = Path(__file__).resolve().parent
    _save_light(static / "critical_difference.png")
    _save_dark(static / "critical_difference_dark.png")
    print("wrote:")
    print(" ", static / "critical_difference.png")
    print(" ", static / "critical_difference_dark.png")
