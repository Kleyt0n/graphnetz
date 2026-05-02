"""Publication-ready Nature-style matplotlib helpers.

The defaults follow Nature figure guidelines: sans-serif Helvetica/Arial,
single-column width 89 mm and double-column 183 mm, thin axes, no top/right
spines, restrained categorical palette, vector output at 300 dpi.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Single-column = 89 mm; double-column = 183 mm; default golden-ratio aspect.
COLUMN_INCHES: dict[str, float] = {
    "single": 3.504,
    "double": 7.205,
}


NATURE_RC: dict[str, object] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "stixsans",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "axes.titleweight": "bold",
    "axes.labelpad": 2.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 1.6,
    "ytick.minor.size": 1.6,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.2,
    "lines.markersize": 3.0,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.0,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "savefig.transparent": False,
    "figure.dpi": 120,
    "figure.figsize": (COLUMN_INCHES["single"], COLUMN_INCHES["single"] / 1.45),
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

NATURE_COLORS: tuple[str, ...] = (
    "#22333B",  # Jet Black
    "#5E503F",  # Stone Brown
    "#C6AC8F",  # Khaki Beige
    "#0A0908",  # Black
    "#EAE0D5",  # Almond Cream
)


def set_plot_style() -> None:
    """Apply the Nature-style rcParams and color cycle."""
    from cycler import cycler

    mpl.rcParams.update(NATURE_RC)
    mpl.rcParams["axes.prop_cycle"] = cycler(color=list(NATURE_COLORS))


def figure(
    width: str | float = "single",
    aspect: float = 1.45,
    nrows: int = 1,
    ncols: int = 1,
    **kwargs: object,
) -> tuple[plt.Figure, np.ndarray | plt.Axes]:
    """Create a sized figure.

    ``width`` is either ``"single"``, ``"double"`` or a float in inches.
    """
    set_plot_style()
    w = COLUMN_INCHES[width] if isinstance(width, str) else float(width)
    h = w / aspect * (nrows / max(ncols, 1)) ** 0.0  # default height per row
    h = (w / aspect) * nrows / max(1, 1)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(w, h), **kwargs)  # type: ignore[call-overload]
    return fig, axes


def panel_label(ax: plt.Axes, text: str, x: float = -0.18, y: float = 1.05) -> None:
    """Add a bold panel label (``a``, ``b``, ...) to an axis."""
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def save_figure(
    fig: plt.Figure,
    path: str | Path,
    formats: Sequence[str] = ("pdf", "png"),
    dpi: int = 300,
) -> list[Path]:
    """Save ``fig`` to one path stem in multiple formats; returns the saved paths."""
    base = Path(path).with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for fmt in formats:
        target = base.with_suffix(f".{fmt}")
        fig.savefig(target, dpi=dpi)
        out.append(target)
    return out


def _epochs_axis(values: Sequence[float]) -> np.ndarray:
    return np.arange(1, len(values) + 1)


def plot_history(
    history: Mapping[str, Sequence[float]],
    ax: plt.Axes | None = None,
    title: str | None = None,
    std: Mapping[str, Sequence[float]] | None = None,
    legend_loc: str = "best",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a training history dict.

    ``loss``-keys go on the primary axis; metric-keys on a twin axis with
    dashed lines. ``std`` (optional) provides per-epoch standard deviation
    rendered as a translucent band.
    """
    set_plot_style()
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure  # type: ignore[assignment]

    loss_keys = [k for k in history if "loss" in k.lower()]
    metric_keys = [k for k in history if k not in loss_keys]
    epochs = _epochs_axis(next(iter(history.values())))

    def _plot(target: plt.Axes, key: str, color: str, dashed: bool) -> None:
        y = np.asarray(history[key], dtype=float)
        target.plot(epochs, y, color=color, linestyle=("--" if dashed else "-"), label=key)
        if std is not None and key in std:
            s = np.asarray(std[key], dtype=float)
            target.fill_between(epochs, y - s, y + s, color=color, alpha=0.15, linewidth=0)

    for i, k in enumerate(loss_keys):
        _plot(ax, k, NATURE_COLORS[i % len(NATURE_COLORS)], dashed=False)
    ax.set_xlabel("Epoch")
    if loss_keys:
        ax.set_ylabel("Loss")

    if metric_keys:
        ax2 = ax.twinx()
        ax2.spines.right.set_visible(True)
        for j, k in enumerate(metric_keys):
            _plot(ax2, k, NATURE_COLORS[(j + len(loss_keys)) % len(NATURE_COLORS)], dashed=True)
        ax2.set_ylabel("Metric")
        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [str(ln.get_label()) for ln in lines], loc=legend_loc, borderaxespad=0.4)
    elif loss_keys:
        ax.legend(loc=legend_loc, borderaxespad=0.4)

    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_grouped_bars(
    values: Mapping[str, Mapping[str, float]],
    errors: Mapping[str, Mapping[str, float]] | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    ylabel: str = "metric",
    annotate: bool = True,
    legend_loc: str = "outside bottom",
    legend_ncol: int | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Grouped bar chart from a ``{group: {series: value}}`` mapping.

    Optional ``errors`` of the same shape draws symmetric error bars.
    """
    set_plot_style()
    groups = list(values)
    series: list[str] = []
    for per_group in values.values():
        for s in per_group:
            if s not in series:
                series.append(s)

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(COLUMN_INCHES["single"], 0.7 * len(groups) + 1.0), 2.4))
    else:
        fig = ax.figure  # type: ignore[assignment]

    width = 0.8 / max(len(series), 1)
    for j, s in enumerate(series):
        xs: list[float] = []
        ys: list[float] = []
        es: list[float] = []
        for i, g in enumerate(groups):
            if s in values[g]:
                xs.append(i + j * width - 0.4 + width / 2)
                ys.append(values[g][s])
                if errors is not None and s in errors.get(g, {}):
                    es.append(errors[g][s])
                else:
                    es.append(0.0)
        ax.bar(
            xs,
            ys,
            width=width,
            label=s,
            color=NATURE_COLORS[j % len(NATURE_COLORS)],
            edgecolor="white",
            linewidth=0.4,
        )
        if any(e > 0 for e in es):
            ax.errorbar(xs, ys, yerr=es, fmt="none", ecolor="black", elinewidth=0.6, capsize=1.6)
        if annotate:
            for x, y in zip(xs, ys, strict=False):
                ax.text(x, y, f"{y:.2f}", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=0, ha="center")
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linewidth=0.3, alpha=0.4)

    ncol = legend_ncol or min(len(series), 4)
    if legend_loc == "outside top":
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=ncol,
            frameon=False,
            handlelength=1.4,
            handletextpad=0.4,
            columnspacing=1.2,
            borderaxespad=0.0,
        )
    elif legend_loc == "outside bottom":
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=ncol,
            frameon=False,
            handlelength=1.4,
            handletextpad=0.4,
            columnspacing=1.2,
            borderaxespad=0.0,
        )
    elif legend_loc == "outside right":
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            ncol=1,
            frameon=False,
        )
    else:
        ax.legend(loc=legend_loc, ncol=ncol)

    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax


__all__ = [
    "COLUMN_INCHES",
    "NATURE_COLORS",
    "NATURE_RC",
    "figure",
    "panel_label",
    "plot_grouped_bars",
    "plot_history",
    "save_figure",
    "set_plot_style",
]
