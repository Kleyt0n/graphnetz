"""Publication-ready matplotlib helpers.

The defaults follow figures guidelines: sans-serif Helvetica/Arial,
generous single-/double-column widths, thin axes, no top/right spines,
restrained categorical palette, vector output at 600 dpi.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Enlarged from the 89 mm / 183 mm journal columns (3.504"/7.205") by ~1.4x
# for bigger, higher-resolution figures while keeping the golden-ratio aspect.
COLUMN_INCHES: dict[str, float] = {
    "single": 5.0,
    "double": 10.0,
}


# Nature-style conventions: soft-black ink (not pure #000), normal-weight
# left-aligned panel titles, no minor-tick clutter, hairline grids, and
# generous label/tick padding so nothing touches.
_INK = "0.15"

NATURE_RC: dict[str, object] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "stixsans",
    "font.size": 9.5,
    "text.color": _INK,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "normal",
    "axes.titlelocation": "left",
    "axes.titlepad": 8.0,
    "axes.labelpad": 4.5,
    "axes.edgecolor": _INK,
    "axes.labelcolor": _INK,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.color": _INK,
    "ytick.color": _INK,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3.2,
    "ytick.major.size": 3.2,
    "xtick.major.pad": 3.0,
    "ytick.major.pad": 3.0,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.4,
    "lines.markersize": 4.0,
    "grid.color": "0.88",
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.2,
    "legend.borderaxespad": 0.6,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "savefig.transparent": False,
    "figure.dpi": 150,
    "figure.figsize": (COLUMN_INCHES["single"], COLUMN_INCHES["single"] / 1.45),
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# Ordered for maximum adjacent contrast (bright / very dark / light / mid)
# so consecutive series stay distinguishable even in greyscale print.
NATURE_COLORS: tuple[str, ...] = (
    "#0466c8",  # Smart Blue
    "#001845",  # Prussian Blue 2
    "#979dac",  # Lavender Grey
    "#33415c",  # Twilight Indigo
    "#5c677d",  # Blue Slate
    "#023e7d",  # Regal Navy
    "#7d8597",  # Slate Grey
)

# Human-readable axis/legend labels for the metric keys recorded in
# training histories.  Unknown keys fall back to a cleaned-up version of
# the raw key ("my_metric" -> "My metric") instead of leaking snake_case.
_PRETTY_METRIC: dict[str, str] = {
    "train_loss": "Training loss",
    "val_loss": "Validation loss",
    "test_loss": "Test loss",
    "train_acc": "Training accuracy",
    "val_acc": "Validation accuracy",
    "test_acc": "Test accuracy",
    "train_auc": "Training AUC",
    "val_auc": "Validation AUC",
    "test_auc": "Test AUC",
    "train_mae": "Training MAE",
    "val_mae": "Validation MAE",
    "test_mae": "Test MAE",
}


def pretty_metric(key: str) -> str:
    """Map a history metric key (``"test_acc"``) to an axis label (``"Test accuracy"``)."""
    if key in _PRETTY_METRIC:
        return _PRETTY_METRIC[key]
    return key.replace("_", " ").strip().capitalize()


def set_plot_style() -> None:
    """Apply the rcParams and color cycle."""
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
    h = (w / aspect) * nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(w, h), **kwargs)  # type: ignore[call-overload]
    return fig, axes


def panel_label(ax: plt.Axes, text: str, x: float = -0.18, y: float = 1.05) -> None:
    """Add a bold panel label (``a``, ``b``, ...) to an axis."""
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def save_figure(
    fig: plt.Figure,
    path: str | Path,
    formats: Sequence[str] = ("pdf", "png"),
    dpi: int = 600,
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
        target.plot(epochs, y, color=color, linestyle=("--" if dashed else "-"), label=pretty_metric(key))
        if std is not None and key in std:
            s = np.asarray(std[key], dtype=float)
            target.fill_between(epochs, y - s, y + s, color=color, alpha=0.15, linewidth=0)

    for i, k in enumerate(loss_keys):
        _plot(ax, k, NATURE_COLORS[i % len(NATURE_COLORS)], dashed=False)
    ax.set_xlabel("Epoch")
    if loss_keys:
        ax.set_ylabel("Loss")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    ax.margins(x=0.02)

    if metric_keys:
        ax2 = ax.twinx()
        ax2.spines.right.set_visible(True)
        ax2.tick_params(which="minor", right=False)
        for j, k in enumerate(metric_keys):
            _plot(ax2, k, NATURE_COLORS[(j + len(loss_keys)) % len(NATURE_COLORS)], dashed=True)
        ax2.set_ylabel(" / ".join(pretty_metric(k) for k in metric_keys))
        ax2.margins(x=0.02)
        lines = ax.get_lines() + ax2.get_lines()
        # One-row legend above the axes: twin-axis charts have curves in
        # every corner, so any in-axes placement collides with data.
        ax.legend(
            lines,
            [str(ln.get_label()) for ln in lines],
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(len(lines), 3),
            frameon=False,
            handlelength=1.6,
            handletextpad=0.5,
            columnspacing=1.2,
            borderaxespad=0.0,
        )
    elif loss_keys:
        ax.legend(loc=legend_loc, borderaxespad=0.4)

    if title:
        # Title above the legend row when a twin axis pushed the legend up.
        ax.set_title(title, pad=22 if metric_keys else 6)
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
    y_top = 0.0
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
            ax.errorbar(xs, ys, yerr=es, fmt="none", ecolor="0.2", elinewidth=0.8, capsize=2.0)
        if annotate:
            # Place the value above the error-bar cap (not the bar top) so
            # the text never collides with the whiskers, rotated upright so
            # neighbouring annotations cannot run into each other.
            for x, y, e in zip(xs, ys, es, strict=False):
                ax.annotate(
                    f"{y:.2f}",
                    xy=(x, y + e),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    rotation=90 if len(series) > 2 else 0,
                    color="0.25",
                )
        y_top = max(y_top, max((y + e for y, e in zip(ys, es, strict=False)), default=0.0))

    if annotate and y_top > 0:
        # Headroom so rotated annotations stay inside the axes.
        ax.set_ylim(top=y_top * (1.14 if len(series) > 2 else 1.08))
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=0, ha="center")
    ax.tick_params(axis="x", which="both", length=0)
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)

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
    "pretty_metric",
    "save_figure",
    "set_plot_style",
]
