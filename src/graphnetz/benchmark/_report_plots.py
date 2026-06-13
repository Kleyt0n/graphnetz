"""Plotting methods of :class:`BenchmarkReport`, split out as a mixin.

The mixin only references attributes/methods defined by
:class:`~graphnetz.benchmark._report.BenchmarkReport` (``histories``,
``final_metrics``, ``_ci_half``, ``metric_name``, ``pairwise``); it exists
purely to keep the statistics and the figure code in separate modules.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from graphnetz.benchmark._stats import _LOWER_IS_BETTER, _auto_metric_key
from graphnetz.plotting import NATURE_COLORS, plot_grouped_bars, set_plot_style


class _ReportPlotsMixin:
    # ----- Plotting ----------------------------------------------------------

    def plot(
        self,
        ax: plt.Axes | None = None,
        *,
        ci: float = 0.95,
        ylabel: str | None = None,
        title: str | None = None,
        annotate: bool = True,
        pretty_tasks: Mapping[str, str] | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Grouped bar chart of mean ± CI half-width across seeds."""
        finals = self.final_metrics()
        pretty = dict(pretty_tasks or {})
        values: dict[str, dict[str, float]] = {}
        errors: dict[str, dict[str, float]] = {}
        for task, per_task in finals.items():
            label = pretty.get(task, task)
            values[label] = {}
            errors[label] = {}
            for model, vals in per_task.items():
                arr = np.asarray(vals, dtype=float)
                values[label][model] = float(arr.mean())
                errors[label][model] = self._ci_half(arr, ci)
        from graphnetz.plotting import pretty_metric

        return plot_grouped_bars(
            values,
            errors=errors,
            ax=ax,
            title=title,
            ylabel=ylabel or pretty_metric(self.metric_name()),
            annotate=annotate,
        )

    def plot_forest(
        self,
        ax: plt.Axes | None = None,
        *,
        ci: float = 0.95,
        pretty_tasks: Mapping[str, str] | None = None,
        xlabel: str | None = None,
        height_per_task: float = 0.42,
        sort_within: bool = False,
        band: bool = True,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Forest plot, one row per task with models jittered within the row.

        Height scales with the number of *tasks* only -- adding more models
        widens the within-row jitter rather than adding new rows -- so the
        figure stays compact for many models.

        ``sort_within=True`` orders the jittered positions per-task so the
        best mean lands at the top of the row (helps spot leaders when there
        are many models).  Each model keeps a stable colour across tasks.

        ``band=True`` shades alternating task rows (banded reading aid).
        """
        from graphnetz.plotting import COLUMN_INCHES, pretty_metric

        set_plot_style()
        finals = self.final_metrics()
        tasks = sorted(finals)
        models = sorted({m for per in finals.values() for m in per})
        pretty = dict(pretty_tasks or {})
        n_tasks = len(tasks)
        n_models = len(models)
        metric = self.metric_name()
        lower_is_better = metric in _LOWER_IS_BETTER

        if ax is None:
            height = max(1.6, height_per_task * n_tasks + 1.0)
            fig, ax = plt.subplots(figsize=(COLUMN_INCHES["single"] * 1.05, height))
        else:
            fig = ax.figure  # type: ignore[assignment]

        jitter_span = 0.7
        slot_positions = (
            np.linspace(-jitter_span / 2, jitter_span / 2, n_models) if n_models > 1 else np.zeros(n_models)
        )

        # Precompute per-task offsets (mapping model_index -> within-row offset).
        per_task_offset: dict[str, dict[str, float]] = {}
        for task in tasks:
            present = [m for m in models if m in finals[task]]
            if sort_within and len(present) > 1:
                means = np.array([float(np.mean(finals[task][m])) for m in present])
                order = np.argsort(means if lower_is_better else -means)
                ordered = [present[i] for i in order]
            else:
                ordered = present
            row_offsets = (
                np.linspace(-jitter_span / 2, jitter_span / 2, len(ordered))
                if len(ordered) > 1
                else np.zeros(len(ordered))
            )
            per_task_offset[task] = dict(zip(ordered, row_offsets, strict=False))

        if band:
            for i in range(n_tasks):
                if i % 2 == 0:
                    ax.axhspan(
                        i - 0.5,
                        i + 0.5,
                        facecolor="0.96",
                        edgecolor="none",
                        zorder=0,
                    )

        for j, model in enumerate(models):
            xs: list[float] = []
            ys: list[float] = []
            errs: list[float] = []
            for i, task in enumerate(tasks):
                if model not in finals[task]:
                    continue
                arr = np.asarray(finals[task][model], dtype=float)
                xs.append(float(arr.mean()))
                offset = per_task_offset[task].get(model, slot_positions[j])
                ys.append(i + offset)
                errs.append(self._ci_half(arr, ci))
            if xs:
                color = NATURE_COLORS[j % len(NATURE_COLORS)]
                ax.errorbar(
                    xs,
                    ys,
                    xerr=[errs, errs],
                    fmt="o",
                    color=color,
                    ecolor=color,
                    elinewidth=1.0,
                    capsize=2.0,
                    markersize=3.5,
                    label=model,
                    zorder=3,
                )

        for i in range(n_tasks - 1):
            ax.axhline(i + 0.5, color="0.85", linewidth=0.3, zorder=1)

        ax.set_yticks(range(n_tasks))
        ax.set_yticklabels([pretty.get(t, t) for t in tasks])
        ax.set_ylim(n_tasks - 0.5, -0.5)
        ax.tick_params(axis="y", which="both", length=0)
        ax.tick_params(axis="y", which="minor", left=False)
        ax.set_xlabel(xlabel or pretty_metric(metric))
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, zorder=1)
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(n_models, 4),
            frameon=False,
            handlelength=1.2,
            handletextpad=0.4,
            columnspacing=1.0,
        )
        fig.tight_layout()
        return fig, ax

    def plot_pairwise(
        self,
        ax: plt.Axes | None = None,
        *,
        ci: float = 0.95,
        alpha: float = 0.05,
        pretty_tasks: Mapping[str, str] | None = None,
        layout: str = "matrix",
        max_cols: int = 3,
        method: str | None = None,
    ) -> tuple[plt.Figure, Any]:
        """Pairwise comparison plot, with two layouts that scale differently.

        ``layout="matrix"`` (default) -- one significance heatmap per task,
        with the lower triangle holding $-\\log_{10}(p_{\\text{Holm}})$ and the
        upper triangle holding the signed mean difference.  Scales to many
        models and many tasks (panels arranged in a grid with at most
        ``max_cols`` columns).

        ``layout="list"`` -- one row per pairwise comparison with CI whiskers
        and a significance marker.  Best for small numbers of comparisons.

        ``method`` overrides ``self.pairwise_method`` (``"t"`` or
        ``"wilcoxon"``) for this call only.
        """
        if layout == "list":
            return self._plot_pairwise_list(ax=ax, ci=ci, alpha=alpha, pretty_tasks=pretty_tasks, method=method)
        if layout == "matrix":
            return self._plot_pairwise_matrix(
                ci=ci, alpha=alpha, pretty_tasks=pretty_tasks, max_cols=max_cols, method=method
            )
        msg = f"Unknown pairwise layout: {layout!r}; choices: 'matrix', 'list'"
        raise ValueError(msg)

    def _plot_pairwise_matrix(
        self,
        *,
        ci: float = 0.95,
        alpha: float = 0.05,
        pretty_tasks: Mapping[str, str] | None = None,
        max_cols: int = 3,
        method: str | None = None,
    ) -> tuple[plt.Figure, np.ndarray]:
        from matplotlib.colors import TwoSlopeNorm

        from graphnetz.plotting import COLUMN_INCHES

        set_plot_style()
        finals = self.final_metrics()
        df = self.pairwise(alpha=alpha, method=method)
        tasks = sorted(finals)
        pretty = dict(pretty_tasks or {})
        n_tasks = len(tasks)

        # Per-task model lists (intersection used for matrix axes).
        per_task_models = {t: sorted(finals[t]) for t in tasks}
        max_models = max((len(per_task_models[t]) for t in tasks), default=0)
        if max_models < 2:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "fewer than two models per task", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            return fig, np.array([[ax]])

        cols = max(1, min(max_cols, n_tasks))
        rows = (n_tasks + cols - 1) // cols
        fig_w = COLUMN_INCHES["double"] if cols > 1 else COLUMN_INCHES["single"]
        cell = max(0.55, 1.8 / max_models)
        fig_h = (cell * max_models + 1.6) * rows
        fig, axes_obj = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)

        diff_max = max(1e-9, df["mean_diff"].abs().max() if not df.empty else 1.0)
        norm = TwoSlopeNorm(vmin=-diff_max, vcenter=0, vmax=diff_max)

        for k, task in enumerate(tasks):
            r, c = divmod(k, cols)
            ax = axes_obj[r, c]
            models_t = per_task_models[task]
            n = len(models_t)
            mat = np.full((n, n), np.nan)  # lower: -log10(p), upper: mean diff
            sub = df[df["task"] == task] if not df.empty else df
            for _, row in sub.iterrows():
                ia = models_t.index(row["model_a"])
                ib = models_t.index(row["model_b"])
                if ia == ib:
                    continue
                lo, hi = (ia, ib) if ia < ib else (ib, ia)
                p_holm = row["p_holm"]
                if not np.isnan(p_holm):
                    mat[hi, lo] = -np.log10(max(p_holm, 1e-12))
                mat[lo, hi] = row["mean_diff"] if ia < ib else -row["mean_diff"]

            mask_lower = np.tri(n, n, -1, dtype=bool)
            mask_upper = mask_lower.T

            # Two passes: lower triangle (significance), upper triangle (effect).
            lower = np.where(mask_lower, mat, np.nan)
            upper = np.where(mask_upper, mat, np.nan)

            ax.imshow(lower, cmap="Blues", vmin=0.0, vmax=3.0, aspect="equal")
            ax.imshow(upper, cmap="RdBu_r", norm=norm, aspect="equal")
            # Neutral diagonal + white separators so cells read as a grid.
            ax.imshow(
                np.where(np.eye(n, dtype=bool), 0.35, np.nan),
                cmap="Greys",
                vmin=0.0,
                vmax=1.0,
                aspect="equal",
                alpha=0.4,
            )
            ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
            ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
            ax.grid(which="minor", color="white", linewidth=1.2)

            # Annotate cells.
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if mask_lower[i, j]:
                        # significance cell: show p_holm
                        sub_match = sub[
                            ((sub["model_a"] == models_t[j]) & (sub["model_b"] == models_t[i]))
                            | ((sub["model_a"] == models_t[i]) & (sub["model_b"] == models_t[j]))
                        ]
                        if sub_match.empty:
                            continue
                        p = float(sub_match["p_holm"].iloc[0])
                        text = "n/a" if np.isnan(p) else f"{p:.2g}"
                        is_sig = (not np.isnan(p)) and p < alpha
                        if is_sig:
                            text += "*"
                        color = "white" if (not np.isnan(p) and -np.log10(max(p, 1e-12)) > 1.5) else "black"
                        weight = "bold" if is_sig else "normal"
                        ax.text(j, i, text, ha="center", va="center", fontsize=7.5, color=color, fontweight=weight)
                    elif mask_upper[i, j]:
                        d = mat[i, j]
                        if np.isnan(d):
                            continue
                        color = "white" if abs(d) > 0.6 * diff_max else "black"
                        ax.text(j, i, f"{d:+.2f}", ha="center", va="center", fontsize=7.5, color=color)

            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(models_t, rotation=35, ha="right")
            ax.set_yticklabels(models_t)
            ax.tick_params(which="both", length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(pretty.get(task, task), loc="center")

        # Hide unused panels.
        for k in range(n_tasks, rows * cols):
            r, c = divmod(k, cols)
            axes_obj[r, c].axis("off")

        # Caption-style legend strip below the panels.
        fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=2.4, w_pad=1.6)
        fig.text(
            0.5,
            0.015,
            r"Lower triangle: Holm-adjusted $p$ (darker blue = more significant, $*$: $p<\alpha$)."
            r"  Upper triangle: mean difference, row $-$ column (red $>0$, blue $<0$).",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="0.3",
        )
        return fig, axes_obj

    def _plot_pairwise_list(
        self,
        ax: plt.Axes | None = None,
        *,
        ci: float = 0.95,
        alpha: float = 0.05,
        pretty_tasks: Mapping[str, str] | None = None,
        method: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        from matplotlib.lines import Line2D

        from graphnetz.plotting import COLUMN_INCHES

        set_plot_style()
        finals = self.final_metrics()
        df = self.pairwise(alpha=alpha, method=method)
        if df.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "no pairwise comparisons", ha="center", va="center", transform=ax.transAxes)
            return fig, ax

        pretty = dict(pretty_tasks or {})
        # Group comparisons by task so each task reads as a block with a
        # single bold header instead of repeating the task in every label.
        by_task: dict[str, list[tuple[str, str, float, float, bool]]] = {}
        for _, row in df.iterrows():
            a = np.asarray(finals[row["task"]][row["model_a"]], dtype=float)
            b = np.asarray(finals[row["task"]][row["model_b"]], dtype=float)
            diff_per_seed = a - b
            mean = float(diff_per_seed.mean())
            half = self._ci_half(diff_per_seed, ci)
            by_task.setdefault(pretty.get(row["task"], row["task"]), []).append(
                (row["model_a"], row["model_b"], mean, half, bool(row["significant"]))
            )

        n_rows = sum(len(v) for v in by_task.values()) + len(by_task)  # + header rows
        if ax is None:
            fig, ax = plt.subplots(figsize=(COLUMN_INCHES["single"] * 1.1, 0.3 * n_rows + 1.0))
        else:
            fig = ax.figure  # type: ignore[assignment]

        ytick_positions: list[float] = []
        ytick_labels: list[str] = []
        header_positions: list[tuple[float, str]] = []
        y = 0.0
        for t_idx, (task_label, comparisons) in enumerate(by_task.items()):
            header_positions.append((y, task_label))
            y += 1.0
            for ma, mb, mean, half, sig in comparisons:
                color = NATURE_COLORS[0] if sig else NATURE_COLORS[2]
                ax.errorbar(
                    mean,
                    y,
                    xerr=[[half], [half]],
                    fmt="o" if sig else "s",
                    color=color,
                    ecolor=color,
                    elinewidth=1.2,
                    capsize=2.2,
                    markersize=4.5 if sig else 3.5,
                )
                ytick_positions.append(y)
                ytick_labels.append(f"{ma} vs {mb}")
                y += 1.0
            if t_idx < len(by_task) - 1:
                ax.axhline(y - 0.5, color="0.88", linewidth=0.6)

        ax.axvline(0, color="0.4", linewidth=0.6, linestyle="--")
        ax.set_yticks(ytick_positions)
        ax.set_yticklabels(ytick_labels)
        # Bold task headers in the label gutter, aligned with their block.
        for hy, label in header_positions:
            ax.text(
                -0.02,
                hy,
                label,
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=9.5,
                fontweight="bold",
            )
        ax.set_ylim(y - 0.5, -0.5)
        ax.tick_params(axis="y", which="both", length=0)
        ax.tick_params(axis="y", which="minor", left=False)
        ax.set_xlabel(rf"Mean difference ({int(ci * 100)}% CI, paired)")
        ax.set_axisbelow(True)
        ax.xaxis.grid(True)

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color=NATURE_COLORS[0],
                linestyle="",
                markersize=4.5,
                label=rf"$p_{{\mathrm{{Holm}}}} < {alpha}$",
            ),
            Line2D([0], [0], marker="s", color=NATURE_COLORS[2], linestyle="", markersize=3.5, label="not significant"),
        ]
        ax.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            frameon=False,
            handlelength=1.2,
            handletextpad=0.4,
        )
        fig.tight_layout()
        return fig, ax

    def plot_critical_difference(
        self,
        *,
        alpha: float = 0.05,
        title: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        r"""Demšar critical-difference (CD) diagram.

        Computes mean ranks of every model across tasks and overlays the
        Nemenyi critical difference at level ``alpha``.  Models within
        ``CD`` of each other are joined by a thick horizontal "clique" bar
        (i.e., not significantly different).  This is the canonical
        scalable visualization for multi-method, multi-dataset benchmarks
        (Demšar, 2006).

        Only models present in *every* task are included.  Requires at
        least two tasks and at least two such models.
        """
        from scipy.stats import studentized_range

        from graphnetz.plotting import COLUMN_INCHES

        set_plot_style()
        finals = self.final_metrics()

        common: set[str] = set.intersection(*[set(per.keys()) for per in finals.values()]) if finals else set()
        if len(common) < 2 or len(finals) < 2:
            fig, ax = plt.subplots(figsize=(COLUMN_INCHES["single"], 1.6))
            ax.text(
                0.5,
                0.5,
                "CD diagram needs >= 2 tasks and >= 2 models common to all tasks",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
            )
            ax.axis("off")
            return fig, ax

        models = sorted(common)
        tasks = sorted(finals)
        means = np.array([[float(np.mean(finals[t][m])) for m in models] for t in tasks])
        # Direction (lower-is-better) is detected *per task* so the CD
        # diagram is correct on heterogeneous benchmarks where some tasks
        # use accuracy (higher better) and others use loss (lower better).
        rows: list[np.ndarray] = []
        for i, task in enumerate(tasks):
            sample = next(iter(self.histories[task].values()))[0]
            task_metric = _auto_metric_key(sample)
            sign = 1.0 if task_metric in _LOWER_IS_BETTER else -1.0
            rows.append(stats.rankdata(sign * means[i], method="average"))
        ranks = np.array(rows)
        avg_ranks = ranks.mean(axis=0)
        # Ranks are always lower-is-better by construction.

        k = len(models)
        n = len(tasks)
        # Friedman omnibus: only interpret Nemenyi after the global null is
        # rejected (Demšar, 2006). We compute it from the same rank table.
        avg_for_chi2 = ranks.mean(axis=0)
        chi2 = (12.0 * n) / (k * (k + 1)) * (float(np.sum(avg_for_chi2**2)) - k * (k + 1) ** 2 / 4.0)
        friedman_p = float(stats.chi2.sf(chi2, df=k - 1))
        friedman_rejected = friedman_p < alpha
        q = float(studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2))
        cd = q * float(np.sqrt(k * (k + 1) / (6 * n)))

        order = np.argsort(avg_ranks)
        sorted_models = [models[i] for i in order]
        sorted_ranks = avg_ranks[order]

        # Maximal cliques: contiguous runs in rank order whose span < CD.
        cliques_raw: list[tuple[int, int]] = []
        i = 0
        while i < k:
            j = i
            while j + 1 < k and sorted_ranks[j + 1] - sorted_ranks[i] < cd:
                j += 1
            if j > i:
                cliques_raw.append((i, j))
            i += 1
        cliques: list[tuple[int, int]] = []
        for a, b in sorted(set(cliques_raw)):
            if any(c <= a and b <= d for c, d in cliques):
                continue
            cliques = [(c, d) for c, d in cliques if not (a <= c and d <= b)]
            cliques.append((a, b))

        # Layout coordinates.
        fig_w = COLUMN_INCHES["double"]
        fig_h = max(2.2, 1.6 + 0.22 * k)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        rank_y = 0.0
        x_min, x_max = 1.0, float(k)
        ax.plot([x_min, x_max], [rank_y, rank_y], color="black", linewidth=0.8)
        for r in range(int(x_min), int(x_max) + 1):
            ax.plot([r, r], [rank_y, rank_y - 0.04], color="black", linewidth=0.6)
            ax.text(r, rank_y - 0.08, f"{r}", ha="center", va="top", fontsize=8)

        # Method leaders + side labels (left for top half, right for bottom half).
        half = (k + 1) // 2
        label_y_step = 0.16
        label_y_top = 0.32
        label_x_left = x_min - 0.5
        label_x_right = x_max + 0.5
        for idx, (model, r) in enumerate(zip(sorted_models, sorted_ranks, strict=False)):
            color = NATURE_COLORS[idx % len(NATURE_COLORS)]
            if idx < half:
                label_x = label_x_left
                ha = "right"
                ly = label_y_top + (half - idx - 1) * label_y_step
            else:
                label_x = label_x_right
                ha = "left"
                ly = label_y_top + (idx - half) * label_y_step
            ax.plot([r, r], [rank_y, ly], color="0.55", linewidth=0.5, zorder=1)
            ax.plot([r, label_x], [ly, ly], color="0.55", linewidth=0.5, zorder=1)
            ax.plot([r], [rank_y], marker="o", markersize=5.0, color=color, zorder=2)
            # Model name in ink for readability; the rank in muted grey.
            ax.text(
                label_x + (-0.05 if ha == "right" else 0.05),
                ly,
                f"{model} ({r:.2f})",
                va="center",
                ha=ha,
                fontsize=9,
                color="0.15",
            )

        # Clique bars below the rank axis (start below the tick labels).
        bar_y = rank_y - 0.16
        for a, b in cliques:
            ax.plot(
                [sorted_ranks[a] - 0.06, sorted_ranks[b] + 0.06],
                [bar_y, bar_y],
                color="black",
                linewidth=3.5,
                solid_capstyle="round",
                zorder=3,
            )
            bar_y -= 0.06

        # CD scale at the top.
        cd_y = label_y_top + max(half - 1, 0) * label_y_step + 0.22
        ax.plot([x_min, x_min + cd], [cd_y, cd_y], color="black", linewidth=1.0)
        ax.plot([x_min, x_min], [cd_y - 0.025, cd_y + 0.025], color="black", linewidth=1.0)
        ax.plot(
            [x_min + cd, x_min + cd],
            [cd_y - 0.025, cd_y + 0.025],
            color="black",
            linewidth=1.0,
        )
        ax.text(
            x_min + cd / 2,
            cd_y + 0.04,
            rf"CD = {cd:.3f} (Nemenyi, $\alpha={alpha}$, $k={k}$, $N={n}$)",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        friedman_color = "0.15" if friedman_rejected else "0.4"
        ax.text(
            x_min + cd / 2,
            cd_y + 0.18,
            rf"Friedman $\chi^2_{{{k - 1}}} = {chi2:.2f}$, $p = {friedman_p:.3g}$"
            + (" (reject)" if friedman_rejected else " (do not reject)"),
            ha="center",
            va="bottom",
            fontsize=7,
            color=friedman_color,
        )

        # Direction caption below all clique bars.
        caption_y = bar_y - 0.04
        ax.text(
            (x_min + x_max) / 2,
            caption_y,
            "Mean rank (lower rank = better)",
            ha="center",
            va="top",
            fontsize=8,
            color="0.3",
        )

        ax.set_xlim(label_x_left - 1.2, label_x_right + 1.2)
        ax.set_ylim(caption_y - 0.12, cd_y + 0.2)
        ax.axis("off")
        if title is not None:
            ax.set_title(title)
        fig.tight_layout()
        return fig, ax

    def plot_learning_curves(
        self,
        *,
        ci: float = 0.95,
        metric_key: str | None = None,
        pretty_tasks: Mapping[str, str] | None = None,
        ylabel: str | None = None,
        max_cols: int = 4,
    ) -> tuple[plt.Figure, np.ndarray]:
        """Mean ± t-CI learning curves, one panel per task, sharing y-axis.

        Panels wrap into rows of at most ``max_cols`` so individual panels
        stay readable on benchmarks with many tasks; a single legend is
        shared by all panels below the figure.
        """
        set_plot_style()
        from string import ascii_lowercase

        from graphnetz.plotting import COLUMN_INCHES, panel_label, pretty_metric

        tasks = list(self.histories)
        n_tasks = max(len(tasks), 1)
        ncols = max(1, min(max_cols, n_tasks))
        nrows = (n_tasks + ncols - 1) // ncols
        width = COLUMN_INCHES["double"]
        panel_w = width / ncols
        height = panel_w / 1.3 * nrows + 0.35  # + room for the shared legend
        fig, axes_obj = plt.subplots(nrows, ncols, figsize=(width, height), sharey=True, squeeze=False)
        axes = axes_obj.ravel()
        pretty = dict(pretty_tasks or {})
        resolved_key = metric_key
        for idx, task in enumerate(tasks):
            ax = axes[idx]
            per_task = self.histories[task]
            for j, model in enumerate(per_task):
                seed_histories = per_task[model]
                if not seed_histories:
                    continue
                key = metric_key or _auto_metric_key(seed_histories[0])
                resolved_key = resolved_key or key
                arr = np.array([h[key] for h in seed_histories], dtype=float)
                mean = arr.mean(axis=0)
                n = arr.shape[0]
                if n > 1:
                    sem = arr.std(axis=0, ddof=1) / np.sqrt(n)
                    half = sem * stats.t.ppf((1 + ci) / 2, n - 1)
                else:
                    half = np.zeros_like(mean)
                epochs_axis = np.arange(1, mean.size + 1)
                color = NATURE_COLORS[j % len(NATURE_COLORS)]
                ax.plot(epochs_axis, mean, color=color, label=model, linewidth=1.4)
                ax.fill_between(epochs_axis, mean - half, mean + half, color=color, alpha=0.18, linewidth=0)
            if idx // ncols == nrows - 1 or idx + ncols >= n_tasks:
                ax.set_xlabel("Epoch")
            ax.set_title(pretty.get(task, task))
            ax.set_axisbelow(True)
            ax.yaxis.grid(True)
            ax.margins(x=0.02)
            if idx % ncols == 0:
                ax.set_ylabel(ylabel or pretty_metric(resolved_key or "metric"))
            else:
                ax.tick_params(labelleft=False)
            if n_tasks > 1:
                panel_label(ax, ascii_lowercase[idx % 26], x=-0.08 if idx % ncols else -0.18)
        for idx in range(n_tasks, nrows * ncols):
            axes[idx].axis("off")

        # One shared legend below all panels instead of crowding panel one.
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=min(len(labels), 5),
                frameon=False,
                handlelength=1.8,
                columnspacing=1.4,
                bbox_to_anchor=(0.5, 0.0),
            )
        fig.tight_layout(rect=(0, 0.35 / height, 1, 1), h_pad=2.2, w_pad=1.2)
        return fig, axes_obj if nrows > 1 else np.atleast_1d(axes_obj[0])
