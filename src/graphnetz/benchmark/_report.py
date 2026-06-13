"""The :class:`BenchmarkReport` container: statistics and LaTeX tables."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from graphnetz.benchmark._report_plots import _ReportPlotsMixin
from graphnetz.benchmark._stats import (
    _LOWER_IS_BETTER,
    _auto_metric_key,
    _holm_correction,
    _paired_pvalue,
    _resolve_ci_half,
)

# --------------------------------------------------------------------------- #
# Benchmark report
# --------------------------------------------------------------------------- #


@dataclass
class BenchmarkReport(_ReportPlotsMixin):
    """Structured outcome of a multi-seed benchmark run.

    ``histories[task][model]`` is a list with one history dict per seed (in
    seed order). The report is also a read-only mapping ``task -> {model:
    history_seed_0}`` for backward compatibility with single-seed callers.
    """

    seeds: tuple[int, ...]
    histories: dict[str, dict[str, list[dict[str, list[float]]]]]
    config: dict[str, Any] = field(default_factory=dict)
    ci_method: str = "t"
    bootstrap_n: int = 10000
    bootstrap_seed: int = 0
    pairwise_method: str = "t"

    def _ci_half(
        self,
        values: np.ndarray,
        ci: float,
        method: str | None = None,
    ) -> float:
        return _resolve_ci_half(
            values,
            ci,
            method or self.ci_method,
            self.bootstrap_n,
            self.bootstrap_seed,
        )

    # ----- Pickle compatibility ---------------------------------------------

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore from pickle, backfilling fields added since serialisation.

        Older :class:`BenchmarkReport` pickles predate the ``ci_method`` /
        ``bootstrap_*`` / ``pairwise_method`` fields. ``__setstate__``
        ensures they load cleanly with sensible defaults so the experiment
        cache (``_cache_*.pkl``) survives library upgrades.
        """
        self.__dict__.update(state)
        self.__dict__.setdefault("ci_method", "t")
        self.__dict__.setdefault("bootstrap_n", 10000)
        self.__dict__.setdefault("bootstrap_seed", 0)
        self.__dict__.setdefault("pairwise_method", "t")
        self.__dict__.setdefault("config", {})

    # ----- Mapping protocol (backward compat with the legacy dict shape) -----

    def __iter__(self):
        return iter(self.histories)

    def __len__(self) -> int:
        return len(self.histories)

    def __getitem__(self, task_type: str) -> dict[str, dict[str, list[float]]]:
        per_task = self.histories[task_type]
        return {model: per_task[model][0] for model in per_task}

    def items(self):
        for task in self.histories:
            yield task, self[task]

    def keys(self):
        return self.histories.keys()

    def values(self):
        return [self[task] for task in self.histories]

    # ----- Statistics --------------------------------------------------------

    def final_metrics(self, key: str | None = None) -> dict[str, dict[str, list[float]]]:
        """Final metric value per (task, model, seed)."""
        out: dict[str, dict[str, list[float]]] = {}
        for task, per_task in self.histories.items():
            out[task] = {}
            for model, seed_histories in per_task.items():
                vals: list[float] = []
                for h in seed_histories:
                    k = key or _auto_metric_key(h)
                    vals.append(float(h[k][-1]))
                out[task][model] = vals
        return out

    def metric_name(self) -> str:
        for per_task in self.histories.values():
            for seed_histories in per_task.values():
                if seed_histories:
                    return _auto_metric_key(seed_histories[0])
        return "metric"

    def summary(self, ci: float = 0.95, method: str | None = None) -> pd.DataFrame:
        """Per-(task, model) mean, std, sem, CI half-width and bounds.

        ``method`` overrides ``self.ci_method`` for this call only; choose
        ``"t"`` for Student's-t intervals (default) or ``"bootstrap"`` for
        percentile-bootstrap intervals (better for non-Gaussian metrics
        such as Hits@K, MRR, or AUC).
        """
        rows = []
        for task, per_task in self.final_metrics().items():
            for model, values in per_task.items():
                arr = np.asarray(values, dtype=float)
                mean = float(arr.mean())
                std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
                sem = float(stats.sem(arr)) if arr.size > 1 else 0.0
                half = self._ci_half(arr, ci, method=method)
                rows.append(
                    {
                        "task": task,
                        "model": model,
                        "n_seeds": arr.size,
                        "mean": mean,
                        "std": std,
                        "sem": sem,
                        "ci_low": mean - half,
                        "ci_high": mean + half,
                    }
                )
        return pd.DataFrame(rows).set_index(["task", "model"]).sort_index()

    def pairwise(self, alpha: float = 0.05, method: str | None = None) -> pd.DataFrame:
        """Paired pairwise tests between models per task with Holm adjustment.

        ``method`` overrides ``self.pairwise_method`` for this call only:

        - ``"t"`` (default) -- paired Student's t-test on per-seed final metrics.
        - ``"wilcoxon"`` -- non-parametric Wilcoxon signed-rank test on the
          paired differences. Recommended at small seed counts where the
          paired t-test's normality assumption is most fragile; see
          Benavoli et al., *JMLR* 17(5):1-36, 2016.
        """
        finals = self.final_metrics()
        test = method or self.pairwise_method
        rows = []
        for task, per_task in finals.items():
            models = sorted(per_task)
            pairs: list[tuple[str, str, float, float]] = []
            ps: list[float] = []
            for i, model_a in enumerate(models):
                for model_b in models[i + 1 :]:
                    a = np.asarray(per_task[model_a], dtype=float)
                    b = np.asarray(per_task[model_b], dtype=float)
                    p = _paired_pvalue(a, b, test)
                    pairs.append((model_a, model_b, float(a.mean() - b.mean()), p))
                    ps.append(p)
            adj = _holm_correction(np.asarray(ps, dtype=float))
            for (model_a, model_b, diff, p_raw), p_holm in zip(pairs, adj, strict=False):
                rows.append(
                    {
                        "task": task,
                        "model_a": model_a,
                        "model_b": model_b,
                        "mean_diff": diff,
                        "p_raw": p_raw,
                        "p_holm": p_holm,
                        "significant": (not np.isnan(p_holm)) and p_holm < alpha,
                    }
                )
        return pd.DataFrame(rows)

    def friedman(self, alpha: float = 0.05) -> dict[str, float | int | bool]:
        r"""Friedman omnibus test on per-task ranks of seed-mean metrics.

        Returns a dict with the statistic ``chi2``, the asymptotic
        $\chi^2_{k-1}$ p-value, the rejection flag at ``alpha``, and the
        $(k, N)$ shape used. The Nemenyi post-hoc surfaced in
        :meth:`plot_critical_difference` should only be interpreted when
        ``rejected`` is true (Demšar, 2006).

        Only models present in every task are included; per-task ranks
        use the metric direction (lower-is-better for ``val_mae`` and
        ``train_loss``).
        """
        finals = self.final_metrics()
        if not finals:
            return {"chi2": float("nan"), "p_value": float("nan"), "k": 0, "n": 0, "rejected": False}
        common: set[str] = set.intersection(*[set(per.keys()) for per in finals.values()])
        if len(common) < 2 or len(finals) < 2:
            return {
                "chi2": float("nan"),
                "p_value": float("nan"),
                "k": len(common),
                "n": len(finals),
                "rejected": False,
            }
        models = sorted(common)
        tasks = sorted(finals)
        means = np.array([[float(np.mean(finals[t][m])) for m in models] for t in tasks])
        rows: list[np.ndarray] = []
        for i, task in enumerate(tasks):
            sample = next(iter(self.histories[task].values()))[0]
            sign = 1.0 if _auto_metric_key(sample) in _LOWER_IS_BETTER else -1.0
            rows.append(stats.rankdata(sign * means[i], method="average"))
        ranks = np.array(rows)
        k = len(models)
        n = len(tasks)
        avg = ranks.mean(axis=0)
        chi2 = (12.0 * n) / (k * (k + 1)) * (float(np.sum(avg**2)) - k * (k + 1) ** 2 / 4.0)
        p = float(stats.chi2.sf(chi2, df=k - 1))
        return {"chi2": float(chi2), "p_value": p, "k": k, "n": n, "rejected": bool(p < alpha)}

    # ----- Reporting helpers -------------------------------------------------

    def _best_per_task(self) -> dict[str, str]:
        finals = self.final_metrics()
        metric = self.metric_name()
        lower_is_better = metric in _LOWER_IS_BETTER
        best: dict[str, str] = {}
        for task, per_task in finals.items():
            scored = [(model, float(np.mean(values))) for model, values in per_task.items()]
            if lower_is_better:
                best[task] = min(scored, key=lambda x: x[1])[0]
            else:
                best[task] = max(scored, key=lambda x: x[1])[0]
        return best

    def to_latex(
        self,
        path: str | Path,
        *,
        ci: float = 0.95,
        bold_best: bool = True,
        pretty_tasks: Mapping[str, str] | None = None,
        caption: str | None = None,
        label: str | None = None,
        method: str | None = None,
    ) -> Path:
        """Booktabs LaTeX table of mean ± CI half-width with bold-best per task.

        ``method`` overrides ``self.ci_method`` (``"t"`` or ``"bootstrap"``).
        """
        finals = self.final_metrics()
        tasks = sorted(finals)
        models = sorted({m for per in finals.values() for m in per})
        best = self._best_per_task() if bold_best else {}
        pretty = dict(pretty_tasks or {})

        lines: list[str] = []
        if caption is not None or label is not None:
            lines.extend([r"\begin{table}[t]", r"  \centering"])
            if caption is not None:
                lines.append(rf"  \caption{{{caption}}}")
            if label is not None:
                lines.append(rf"  \label{{{label}}}")
        lines.append(r"\begin{tabular}{l" + "c" * len(tasks) + "}")
        lines.append(r"\toprule")
        header = "Model & " + " & ".join(pretty.get(t, t) for t in tasks) + r" \\"
        lines.append(header)
        lines.append(r"\midrule")
        for model in models:
            cells = []
            for task in tasks:
                values = np.asarray(finals[task].get(model, []), dtype=float)
                if values.size == 0:
                    cells.append("--")
                    continue
                mean = float(values.mean())
                half = self._ci_half(values, ci, method=method)
                if bold_best and best.get(task) == model:
                    cell = rf"$\mathbf{{{mean:.3f} \pm {half:.3f}}}$"
                else:
                    cell = rf"${mean:.3f} \pm {half:.3f}$"
                cells.append(cell)
            lines.append(f"{model} & " + " & ".join(cells) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        if caption is not None or label is not None:
            lines.append(r"\end{table}")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n")
        return out

    def pairwise_to_latex(
        self,
        path: str | Path,
        *,
        alpha: float = 0.05,
        caption: str | None = None,
        label: str | None = None,
        method: str | None = None,
    ) -> Path:
        """LaTeX booktabs table of pairwise Holm-adjusted p-values.

        ``method`` overrides ``self.pairwise_method`` (``"t"`` or
        ``"wilcoxon"``) for this call only.
        """
        df = self.pairwise(alpha=alpha, method=method)
        lines: list[str] = []
        if caption is not None or label is not None:
            lines.extend([r"\begin{table}[t]", r"  \centering"])
            if caption is not None:
                lines.append(rf"  \caption{{{caption}}}")
            if label is not None:
                lines.append(rf"  \label{{{label}}}")
        lines.append(r"\begin{tabular}{llcccl}")
        lines.append(r"\toprule")
        lines.append(r"Task & Comparison & $\Delta\mu$ & $p_{\text{raw}}$ & $p_{\text{Holm}}$ & Sig. \\")
        lines.append(r"\midrule")
        for _, row in df.iterrows():
            sig = r"\textbf{*}" if row["significant"] else ""
            p_raw = "n/a" if pd.isna(row["p_raw"]) else f"{row['p_raw']:.3g}"
            p_holm = "n/a" if pd.isna(row["p_holm"]) else f"{row['p_holm']:.3g}"
            lines.append(
                f"{row['task']} & {row['model_a']} vs.\\ {row['model_b']} & "
                f"${row['mean_diff']:+.3f}$ & {p_raw} & {p_holm} & {sig} \\\\"
            )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        if caption is not None or label is not None:
            lines.append(r"\end{table}")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n")
        return out
