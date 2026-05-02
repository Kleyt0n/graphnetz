# Reading the report

Every `run_benchmark` call returns a
{py:class}`~graphnetz.benchmark.BenchmarkReport`. It carries the full
per-seed history and exposes a small, opinionated set of views — each
designed to answer a different question.

## Choosing the right view

| Question | Method | Returns |
|---|---|---|
| What's the headline number per *(task, model)*? | `summary()` | DataFrame: `n_seeds, mean, std, sem, ci_low, ci_high` |
| Are these two models significantly different on this task? | `pairwise()` | DataFrame: per-pair `p_raw, p_holm, significant` (parametric or Wilcoxon) |
| Which model wins overall across many tasks? | `plot_critical_difference()` | Demšar / Nemenyi CD diagram |
| How do all tasks compare side by side? | `plot_forest()` | one row per task, models jittered within row |
| Is the win on task X driven by an outlier seed? | `plot_pairwise(layout="list")` | per-comparison CI whiskers |
| How fast did each model converge? | `plot_learning_curves()` | mean ± *t*-CI band, one panel per task |
| I want raw seed values for my own analysis. | `final_metrics()` | nested dict `task → model → [seed_values]` |

## Summary table

```python
print(report.summary())                     # default: Student's t CI
print(report.summary(method="bootstrap"))   # percentile-bootstrap CI
```

Per-(task, model) DataFrame with `n_seeds, mean, std, sem, ci_low, ci_high`.
The metric is auto-detected from the per-epoch history; call
`report.metric_name()` to confirm. Set `report.ci_method = "bootstrap"`
once if you want every downstream plot/table to follow.

## Pairwise tests

```python
print(report.pairwise(alpha=0.05))              # default: paired t-test
print(report.pairwise(alpha=0.05, method="wilcoxon"))  # non-parametric
```

Paired *t*-test (default) or Wilcoxon signed-rank test across seed-aligned
final metrics per task with Holm step-down adjustment. Columns: `task,
model_a, model_b, mean_diff, p_raw, p_holm, significant`.

The ``method`` argument accepts ``"t"`` (parametric, default) or
``"wilcoxon"`` (non-parametric).  You can also set
``report.pairwise_method = "wilcoxon"`` once to change the default for all
downstream calls.  Wilcoxon is recommended at small seed counts where the
paired *t*-test's normality assumption is most fragile (Benavoli et al.,
*JMLR* 17(5):1-36, 2016).

If every paired difference is exactly zero the signed-rank statistic is
undefined; the row reports ``NaN`` rather than an artificial ``p = 1.0``.

```{note}
``mean_diff = mean(model_a) − mean(model_b)``. For lower-is-better metrics
(MAE), a *negative* ``mean_diff`` means ``model_a`` is the better model.
```

## Forest plot

```python
fig, _ = report.plot_forest(ci=0.95, sort_within=True, band=True)
```

One row per task; models are jittered within the row. The figure scales
with the number of *tasks* — adding more models widens the within-row
jitter rather than adding new rows — so you can compare a dozen models on a
dozen tasks in a single column-width figure.

## Pairwise heatmap

```python
fig, _ = report.plot_pairwise(layout="matrix", alpha=0.05)
```

One heatmap panel per task. The lower triangle shows
$-\log_{10}(p_{\text{Holm}})$ (darker = more significant); the upper triangle
shows the signed mean difference. Switch to `layout="list"` for one row per
comparison with CI whiskers — better when you have only a few pairs.

Pass ``method="wilcoxon"`` to use the non-parametric p-values in the plot
(and optionally set ``report.pairwise_method = "wilcoxon"`` as a default).

## Critical-difference diagram

```python
fig, _ = report.plot_critical_difference(alpha=0.05)
```

A Friedman–Nemenyi diagram with average ranks across tasks and horizontal
clique bars joining models within the Nemenyi critical difference. This is
the canonical scalable view for multi-method × multi-dataset comparisons.

```{tip}
Needs ≥ 2 tasks and ≥ 2 models common to *every* task. With $N \le 4$
tasks the diagram is informative for rank ordering but rarely strong enough
to certify pairwise differences — fall back to the pairwise heatmap at that
scale.
```

## Learning curves

```python
fig, _ = report.plot_learning_curves(ci=0.95, ylabel="Test accuracy")
```

One panel per task, sharing the y-axis, with the mean ± *t*-CI band across
seeds. Useful for diagnosing under- vs over-training before you commit to a
final epoch budget.

## LaTeX exports

```python
report.to_latex("results.tex", ci=0.95, bold_best=True)
report.pairwise_to_latex("pairwise.tex")
report.pairwise_to_latex("pairwise_wilcoxon.tex", method="wilcoxon")
```

`to_latex` writes a booktabs table with the row-best in bold green; ties
get an almond-cream shade. `pairwise_to_latex` writes the Holm-adjusted
pairwise comparison table with significance markers.  Pass
``method="wilcoxon"`` to emit the non-parametric variant.

## Reading a single cell

```python
finals = report.final_metrics()                # dict[task][model] -> [v_seed_0, ...]
arr = finals["cora"]["GCN"]                    # ten seeds for GCN on Cora
mean = sum(arr) / len(arr)
```

`final_metrics()` is the canonical access path for downstream analyses
(custom plots, ad-hoc statistics, exporting to W&B / MLflow).

## Persistence

`BenchmarkReport` is a plain dataclass; pickle it as-is to cache between
sessions:

```python
import pickle
with open("report_social.pkl", "wb") as f:
    pickle.dump(report, f)
```

See `examples/experiment.py` for a small driver that caches per-category
reports under `examples/_artifacts/` and emits figures + LaTeX tables on
top of them.

## Legacy mapping access

`BenchmarkReport` implements the read-only mapping protocol so older code
that expected a plain `dict[task][model][history]` still works — at the cost
of seeing only seed 0:

```python
for task, per_task in report.items():
    for model, history in per_task.items():
        # history is the seed-0 per-epoch dict (legacy single-seed view)
        ...
```

This view is intentionally lossy. Use `report.histories` for the full
nested dict and `report.final_metrics()` for analysis-ready arrays.
