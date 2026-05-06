# Benchmark protocol

The benchmark is built around a single guarantee: **every cell in the report
goes through the same pipeline.** Whatever model and task you throw at
`run_benchmark`, the seeds are drawn the same way, the metrics are reduced
the same way, and the resulting {py:class}`~graphnetz.benchmark.BenchmarkReport`
exposes the same set of statistical methods. This page walks through the
five stages of that pipeline and how to drive it.

## The five stages

| # | Stage | What happens |
|---|---|---|
| 1 | **Catalogue** | The chosen `category` is mapped to a list of curated tasks via {py:data}`graphnetz.benchmark.BENCHMARK_TASKS`. |
| 2 | **Encoders** | Each model declares the task types it supports; incompatible *(model, task)* pairs are dropped before training. |
| 3 | **Training** | For every $(t, m, s)$ triple the runner reseeds Python `random`, NumPy, Torch CPU, and Torch CUDA, then trains for $E$ epochs through the appropriate trainer. |
| 4 | **Statistics** | Per-seed final metrics feed three reducers: per-cell mean ± CI, Holm-corrected paired *t*-tests (or Wilcoxon signed-rank) within each task, Friedman ranks + Nemenyi CD across tasks. |
| 5 | **Report** | Histories, summaries, and one-call exporters live on the {py:class}`~graphnetz.benchmark.BenchmarkReport`. |

## Running

The standard call:

```python
from graphnetz import GAT, GCN, GraphSAGE, GraphTransformer, run_benchmark

report = run_benchmark(
    category="social",
    models={"GCN": GCN, "GAT": GAT, "GraphSAGE": GraphSAGE, "GraphTransformer": GraphTransformer},
    seeds=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    task_type="node_cls",          # restrict to one task family (optional)
    epochs=100,               # override the per-task default (optional)
    only=["cora", "citeseer"],# subset of tasks (optional)
)
```

A single class instead of a dict is also accepted — handy for one-off
sanity checks:

```python
run_benchmark("infrastructure", GAT, task_type="link_pred")
```

```{tip}
The first run of a real benchmark downloads + processes datasets to
``data/benchmark/<category>/<task>/`` (overridable with ``root=``). Reruns
hit the on-disk cache, so iteration on plotting / report logic is fast.
```

## The report object

Every method below operates on the same per-seed `final_metrics()` table; the
choice of method just decides which view to render.

```{eval-rst}
.. autoclass:: graphnetz.benchmark.BenchmarkReport
   :noindex:
   :members: summary, pairwise, final_metrics, metric_name,
             to_latex, pairwise_to_latex,
             plot, plot_forest, plot_pairwise,
             plot_critical_difference, plot_learning_curves
```

### One-call publication artefacts

```python
# Mean ± t-CI table; row-best in bold green, ties in almond cream
report.to_latex("results.tex", ci=0.95, bold_best=True)

# Holm-corrected pairwise test table
report.pairwise_to_latex("pairwise.tex")

# Per-task forest plot, models jittered within rows
fig, _ = report.plot_forest(ci=0.95)

# Pairwise significance heatmap (one panel per task)
fig, _ = report.plot_pairwise(layout="matrix")

# Demšar critical-difference diagram across tasks
fig, _ = report.plot_critical_difference(alpha=0.05)
```

For interactive analysis, see [Reading the report](report.md).

## Statistical guarantees

The library is opinionated about *which* tests are appropriate for *which*
question. The table below states each one explicitly so you can cite it
without re-deriving:

**Per-cell CI (default Student's *t*)** over $n$ seeds:

$$\bar{x} \;\pm\; t_{1-\alpha/2,\,n-1}\,\dfrac{s}{\sqrt{n}}$$

For non-Gaussian metrics (Hits@K, MRR, AUC on imbalanced splits), pass
`method="bootstrap"` to {py:meth}`~graphnetz.benchmark.BenchmarkReport.summary`
or set `report.ci_method = "bootstrap"`.

**Holm pairwise.** The default is a paired *t*-test per task across
seed-aligned final metrics, then Holm step-down adjustment so the family-wise
error rate is controlled:

$$p_i^{\text{adj}} \;=\; \min\!\big(p_{(i)}\,(k - i),\, 1\big)$$

For small seed counts (typically $n < 10$) where the normality assumption of
the paired *t*-test is fragile, pass ``method="wilcoxon"`` to
{py:meth}`~graphnetz.benchmark.BenchmarkReport.pairwise`,
{py:meth}`~graphnetz.benchmark.BenchmarkReport.pairwise_to_latex`, or
{py:meth}`~graphnetz.benchmark.BenchmarkReport.plot_pairwise`, or set
``report.pairwise_method = "wilcoxon"`` to change the default for the whole
report (Benavoli et al., *JMLR* 17(5):1-36, 2016).

**Friedman + Nemenyi.** Average ranks across $N$ tasks; clique bars in the
CD diagram join models within the Nemenyi critical difference:

$$CD_\alpha \;=\; q_\alpha\,\sqrt{\dfrac{k(k+1)}{6N}}$$

The CD diagram is the canonical scalable visualisation for multi-method,
multi-dataset comparisons (Demšar, 2006); the implementation handles
heterogeneous metric directions per-task before averaging ranks.

## Reproducibility

`run_benchmark` reseeds every RNG the training code reaches —
`random.seed`, `numpy.random.seed`, `torch.manual_seed`, and
`torch.cuda.manual_seed_all` — before each `(task, model, seed)` triple.
Combinatorial loaders thread the seed through to their data generator, so
cross-seed variance reflects *both* model initialisation and data
resampling, not only the former.

A run with the same seed list and software stack reproduces bit-for-bit on
the same hardware.
