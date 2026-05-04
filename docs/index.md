---
hide-toc: true
---

# GraphNetz

**A GNN benchmark whose default output is a statistical report, not a leaderboard.**

GraphNetz turns the standard "train, eval, table of accuracies" loop into a
reproducible pipeline that produces the artefacts a reviewer would otherwise
ask for: confidence intervals on every cell, paired pairwise tests with
multiple-comparison correction, and a Demšar critical-difference diagram
across datasets.

## Install

```bash
pip install graphnetz
# or
uv add graphnetz
```

Requires Python ≥ 3.10, PyTorch ≥ 2.6, torch-geometric ≥ 2.6.

## A 30-second example

```python
from graphnetz import GAT, GCN, GraphSAGE, run_benchmark

report = run_benchmark(
    "social",
    {"GCN": GCN, "GAT": GAT, "GraphSAGE": GraphSAGE},
    seeds=range(10),
    kind="node_cls",
)

print(report.summary())          # per-(task, model) mean ± t-CI
print(report.pairwise())         # Holm-corrected paired t-tests
report.plot_critical_difference(alpha=0.05)
report.to_latex("results.tex")   # publication-ready table
```

→ Walk through this end-to-end in **[Getting started](getting-started.md)**.

## Why GraphNetz

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Honest comparisons by default
Per-cell Student's-*t* (or percentile-bootstrap) CIs, Holm-adjusted paired
*t*-tests within each task, Friedman ranks plus Nemenyi CD across tasks —
no extra bookkeeping.
:::

:::{grid-item-card} One call, every metric
`run_benchmark(category, models, seeds=...)` trains every compatible
*(task, model, seed)* triple and returns a
{py:class}`~graphnetz.benchmark.BenchmarkReport`.
:::

:::{grid-item-card} Publication-ready artefacts
`report.to_latex(...)`, `plot_forest()`, `plot_pairwise()`,
`plot_critical_difference()`
:::

:::{grid-item-card} Pluggable models
Decorator, class attribute, or inline tuple — your encoder runs through the
same statistical pipeline as the built-ins.
:::

::::

## At a glance

| | |
|---|---|
| **Task kinds** | `node_cls` · `graph_cls` · `graph_reg` · `link_pred` |
| **Architectures** | GCN · GAT · GIN · GraphSAGE · GraphTransformer (DGI as a pre-training utility) |
| **Loaders** | 63 across 10 categories (combinatorial, biology, social, knowledge, infrastructure, finance, computing, vision, physics, security) |
| **Default report** | per-cell mean ± Student's-*t* CI · Holm-adjusted paired *t* · Demšar/Nemenyi CD |
| **Source** | [github.com/quant-sci/graphnetz](https://github.com/quant-sci/graphnetz) |

## Documentation

**Start here**
- [Getting started](getting-started.md) — install and run your first benchmark in five minutes.

**Concepts**
- [Dataset taxonomy](datasets.md) — the full *category × task* grid and how to pick a loader.
- [Models & adapters](models.md) — built-in encoders and three ways to plug in your own.
- [Benchmark protocol](benchmark.md) — the five-stage pipeline that turns raw histories into a publishable report.
- [Reading the report](report.md) — which view to use for which question.

**Reference**
- [API reference](api/index) — modules, classes, and functions.
- [Contributing](contributing.md) — add a loader, a model, or a new task kind.

```{toctree}
:maxdepth: 2
:hidden:
:caption: Start here

getting-started
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Concepts

datasets
models
benchmark
report
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Reference

api/index
contributing
```
