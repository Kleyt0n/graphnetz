---
hide-toc: true
---

# GraphNetz

**A benchmark whose default output is a statistical report, not a leaderboard.**

GraphNetz turns the standard "train, eval, table of accuracies" loop into a
reproducible pipeline that always produces the artefacts a reviewer would
otherwise ask for: confidence intervals on every cell, paired pairwise tests
with multiple-comparison correction, and a Demšar critical-difference diagram
across datasets. The catalogue is organised along a *category × task*
taxonomy so the same call signature works whether you're benchmarking on
citation graphs, molecular regression, or knowledge-graph link prediction.

```{image} _static/hero-animation.svg
:alt: A graph network resolving into per-model confidence intervals.
:class: gn-hero
```

## What you get

- **One call, every metric.** `run_benchmark(category, models, seeds=...)`
  trains every compatible *(task, model, seed)* triple and returns a
  {py:class}`~graphnetz.benchmark.BenchmarkReport`.
- **Honest comparisons.** Per-cell Student's-*t* (or percentile-bootstrap)
  CIs, Holm-adjusted paired *t*-tests within each task, Friedman ranks +
  Nemenyi CD across tasks.
- **Publication artefacts.** `report.to_latex(...)`, `plot_forest()`,
  `plot_pairwise()`, `plot_critical_difference()` — Nature-styled figures
  and booktabs tables, no extra bookkeeping.
- **Pluggable models.** Decorator, class attribute, or inline tuple — your
  encoder runs through the same statistical pipeline as the built-ins.

## At a glance

| | |
|---|---|
| **Task kinds** | `node_cls` · `graph_cls` · `graph_reg` · `link_pred` |
| **Architectures** | GCN · GAT · GIN · GraphSAGE · GraphTransformer (DGI as a pre-training utility) |
| **Loaders** | 58 across 10 categories (combinatorial, biology, social, knowledge, infrastructure, finance, computing, vision, physics, security) |
| **Default report** | per-cell mean ± Student's *t*-CI · Holm-adjusted paired *t* · Demšar/Nemenyi CD |
| **Source** | [github.com/quant-sci/graphnetz](https://github.com/quant-sci/graphnetz) |

## Where to next

- **[Getting started](getting-started.md)** — install and run your first
  benchmark in five minutes.
- **[Dataset taxonomy](datasets.md)** — the full *category × task* grid and
  how to pick a loader.
- **[Models & adapters](models.md)** — built-in encoders and three ways to
  plug in your own.
- **[Benchmark protocol](benchmark.md)** — the five-stage pipeline that
  turns raw histories into a publishable report.
- **[Reading the report](report.md)** — which view to use for which
  question.
- **[Contributing](contributing.md)** — add a loader, a model, or a new
  task kind.

```{toctree}
:maxdepth: 2
:hidden:

getting-started
datasets
models
benchmark
report
contributing
api/index
```
